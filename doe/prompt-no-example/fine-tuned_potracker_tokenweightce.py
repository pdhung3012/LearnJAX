# pip install -U bitsandbytes peft accelerate transformers datasets peft nltk
import os
import torch
import pandas as pd
from datasets import Dataset
from doe.combined_metrics import *
import torch.nn.functional as F

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
)

from peft import LoraConfig, get_peft_model

# from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

model_name = "/home/hungphd/git/pretrained_open_llms/Qwen2.5-7B-Instruct/"
folder_output = "/home/hungphd/git/potracker_tokenweightce/"

fp_file_tuning_train = "../data-all/label-split/finetune_noex_train.csv"
fp_file_tuning_valid = "../data-all/label-split/finetune_noex_valid.csv"
num_samples=100
df_train = pd.read_csv(fp_file_tuning_train, dtype=str, keep_default_na=False, na_filter=False)#.head(num_samples)
df_valid = pd.read_csv(fp_file_tuning_valid, dtype=str, keep_default_na=False, na_filter=False)#.head(num_samples)

for df in (df_train, df_valid):
    df["prompt"] = df["prompt"].fillna("").astype(str)
    df["response"] = df["response"].fillna("").astype(str)

train_ds = Dataset.from_pandas(df_train[["prompt", "response"]], preserve_index=False)
valid_ds = Dataset.from_pandas(df_valid[["prompt", "response"]], preserve_index=False)

real_model_name = model_name.split("/")[-2]
fop_output_model = folder_output + real_model_name + "/"

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

MAX_LEN = 2048

import re
import torch
import torch.nn.functional as F
from transformers import Trainer

_XML_TAG_RE = re.compile(r"<|>|</|/>")

def build_token_weight_vector(tokenizer,
                              w_tag: float = 8.0,
                              w_slash: float = 5.0,
                              w_attr: float = 3.0,
                              w_default: float = 1.0) -> torch.Tensor:
    """
    Returns a (vocab_size,) float32 tensor of weights.
    Heuristic based on token text; no dataset changes needed.
    """
    vocab_size = len(tokenizer)
    weights = torch.full((vocab_size,), float(w_default), dtype=torch.float32)

    # Fast-ish: token string via convert_ids_to_tokens. (decode is slower)
    for tid in range(vocab_size):
        s = tokenizer.convert_ids_to_tokens(tid)
        if s is None:
            s = ""

        # Many tokenizers include special markers like Ġ / ▁ — keep them, just search symbols.
        if "<" in s or ">" in s:
            weights[tid] = float(w_tag)
        elif "/" in s:
            weights[tid] = float(w_slash)
        elif "=" in s or "\"" in s or "'" in s:
            weights[tid] = float(w_attr)

    # Make special tokens neutral (optional)
    for tok_id in getattr(tokenizer, "all_special_ids", []):
        weights[tok_id] = float(w_default)

    return weights


def format_prompt(p: str) -> str:
    return f"### Instruction:\n{p}\n\n### Response:\n"

def tokenize_prompt_response(example):
    prompt_text = format_prompt(example["prompt"])
    response_text = example["response"] + tokenizer.eos_token

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    resp_ids   = tokenizer(response_text, add_special_tokens=False)["input_ids"]

    # Truncate to MAX_LEN while prioritizing keeping response
    if len(prompt_ids) + len(resp_ids) > MAX_LEN:
        if len(resp_ids) >= MAX_LEN:
            resp_ids = resp_ids[:MAX_LEN]
            prompt_ids = []
        else:
            keep_prompt = MAX_LEN - len(resp_ids)
            prompt_ids = prompt_ids[-keep_prompt:]

    input_ids = prompt_ids + resp_ids
    attention_mask = [1] * len(input_ids)
    labels = ([-100] * len(prompt_ids)) + resp_ids  # loss only on response

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

train_tokenized = train_ds.map(tokenize_prompt_response, remove_columns=train_ds.column_names)
valid_tokenized = valid_ds.map(tokenize_prompt_response, remove_columns=valid_ds.column_names)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
)

model.config.use_cache = False
model.gradient_checkpointing_enable()
model.enable_input_require_grads()

peft_config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
)
model = get_peft_model(model, peft_config)

training_args = TrainingArguments(
    output_dir=fop_output_model,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=10,
    num_train_epochs=2,
    logging_steps=10,
    save_strategy="epoch",
    fp16=False,
    bf16=torch.cuda.is_available(),
    optim="paged_adamw_8bit",
    report_to="none",
)

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    padding=True,
    label_pad_token_id=-100,
    pad_to_multiple_of=8,
)

class TokenWeightedCETrainer(Trainer):
    def __init__(self,
                 *args,
                 xml_w_tag: float = 8.0,
                 xml_w_slash: float = 5.0,
                 xml_w_attr: float = 3.0,
                 normalize_weights: bool = True,
                 **kwargs):
        super().__init__(*args, **kwargs)

        # Build once; keep on CPU and move to device during compute_loss
        self.token_weight_cpu = build_token_weight_vector(
            self.tokenizer,
            w_tag=xml_w_tag,
            w_slash=xml_w_slash,
            w_attr=xml_w_attr,
            w_default=1.0,
        )
        self.normalize_weights = bool(normalize_weights)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        logits = outputs.logits              # (B, T, V)
        labels = inputs["labels"]            # (B, T), -100 = ignore

        # Causal LM shift (match HF CE)
        shift_logits = logits[:, :-1, :].contiguous()  # (B, T-1, V)
        shift_labels = labels[:, 1:].contiguous()      # (B, T-1)

        B, Tm1, V = shift_logits.shape

        # Per-token CE (no reduction)
        per_token_ce = F.cross_entropy(
            shift_logits.view(-1, V),
            shift_labels.view(-1),
            reduction="none",
            ignore_index=-100,
        ).view(B, Tm1)

        valid = (shift_labels != -100).to(per_token_ce.dtype)

        # Gather token weights by label id (need safe index for -100)
        safe = shift_labels.clone()
        safe[safe == -100] = 0

        token_weight = self.token_weight_cpu.to(device=shift_logits.device)
        w = token_weight[safe] * valid   # (B, T-1)

        # Optional: normalize weights so the overall loss scale stays CE-like
        # (prevents effective LR changes when many tokens are upweighted)
        if self.normalize_weights:
            denom = w.sum().clamp(min=1.0)
            w = w * (valid.sum().clamp(min=1.0) / denom)

        # Weighted CE
        total_loss = (per_token_ce * w).sum() / valid.sum().clamp(min=1.0)

        # Logging (optional)
        self.log({
            "total_loss": float(total_loss.detach().cpu()),
        })

        return (total_loss, outputs) if return_outputs else total_loss



trainer = TokenWeightedCETrainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=valid_tokenized,
    processing_class=tokenizer,   # if you’re on Transformers v5 style
    data_collator=data_collator,
    xml_w_tag=8.0,
    xml_w_slash=5.0,
    xml_w_attr=3.0,
    normalize_weights=True,
)

trainer.train()
trainer.save_model(fop_output_model)
