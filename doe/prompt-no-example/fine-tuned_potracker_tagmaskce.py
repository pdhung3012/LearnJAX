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
folder_output = "/home/hungphd/git/potracker_tagmaskce/"

fp_file_tuning_train = "../data-all/label-split/finetune_noex_train.csv"
fp_file_tuning_valid = "../data-all/label-split/finetune_noex_valid.csv"
num_samples=100
df_train = pd.read_csv(fp_file_tuning_train, dtype=str, keep_default_na=False, na_filter=False) .head(num_samples)
df_valid = pd.read_csv(fp_file_tuning_valid, dtype=str, keep_default_na=False, na_filter=False) .head(num_samples)

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

import re
from typing import Dict, Any, List, Tuple

TAG_RE = re.compile(r"<[^>]*>")  # spans like <tag ...> </tag> <tag/>

def find_response_span(full_text: str, response_delim: str = "### Response:\n") -> Tuple[int, int]:
    """
    Returns (resp_char_start, resp_char_end) in full_text.
    If delimiter not found, treat whole text as response.
    """
    idx = full_text.find(response_delim)
    if idx == -1:
        return 0, len(full_text)
    resp_start = idx + len(response_delim)
    return resp_start, len(full_text)

def collect_tag_spans(text: str) -> List[Tuple[int, int]]:
    """Return list of (start,end) char spans for tag substrings like <...>."""
    return [(m.start(), m.end()) for m in TAG_RE.finditer(text)]

def overlap(a0, a1, b0, b1) -> bool:
    """True if [a0,a1) overlaps [b0,b1)."""
    return (a0 < b1) and (b0 < a1)

def tokenize_with_tag_mask(example: Dict[str, Any],
                           tokenizer,
                           max_length: int = 2048,
                           response_delim: str = "### Response:\n") -> Dict[str, Any]:
    """
    Outputs:
      input_ids, attention_mask, labels, tag_mask (same shape as labels)
    Requires a *fast* tokenizer for return_offsets_mapping=True.
    """
    # 1) Build full text
    # Use your own construction if you have prompt/response fields.
    if "text" in example:
        full_text = example["text"]
    else:
        # minimal fallback (adjust to your schema)
        full_text = example["prompt"] + response_delim + example["response"]

    # 2) Find response char span in full_text
    resp_char_start, resp_char_end = find_response_span(full_text, response_delim)

    # 3) Collect tag spans inside the response substring
    resp_text = full_text[resp_char_start:resp_char_end]
    resp_tag_spans = collect_tag_spans(resp_text)  # spans relative to resp_text
    # Convert to full_text-global char spans
    global_tag_spans = [(resp_char_start + s, resp_char_start + e) for (s, e) in resp_tag_spans]

    # 4) Tokenize full_text with offsets
    enc = tokenizer(
        full_text,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_offsets_mapping=True,
    )

    input_ids = enc["input_ids"]
    attention_mask = enc.get("attention_mask", [1] * len(input_ids))
    offsets = enc["offset_mapping"]  # list of (start,end) per token (chars in full_text)

    # 5) Build labels: train only on response tokens; prompt tokens -> -100
    # Find first token whose offset starts at/after resp_char_start
    # (skip special tokens where offset might be (0,0) in some tokenizers)
    resp_tok_start = None
    for i, (s, e) in enumerate(offsets):
        if e <= s:  # special / empty
            continue
        if s >= resp_char_start:
            resp_tok_start = i
            break
    if resp_tok_start is None:
        # if response got truncated away, just mask everything
        resp_tok_start = len(input_ids)

    labels = input_ids.copy()
    for i in range(resp_tok_start):
        labels[i] = -100

    # 6) Build tag_mask aligned to tokens (1 if token overlaps ANY <...> span)
    tag_mask = [0] * len(input_ids)
    if global_tag_spans:
        for i, (s, e) in enumerate(offsets):
            if e <= s:  # special/empty
                continue
            if labels[i] == -100:
                continue  # we don't train on prompt tokens anyway
            # overlap with any tag span
            for ts, te in global_tag_spans:
                if overlap(s, e, ts, te):
                    tag_mask[i] = 1
                    break

    # return without offsets (don’t store them in dataset)
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "tag_mask": tag_mask,
    }

import torch
from dataclasses import dataclass
from transformers import PreTrainedTokenizerBase

@dataclass
@dataclass
class TagMaskDataCollator:
    tokenizer: PreTrainedTokenizerBase
    pad_to_multiple_of: int = None

    def __call__(self, features):
        """
        Pad input_ids / attention_mask / labels / tag_mask to a common length.

        Note: tokenizer.pad() may drop unknown fields like `tag_mask` depending on
        transformers version, so we pad manually to guarantee `tag_mask` exists.
        """
        # Basic sanity check (helps catch accidental use of a dataset without tag_mask)
        if len(features) == 0:
            return {}

        if "tag_mask" not in features[0]:
            raise KeyError(
                "tag_mask is missing from dataset features. "
                "Make sure your dataset.map(tokenize_with_tag_mask, ...) is the one passed "
                "to the Trainer (train_dataset=tokenized_train / eval_dataset=tokenized_valid)."
            )

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = getattr(self.tokenizer, "eos_token_id", 0)

        max_len = max(len(f["input_ids"]) for f in features)
        if self.pad_to_multiple_of:
            m = int(self.pad_to_multiple_of)
            max_len = ((max_len + m - 1) // m) * m

        batch_input_ids = []
        batch_attention = []
        batch_labels = []
        batch_tag_mask = []

        for f in features:
            ids = list(f["input_ids"])
            attn = list(f.get("attention_mask", [1] * len(ids)))
            labels = list(f.get("labels", [-100] * len(ids)))
            tag_mask = list(f.get("tag_mask", [0] * len(ids)))

            # Safety: ensure same length per example
            if not (len(ids) == len(attn) == len(labels) == len(tag_mask)):
                raise ValueError(
                    f"Length mismatch: input_ids={len(ids)}, attention_mask={len(attn)}, "
                    f"labels={len(labels)}, tag_mask={len(tag_mask)}"
                )

            pad_len = max_len - len(ids)
            batch_input_ids.append(ids + [pad_id] * pad_len)
            batch_attention.append(attn + [0] * pad_len)

            # IMPORTANT: pad labels with -100 so padding doesn't contribute to CE loss
            batch_labels.append(labels + [-100] * pad_len)

            # Pad tag_mask with 0 (non-tag)
            batch_tag_mask.append(tag_mask + [0] * pad_len)

        batch = {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(batch_attention, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
            "tag_mask": torch.tensor(batch_tag_mask, dtype=torch.long),
        }
        return batch


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
    remove_unused_columns=False
)

# data_collator = DataCollatorForSeq2Seq(
#     tokenizer=tokenizer,
#     padding=True,
#     label_pad_token_id=-100,
#     pad_to_multiple_of=8,
# )

import torch
import torch.nn.functional as F
from transformers import Trainer

class TagMaskWeightedCETrainer(Trainer):
    def __init__(self, *args,
                 w_tag: float = 8.0,          # weight for tokens inside <...>
                 w_text: float = 1.0,         # weight for normal response tokens
                 normalize_weights: bool = True,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.w_tag = float(w_tag)
        self.w_text = float(w_text)
        self.normalize_weights = bool(normalize_weights)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        logits = outputs.logits                  # (B, T, V)
        labels = inputs["labels"]                # (B, T)
        tag_mask = inputs["tag_mask"]            # (B, T)  0/1

        # Causal shift
        shift_logits = logits[:, :-1, :].contiguous()   # predicts token t+1
        shift_labels = labels[:, 1:].contiguous()
        shift_tag    = tag_mask[:, 1:].contiguous()

        B, Tm1, V = shift_logits.shape

        per_token_ce = F.cross_entropy(
            shift_logits.view(-1, V),
            shift_labels.view(-1),
            reduction="none",
            ignore_index=-100,
        ).view(B, Tm1)

        valid = (shift_labels != -100).to(per_token_ce.dtype)

        # weights: tag tokens get w_tag, others get w_text (only where valid)
        w = (shift_tag.to(per_token_ce.dtype) * (self.w_tag - self.w_text) + self.w_text) * valid

        # Optional: normalize so average weight ~1 to keep CE-like scale
        if self.normalize_weights:
            denom = w.sum().clamp(min=1.0)
            target = valid.sum().clamp(min=1.0) * 1.0  # average weight ≈ 1
            w = w * (target / denom)

        loss = (per_token_ce * w).sum() / valid.sum().clamp(min=1.0)

        self.log({
            "total_loss": float(loss.detach().cpu()),
            "tag_frac": float((shift_tag * valid).sum().detach().cpu() / valid.sum().clamp(min=1.0).detach().cpu()),
        })

        return (loss, outputs) if return_outputs else loss

# Tokenize dataset
tokenized_train = train_ds.map(
    lambda ex: tokenize_with_tag_mask(ex, tokenizer, max_length=MAX_LEN, response_delim="### Response:\n"),
    remove_columns=train_ds.column_names,
)
print(tokenized_train.column_names)
print(tokenized_train[0].keys())
# input('aaa')

tokenized_valid = valid_ds.map(
    lambda ex: tokenize_with_tag_mask(ex, tokenizer, max_length=MAX_LEN, response_delim="### Response:\n"),
    remove_columns=valid_ds.column_names,
)

data_collator = TagMaskDataCollator(tokenizer)

trainer = TagMaskWeightedCETrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_valid,
    processing_class=tokenizer,   # transformers v5 style (or tokenizer=... in older)
    data_collator=data_collator,
    w_tag=8.0,
    w_text=1.0,
    normalize_weights=True,

)

trainer.train()
trainer.save_model(fop_output_model)
