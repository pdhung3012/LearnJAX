# pip install -U bitsandbytes peft accelerate transformers datasets peft nltk
import os
import torch
import pandas as pd
from datasets import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
)

from peft import LoraConfig, get_peft_model

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

model_name = "/home/hungphd/git/pretrained_open_llms/Qwen2.5-7B-Instruct/"
folder_output = "/home/hungphd/git/potracker_adapter_weights/"

fp_file_tuning_train = "/doe/data-all/label-split/finetune_train.csv"
fp_file_tuning_valid = "/doe/data-all/label-split/finetune_valid.csv"

df_train = pd.read_csv(fp_file_tuning_train, dtype=str, keep_default_na=False, na_filter=False)
df_valid = pd.read_csv(fp_file_tuning_valid, dtype=str, keep_default_na=False, na_filter=False)

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
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
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

# -----------------------------
# Custom BLEU-augmented Trainer
# -----------------------------
class BleuLossTrainer(Trainer):
    """
    Adds a non-differentiable BLEU penalty on top of the standard CE loss.
    Gradients still come from CE; BLEU term is a scalar regularizer.
    """
    def __init__(
        self,
        *args,
        bleu_weight: float = 0.2,
        bleu_mode: str = "1-minus",  # "1-minus" or "inv"
        bleu_eps: float = 1e-6,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.bleu_weight = float(bleu_weight)
        self.bleu_mode = bleu_mode
        self.bleu_eps = float(bleu_eps)
        self._smooth = SmoothingFunction().method1

    def _batch_bleu(self, pred_ids: torch.Tensor, label_ids: torch.Tensor) -> float:
        """
        pred_ids: (B, T) predicted token ids (already aligned with labels)
        label_ids: (B, T) label token ids where -100 marks non-response/pad
        """
        pred_ids = pred_ids.detach().cpu()
        label_ids = label_ids.detach().cpu()

        scores = []
        for p, y in zip(pred_ids, label_ids):
            # keep only response tokens (labels != -100)
            mask = (y != -100)
            y_resp = y[mask]
            p_resp = p[mask]

            if y_resp.numel() == 0:
                continue

            # decode; strip special tokens
            ref_text = self.tokenizer.decode(y_resp.tolist(), skip_special_tokens=True).strip()
            hyp_text = self.tokenizer.decode(p_resp.tolist(), skip_special_tokens=True).strip()

            # tokenization for BLEU (simple whitespace)
            ref_tok = ref_text.split()
            hyp_tok = hyp_text.split()

            if len(ref_tok) == 0 or len(hyp_tok) == 0:
                scores.append(0.0)
                continue

            bleu = sentence_bleu([ref_tok], hyp_tok, smoothing_function=self._smooth)
            scores.append(float(bleu))

        if not scores:
            return 0.0
        return sum(scores) / len(scores)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Standard forward pass -> CE loss (because labels provided)
        outputs = model(**inputs)
        ce_loss = outputs.loss

        # Build token predictions aligned to labels length (teacher-forced)
        # logits: (B, T, V) for each position
        logits = outputs.logits
        pred_ids = torch.argmax(logits, dim=-1)  # (B, T)

        labels = inputs["labels"]

        # Compute BLEU on current batch (non-differentiable scalar)
        with torch.no_grad():
            bleu = self._batch_bleu(pred_ids, labels)

        if self.bleu_mode == "inv":
            bleu_penalty = 1.0 / (bleu + self.bleu_eps)
        else:
            # default: (1 - BLEU)
            bleu_penalty = 1.0 - bleu

        total_loss = ce_loss + (self.bleu_weight * ce_loss.new_tensor(bleu_penalty))

        # (optional) log bleu occasionally
        if self.state.global_step % max(self.args.logging_steps, 1) == 0:
            self.log({
                "train_bleu": bleu,
                "bleu_penalty": float(bleu_penalty),
                "ce_loss": float(ce_loss.detach().cpu()),
            })

        return (total_loss, outputs) if return_outputs else total_loss


trainer = BleuLossTrainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=valid_tokenized,
    processing_class=tokenizer,
    data_collator=data_collator,
    bleu_weight=0.2,      # tune this
    bleu_mode="1-minus",  # or "inv"
)

trainer.train()
trainer.save_model(fop_output_model)
