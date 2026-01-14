# pip install -U bitsandbytes peft accelerate transformers datasets peft
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

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

model_name = "/home/hungphd/git/pretrained_open_llms/Qwen2.5-7B-Instruct/"
folder_output = "/home/hungphd/git/orgfinetuned_adapter_weights/"

fp_file_tuning_train = "/doe/data-all/label-split/finetune_train.csv"
fp_file_tuning_valid = "/doe/data-all/label-split/finetune_valid.csv"

df_train = pd.read_csv(fp_file_tuning_train, dtype=str, keep_default_na=False, na_filter=False)
df_valid = pd.read_csv(fp_file_tuning_valid, dtype=str, keep_default_na=False, na_filter=False)

# Safety: ensure strings
for df in (df_train, df_valid):
    df["prompt"] = df["prompt"].fillna("").astype(str)
    df["response"] = df["response"].fillna("").astype(str)

train_ds = Dataset.from_pandas(df_train[["prompt", "response"]], preserve_index=False)
valid_ds = Dataset.from_pandas(df_valid[["prompt", "response"]], preserve_index=False)

arr_model_path = model_name.split("/")
real_model_name = arr_model_path[-2]
fop_output_model = folder_output + real_model_name + "/"

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
# Qwen instruct usually works fine with EOS as pad for training
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

MAX_LEN = 512

def format_prompt(p: str) -> str:
    # Prompt-only template (no response here)
    return f"### Instruction:\n{p}\n\n### Response:\n"

def tokenize_prompt_response(example):
    prompt_text = format_prompt(example["prompt"])
    # Add EOS to response so model learns to stop
    response_text = example["response"] + tokenizer.eos_token

    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    resp_ids   = tokenizer(response_text, add_special_tokens=False)["input_ids"]

    # Truncate to MAX_LEN, prefer keeping the full response
    if len(prompt_ids) + len(resp_ids) > MAX_LEN:
        # if response itself is too long, truncate response
        if len(resp_ids) >= MAX_LEN:
            resp_ids = resp_ids[:MAX_LEN]
            prompt_ids = []
        else:
            keep_prompt = MAX_LEN - len(resp_ids)
            # truncate prompt from the left to keep the most recent instruction context
            prompt_ids = prompt_ids[-keep_prompt:]

    input_ids = prompt_ids + resp_ids
    attention_mask = [1] * len(input_ids)

    # Mask prompt tokens in labels so loss is computed only on response tokens
    labels = ([-100] * len(prompt_ids)) + resp_ids

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

train_tokenized = train_ds.map(tokenize_prompt_response, remove_columns=train_ds.column_names)
valid_tokenized = valid_ds.map(tokenize_prompt_response, remove_columns=valid_ds.column_names)

# 4-bit load
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

# Make training-friendly
model.config.use_cache = False
model.gradient_checkpointing_enable()
model.enable_input_require_grads()

# LoRA config (Qwen uses similar proj names; keep as you had)
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
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

# IMPORTANT: use a collator that preserves labels and pads labels with -100
data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    padding=True,
    label_pad_token_id=-100,
    pad_to_multiple_of=8,   # optional but often helps tensor cores
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=valid_tokenized,
    processing_class=tokenizer,  # transformers v5+
    data_collator=data_collator,
)

trainer.train()
trainer.save_model(fop_output_model)
