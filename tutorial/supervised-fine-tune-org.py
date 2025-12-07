import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)
import torch

# 1. Load and preprocess data
df = pd.read_csv("sample_data.csv")

# Format: instruction tuning style
def format_prompt(example):
    return f"### Instruction:\n{example['prompt']}\n\n### Response:\n{example['response']}"

df["prompt_formatted"] = df.apply(format_prompt, axis=1)
dataset = Dataset.from_pandas(df[["prompt_formatted", "response"]])

# 2. Load tokenizer and model
model_name = "tiiuae/falcon-rw-1b"  # or "EleutherAI/pythia-70m", etc.
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name)

# 3. Tokenize with labels masked outside the response
def tokenize_supervised(example):
    prompt = example["prompt_formatted"]
    response = example["response"]

    full_text = prompt + response
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, truncation=True, max_length=512, padding="max_length")

    labels = full_ids["input_ids"].copy()
    # Mask loss for prompt tokens
    labels[:len(prompt_ids)] = [-100] * len(prompt_ids)

    full_ids["labels"] = labels
    return full_ids

tokenized_dataset = dataset.map(tokenize_supervised, remove_columns=dataset.column_names)

# 4. Training arguments
training_args = TrainingArguments(
    output_dir="./sft-model",
    per_device_train_batch_size=2,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=50,
    save_total_limit=1,
    evaluation_strategy="no",
    fp16=torch.cuda.is_available(),
    report_to="none"
)

# 5. Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
)

# 6. Fine-tune
trainer.train()
trainer.save_model("./sft-model")

