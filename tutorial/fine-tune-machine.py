# pip install -U bitsandbytes peft accelerate transformers
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from transformers import DataCollatorForLanguageModeling, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
import pandas as pd
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
from datasets import Dataset
import torch
from utils import *


# Load sample data
df = pd.read_csv("code_finetuned.csv")
df["text"] = "### Instruction:\n" + df["prompt"] + "\n\n### Response:\n" + df["response"]
dataset_all = Dataset.from_pandas(df[["text"]])
splits = dataset_all.train_test_split(test_size=0.2, seed=42)  # 20% test
train_ds = splits["train"]
test_ds  = splits["test"]

model_name = "/lustre/hdd/LAS/jannesar-lab/hungphd/git/pretrained_models_org/Qwen2.5-32B-Instruct/"
folder_output="/lustre/hdd/LAS/jannesar-lab/hungphd/git/adapter_weights/"
arr_model_path='/'.split(model_name)
real_model_name=arr_model_path[-2]
fop_output_model=folder_output+real_model_name+'/'


tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

def tokenize(example):
    return tokenizer(example["text"],  truncation=True, max_length=8)

train_tokenized = train_ds.map(tokenize, batched=True)
test_tokenized = test_ds.map(tokenize, batched=True)



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
model.config.use_cache = False                      # avoid KV cache during training
model.gradient_checkpointing_enable()               # save activation memory
model.enable_input_require_grads()                  # needed for QLoRA

# LoRA on LLaMA/CodeLlama proj layers
peft_config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
)
model = get_peft_model(model, peft_config)

# your dataset pieces...
# train_tokenized, test_tokenized from your script

training_args = TrainingArguments(
    output_dir=fop_output_model,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,      # simulate bigger batch
    num_train_epochs=3,
    logging_steps=10,
    save_strategy="epoch",              # avoid saving every step
    fp16=False,                         # use bf16 if available instead
    bf16=torch.cuda.is_available(),     # A100/H100 etc.
    optim="paged_adamw_8bit",           # memory-efficient optimizer
    report_to="none"
)

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=test_tokenized,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()
trainer.save_model("./finetuned-llm")
