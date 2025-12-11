# pip install -U bitsandbytes peft accelerate transformers nltk
import os
import torch
import numpy as np
import pandas as pd
from datasets import Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
)

from peft import LoraConfig, get_peft_model
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


# =========================
# 1. Load & prepare data
# =========================
df = pd.read_csv("sample_data.csv")
df["text"] = "### Instruction:\n" + df["prompt"] + "\n\n### Response:\n" + df["response"]

dataset_all = Dataset.from_pandas(df[["text"]])
splits = dataset_all.train_test_split(test_size=0.2, seed=42)  # 20% test
train_ds = splits["train"]
test_ds  = splits["test"]

model_name = "/work/LAS/jannesar-lab/hungphd/git/pretrained_open_llms/CodeLlama-7b-hf/"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # safety


def tokenize(example):
    # For real training, you want a bigger max_length,
    # but keep it small for quick tests.
    return tokenizer(example["text"], truncation=True, max_length=512)


train_tokenized = train_ds.map(
    tokenize, batched=True, remove_columns=train_ds.column_names
)
test_tokenized = test_ds.map(
    tokenize, batched=True, remove_columns=test_ds.column_names
)


# =========================
# 2. 4-bit quant + LoRA
# =========================
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
model.enable_input_require_grads()  # needed for QLoRA

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
)
model = get_peft_model(model, peft_config)


# =========================
# 3. Custom Trainer (CE loss)
# =========================
class TextSimilarityTrainer(Trainer):
    """
    Trainer with explicit CE loss. BLEU is computed
    separately in compute_metrics (see below).
    """

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.logits  # (batch, seq_len, vocab)

        # Shift so that tokens <t> predict <t+1>
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )

        return (loss, outputs) if return_outputs else loss


# =========================
# 4. BLEU metric between labels & outputs
# =========================
chencherry = SmoothingFunction()


def compute_bleu(eval_pred):
    """
    eval_pred: (predictions, label_ids) from Trainer
    Returns: dict with BLEU score.
    """
    predictions, labels = eval_pred

    # Trainer sometimes returns a tuple for predictions
    if isinstance(predictions, tuple):
        predictions = predictions[0]

    # predictions are logits: [batch, seq_len, vocab]
    pred_ids = np.argmax(predictions, axis=-1)

    # Replace ignore_index -100 with pad_token_id for decoding
    if tokenizer.pad_token_id is None:
        raise ValueError("tokenizer.pad_token_id is None, but needed for decoding labels")

    labels_for_decode = np.where(labels == -100, tokenizer.pad_token_id, labels)

    pred_texts = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_texts = tokenizer.batch_decode(labels_for_decode, skip_special_tokens=True)

    scores = []
    for p, l in zip(pred_texts, label_texts):
        hyp = p.split()
        ref = [l.split()]  # list of references

        if len(hyp) == 0 or len(ref[0]) == 0:
            scores.append(0.0)
            continue

        s = sentence_bleu(ref, hyp, smoothing_function=chencherry.method1)
        scores.append(s)

    bleu = float(np.mean(scores)) if scores else 0.0
    return {"bleu": bleu}


# =========================
# 5. Training setup
# =========================
training_args = TrainingArguments(
    output_dir="./finetuned-llm",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_train_epochs=3,
    logging_steps=10,
    save_strategy="epoch",
    fp16=False,
    bf16=torch.cuda.is_available(),
    optim="paged_adamw_8bit",
    report_to="none",
    evaluation_strategy="epoch",  # so BLEU gets computed
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

trainer = TextSimilarityTrainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=test_tokenized,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_bleu,  # <--- BLEU between labels & outputs
)

trainer.train()
eval_metrics = trainer.evaluate()
print("Eval metrics:", eval_metrics)

trainer.save_model("./finetuned-llm")
tokenizer.save_pretrained("./finetuned-llm")
