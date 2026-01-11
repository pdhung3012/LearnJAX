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
import numpy as np
from combined_metrics import *
from statistics import mean

class POReportSimilarityTrainer(Trainer):
    """
    Trainer with explicit CE loss. XML Score is computed
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
# XML metric between labels & outputs
# =========================
# chencherry = SmoothingFunction()


def compute_po_tracker(eval_pred):
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
    # print('aaa {} bbb {}'.format(pred_texts,label_texts))

    scores = []
    for p, l in zip(pred_texts, label_texts):
        hyp = p.split()
        ref = [l.split()]  # list of references

        if len(hyp) == 0 or len(ref[0]) == 0:
            scores.append(0.0)
            continue
        print('sample rep:\n{}\nhype:\n{}'.format(ref,hyp))
        score_obj=combined_similarity(ref,hyp)
        s = score_obj['combined_similarity']
        scores.append(s)

    xml_score = float(np.mean(scores)) if scores else 0.0
    return {"xml": xml_score}



model_name = "/home/hungphd/git/pretrained_open_llms/Qwen2.5-7B-Instruct/"
folder_output="/home/hungphd/git/potracker_customloss_adapter_weights/"

# Load sample data
fp_file_tuning_train='/home/hungphd/git/LearnJAX/doe/data-all/label-split/finetune_train.csv'
fp_file_tuning_valid='/home/hungphd/git/LearnJAX/doe/data-all/label-split/finetune_valid.csv'
df_train = pd.read_csv(fp_file_tuning_train, dtype=str, keep_default_na=False, na_filter=False)
df_valid = pd.read_csv(fp_file_tuning_valid, dtype=str, keep_default_na=False, na_filter=False)

# ✅ Extra safety (in case of weird values)
for df in (df_train, df_valid):
    df["prompt"] = df["prompt"].fillna("").astype(str)
    df["response"] = df["response"].fillna("").astype(str)

df_train["text"] = "### Instruction:\n" + df_train["prompt"] + "\n\n### Response:\n" + df_train["response"]
df_valid["text"] = "### Instruction:\n" + df_valid["prompt"] + "\n\n### Response:\n" + df_valid["response"]

train_ds = Dataset.from_pandas(df_train[["text"]], preserve_index=False)
valid_ds = Dataset.from_pandas(df_valid[["text"]], preserve_index=False)

# model_name = "/home/hungphd/git/Qwen2.5-3B-Instruct/"
# model_name = "/home/hungphd/git/pretrained_open_llms/phi-4/"
arr_model_path=model_name.split('/')
real_model_name=arr_model_path[-2]
fop_output_model=folder_output+real_model_name+'/'


tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

def tokenize(example):
    return tokenizer(example["text"],  truncation=True, max_length=2048)

train_tokenized = train_ds.map(tokenize, batched=True)
test_tokenized = valid_ds.map(tokenize, batched=True)



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

trainer = POReportSimilarityTrainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=test_tokenized,
    processing_class=tokenizer,   # ✅ instead of tokenizer=tokenizer
    data_collator=data_collator,
    compute_metrics=compute_po_tracker,

)


trainer.train()
trainer.save_model(fop_output_model)
