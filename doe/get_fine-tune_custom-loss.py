# pip install -U bitsandbytes peft accelerate transformers
import os
import traceback

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
import torch.nn.functional as F
import numpy as np
from transformers import Trainer
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union


def _xml_reward(tokenizer, pred_ids, ref_text):
    pred_text = tokenizer.decode(pred_ids, skip_special_tokens=True)
    hyp = pred_text.split()
    ref = [ref_text.split()]
    if len(hyp) == 0 or len(ref[0]) == 0:
        return 0.0
    # print('ref {}\npred {}'.format(ref_text,pred_text))
    # input('bbb')
    score_obj = combined_similarity(ref_text, pred_text,show_errors=False)
    return float(score_obj["combined_similarity"])

class POReportSimilarityTrainer(Trainer):
    """
    Trainer with explicit CE loss. XML Score is computed
    separately in compute_metrics (see below).
    """

    def __init__(
            self,
            *args,
            rl_weight=0.05,
            rl_every_n_steps=10,
            gen_max_new_tokens=256,
            gen_temperature=0.7,
            gen_top_p=0.9,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.rl_weight = rl_weight
        self.rl_every_n_steps = rl_every_n_steps
        self.gen_max_new_tokens = gen_max_new_tokens
        self.gen_temperature = gen_temperature
        self.gen_top_p = gen_top_p

    #
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # 1) Normal supervised CE loss (stable)
        outputs = model(**inputs)
        ce_loss = outputs.loss
        # print('go here')

        # Optional: only do RL loss every N steps (saves a LOT of time)
        do_rl = (self.state.global_step % self.rl_every_n_steps == 0)

        if (not do_rl) or (self.rl_weight <= 0):
            return (ce_loss, outputs) if return_outputs else ce_loss

        # 2) SCST reward term (non-differentiable reward -> REINFORCE)
        # We need prompt-only input for generation; prompt tokens have label == -100
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        labels = inputs["labels"]

        batch_size = input_ids.size(0)
        device = input_ids.device

        rl_losses = []
        for b in range(batch_size):
            # Find where response starts: first non -100 in labels
            lab = labels[b].tolist()
            try:
                resp_start = lab.index(next(x for x in lab if x != -100))
            except StopIteration:
                # no supervised tokens, skip
                continue

            prompt_ids = input_ids[b, :resp_start].unsqueeze(0)
            prompt_mask = attention_mask[b, :resp_start].unsqueeze(0)

            # Reference response text (decode labels excluding -100)
            ref_ids = [t for t in lab if t != -100 and t != tokenizer.pad_token_id]
            ref_text = tokenizer.decode(ref_ids, skip_special_tokens=True)

            # --- Baseline: greedy ---
            with torch.no_grad():
                greedy = model.generate(
                    input_ids=prompt_ids,
                    attention_mask=prompt_mask,
                    max_new_tokens=self.gen_max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
                greedy_new = greedy[0, resp_start:] if greedy.size(1) > resp_start else greedy[0, resp_start:]
                r_baseline = _xml_reward(tokenizer, greedy_new.tolist(), ref_text)

            # --- Sampled: stochastic generation (policy) ---
            sampled = model.generate(
                input_ids=prompt_ids,
                attention_mask=prompt_mask,
                max_new_tokens=self.gen_max_new_tokens,
                do_sample=True,
                temperature=self.gen_temperature,
                top_p=self.gen_top_p,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            sampled_new = sampled[0, resp_start:] if sampled.size(1) > resp_start else sampled[0, resp_start:]
            r_sample = _xml_reward(tokenizer, sampled_new.tolist(), ref_text)

            advantage = (r_sample - r_baseline)

            # Compute logprob of sampled tokens under current model
            # seq = prompt + sampled_new (full sampled already contains prompt)
            seq = sampled.unsqueeze(0) if sampled.dim() == 1 else sampled  # [1, T]
            seq = seq.to(device)

            out = model(input_ids=seq)
            logits = out.logits  # [1, T, V]

            # Logprobs for tokens at positions resp_start .. T-1
            # token at position j is predicted by logits at j-1
            T = seq.size(1)
            if resp_start >= T:
                continue

            token_ids = seq[0, resp_start:T]  # [L]
            pred_logits = logits[0, resp_start - 1:T - 1, :]  # [L, V]

            log_probs = F.log_softmax(pred_logits, dim=-1)  # [L, V]
            token_logp = log_probs.gather(1, token_ids.unsqueeze(1)).squeeze(1)  # [L]
            seq_logp = token_logp.sum() / max(1, token_logp.numel())

            # REINFORCE loss: -advantage * logp
            rl_loss = -(advantage) * seq_logp
            rl_losses.append(rl_loss)

        if len(rl_losses) == 0:
            total_loss = ce_loss
        else:
            rl_loss_mean = torch.stack(rl_losses).mean()
            total_loss = ce_loss + self.rl_weight * rl_loss_mean
        # print('end go here')
        return (total_loss, outputs) if return_outputs else total_loss

    # def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
    #     labels = inputs.get("labels")
    #     outputs = model(**inputs)
    #     logits = outputs.logits  # (batch, seq_len, vocab)
    #
    #     # Shift so that tokens <t> predict <t+1>
    #     shift_logits = logits[..., :-1, :].contiguous()
    #     shift_labels = labels[..., 1:].contiguous()
    #
    #     loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
    #     loss = loss_fct(
    #         shift_logits.view(-1, shift_logits.size(-1)),
    #         shift_labels.view(-1),
    #     )
    #
    #     return (loss, outputs) if return_outputs else loss

# =========================
# XML metric between labels & outputs
# =========================
# chencherry = SmoothingFunction()

def preprocess_logits_for_metrics(logits, labels):
    # logits can be tuple in some models
    if isinstance(logits, tuple):
        logits = logits[0]
    return torch.argmax(logits, dim=-1)  # [batch, seq_len]
def compute_po_tracker(eval_pred):
    pred_ids, labels = eval_pred  # pred_ids is now [batch, seq_len]

    labels_for_decode = np.where(labels == -100, tokenizer.pad_token_id, labels)

    pred_texts = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_texts = tokenizer.batch_decode(labels_for_decode, skip_special_tokens=True)

    scores = []
    for p, l in zip(pred_texts, label_texts):
        if not p.strip() or not l.strip():
            scores.append(0.0)
            continue
        score_obj = combined_similarity(l, p, show_errors=False)  # use strings
        scores.append(float(score_obj["combined_similarity"]))

    return {"xml": float(np.mean(scores)) if scores else 0.0}




model_name = "/home/hungphd/git/pretrained_open_llms/Qwen2.5-7B-Instruct/"
folder_output="/home/hungphd/git/potracker_customloss_adapter_weights/"
arr_model_path=model_name.split('/')
real_model_name=arr_model_path[-2]
fop_output_model=folder_output+real_model_name+'/'

# Load sample data
fp_file_tuning_train='/home/hungphd/git/LearnJAX/doe/data-all/label-split/finetune_train.csv'
fp_file_tuning_valid='/home/hungphd/git/LearnJAX/doe/data-all/label-split/finetune_valid.csv'

# Load CSVs
df_train = pd.read_csv(fp_file_tuning_train, dtype=str, keep_default_na=False, na_filter=False)
df_valid = pd.read_csv(fp_file_tuning_valid, dtype=str, keep_default_na=False, na_filter=False)

for df in (df_train, df_valid):
    df["prompt"] = df["prompt"].fillna("").astype(str)
    df["response"] = df["response"].fillna("").astype(str)

train_ds = Dataset.from_pandas(df_train[["prompt", "response"]], preserve_index=False)
valid_ds = Dataset.from_pandas(df_valid[["prompt", "response"]], preserve_index=False)

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

MAX_LEN = 256

def tokenize_supervised(batch):
    prompts = batch["prompt"]
    responses = batch["response"]

    input_ids_list = []
    attention_mask_list = []
    labels_list = []

    for p, r in zip(prompts, responses):
        p_ids = tokenizer.encode(p, add_special_tokens=False)
        r_ids = tokenizer.encode(r, add_special_tokens=False) + [tokenizer.eos_token_id]

        # input = prompt + response
        input_ids = p_ids + r_ids
        input_ids = input_ids[:MAX_LEN]

        # labels: ignore prompt tokens, supervise only response tokens
        labels = ([-100] * len(p_ids) + r_ids)[:MAX_LEN]

        attn = [1] * len(input_ids)

        input_ids_list.append(input_ids)
        attention_mask_list.append(attn)
        labels_list.append(labels)

    return {
        "input_ids": input_ids_list,
        "attention_mask": attention_mask_list,
        "labels": labels_list,
    }

train_tokenized = train_ds.map(tokenize_supervised, batched=True, remove_columns=train_ds.column_names)
eval_tokenized  = valid_ds.map(tokenize_supervised, batched=True, remove_columns=valid_ds.column_names)




# tokenizer = AutoTokenizer.from_pretrained(model_name)
# tokenizer.pad_token = tokenizer.eos_token
#
# def tokenize(example):
#     return tokenizer(example["text"],  truncation=True, max_length=2048)
#
# train_tokenized = train_ds.map(tokenize, batched=True)
# test_tokenized = valid_ds.map(tokenize, batched=True)



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
    save_strategy="epoch",
    # eval_strategy="epoch",# avoid saving every step
    fp16=False,                         # use bf16 if available instead
    bf16=torch.cuda.is_available(),     # A100/H100 etc.
    optim="paged_adamw_8bit",           # memory-efficient optimizer
    report_to="none"
)

from transformers import DataCollatorWithPadding
# data_collator = CausalLMDataCollatorWithLabelPadding(tokenizer=tokenizer, padding=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)

trainer = POReportSimilarityTrainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=eval_tokenized,
    processing_class=tokenizer,
    data_collator=data_collator,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    compute_metrics=compute_po_tracker,   # still fine for eval
    rl_weight=0.05,                       # tune this
    rl_every_n_steps=10,                  # compute RL loss every 10 steps
    gen_max_new_tokens=MAX_LEN,               # keep small for speed
)


trainer.train()
eval_metrics = trainer.evaluate()
print("Eval metrics:", eval_metrics)

trainer.save_model(fop_output_model)
