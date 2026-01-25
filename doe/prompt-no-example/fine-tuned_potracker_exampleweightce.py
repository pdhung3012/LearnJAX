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
folder_output = "/home/hungphd/git/finetuned_weights_noex_potracker/"

fp_file_tuning_train = "../data-all/label-split/finetune_noex_train.csv"
fp_file_tuning_valid = "../data-all/label-split/finetune_noex_valid.csv"
num_samples=10
df_train = pd.read_csv(fp_file_tuning_train, dtype=str, keep_default_na=False, na_filter=False) #.head(num_samples)
df_valid = pd.read_csv(fp_file_tuning_valid, dtype=str, keep_default_na=False, na_filter=False) #.head(num_samples)

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
class POTrackerLossTrainer(Trainer):
    """
    Example-weighted CE using combined_similarity, computed every N batches.

    - Compute per-example CE loss (teacher-forced)
    - Occasionally (every potracker_every batches) compute POTracker scores from argmax decode
    - Convert penalty into a weight alpha_i that scales CE_i
    - Use EMA baseline so it still works when batch_size == 1
    """

    def __init__(
        self,
        *args,
        potracker_weight: float = 1.0,        # how strongly weights affect CE
        potracker_mode: str = "1-minus",      # "1-minus" or "inv"
        potracker_eps: float = 1e-6,
        potracker_every: int = 20,            # compute combined_similarity every N batches
        alpha_text: float = 0.1,              # passed to combined_similarity(..., alpha=alpha_text)
        weight_clip_min: float = 0.5,
        weight_clip_max: float = 2.0,
        ema_beta: float = 0.95,               # EMA smoothing for baseline
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.potracker_weight = float(potracker_weight)
        self.potracker_mode = potracker_mode
        self.potracker_eps = float(potracker_eps)

        self.potracker_every = int(max(1, potracker_every))
        self.alpha_text = float(alpha_text)
        self.weight_clip_min = float(weight_clip_min)
        self.weight_clip_max = float(weight_clip_max)

        self.ema_beta = float(ema_beta)
        self._potracker_call_count = 0
        self._penalty_ema = None  # float baseline (EMA)

    def _batch_potracker_per_example(self, pred_ids: torch.Tensor, label_ids: torch.Tensor) -> torch.Tensor:
        """
        Returns: scores tensor of shape (B,) on CPU (float32), each in [0,1] typically.
        """
        pred_ids = pred_ids.detach().cpu()
        label_ids = label_ids.detach().cpu()

        scores = []
        for p, y in zip(pred_ids, label_ids):
            mask = (y != -100)
            y_resp = y[mask]
            p_resp = p[mask]

            if y_resp.numel() == 0:
                scores.append(0.0)
                continue

            ref_text = self.tokenizer.decode(y_resp.tolist(), skip_special_tokens=True).strip()
            hyp_text = self.tokenizer.decode(p_resp.tolist(), skip_special_tokens=True).strip()

            if len(ref_text) == 0 or len(hyp_text) == 0:
                scores.append(0.0)
                continue

            score_obj = combined_similarity(ref_text, hyp_text, show_errors=False, alpha=self.alpha_text)
            scores.append(float(score_obj["combined_similarity"]))

        if not scores:
            return torch.zeros(1, dtype=torch.float32)

        return torch.tensor(scores, dtype=torch.float32)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        logits = outputs.logits                      # (B, T, V)
        labels = inputs["labels"]                    # (B, T)

        # ---- per-example CE (teacher-forced causal LM with shift) ----
        shift_logits = logits[:, :-1, :].contiguous()  # (B, T-1, V)
        shift_labels = labels[:, 1:].contiguous()      # (B, T-1)

        B, Tm1, V = shift_logits.shape

        per_token_ce = F.cross_entropy(
            shift_logits.view(-1, V),
            shift_labels.view(-1),
            reduction="none",
            ignore_index=-100,
        ).view(B, Tm1)

        valid = (shift_labels != -100).to(per_token_ce.dtype)
        tok_cnt = valid.sum(dim=1).clamp(min=1.0)
        ce_per_ex = (per_token_ce * valid).sum(dim=1) / tok_cnt   # (B,)

        # ---- decide whether to compute POTracker weights this batch ----
        self._potracker_call_count += 1
        do_potracker = (self._potracker_call_count % self.potracker_every == 0)

        if do_potracker and self.potracker_weight > 0:
            with torch.no_grad():
                pred_ids = torch.argmax(logits, dim=-1)  # (B, T)
                scores_cpu = self._batch_potracker_per_example(pred_ids, labels)  # (B,) CPU

                scores = scores_cpu.to(device=logits.device)
                # penalty: higher means worse
                if self.potracker_mode == "inv":
                    penalty = 1.0 / (scores + self.potracker_eps)
                else:
                    penalty = 1.0 - scores

                # EMA baseline (CRUCIAL when batch size == 1)
                batch_mean = float(penalty.mean().detach().cpu())
                if self._penalty_ema is None:
                    self._penalty_ema = batch_mean
                else:
                    self._penalty_ema = self.ema_beta * self._penalty_ema + (1.0 - self.ema_beta) * batch_mean

                baseline = penalty.new_tensor(self._penalty_ema)

                # weights: centered around 1.0; clamped for stability
                weights = 1.0 + self.potracker_weight * (penalty - baseline)
                weights = torch.clamp(weights, self.weight_clip_min, self.weight_clip_max).detach()

        else:
            scores = None
            penalty = None
            weights = torch.ones_like(ce_per_ex).detach()

        # IMPORTANT:
        # Use mean(weights * ce) (NOT normalized by weights.sum())
        # so it still has an effect under batch_size=1 and grad accumulation.
        total_loss = (weights * ce_per_ex).mean()

        if float(do_potracker)==1.0:
            # ---- logging ----
            log_dict = {
                "total_loss": float(total_loss.detach().cpu()),
                "ce_loss_mean": float(ce_per_ex.mean().detach().cpu()),
                "potracker_do": float(do_potracker),
                "w_mean": float(weights.mean().detach().cpu()),
            }
            if scores is not None:
                log_dict.update({
                    "potracker_score_mean": float(scores.mean().detach().cpu()),
                    "potracker_penalty_mean": float(penalty.mean().detach().cpu()),
                    "potracker_penalty_ema": float(self._penalty_ema),
                })

            self.log(log_dict)

        return (total_loss, outputs) if return_outputs else total_loss



trainer = POTrackerLossTrainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset=valid_tokenized,
    processing_class=tokenizer,
    data_collator=data_collator,
    potracker_weight=0.5,       # try 0.2–1.0
    potracker_mode="1-minus",
    potracker_every=10,         # compute POTracker every 50 batches
    alpha_text=0.1,
    weight_clip_min=0.5,
    weight_clip_max=2.0,
    ema_beta=0.95,
)


trainer.train()
trainer.save_model(fop_output_model)
