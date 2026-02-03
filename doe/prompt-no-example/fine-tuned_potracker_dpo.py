# pip install -U "transformers>=5" datasets accelerate bitsandbytes peft trl
import os
import re
import random
import pandas as pd
import torch
from datasets import Dataset
import xml.etree.ElementTree as ET

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# TRL (DPO)
try:
    from trl import DPOTrainer, DPOConfig
except Exception as e:
    raise ImportError(
        "Missing TRL. Install with: pip install -U trl"
    ) from e


# =========================
# 0) Paths / settings
# =========================
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Base model (policy init + reference model weights)
model_name = "/home/hungphd/git/pretrained_open_llms/Qwen2.5-7B-Instruct/"

# Output folder (LoRA adapter + trainer checkpoints)
folder_output = "/home/hungphd/git/potracker_dpo/"
#
# # Your original SFT CSVs (prompt, response)
# fp_file_sft_train = "/mnt/data/finetune_train.csv"
# fp_file_sft_valid = "/mnt/data/finetune_valid.csv"

# DPO CSVs to be created/used (prompt, response, bad_response)
fp_file_dpo_train = "../data-all/label-split/train_dpo.csv"
fp_file_dpo_valid = "../data-all/label-split/valid_dpo.csv"

SEED = 42

# DPO sequence limits (adjust if you OOM)
MAX_LENGTH = 2048
MAX_PROMPT_LENGTH = 1024


# =========================
# 1) Build DPO CSVs
# =========================
def _ensure_xml_decl(s: str) -> str:
    s = (s or "").strip()
    if s.startswith("<?xml"):
        return s
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + s

def _fallback_corrupt(text: str, rng: random.Random) -> str:
    t = (text or "").strip()
    if len(t) < 40:
        return _ensure_xml_decl("<bad>INVALID</bad>")
    cut_lo = max(20, len(t) // 5)
    cut_hi = max(40, len(t) // 2)
    cut = rng.randint(cut_lo, cut_hi)
    t2 = t[:cut].rstrip()
    if not t2.endswith(">"):
        t2 += ">"
    # inject a plausible-but-wrong field + broken closing tags
    t2 += "\n<metersServed>999999</metersServed>\n</Outage>\n</PubOutages>"
    return _ensure_xml_decl(t2)

def make_bad_response(good_xml: str, rng: random.Random, other_good: str | None = None) -> str:
    good_xml = (good_xml or "").strip()
    try:
        root = ET.fromstring(good_xml)
    except Exception:
        return _fallback_corrupt(good_xml, rng)

    elems = [e for e in root.iter()]
    parent_map = {c: p for p in root.iter() for c in p}
    leaves = [e for e in elems if len(list(e)) == 0 and (e.text is not None) and (e.text.strip() != "")]

    other_texts = []
    if other_good:
        try:
            oroot = ET.fromstring((other_good or "").strip())
            other_texts = [
                e.text.strip() for e in oroot.iter()
                if len(list(e)) == 0 and e.text and e.text.strip()
            ]
        except Exception:
            other_texts = []

    strategies = ["swap_leaf_text", "replace_leaf_text", "remove_element", "numeric_perturb"]
    strat = rng.choice(strategies)

    if strat == "swap_leaf_text" and len(leaves) >= 2:
        a, b = rng.sample(leaves, 2)
        a.text, b.text = b.text, a.text

    elif strat == "replace_leaf_text" and leaves:
        target = rng.choice(leaves)
        if other_texts:
            target.text = rng.choice(other_texts)
        else:
            target.text = rng.choice(["UNKNOWN", "N/A", "0", "999999", "Pending Investigation"])

    elif strat == "remove_element":
        candidates = [e for e in elems[1:] if e in parent_map]  # non-root
        if candidates:
            victim = rng.choice(candidates)
            parent_map[victim].remove(victim)
        elif leaves:
            rng.choice(leaves).text = "UNKNOWN"

    elif strat == "numeric_perturb" and leaves:
        target = rng.choice(leaves)
        txt = target.text.strip()
        if re.fullmatch(r"-?\d+(\.\d+)?", txt):
            try:
                val = float(txt)
                val = val + rng.choice([-1, 1]) * rng.uniform(1, 1000)
                target.text = str(int(val)) if float(val).is_integer() else str(val)
            except Exception:
                target.text = "0"
        else:
            target.text = rng.choice(["0", "UNKNOWN", "999999"])

    bad_body = ET.tostring(root, encoding="unicode")
    bad = _ensure_xml_decl(bad_body)

    # Ensure rejected differs from chosen
    if bad.strip() == _ensure_xml_decl(good_xml).strip() or len(bad) < 60:
        bad = _fallback_corrupt(good_xml, rng)
    return bad

def build_dpo_csv(inp_path: str, out_path: str, seed: int) -> None:
    df = pd.read_csv(inp_path, dtype=str, keep_default_na=False, na_filter=False)
    df["prompt"] = df["prompt"].fillna("").astype(str)
    df["response"] = df["response"].fillna("").astype(str)

    rng = random.Random(seed)
    all_good = df["response"].tolist()

    bad_list = []
    for good in all_good:
        other = all_good[rng.randrange(0, len(all_good))] if len(all_good) > 1 else None
        bad_list.append(make_bad_response(good, rng, other_good=other))

    df_out = df.copy()
    df_out["bad_response"] = bad_list

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df_out.to_csv(out_path, index=False)
    print(f"Wrote {out_path} with {len(df_out)} rows")

# if (not os.path.exists(fp_file_dpo_train)) or (not os.path.exists(fp_file_dpo_valid)):
#     print("DPO CSVs not found -> generating from SFT CSVs...")
#     build_dpo_csv(fp_file_sft_train, fp_file_dpo_train, seed=SEED)
#     build_dpo_csv(fp_file_sft_valid, fp_file_dpo_valid, seed=SEED + 1)


# =========================
# 2) Load DPO dataset
# =========================
df_train = pd.read_csv(fp_file_dpo_train, dtype=str, keep_default_na=False, na_filter=False)
df_valid = pd.read_csv(fp_file_dpo_valid, dtype=str, keep_default_na=False, na_filter=False)

for df in (df_train, df_valid):
    for c in ("prompt", "response", "bad_response"):
        df[c] = df[c].fillna("").astype(str)

train_ds = Dataset.from_pandas(df_train[["prompt", "response", "bad_response"]], preserve_index=False)
valid_ds = Dataset.from_pandas(df_valid[["prompt", "response", "bad_response"]], preserve_index=False)

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

def format_prompt(p: str) -> str:
    # Keep your original instruction template to match SFT formatting
    return f"### Instruction:\n{p}\n\n### Response:\n"

def to_dpo(example):
    prompt = format_prompt(example["prompt"])
    chosen = (example["response"] or "").strip()
    rejected = (example["bad_response"] or "").strip()

    # Ensure EOS at end of completions (helps truncation)
    if not chosen.endswith(tokenizer.eos_token):
        chosen = chosen + tokenizer.eos_token
    if not rejected.endswith(tokenizer.eos_token):
        rejected = rejected + tokenizer.eos_token

    return {"prompt": prompt, "chosen": chosen, "rejected": rejected}

train_dpo = train_ds.map(to_dpo, remove_columns=train_ds.column_names)
valid_dpo = valid_ds.map(to_dpo, remove_columns=valid_ds.column_names)


# =========================
# 3) Load policy + ref models (QLoRA)
# =========================
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
)

def load_base():
    m = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
    )
    m.config.use_cache = False
    return m

# Policy model = base + LoRA (trainable)
policy = load_base()
policy = prepare_model_for_kbit_training(policy)

peft_config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    task_type="CAUSAL_LM",
)
policy = get_peft_model(policy, peft_config)

# Reference model = frozen base (no LoRA)
ref_model = load_base()
ref_model.eval()
for p in ref_model.parameters():
    p.requires_grad_(False)


# =========================
# 4) DPO training
# =========================
real_model_name = model_name.rstrip("/").split("/")[-1]
output_dir = os.path.join(folder_output, real_model_name)

# Build DPOConfig with version-compat fallbacks (Transformers v4 vs v5 naming)
try:
    training_args = DPOConfig(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=1,
        learning_rate=1e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",  # Transformers v5+
        bf16=torch.cuda.is_available(),
        fp16=False,
        report_to="none",
        optim="paged_adamw_8bit",
        beta=0.1,  # DPO temperature (tune 0.05 ~ 0.5)
        max_length=MAX_LENGTH,
        max_prompt_length=MAX_PROMPT_LENGTH,
    )
except TypeError:
    training_args = DPOConfig(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=1,
        learning_rate=1e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        save_strategy="epoch",
        evaluation_strategy="epoch",  # Transformers v4
        bf16=torch.cuda.is_available(),
        fp16=False,
        report_to="none",
        optim="paged_adamw_8bit",
        beta=0.1,
        max_length=MAX_LENGTH,
        max_prompt_length=MAX_PROMPT_LENGTH,
    )

# TRL/Transformers version compatibility: some versions use `tokenizer=...`,
# others use `processing_class=...` like Transformers v5.
try:
    dpo_trainer = DPOTrainer(
        model=policy,
        ref_model=ref_model,
        args=training_args,
        train_dataset=train_dpo,
        eval_dataset=valid_dpo,
        processing_class=tokenizer,
    )
except TypeError:
    dpo_trainer = DPOTrainer(
        model=policy,
        ref_model=ref_model,
        args=training_args,
        train_dataset=train_dpo,
        eval_dataset=valid_dpo,
        tokenizer=tokenizer,
    )

dpo_trainer.train()
dpo_trainer.save_model(output_dir)
print(f"Saved DPO model to: {output_dir}")
