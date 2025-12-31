import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import AutoPeftModelForCausalLM, PeftModel

adapter_dir = "/home/hungphd/git/finetuned/CodeLlama-7b-Instruct-hf/"   # where trainer.save_model(...) wrote
base_model = "/home/hungphd/git/CodeLlama-7b-Instruct-hf/"

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model)
tokenizer.pad_token = tokenizer.eos_token

# 4-bit (same as training)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
)

# Option A: load adapter directly (works if adapter saves base_model_name_or_path)
try:
    model = AutoPeftModelForCausalLM.from_pretrained(
        adapter_dir,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
        quantization_config=bnb_config,
    )
except Exception:
    # Option B: manually load base then attach adapter
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
    )
    model = PeftModel.from_pretrained(base, adapter_dir)

model.eval()

# Use the same template as training
prompt = (
    "### Instruction:\n"
    "Explain what a binary search tree is and why it's useful.\n\n"
    "### Response:\n"
)

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=True, temperature=0.7, top_p=0.9,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )

print(tokenizer.decode(output_ids[0], skip_special_tokens=True))
