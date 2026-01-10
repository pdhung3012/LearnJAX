import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

base_model_path = "/home/hungphd/git/pretrained_open_llms/Llama-3.1-8B/"
adapter_dir     = "/home/hungphd/git/potracker_adapter_weights/Llama-3.1-8B/"  # <-- your saved adapter folder

tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"  # often better for generation

# Load base (choose dtype + device_map that fits your GPU)
base = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
)

# Attach adapter
model = PeftModel.from_pretrained(base, adapter_dir)
model.eval()

def ask(prompt: str, max_new_tokens=512):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    text = tokenizer.decode(out[0], skip_special_tokens=True)
    return text

query = "### Instruction:\nConvert this JSON into standard_xml:\n{...}\n\n### Response:\n"
print(ask(query))
