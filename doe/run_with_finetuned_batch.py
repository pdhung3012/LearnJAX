import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

base_model_path = "/home/hungphd/git/pretrained_open_llms/Llama-3.1-8B/"
adapter_dir     = "/home/hungphd/git/potracker_adapter_weights/Llama-3.1-8B/"

tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"  # good for generation with padding

base = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
)

model = PeftModel.from_pretrained(base, adapter_dir)
model.eval()

@torch.inference_mode()
def ask_batch(prompts, max_new_tokens=512, do_sample=True, temperature=0.7, top_p=0.9):
    """
    prompts: List[str]
    returns: List[str] (answers only, not including the prompt text)
    """
    if isinstance(prompts, str):
        prompts = [prompts]
    for i in range(0,100):
        prompts.append(prompts[0])

    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    enc = {k: v.to(model.device) for k, v in enc.items()}

    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        return_dict_in_generate=False,
    )

    # Decode full sequences
    decoded = tokenizer.batch_decode(out, skip_special_tokens=True)

    # Option 1 (recommended): return only newly generated tokens (answer-only)
    # We do this by slicing off the prompt length (in tokens) per row.
    input_lens = enc["attention_mask"].sum(dim=1).tolist()
    answers = []
    for i, seq in enumerate(out):
        gen_tokens = seq[input_lens[i]:]  # tokens after the prompt
        answers.append(tokenizer.decode(gen_tokens, skip_special_tokens=True).strip())

    # If you instead want the full text (prompt + answer), return `decoded`.
    return answers

# Example usage
queries = [
    "### Instruction:\nConvert this JSON into standard_xml:\n{...}\n\n### Response:\n",
    "### Instruction:\nSummarize this:\nHello world\n\n### Response:\n",
]

answers = ask_batch(queries, max_new_tokens=256)
for i, a in enumerate(answers, 1):
    print(f"\n--- Answer {i} ---\n{a}")
