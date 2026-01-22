"""
requirements:
  pip install vllm transformers

Notes:
- fp_local_model must be a local HuggingFace-style folder (config.json, tokenizer.json,
  model.safetensors/bin, etc.). Examples: Llama 3 Instruct, Mistral Instruct, Qwen2.5 Instruct.
- vLLM uses GPU; set CUDA_VISIBLE_DEVICES if needed.
"""

import json
from copy import deepcopy
from typing import Any, Dict, List, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from doe.utils import *

# ---------------- Hard "no-reasoning" guard ----------------
NO_REASONING_RULES = """
CRITICAL OUTPUT RULES:
- Output ONLY the final XML document.
- Do NOT include explanations, reasoning, analysis, or steps.
- Do NOT include Markdown code fences, labels, or comments.
- If you cannot produce valid XML, output exactly: <error/>
""".strip()

# ---------------- Your prompt template ----------------
class LabelGenerationPromptTemplate:
    system_template='''
You are an expert in XML generation. Your task is to generate the standard XML from non-standard input json text. 
    '''
    prompt_template='''
You are an expert in XML generation. Your task is to generate the standard XML from non-standard input json text. 
Translating this input to standard XML file.
1. Input JSON string:
"""
{INPUT_XML}
"""


2. Please provide the output as standard XML and return the XML as output only.
'''.strip()




# ---------------- Helpers ----------------
def keep_xml_only(text: str) -> str:
    """Defensively keep only the first XML-looking block."""
    if not text:
        return text
    start = text.find("<")
    end = text.rfind(">")
    return text[start:end + 1].strip() if start != -1 and end > start else text.strip()


def build_messages(item: Dict[str, Any], tmpl: LabelGenerationPromptTemplate) -> List[Dict[str, str]]:
    # print(item)
    item.pop('standard_xml')
    json_str = json.dumps(item, ensure_ascii=False, indent=2)
    # print(json_str)
    # input('aaaa')
    user_prompt = tmpl.prompt_template.replace("{INPUT_XML}", json_str)
    return [
        {"role": "system", "content": tmpl.system_template},
        {"role": "user", "content": user_prompt},
    ]


def format_with_chat_template(tokenizer: AutoTokenizer, messages: List[Dict[str, str]]) -> str:
    """Use the model's native chat template if available; fall back to a simple transcript."""
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            pass
    # Fallback: plain role-tagged transcript
    parts = []
    for m in messages:
        role = m.get("role", "user")
        parts.append(f"<|{role}|>\n{m.get('content','').strip()}\n")
    parts.append("<|assistant|>\n")
    return "\n".join(parts)


def ask_batch(prompts,base,model,tokenizer, max_new_tokens=512, do_sample=True, temperature=0.7, top_p=0.9):
    """
    prompts: List[str]
    returns: List[str] (answers only, not including the prompt text)
    """
    if isinstance(prompts, str):
        prompts = [prompts]

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

# ---------------- vLLM pipeline ----------------
def process_items_vllm(
    items: List[Dict[str, Any]],
    output_path: str,
    fp_local_model: str,
    fp_adapter_weight:str,
    *,
    output_field: str = "predicted_xml",
    max_new_tokens: int = 1200,
    temperature: float = 0.0,
    top_p: float = 0.7,
    tensor_parallel_size: int = 1,
    dtype: str = "bfloat16",            # or "float16" / "auto"
    max_model_len: int = 8192,
    gpu_memory_utilization: float = 0.9,
    stop: Optional[List[str]] = None,   # e.g., custom stop strings if you have a sentinel
) -> None:
    """
    Uses vLLM to generate XML for each input dict with a local model at `fp_local_model`.
    Saves a JSON array of cloned dicts (with `output_field`) to `output_path`.
    """
    tmpl = LabelGenerationPromptTemplate()

    tokenizer = AutoTokenizer.from_pretrained(fp_local_model, use_fast=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # good for generation with padding

    base = AutoModelForCausalLM.from_pretrained(
        fp_local_model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )

    model = PeftModel.from_pretrained(base, fp_adapter_weight)
    model.eval()

    # Build prompts (batched for efficiency)
    prompts: List[str] = []
    for item in items:
        messages = build_messages(item, tmpl)
        prompt = format_with_chat_template(tokenizer, messages)
        # print(prompt)
        # f1=open('sample.txt','w')
        # f1.write(prompt)
        # f1.close()
        # input('aaa')
        prompts.append(prompt)
    # prompts=prompts[:10]
    len_prompt=len(prompts)
    cache_size=10
    batch_num=len_prompt//cache_size
    answers=[]
    for ind in range(0,batch_num):
        indStart=ind*cache_size
        indEnd=(ind+1)*cache_size-1
        if indEnd>=(len_prompt-1):
            indEnd=len_prompt-1
        sub_prompts=prompts[indStart:(indEnd+1)]
        enc = tokenizer(
            sub_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        enc = {k: v.to(model.device) for k, v in enc.items()}

        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=True,
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
        sub_answers = []
        for i, seq in enumerate(out):
            gen_tokens = seq[input_lens[i]:]  # tokens after the prompt
            sub_answers.append(tokenizer.decode(gen_tokens, skip_special_tokens=True).strip())
        answers+=sub_answers
        print('finish from {} to {}'.format(indStart,indEnd))

    # Collect results in list_output
    list_output: List[Dict[str, Any]] = []
    idx=0
    cache_size=20
    for item, out in zip(items, answers):
        is_success=False
        try:
            # Take the top candidate
            xml = keep_xml_only(out)
            print(xml)
            input('bbb')
            cloned = deepcopy(item)
            cloned[output_field] = xml
            list_output.append(cloned)
            is_success = True
        except Exception as e:
            cloned = deepcopy(item)
            cloned[output_field] = None
            cloned["error"] = f"{type(e).__name__}: {e}"
            list_output.append(cloned)
            print('error {}'.format(str(e)))
        print('handle index {} {}'.format(idx,is_success))
        if (idx + 1) % cache_size == 0:
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(list_output, f, ensure_ascii=False, indent=2)
                f.write("\n")
            print('cache result at {}'.format(idx + 1))
        idx+=1

    # Save to file
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(list_output, f, ensure_ascii=False, indent=2)
        f.write("\n")


# ---------------- Example ----------------
if __name__ == "__main__":
    fp_local_model = "/home/hungphd/git/pretrained_open_llms/Qwen2.5-7B-Instruct/"  # e.g., "/models/Llama-3.1-8B-Instruct"

    model_name=fp_local_model.split('/')[-2]
    fop_adapter_weight='/home/hungphd/git/finetuned_weights_noex_potracker/'+model_name+'/'
    name_output_folder=fop_adapter_weight.split('/')[-3]
    fop_output_result='../data-all/results/'+name_output_folder+'/'
    fp_output=fop_output_result+'test.'+model_name+'.json'
    fp_input_file= '../data-all/label-split/test.json'
    input_items =load_list_from_file(fp_input_file)
    # for i in range(0,len(input_items)):
    #     print(input_items[i])
    #     input_items[i].pop('standard_xml',None)
    ensure_dir_os(fop_output_result)
    # input_items = [
    #     {"utility_name": "Example Utility A", "county": "Shelby", "outages": 123},
    #     {"utility_name": "Example Utility B", "zip": "37996", "outages": 45},
    # ]
    process_items_vllm(
        input_items,
        output_path=fp_output,
        fp_local_model=fp_local_model,
        fp_adapter_weight=fop_adapter_weight,
        dtype="bfloat16",
        tensor_parallel_size=1,
        max_model_len=2048,
        temperature=0.7,
        top_p=0.95,
        stop=None,  # add custom stop strings if you use a sentinel
    )
    print("Done. Saved to {}".format(fp_output))
