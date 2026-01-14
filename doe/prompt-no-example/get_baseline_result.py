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
Translating this input to standard XML file:
1. Input JSON string:
"""
{INPUT_XML}
"""


2. Please provide the output as standard XML and return the XML as output only
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


# ---------------- vLLM pipeline ----------------
def process_items_vllm(
    items: List[Dict[str, Any]],
    output_path: str,
    fp_local_model: str,
    *,
    output_field: str = "predicted_xml",
    max_new_tokens: int = 512,
    temperature: float = 0.0,
    top_p: float = 0.9,
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

    # Load tokenizer for chat template formatting
    tokenizer = AutoTokenizer.from_pretrained(fp_local_model, use_fast=True, trust_remote_code=True)

    # Initialize vLLM engine
    llm = LLM(
        model=fp_local_model,
        dtype=dtype,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
        trust_remote_code=True,
        gpu_memory_utilization=gpu_memory_utilization,
    )

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

    # vLLM sampling params
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_new_tokens,
        stop=stop or [],
    )

    # Generate in one batched call
    outputs = llm.generate(prompts, sampling_params)
    cache_size=20

    # Collect results in list_output
    list_output: List[Dict[str, Any]] = []
    idx=0
    for item, out in zip(items, outputs):
        is_success=False
        try:
            # Take the top candidate
            text = out.outputs[0].text.strip()
            # print('the text: {}'.format(text))
            # input('bbbb')
            xml = keep_xml_only(text)
            # print(xml)
            # input('bbb')
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
    # fp_local_model = "/home/hungphd/git/pretrained_open_llms/phi-4/"  # e.g., "/models/Llama-3.1-8B-Instruct"
    # fp_local_model = "/home/hungphd/git/pretrained_open_llms/Llama-3.1-8B/"

    model_name=fp_local_model.split('/')[-2]
    fop_output_result='../data-all/results/baselines_no_example/'
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
        dtype="bfloat16",
        tensor_parallel_size=1,
        max_model_len=2048,
        temperature=0.0,
        top_p=0.95,
        stop=None,  # add custom stop strings if you use a sentinel
    )
    print("Done. Saved to {}".format(fp_output))
