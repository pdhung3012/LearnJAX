import json
import csv
from prompt_template import BasicPromptTemplate

file_path = "../kernelbench_pytorch_jax_pairs_inline.jsonl"  # update path if needed
fp_output = "code_finetuned.csv"  # output CSV

def load_kernelbench_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records

if __name__ == "__main__":
    data = load_kernelbench_jsonl(file_path)
    print(f"Loaded {len(data)} records")

    with open(fp_output, "w", encoding="utf-8", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=["prompt", "response"])
        writer.writeheader()

        for index, item in enumerate(data):
            # grab code strings
            pytorch_code = item.get("pytorch_code", "")
            jax_code     = item.get("jax_code", "")

            # build training pair
            text = BasicPromptTemplate.str_query.replace("{PYTORCH_CODE}", pytorch_code)
            label = BasicPromptTemplate.str_answer.replace("{JAX_CODE}", jax_code)

            writer.writerow({"prompt": text, "response": label})

            # optional peek
            # if index < 1:
            #     print("\n=== text (truncated) ===")
            #     print(text[:400], "...")
            #     print("\n=== label (truncated) ===")
            #     print(label[:400], "...")

    print(f"Wrote: {fp_output}")
