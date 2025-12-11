import json

file_path = "../kernelbench_pytorch_jax_pairs_inline.jsonl"  # update path if needed

def load_kernelbench_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            records.append(obj)
    return records

if __name__ == "__main__":
    data = load_kernelbench_jsonl(file_path)
    print(f"Loaded {len(data)} records")

    # Peek at the first one
    first = data[0]
    print("Keys:", list(first.keys()))
    print("kb_id:", first["kb_id"])
    print("level:", first["level"])
    print("success:", first["success"])
    print("AQS:", first["aqs"])
    print("PyTorch path:", first["pytorch_path"])
    print("JAX path:", first["jax_path"])

    # If you want the actual code strings:
    pytorch_code = first["pytorch_code"]
    jax_code     = first["jax_code"]

    print("\n=== PyTorch code (truncated) ===")
    print(pytorch_code[:400], "...")
    print("\n=== JAX code (truncated) ===")
    print(jax_code[:400], "...")
