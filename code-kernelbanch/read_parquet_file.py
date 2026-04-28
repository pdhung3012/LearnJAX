import pandas as pd
from pathlib import Path
import re

fop_parquet=''
# Folder containing multiple parquet files
folder_path = Path("/home/hungphd/git/KB_datasets/KernelBench/data/")

# Read all parquet files
dfs = []

for file_path in folder_path.glob("*.parquet"):
    df = pd.read_parquet(file_path, engine="pyarrow")
    dfs.append(df)

# Combine all files into one DataFrame
combined_df = pd.concat(dfs, ignore_index=True)

print(combined_df.head())
print(combined_df.shape)
columns = list(combined_df.columns)
print(columns)

# Output folder
output_dir = Path("data_kb/")
output_dir.mkdir(parents=True, exist_ok=True)

def safe_filename(text):
    """
    Make text safe for filenames by replacing unsafe characters.
    """
    text = str(text)
    text = re.sub(r'[\\/*?:"<>|]', "_", text)
    text = text.strip()
    return text

for _, row in combined_df.iterrows():
    problem_id = safe_filename(row["problem_id"])
    level = safe_filename(row["level"])
    name = safe_filename(row["name"])

    file_name = f"{problem_id}_{level}_{name}.py"
    file_path = output_dir / file_name

    code_content = row["code"]

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(str(code_content))

print(f"Created {len(combined_df)} Python files in {output_dir}")