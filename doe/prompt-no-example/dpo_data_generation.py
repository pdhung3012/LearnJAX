#!/usr/bin/env python3
"""
Generate train_dpo.csv / valid_dpo.csv from finetune_train.csv / finetune_valid.csv

Input CSV columns required:
  - prompt
  - response   (this is the "chosen" / good output)

Output CSV columns:
  - prompt
  - response
  - bad_response   (synthetic "rejected" output)

Usage:
  python make_dpo_csv.py \
    --train_in finetune_train.csv --valid_in finetune_valid.csv \
    --train_out train_dpo.csv --valid_out valid_dpo.csv
"""

import argparse
import os
import random
import re
from typing import Optional

import pandas as pd
import xml.etree.ElementTree as ET


def ensure_xml_decl(s: str) -> str:
    s = (s or "").strip()
    if s.startswith("<?xml"):
        return s
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + s


def fallback_corrupt(text: str, rng: random.Random) -> str:
    """Make something XML-ish but wrong: truncate + inject broken closing tags."""
    t = (text or "").strip()
    if len(t) < 40:
        return ensure_xml_decl("<bad>INVALID</bad>")

    cut_lo = max(20, len(t) // 5)
    cut_hi = max(40, len(t) // 2)
    cut = rng.randint(cut_lo, cut_hi)
    t2 = t[:cut].rstrip()

    if not t2.endswith(">"):
        t2 += ">"

    # inject a plausible-but-wrong field + broken closing tags
    t2 += "\n<metersServed>999999</metersServed>\n</Outage>\n</PubOutages>"
    return ensure_xml_decl(t2)


def make_bad_response(good_xml: str, rng: random.Random, other_good: Optional[str] = None) -> str:
    """Corrupt the chosen XML in simple ways to create a 'rejected' example."""
    good_xml = (good_xml or "").strip()

    # If parsing fails, just return a corrupted fallback.
    try:
        root = ET.fromstring(good_xml)
    except Exception:
        return fallback_corrupt(good_xml, rng)

    elems = list(root.iter())
    parent_map = {c: p for p in root.iter() for c in p}

    # Leaf nodes with text content
    leaves = [
        e for e in elems
        if len(list(e)) == 0 and e.text is not None and e.text.strip() != ""
    ]

    # Collect leaf texts from another sample (optional) for "mixing" corruption
    other_texts = []
    if other_good:
        try:
            oroot = ET.fromstring((other_good or "").strip())
            other_texts = [
                e.text.strip()
                for e in oroot.iter()
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
        # Remove a random non-root element if possible
        candidates = [e for e in elems[1:] if e in parent_map]
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
    bad = ensure_xml_decl(bad_body)

    # Ensure it's not identical (or too short); otherwise fallback-corrupt.
    if bad.strip() == ensure_xml_decl(good_xml).strip() or len(bad) < 60:
        bad = fallback_corrupt(good_xml, rng)

    return bad


def build_dpo_csv(inp_path: str, out_path: str, seed: int) -> None:
    df = pd.read_csv(inp_path, dtype=str, keep_default_na=False, na_filter=False)

    # Validate required columns
    for col in ("prompt", "response"):
        if col not in df.columns:
            raise ValueError(f"Missing required column '{col}' in {inp_path}")

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

    out_dir = os.path.dirname(os.path.abspath(out_path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    df_out.to_csv(out_path, index=False)
    print(f"Wrote {out_path} with {len(df_out)} rows")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_in", help="Path to finetune_train.csv",default="../data-all/label-split/finetune_noex_train.csv")
    ap.add_argument("--valid_in",  help="Path to finetune_valid.csv",default="../data-all/label-split/finetune_noex_valid.csv")
    ap.add_argument("--train_out", default="../data-all/label-split/train_dpo.csv", help="Output path for train_dpo.csv")
    ap.add_argument("--valid_out", default="../data-all/label-split/valid_dpo.csv", help="Output path for valid_dpo.csv")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed for train bad_response generation")
    ap.add_argument("--valid_seed_offset", type=int, default=1, help="Offset added to seed for valid split")
    args = ap.parse_args()

    build_dpo_csv(args.train_in, args.train_out, seed=args.seed)
    build_dpo_csv(args.valid_in, args.valid_out, seed=args.seed + args.valid_seed_offset)


if __name__ == "__main__":
    main()
