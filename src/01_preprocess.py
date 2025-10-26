#!/usr/bin/env python
"""
01_preprocess.py
Clean and prepare Amazon Movies & TV reviews into a single Parquet file.

Usage:
  python src/01_preprocess.py --input_dir data/raw --output_parquet data/processed/movies_reviews.parquet --drop_neutral
"""
import argparse, os, glob, json, gzip, hashlib
import pandas as pd
from tqdm import tqdm

def iter_json_lines(path):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--output_parquet", required=True)
    ap.add_argument("--drop_neutral", action="store_true", help="Drop 3-star reviews")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.input_dir, "*.json*")))
    rows = []
    for fp in tqdm(files, desc="Reading"):
        for obj in iter_json_lines(fp):
            text = (obj.get("summary","") + ". " + obj.get("reviewText","")).strip()
            stars = obj.get("overall", None)
            if not text or stars is None:
                continue
            # Label mapping
            label = None
            if stars in [1,2]:
                label = 0
            elif stars in [4,5]:
                label = 1
            elif stars == 3 and not args.drop_neutral:
                label = 2
            else:
                continue
            rid = hashlib.md5((obj.get("asin","") + "|" + text).encode("utf-8")).hexdigest()
            rows.append({
                "id": rid,
                "asin": obj.get("asin"),
                "reviewTime": obj.get("reviewTime"),
                "verified": obj.get("verified"),
                "stars": stars,
                "label": label,
                "text": text
            })
    df = pd.DataFrame(rows).drop_duplicates(subset=["id"])
    # Optional: time-based split if reviewTime is parsable
    df["reviewTime"] = pd.to_datetime(df["reviewTime"], errors="coerce")
    df = df.sort_values("reviewTime")
    n = len(df)
    train_end = int(0.8*n)
    val_end = int(0.9*n)
    df.loc[:,"split"] = "train"
    df.iloc[train_end:val_end, df.columns.get_loc("split")] = "val"
    df.iloc[val_end:, df.columns.get_loc("split")] = "test"
    os.makedirs(os.path.dirname(args.output_parquet), exist_ok=True)
    df.to_parquet(args.output_parquet, index=False)
    print("Saved:", args.output_parquet, "rows:", len(df), "label counts:\n", df["label"].value_counts())

if __name__ == "__main__":
    main()
