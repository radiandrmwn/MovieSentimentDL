#!/usr/bin/env python
"""
01_preprocess_imdb.py
Clean and prepare IMDb reviews into a single Parquet file.

Usage:
  python src/01_preprocess_imdb.py --input_dir data/imdb_raw/aclImdb --output_parquet data/processed/imdb_reviews.parquet
"""
import argparse
import os
import hashlib
import pandas as pd
from tqdm import tqdm


def read_imdb_folder(folder_path, label, split_name):
    """Read all text files from a folder and assign label"""
    rows = []
    if not os.path.exists(folder_path):
        print(f"Warning: {folder_path} does not exist, skipping...")
        return rows

    files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]
    for filename in tqdm(files, desc=f"Reading {split_name}/{label}"):
        filepath = os.path.join(folder_path, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read().strip()
                if not text:
                    continue
                # Create unique ID from filename and text
                rid = hashlib.md5((filename + text).encode("utf-8")).hexdigest()
                rows.append({
                    "id": rid,
                    "label": label,  # 0=negative, 1=positive
                    "text": text,
                    "split": split_name
                })
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            continue

    return rows


def main():
    ap = argparse.ArgumentParser(description="Preprocess IMDb dataset")
    ap.add_argument("--input_dir", required=True, help="Path to aclImdb folder")
    ap.add_argument("--output_parquet", required=True, help="Output parquet file path")
    args = ap.parse_args()

    print(f"Processing IMDb dataset from: {args.input_dir}")

    all_rows = []

    # Read training data
    print("\nProcessing training data...")
    train_pos = read_imdb_folder(os.path.join(args.input_dir, "train/pos"), label=1, split_name="train")
    train_neg = read_imdb_folder(os.path.join(args.input_dir, "train/neg"), label=0, split_name="train")
    all_rows.extend(train_pos)
    all_rows.extend(train_neg)

    # Read test data
    print("\nProcessing test data...")
    test_pos = read_imdb_folder(os.path.join(args.input_dir, "test/pos"), label=1, split_name="test")
    test_neg = read_imdb_folder(os.path.join(args.input_dir, "test/neg"), label=0, split_name="test")
    all_rows.extend(test_pos)
    all_rows.extend(test_neg)

    # Create dataframe
    df = pd.DataFrame(all_rows)

    # Remove duplicates based on ID
    df = df.drop_duplicates(subset=["id"])

    # Create validation split from training data (10% of train)
    train_df = df[df['split'] == 'train'].copy()
    test_df = df[df['split'] == 'test'].copy()

    # Shuffle train data
    train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Split train into train and validation (90/10)
    n_train = len(train_df)
    val_idx = int(0.9 * n_train)

    val_df = train_df.iloc[val_idx:].copy()
    val_df['split'] = 'val'
    train_df = train_df.iloc[:val_idx].copy()

    # Combine all splits
    df_final = pd.concat([train_df, val_df, test_df], ignore_index=True)

    # Save to parquet
    os.makedirs(os.path.dirname(args.output_parquet), exist_ok=True)
    df_final.to_parquet(args.output_parquet, index=False)

    print(f"\n{'='*60}")
    print(f"Saved: {args.output_parquet}")
    print(f"{'='*60}")
    print(f"Total rows: {len(df_final)}")
    print(f"\nLabel distribution:")
    print(df_final['label'].value_counts().sort_index())
    print(f"\nSplit distribution:")
    print(df_final['split'].value_counts())
    print(f"\nLabel distribution by split:")
    print(df_final.groupby(['split', 'label']).size().unstack(fill_value=0))
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
