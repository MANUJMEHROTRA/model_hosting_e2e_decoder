"""
Step 1: Download CNN/DailyMail dataset from HuggingFace and save as CSV.

We download a small chunk (train: 5000, val: 500, test: 500) to keep things
manageable for fine-tuning on a local machine / Colab.
"""

import os
from datasets import load_dataset
import pandas as pd

# ── Config ──────────────────────────────────────────────────────────────────
DATASET_NAME   = "abisee/cnn_dailymail"
DATASET_CONFIG = "3.0.0"          # version of the dataset on HF
SAVE_DIR       = os.path.dirname(os.path.abspath(__file__))  # same dir as script

CHUNK_SIZES = {
    "train": 5_000,   # enough samples to show meaningful fine-tuning
    "validation": 500,
    "test": 500,
}

# ── Download & Save ──────────────────────────────────────────────────────────
def download_and_save():
    print(f"Loading '{DATASET_NAME}' (config={DATASET_CONFIG}) from HuggingFace…")
    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG)

    for split, n in CHUNK_SIZES.items():
        hf_split = "validation" if split == "validation" else split
        data = dataset[hf_split].select(range(min(n, len(dataset[hf_split]))))

        df = pd.DataFrame({
            "article":   data["article"],     # source text (news article)
            "highlights": data["highlights"], # reference summary
            "id":         data["id"],
        })

        out_path = os.path.join(SAVE_DIR, f"cnn_dailymail_{split}.csv")
        df.to_csv(out_path, index=False)
        print(f"  ✓ Saved {len(df):,} rows → {out_path}")

    print("\nDataset statistics:")
    df_train = pd.read_csv(os.path.join(SAVE_DIR, "cnn_dailymail_train.csv"))
    print(f"  article length  (mean chars): {df_train['article'].str.len().mean():.0f}")
    print(f"  highlight length(mean chars): {df_train['highlights'].str.len().mean():.0f}")


if __name__ == "__main__":
    download_and_save()
