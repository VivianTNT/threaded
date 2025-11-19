# recsys/src/build_train_pairs_hm.py

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import gc

DATA = Path("recsys/data")
ART = Path("recsys/artifacts")
OUT = DATA / "train_pairs_hm"
OUT.mkdir(parents=True, exist_ok=True)

print("[HM] Loading events.parquet...")
events = pd.read_parquet(DATA / "events_hm.parquet")

# H&M IDs are small: < 3,000,000
hm = events[(events["user_id"] < 3_000_000) & (events["item_id"] < 3_000_000)].copy()
hm = hm[["user_id", "item_id"]].drop_duplicates()

print(f"[HM] H&M interactions: {len(hm):,}")

users = hm["user_id"].unique()
items = hm["item_id"].unique()

print(f"[HM] Users: {len(users):,}   Items: {len(items):,}")

# positives
positives = hm.copy()
positives["label"] = 1

# negative sampling ratio
num_neg = len(positives) * 4

print(f"[HM] Sampling {num_neg:,} negatives...")

neg_users = np.random.choice(users, size=num_neg, replace=True)
neg_items = np.random.choice(items, size=num_neg, replace=True)

negatives = pd.DataFrame({
    "user_id": neg_users,
    "item_id": neg_items,
    "label": np.zeros(num_neg, dtype=np.int8)
})

# drop accidental positives
merged = pd.merge(
    negatives,
    positives[["user_id", "item_id"]],
    on=["user_id", "item_id"],
    how="left",
    indicator=True
)

negatives = merged[merged["_merge"] == "left_only"].drop(columns=["_merge"])

pairs = pd.concat([positives, negatives], ignore_index=True)
pairs = pairs.sample(frac=1.0).reset_index(drop=True)

print(f"[HM] Total training pairs: {len(pairs):,}")

# chunk into manageable files
chunk_size = 2_000_000
num_chunks = int(np.ceil(len(pairs) / chunk_size))

for i in range(num_chunks):
    chunk = pairs.iloc[i*chunk_size:(i+1)*chunk_size]
    out_path = OUT / f"train_pairs_hm_part{i:03d}.parquet"
    chunk.to_parquet(out_path, index=False)
    print(f"[HM] Saved chunk {i+1}/{num_chunks}")

print("[HM] Done.")