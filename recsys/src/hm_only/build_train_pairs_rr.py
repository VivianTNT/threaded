"""Build training pairs for RR (like build_train_pairs_hm)."""
import pandas as pd
import numpy as np
from pathlib import Path

DATA = Path("recsys/data")
OUT = DATA / "train_pairs_rr"
OUT.mkdir(parents=True, exist_ok=True)

events = pd.read_parquet(DATA / "events_rr.parquet")
rr = events[["user_id", "item_id"]].drop_duplicates()

print(f"[RR] RR interactions: {len(rr):,}")

users = rr["user_id"].unique()
items = rr["item_id"].unique()

positives = rr.copy()
positives["label"] = 1

num_neg = len(positives) * 4
neg_users = np.random.choice(users, size=num_neg, replace=True)
neg_items = np.random.choice(items, size=num_neg, replace=True)

negatives = pd.DataFrame({
    "user_id": neg_users,
    "item_id": neg_items,
    "label": np.zeros(num_neg, dtype=np.int8)
})

merged = pd.merge(negatives, positives[["user_id", "item_id"]], on=["user_id", "item_id"], how="left", indicator=True)
negatives = merged[merged["_merge"] == "left_only"].drop(columns=["_merge"])

pairs = pd.concat([positives, negatives], ignore_index=True).sample(frac=1.0).reset_index(drop=True)
print(f"[RR] Total pairs: {len(pairs):,}")

chunk_size = 2_000_000
for i in range(int(np.ceil(len(pairs) / chunk_size))):
    chunk = pairs.iloc[i*chunk_size:(i+1)*chunk_size]
    chunk.to_parquet(OUT / f"train_pairs_rr_part{i:03d}.parquet", index=False)
    print(f"[RR] Saved chunk {i+1}")

print("[RR] Done.")
