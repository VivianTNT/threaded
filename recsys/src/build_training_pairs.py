import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

DATA = Path("recsys/data")
NEGATIVE_RATIO = 4  # how many negatives per positive

print("[build_training_pairs] loading events ...")
events = pd.read_parquet(DATA / "events.parquet")

assert {"user_id", "item_id"}.issubset(events.columns), "events.parquet missing columns"
print(f"[build_training_pairs] loaded {len(events):,} events")

# Drop duplicates
events = events.drop_duplicates(subset=["user_id", "item_id"])
n_users = events["user_id"].nunique()
n_items = events["item_id"].nunique()
print(f"[build_training_pairs] {n_users:,} unique users, {n_items:,} unique items")

# --- Positive samples ---
positive_pairs = events[["user_id", "item_id"]].copy()
positive_pairs["label"] = 1

# --- Negative sampling (vectorized) ---
print("[build_training_pairs] sampling negatives ...")

n_pos = len(positive_pairs)
n_neg = n_pos * NEGATIVE_RATIO

print("[build_training_pairs] sampling user_ids for negatives ...")
user_ids = events["user_id"].sample(n=n_neg, replace=True, random_state=42).to_numpy()

print("[build_training_pairs] sampling item_ids for negatives ...")
unique_items = events["item_id"].unique()
item_ids = np.random.choice(unique_items, size=n_neg, replace=True)

print("[build_training_pairs] creating negative pairs dataframe ...")
neg_df = pd.DataFrame({"user_id": user_ids, "item_id": item_ids})

print("[build_training_pairs] filtering out accidental positives ...")
pos_set = set(zip(events["user_id"], events["item_id"]))
neg_df["pair"] = list(zip(neg_df["user_id"], neg_df["item_id"]))
neg_df = neg_df[~neg_df["pair"].isin(pos_set)]
neg_df = neg_df.drop(columns=["pair"]).head(n_neg)
neg_df["label"] = 0

# --- Combine and shuffle ---
train_df = pd.concat([positive_pairs, neg_df], ignore_index=True)
train_df = train_df.sample(frac=1.0, random_state=42).reset_index(drop=True)

print(f"[build_training_pairs] created {len(train_df):,} training pairs "
      f"({train_df['label'].sum():,} positives, {len(train_df)-train_df['label'].sum():,} negatives)")

# --- Optional timestamp ---
if "timestamp" in events.columns:
    ts_map = events.groupby(["user_id", "item_id"])["timestamp"].max().to_dict()
    train_df["timestamp"] = train_df.apply(lambda r: ts_map.get((r["user_id"], r["item_id"]), np.nan), axis=1)

# --- Save in chunks ---
out_dir = DATA / "train_pairs"
out_dir.mkdir(parents=True, exist_ok=True)

chunk_size = 5_000_000
num_chunks = (len(train_df) + chunk_size - 1) // chunk_size

for i in range(num_chunks):
    start_idx = i * chunk_size
    end_idx = min((i + 1) * chunk_size, len(train_df))
    chunk = train_df.iloc[start_idx:end_idx]
    out_path = out_dir / f"train_pairs_part{i:03d}.parquet"
    chunk.to_parquet(out_path, index=False)
    print(f"✅ saved chunk {i+1}/{num_chunks} to {out_path}")

print("✅ done writing all chunks")