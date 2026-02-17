#!/usr/bin/env python3
"""
Export user-item interactions for Apache Mahout / Spark ALS.

Reads from hm_raw (+ hm_images) for H&M and rr_raw for RetailRocket.
Datasets remain strictly separated (no mixed IDs).

Output schema: user,item,value (no negatives).
- Deduplicates (user,item) pairs (aggregates value)
- Remaps IDs to contiguous integers
- Caps max interactions per user/item for faster training
- Saves mapping files for back-conversion
"""
import argparse
from typing import Optional

import pandas as pd
import numpy as np
import joblib
from pathlib import Path

DATA = Path("recsys/data")
MAHOUT_DIR = DATA / "mahout"
MAHOUT_DIR.mkdir(parents=True, exist_ok=True)


def load_hm_interactions(nrows=None, chunksize=500_000):
    """Load H&M interactions from hm_raw/transactions_train.csv."""
    tx_path = DATA / "hm_raw" / "transactions_train.csv"
    if not tx_path.exists():
        raise FileNotFoundError(f"H&M transactions not found: {tx_path}")

    print("[export] Loading H&M transactions (hm_raw)...")
    # Same ID scheme as hm_loader: user_id from customer_id codes, item_id = 2 + article_id
    chunks = []
    for chunk in pd.read_csv(
        tx_path,
        usecols=["customer_id", "article_id"],
        nrows=nrows,
        chunksize=chunksize,
        low_memory=False,
    ):
        chunk["item_id"] = chunk["article_id"].astype(str).apply(lambda x: int("2" + x))
        chunks.append(chunk[["customer_id", "item_id"]])

    df = pd.concat(chunks, ignore_index=True)
    df["user_id"] = df["customer_id"].astype("category").cat.codes.astype("int64")
    df["value"] = 1.0
    return df[["user_id", "item_id", "value"]]


def load_rr_interactions(nrows=None, chunksize=500_000):
    """Load RetailRocket interactions from rr_raw/events.csv."""
    events_path = DATA / "rr_raw" / "events.csv"
    if not events_path.exists():
        raise FileNotFoundError(f"RetailRocket events not found: {events_path}")

    print("[export] Loading RetailRocket events (rr_raw)...")
    keep = {"view", "addtocart", "transaction"}
    chunks = []
    for chunk in pd.read_csv(
        events_path,
        usecols=["visitorid", "event", "itemid"],
        nrows=nrows,
        chunksize=chunksize,
        low_memory=False,
    ):
        chunk = chunk[chunk["event"].isin(keep)].copy()
        chunk = chunk.rename(columns={"visitorid": "user_id", "itemid": "item_id"})
        chunk["value"] = 1.0
        chunks.append(chunk[["user_id", "item_id", "value"]])
    df = pd.concat(chunks, ignore_index=True)
    return df


def export_dataset(
    dataset: str,
    max_per_user: int = 500,
    max_per_item: int = 2000,
    nrows: Optional[int] = None,
) -> None:
    """Export, dedup, remap, cap, and save interactions for a dataset."""
    if dataset == "hm":
        df = load_hm_interactions(nrows=nrows)
    elif dataset == "rr":
        df = load_rr_interactions(nrows=nrows)
    else:
        raise ValueError(f"Unknown dataset: {dataset}. Use 'hm' or 'rr'.")

    print(f"[export] Raw interactions: {len(df):,}")

    # 1. Deduplicate: aggregate (user_id, item_id) -> sum(value)
    df = df.groupby(["user_id", "item_id"], as_index=False)["value"].sum()
    print(f"[export] After dedup (aggregate): {len(df):,}")

    # 2. Cap interactions per user
    rng = np.random.default_rng(42)
    user_counts = df.groupby("user_id").size()
    users_over = user_counts[user_counts > max_per_user].index
    if len(users_over) > 0:
        to_drop = []
        for uid in users_over:
            idx = df[df["user_id"] == uid].index.to_numpy()
            keep = rng.choice(idx, size=max_per_user, replace=False)
            to_drop.extend(idx[~np.isin(idx, keep)])
        df = df.drop(to_drop)
        print(f"[export] After user cap ({max_per_user}): {len(df):,}")

    # 3. Cap interactions per item
    item_counts = df.groupby("item_id").size()
    items_over = item_counts[item_counts > max_per_item].index
    if len(items_over) > 0:
        to_drop = []
        for iid in items_over:
            idx = df[df["item_id"] == iid].index.to_numpy()
            keep = rng.choice(idx, size=min(max_per_item, len(idx)), replace=False)
            to_drop.extend(idx[~np.isin(idx, keep)])
        df = df.drop(to_drop)
        print(f"[export] After item cap ({max_per_item}): {len(df):,}")

    # 4. Remap to contiguous integers
    unique_users = sorted(df["user_id"].unique())
    unique_items = sorted(df["item_id"].unique())

    user_id_to_idx = {int(u): i for i, u in enumerate(unique_users)}
    item_id_to_idx = {int(i): j for j, i in enumerate(unique_items)}
    idx_to_user_id = {i: int(u) for i, u in enumerate(unique_users)}
    idx_to_item_id = {j: int(i) for j, i in enumerate(unique_items)}

    df["user"] = df["user_id"].map(user_id_to_idx)
    df["item"] = df["item_id"].map(item_id_to_idx)
    df = df[["user", "item", "value"]].dropna()

    print(f"[export] Users: {len(unique_users):,}  Items: {len(unique_items):,}  Pairs: {len(df):,}")

    # 5. Save CSV (schema: user,item,value)
    out_csv = MAHOUT_DIR / f"interactions_{dataset}.csv"
    df.to_csv(out_csv, index=False, header=True)
    print(f"[export] Wrote {out_csv}")

    # 6. Save mappings for back-conversion
    mappings = {
        "user_id_to_idx": user_id_to_idx,
        "item_id_to_idx": item_id_to_idx,
        "idx_to_user_id": idx_to_user_id,
        "idx_to_item_id": idx_to_item_id,
        "n_users": len(unique_users),
        "n_items": len(unique_items),
    }
    map_path = MAHOUT_DIR / f"id_mappings_{dataset}.joblib"
    joblib.dump(mappings, map_path)
    print(f"[export] Wrote {map_path}")

    print(f"[export] Done for {dataset}.")


def main():
    parser = argparse.ArgumentParser(description="Export interactions for Mahout/Spark ALS")
    parser.add_argument("--dataset", type=str, required=True, choices=["hm", "rr"])
    parser.add_argument("--max-per-user", type=int, default=500, help="Cap interactions per user")
    parser.add_argument("--max-per-item", type=int, default=2000, help="Cap interactions per item")
    parser.add_argument("--nrows", type=int, default=None, help="Limit rows (for testing)")
    args = parser.parse_args()

    export_dataset(
        dataset=args.dataset,
        max_per_user=args.max_per_user,
        max_per_item=args.max_per_item,
        nrows=args.nrows,
    )


if __name__ == "__main__":
    main()
