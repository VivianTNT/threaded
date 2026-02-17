#!/usr/bin/env python3
"""
Convert Spark ALS output (user_factors.csv, item_factors.csv) into als_*.joblib.

Artifact schema:
  user_factors, item_factors, user_id_to_idx, item_id_to_idx,
  idx_to_user_id, idx_to_item_id
"""
import argparse
import csv
import joblib
import numpy as np
from pathlib import Path

DATA = Path("recsys/data")
MAHOUT_DIR = DATA / "mahout"
ART = Path("recsys/artifacts")
ART.mkdir(parents=True, exist_ok=True)


def load_factors_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load factors from CSV. Returns (ids, factors_matrix)."""
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        feat_cols = sorted(
            [c for c in fieldnames if c.startswith("f") and len(c) > 1 and c[1:].isdigit()],
            key=lambda x: int(x[1:]),
        )
        ids = []
        rows = []
        for row in reader:
            ids.append(int(row["id"]))
            rows.append([float(row[c]) for c in feat_cols])
    return np.array(ids), np.array(rows, dtype=np.float32)


def import_factors(
    dataset: str,
    user_factors_path: str,
    item_factors_path: str,
) -> None:
    """Import Spark ALS factors and save als_{dataset}.joblib."""
    user_factors_path = Path(user_factors_path)
    item_factors_path = Path(item_factors_path)

    if not user_factors_path.exists():
        raise FileNotFoundError(f"User factors not found: {user_factors_path}")
    if not item_factors_path.exists():
        raise FileNotFoundError(f"Item factors not found: {item_factors_path}")

    # Load id mappings from export
    map_path = MAHOUT_DIR / f"id_mappings_{dataset}.joblib"
    if not map_path.exists():
        raise FileNotFoundError(
            f"ID mappings not found: {map_path}. Run export_interactions first."
        )
    mappings = joblib.load(map_path)
    idx_to_user_id = mappings["idx_to_user_id"]
    idx_to_item_id = mappings["idx_to_item_id"]
    user_id_to_idx = mappings["user_id_to_idx"]
    item_id_to_idx = mappings["item_id_to_idx"]

    print(f"[import] Loading user factors from {user_factors_path}...")
    user_ids_from_csv, user_factors = load_factors_csv(user_factors_path)
    print(f"[import] Loading item factors from {item_factors_path}...")
    item_ids_from_csv, item_factors = load_factors_csv(item_factors_path)

    n_users, rank_u = user_factors.shape
    n_items, rank_i = item_factors.shape
    print(f"[import] User factors: {n_users} x {rank_u}, Item factors: {n_items} x {rank_i}")

    # Spark uses 0-based contiguous ids (same as our remapped indices)
    # Build arrays aligned by idx: factors[idx] = vector for idx
    max_user_idx = max(idx_to_user_id.keys()) if idx_to_user_id else 0
    max_item_idx = max(idx_to_item_id.keys()) if idx_to_item_id else 0

    user_factors_aligned = np.zeros((max_user_idx + 1, rank_u), dtype=np.float32)
    for i, idx in enumerate(user_ids_from_csv):
        if idx <= max_user_idx:
            user_factors_aligned[idx] = user_factors[i]

    item_factors_aligned = np.zeros((max_item_idx + 1, rank_i), dtype=np.float32)
    for i, idx in enumerate(item_ids_from_csv):
        if idx <= max_item_idx:
            item_factors_aligned[idx] = item_factors[i]

    payload = {
        "user_factors": user_factors_aligned,
        "item_factors": item_factors_aligned,
        "user_id_to_idx": user_id_to_idx,
        "item_id_to_idx": item_id_to_idx,
        "idx_to_user_id": idx_to_user_id,
        "idx_to_item_id": idx_to_item_id,
    }

    out_path = ART / f"als_{dataset}.joblib"
    joblib.dump(payload, out_path)
    print(f"[import] Saved {out_path}")

    # Legacy: also save as collab_model for hybrid_ranker compatibility
    legacy_path = ART / f"collab_model_{dataset}.joblib"
    legacy = {
        "user_factors": user_factors_aligned,
        "item_factors": item_factors_aligned,
        "user_map": idx_to_user_id,  # idx -> user_id
        "item_map": idx_to_item_id,  # idx -> item_id
    }
    joblib.dump(legacy, legacy_path)
    if dataset == "hm":
        joblib.dump(legacy, ART / "hm_collab_model.joblib")
    print(f"[import] Also saved legacy {legacy_path}")

    # For train_mahout_finetune: dict format {id: vector}
    user_vecs_dict = {
        int(idx_to_user_id[i]): user_factors_aligned[i].tolist()
        for i in range(user_factors_aligned.shape[0])
        if i in idx_to_user_id
    }
    item_vecs_dict = {
        int(idx_to_item_id[j]): item_factors_aligned[j].tolist()
        for j in range(item_factors_aligned.shape[0])
        if j in idx_to_item_id
    }
    joblib.dump(user_vecs_dict, ART / f"user_vectors_mahout_{dataset}.joblib")
    joblib.dump(item_vecs_dict, ART / f"item_vectors_mahout_{dataset}.joblib")
    print(f"[import] Saved user_vectors_mahout_{dataset}.joblib, item_vectors_mahout_{dataset}.joblib")


def main():
    parser = argparse.ArgumentParser(description="Import ALS factors to joblib")
    parser.add_argument("--dataset", type=str, required=True, choices=["hm", "rr"])
    parser.add_argument("--user_factors", type=str, required=True, help="Path to user_factors.csv")
    parser.add_argument("--item_factors", type=str, required=True, help="Path to item_factors.csv")
    args = parser.parse_args()

    import_factors(
        dataset=args.dataset,
        user_factors_path=args.user_factors,
        item_factors_path=args.item_factors,
    )


if __name__ == "__main__":
    main()
