#!/usr/bin/env python3
"""
Train Mahout-compatible ALS model locally using implicit library.
This is equivalent to Mahout's parallelALS but much easier to run.
"""
import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
import joblib
from pathlib import Path
from implicit.als import AlternatingLeastSquares
import sys
from tqdm import tqdm

DATA = Path("recsys/data/mahout")
ART = Path("recsys/artifacts")
ART.mkdir(parents=True, exist_ok=True)

def train_als(dataset="hm", factors=64, regularization=0.05, iterations=15):
    """Train ALS model (Mahout-compatible) on interactions CSV."""
    csv_path = DATA / f"interactions_{dataset}.csv"
    
    if not csv_path.exists():
        print(f"❌ Error: {csv_path} not found!")
        print("   Run: python3 -m recsys.src.mahout.mahout_interface --dataset {dataset} --action export")
        return
    
    print(f"[ALS] Loading {csv_path}...")
    df = pd.read_csv(csv_path, header=None, names=["user_id", "item_id", "pref"])
    print(f"[ALS] Loaded {len(df):,} interactions")
    
    # Build sparse matrix
    print("[ALS] Building sparse matrix...")
    users = df["user_id"].astype("category")
    items = df["item_id"].astype("category")
    
    mat = coo_matrix(
        (df["pref"].values.astype(np.float32), (users.cat.codes, items.cat.codes)),
        shape=(len(users.cat.categories), len(items.cat.categories))
    )
    print(f"[ALS] Matrix shape: {mat.shape} ({mat.nnz:,} non-zero entries)")
    
    # Train ALS
    print(f"[ALS] Training ALS (factors={factors}, lambda={regularization}, iterations={iterations})...")
    print("[ALS] This may take 10-30 minutes depending on data size...")
    
    model = AlternatingLeastSquares(
        factors=factors,
        regularization=regularization,
        iterations=iterations,
        use_gpu=False,
        random_state=42
    )
    
    # Fit model (implicit handles iterations internally)
    model.fit(mat.T)
    
    print("[ALS] ✅ Training complete!")
    
    # Save in Mahout-compatible format
    user_map = {i: int(uid) for i, uid in enumerate(users.cat.categories)}
    item_map = {i: int(iid) for i, iid in enumerate(items.cat.categories)}
    
    # Also save as simple dictionaries for Two-Tower input
    user_vecs_dict = {int(uid): model.user_factors[i] for i, uid in enumerate(users.cat.categories)}
    item_vecs_dict = {int(iid): model.item_factors[i] for i, iid in enumerate(items.cat.categories)}
    
    # Save collaborative filtering model
    cf_model_path = ART / f"collab_model_{dataset}.joblib"
    joblib.dump({
        "user_factors": model.user_factors.astype(np.float32),
        "item_factors": model.item_factors.astype(np.float32),
        "user_map": user_map,
        "item_map": item_map
    }, cf_model_path)
    print(f"[ALS] Saved CF model to {cf_model_path}")
    
    # Legacy path for H&M
    if dataset == "hm":
        joblib.dump({
            "user_factors": model.user_factors.astype(np.float32),
            "item_factors": model.item_factors.astype(np.float32),
            "user_map": user_map,
            "item_map": item_map
        }, ART / "hm_collab_model.joblib")
        print(f"[ALS] Also saved to hm_collab_model.joblib (legacy path)")
    
    # Save vectors for Two-Tower
    user_vecs_path = ART / f"user_vectors_mahout_{dataset}.joblib"
    item_vecs_path = ART / f"item_vectors_mahout_{dataset}.joblib"
    
    joblib.dump(user_vecs_dict, user_vecs_path)
    joblib.dump(item_vecs_dict, item_vecs_path)
    print(f"[ALS] Saved user vectors to {user_vecs_path}")
    print(f"[ALS] Saved item vectors to {item_vecs_path}")
    
    print(f"\n✅ All done! Model ready for hybrid ranker.")
    print(f"   You can now train Two-Tower with:")
    print(f"   python3 -m recsys.src.models.train_mahout_finetune --dataset {dataset}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train ALS (Mahout-compatible) locally")
    parser.add_argument("--dataset", type=str, default="hm", choices=["hm", "rr"])
    parser.add_argument("--factors", type=int, default=64)
    parser.add_argument("--lambda", type=float, default=0.05, dest="regularization")
    parser.add_argument("--iterations", type=int, default=15)
    
    args = parser.parse_args()
    train_als(args.dataset, args.factors, args.regularization, args.iterations)
