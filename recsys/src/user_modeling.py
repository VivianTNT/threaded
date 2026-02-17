# recsys/src/user_modeling.py
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from tqdm import tqdm
import faiss
import scipy.sparse as sp

DATA = Path("recsys/data")
ART = Path("recsys/artifacts")

# ---- Load Artifacts ----
print("[user_modeling] loading data ...")
events = pd.read_parquet(DATA / "events_hm.parquet")
# events_hm is H&M-only; no ID filter needed

faiss_pack = joblib.load(ART / "faiss_items_hm.joblib")
X = faiss_pack["X"]            # item embedding matrix (N_items, dim)
row_map = faiss_pack["row_map"]
index = faiss_pack["index"]
item_ids = faiss_pack["item_ids"]

# ---- Try loading optional user style embeddings ----
style_path = ART / "user_style_embs_hm.joblib"
if style_path.exists():
    user_style_embs = joblib.load(style_path)
    print(f"[user_modeling] loaded {len(user_style_embs)} user style embeddings")
else:
    user_style_embs = {}
    print("[user_modeling] no user style embeddings found (skipping style fusion)")

# ---- Filter to valid items ----
events = events[events["item_id"].isin(row_map.keys())].copy()
events["row_idx"] = events["item_id"].map(row_map)

# ---- Compute recency weights ----
def compute_weights(df, weight_by_recency=True):
    if not weight_by_recency:
        df["weight"] = 1.0
        return df
    tmin, tmax = df["timestamp"].min(), df["timestamp"].max()
    df["recency"] = (df["timestamp"] - tmin) / (tmax - tmin)
    df["weight"] = np.exp(df["recency"] * 5)
    return df

events = compute_weights(events, weight_by_recency=True)

# ---- Build user vectors ----
def build_user_vectors_sparse(alpha=0.6):
    """
    Builds user embeddings via sparse-matrix multiplication and fuses
    optional style embeddings if available.
    alpha: weighting factor for behavior vs style (0.0–1.0)
    """
    print("[user_modeling] building user vectors (sparse-matrix version)...")

    # Map user IDs to integer indices
    user_idx, unique_users = pd.factorize(events["user_id"], sort=True)
    rows = user_idx
    cols = events["row_idx"].values
    vals = events["weight"].values.astype("float32")

    print(f"[user_modeling] building sparse matrix with {len(unique_users):,} users, {X.shape[0]:,} items ...")
    R = sp.csr_matrix((vals, (rows, cols)), shape=(len(unique_users), X.shape[0]))

    print("[user_modeling] multiplying R @ X ... (this may take a few minutes)")
    user_embs = R @ X  # (num_users, dim)
    user_embs = np.asarray(user_embs)

    # Normalize
    norms = np.linalg.norm(user_embs, axis=1, keepdims=True)
    user_embs = user_embs / np.maximum(norms, 1e-8)

    user_vecs = {}
    for i, uid in enumerate(unique_users):
        u_behavior = user_embs[i]
        u_final = u_behavior
        if int(uid) in user_style_embs:
            u_style = user_style_embs[int(uid)]
            u_final = alpha * u_behavior + (1 - alpha) * u_style
            u_final /= np.linalg.norm(u_final)
        user_vecs[int(uid)] = u_final

    print(f"[user_modeling] built {len(user_vecs):,} user vectors (with style fusion)")
    return user_vecs

# ---- Evaluate retrieval quality ----
def evaluate_user_vectors(user_vecs, top_k=10, n_samples=2000):
    print("[evaluation] sampling users...")
    users = np.random.choice(list(user_vecs.keys()), size=min(n_samples, len(user_vecs)), replace=False)
    hits, total = 0, 0
    for uid in tqdm(users):
        user_vec = user_vecs[uid].astype("float32")[None, :]
        user_events = events[events["user_id"] == uid]["item_id"].tolist()
        if len(user_events) < 2:
            continue
        test_item = user_events[-1]
        D, I = index.search(user_vec, top_k)
        retrieved_ids = [int(item_ids[i]) for i in I[0]]
        if test_item in retrieved_ids:
            hits += 1
        total += 1
    recall = hits / total if total > 0 else 0
    print(f"[evaluation] Recall@{top_k}: {recall:.4f}  ({hits}/{total})")

if __name__ == "__main__":
    user_vecs = build_user_vectors_sparse(alpha=0.6)
    ART.mkdir(parents=True, exist_ok=True)
    joblib.dump(user_vecs, ART / "user_vectors_hm.joblib")
    evaluate_user_vectors(user_vecs, top_k=10, n_samples=2000)