# recsys/src/models/hybrid_v2.py

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn

DATA = Path("recsys/data")
ART = Path("recsys/artifacts")

# =====================================================
# Load H&M Artifacts
# =====================================================
print("[hybrid_v2] Loading events_hm...")
events = pd.read_parquet(DATA / "events_hm.parquet")[["user_id", "item_id"]]

print("[hybrid_v2] Loading FAISS H&M embeddings...")
faiss_pack = joblib.load(ART / "faiss_items_hm.joblib")
item_X = faiss_pack["X"].astype("float32")
row_map = faiss_pack["row_map"]

print("[hybrid_v2] Loading content user vectors...")
user_vecs = joblib.load(ART / "user_vectors_hm.joblib")

print("[hybrid_v2] Loading collaborative filtering model...")
# This will be either the implicit one OR the Mahout one if imported
hm_collab = joblib.load(ART / "hm_collab_model.joblib")
cf_item_factors = hm_collab["user_factors"].astype("float32")   # ALS items
cf_user_factors = hm_collab["item_factors"].astype("float32")   # ALS users
cf_user_map = hm_collab["user_map"]                             # idx -> user_id
cf_item_map = hm_collab["item_map"]                             # idx -> item_id

cf_user_id_to_idx = {uid: i for i, uid in cf_user_map.items()}
cf_item_id_to_idx = {iid: i for i, iid in cf_item_map.items()}

# =====================================================
# Two-Tower Model Selection (Mahout Finetuned vs Legacy)
# =====================================================
# Prefer hm-specific artifacts (from train_mahout_finetune + import_factors)
mahout_model_path = ART / "mahout_finetuned_hm.pt"
if not mahout_model_path.exists():
    mahout_model_path = ART / "mahout_finetuned_model.pt"  # legacy name
use_mahout_tt = mahout_model_path.exists()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
two_tower = None
user_vecs_mahout = {}

if use_mahout_tt:
    print("[hybrid_v2] Found Mahout Finetuned Model! Loading...")
    
    # Load Mahout vectors (prefer hm-specific, fallback to legacy)
    mahout_user_path = ART / "user_vectors_mahout_hm.joblib"
    if not mahout_user_path.exists():
        mahout_user_path = ART / "user_vectors_mahout.joblib"
    user_vecs_mahout = joblib.load(mahout_user_path)
    
    # Infer dimensions
    sample_uid = next(iter(user_vecs_mahout))
    user_dim = len(user_vecs_mahout[sample_uid])
    item_dim = item_X.shape[1]
    
    class MahoutTwoTower(nn.Module):
        def __init__(self, user_dim, item_dim):
            super().__init__()
            self.user_tower = nn.Sequential(
                nn.Linear(user_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
            )
            self.item_tower = nn.Sequential(
                nn.Linear(item_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
            )
            self.output = nn.Linear(64, 1)

        def forward(self, u_vec, i_vec):
            u = self.user_tower(u_vec)
            i = self.item_tower(i_vec)
            x = torch.abs(u - i)
            return torch.sigmoid(self.output(x).squeeze(1))

    two_tower = MahoutTwoTower(user_dim, item_dim).to(device)
    two_tower.load_state_dict(torch.load(mahout_model_path, map_location=device))
    two_tower.eval()
    print("[hybrid_v2] Mahout Finetuned Model loaded.")

else:
    print("[hybrid_v2] Loading Legacy Two-Tower v2 model (ID-based)...")
    
    print("[hybrid_v2] Loading Two-Tower v2 ID maps...")
    idmaps = joblib.load(ART / "two_tower_v2_idmaps.joblib")
    user_id_to_idx = idmaps["user_id_to_idx"]
    item_id_to_idx = idmaps["item_id_to_idx"]
    
    print("[hybrid_v2] Loading Two-Tower v2 item_features...")
    item_features = joblib.load(ART / "two_tower_v2_item_features.joblib").astype("float32")
    item_dim = item_features.shape[1]

    class TwoTowerHMV2(nn.Module):
        def __init__(self, n_users, item_dim=384, embed_dim=128):
            super().__init__()
            self.user_tower = nn.Sequential(
                nn.Embedding(n_users, 256),
                nn.ReLU(),
                nn.Linear(256, embed_dim),
                nn.ReLU(),
            )
            self.item_tower = nn.Sequential(
                nn.Linear(item_dim, 256),
                nn.ReLU(),
                nn.Linear(256, embed_dim),
                nn.ReLU(),
            )
            self.scorer = nn.Sequential(
                nn.Linear(embed_dim, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
            )

        def forward(self, user_idx, item_vec):
            u = self.user_tower[0](user_idx)
            u = self.user_tower[1:](u)
            i = self.item_tower(item_vec)
            x = torch.abs(u - i)
            return torch.sigmoid(self.scorer(x).squeeze(1))

    two_tower = TwoTowerHMV2(n_users=len(user_id_to_idx),
                             item_dim=item_dim,
                             embed_dim=128).to(device)
    
    state = torch.load(ART / "two_tower_hm_v2_final.pt", map_location=device)
    two_tower.load_state_dict(state)
    two_tower.eval()
    print("[hybrid_v2] Legacy Two-Tower v2 loaded.")


# =====================================================
# Scoring Functions (Content, CF, Two-Tower)
# =====================================================

def score_content(user_id, item_id):
    """Content-based: dot(user_vec, clip_item_vec)"""
    if user_id not in user_vecs:
        return None
    if item_id not in row_map:
        return None
    u = user_vecs[user_id]
    i = item_X[row_map[item_id]]
    return float(np.dot(u, i))


def score_cf(user_id, item_id):
    """Collaborative filtering ALS score"""
    # Uses whatever is in hm_collab_model (Implicit or Mahout)
    if user_id not in cf_user_id_to_idx:
        return None
    u_idx = cf_user_id_to_idx[user_id]
    u_vec = cf_user_factors[u_idx]

    i_idx = cf_item_id_to_idx.get(item_id, None)
    if i_idx is None:
        return None
    return float(np.dot(u_vec, cf_item_factors[i_idx]))


def score_two_tower(user_id, item_id):
    """Two-Tower score"""
    if two_tower is None:
        return None

    if use_mahout_tt:
        # Mahout Finetuned Version (Vector based)
        if user_id not in user_vecs_mahout:
            return None
        if item_id not in row_map:
            return None
        
        u_vec = torch.tensor(user_vecs_mahout[user_id], dtype=torch.float32, device=device).unsqueeze(0)
        # item_X holds CLIP embeddings
        i_vec = torch.tensor(item_X[row_map[item_id]], dtype=torch.float32, device=device).unsqueeze(0)
        
        with torch.no_grad():
            return float(two_tower(u_vec, i_vec).cpu().numpy()[0])
            
    else:
        # Legacy ID based Version
        if user_id not in user_id_to_idx:
            return None
        if item_id not in item_id_to_idx:
            return None

        u_idx = user_id_to_idx[user_id]
        i_idx = item_id_to_idx[item_id]

        i_vec = torch.tensor(item_features[i_idx], dtype=torch.float32, device=device)
        u_tensor = torch.tensor([u_idx], dtype=torch.long, device=device)

        with torch.no_grad():
            return float(two_tower(u_tensor, i_vec.unsqueeze(0)).cpu().numpy()[0])

# =====================================================
# Hybrid score (weighted combination)
# =====================================================

def score_hybrid(user_id, item_id,
                 w_content=0.4, w_cf=0.4, w_tt=0.2):
    """Weighted sum of content + collab + Two-Tower. Handles missing scores gracefully."""
    s1 = score_content(user_id, item_id)
    s2 = score_cf(user_id, item_id)
    s3 = score_two_tower(user_id, item_id)

    parts, weights = [], []
    if s1 is not None:
        parts.append(s1)
        weights.append(w_content)
    if s2 is not None:
        parts.append(s2)
        weights.append(w_cf)
    if s3 is not None:
        parts.append(2 * (s3 - 0.5))  # map sigmoid [0,1] -> [-1,1]
        weights.append(w_tt)

    if not parts:
        return None

    # Renormalize weights when some scores are missing
    w_sum = sum(weights)
    return sum(p * w for p, w in zip(parts, weights)) / w_sum

# =====================================================
# Evaluation: Recall@K, NDCG@K
# =====================================================

def recall_at_k(rank, k):
    return 1.0 if rank < k else 0.0

def ndcg_at_k(rank, k):
    if rank < k:
        return 1.0 / np.log2(rank + 2)
    return 0.0

# =====================================================
# Main Evaluation Loop
# =====================================================

def evaluate(n_users_eval=200, k=10):

    print(f"[hybrid_v2] Evaluating on {n_users_eval} users...")

    # sample users with >=2 interactions
    vc = events["user_id"].value_counts()
    users = vc[vc >= 2].index.to_numpy()

    rng = np.random.default_rng(123)
    users_eval = rng.choice(users, size=min(n_users_eval, len(users)), replace=False)

    ranks_content = []
    ranks_cf = []
    ranks_tt = []
    ranks_hybrid = []

    all_items = events["item_id"].unique()

    for uid in tqdm(users_eval):

        user_items = events[events["user_id"] == uid]["item_id"].values
        test_item = user_items[-1]
        history = set(user_items[:-1])

        possible_negs = np.setdiff1d(all_items, user_items)
        if len(possible_negs) < 99:
            continue

        negs = rng.choice(possible_negs, size=99, replace=False)
        candidates = np.concatenate([[test_item], negs])

        def get_rank(score_fn):
            scores = []
            for iid in candidates:
                s = score_fn(uid, iid)
                scores.append(s if s is not None else -1e9)
            scores = np.array(scores)
            order = np.argsort(scores)[::-1]
            rank = int(np.where(order == 0)[0][0])
            return rank

        ranks_content.append(get_rank(score_content))
        ranks_cf.append(get_rank(score_cf))
        ranks_tt.append(get_rank(score_two_tower))
        ranks_hybrid.append(get_rank(score_hybrid))

    def summarize(name, ranks):
        r = np.array(ranks)
        rec = np.mean([recall_at_k(x, k) for x in r])
        ndcg = np.mean([ndcg_at_k(x, k) for x in r])
        print(f"{name:10s}  Recall@{k}: {rec:.4f}   NDCG@{k}: {ndcg:.4f}   (n={len(r)})")

    print("\n================ H&M Model Comparison ================")
    summarize("content", ranks_content)
    summarize("collab", ranks_cf)
    summarize("two_tower", ranks_tt)
    summarize("hybrid", ranks_hybrid)


if __name__ == "__main__":
    evaluate(n_users_eval=200, k=10)
