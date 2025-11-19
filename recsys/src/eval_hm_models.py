import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn as nn

DATA = Path("recsys/data")
ART = Path("recsys/artifacts")

# -------------------------------
# Load artifacts
# -------------------------------
print("[eval] loading events_hm ...")
events = pd.read_parquet(DATA / "events_hm.parquet")
events = events[["user_id", "item_id"]].drop_duplicates()

print("[eval] loading content (FAISS H&M) ...")
faiss_pack = joblib.load(ART / "faiss_items_hm.joblib")
item_X = faiss_pack["X"].astype("float32")           # (n_items_hm, dim)
row_map = faiss_pack["row_map"]                      # item_id -> row idx
item_ids_faiss = np.array(list(row_map.keys()))

print("[eval] loading user vectors (content) ...")
user_vecs = joblib.load(ART / "user_vectors_hm.joblib")

print("[eval] loading HM ALS collab model ...")
hm_collab = joblib.load(ART / "hm_collab_model.joblib")
cf_item_factors = hm_collab["user_factors"].astype("float32")
cf_user_factors = hm_collab["item_factors"].astype("float32")
cf_user_map = hm_collab["user_map"]   # idx -> user_id
cf_item_map = hm_collab["item_map"]   # idx -> item_id

# invert maps: user_id -> idx, item_id -> idx
cf_user_id_to_idx = {uid: idx for idx, uid in cf_user_map.items()}
cf_item_id_to_idx = {iid: idx for idx, iid in cf_item_map.items()}

print("[eval] loading two-tower model ...")

class TwoTowerHM(nn.Module):
    def __init__(self, n_users, item_dim=384, embed_dim=128):
        super().__init__()
        self.user_emb = nn.Embedding(n_users, embed_dim)
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
        u = self.user_emb(user_idx)          # (B, embed_dim)
        i = self.item_tower(item_vec)        # (B, embed_dim)
        x = torch.abs(u - i)
        logits = self.scorer(x).squeeze(1)
        return torch.sigmoid(logits)

n_users_cf = len(cf_user_id_to_idx)
item_dim = item_X.shape[1]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
two_tower = TwoTowerHM(n_users=n_users_cf, item_dim=item_dim, embed_dim=128).to(device)
state = torch.load(ART / "two_tower_hm_v2_final.pt", map_location=device)
two_tower.load_state_dict(state)
two_tower.eval()
print(f"[eval] two-tower loaded on {device}")

# -------------------------------
# Helper: candidate set per user
# -------------------------------
rng = np.random.default_rng(42)

def sample_candidates(user_id, all_items, n_neg=99):
    """
    For a given user, we build a candidate set of:
    - 1 held-out positive (test item)
    - n_neg random negatives
    This is a common offline eval approximation.
    """
    user_items = events[events["user_id"] == user_id]["item_id"].values
    if len(user_items) < 2:
        return None, None  # not enough history

    test_item = user_items[-1]
    history = set(user_items[:-1])

    # sample negatives that are not in history and not the test item
    possible_negs = np.setdiff1d(all_items, user_items)
    if len(possible_negs) < n_neg:
        negs = possible_negs
    else:
        negs = rng.choice(possible_negs, size=n_neg, replace=False)

    candidates = np.concatenate([[test_item], negs])
    return test_item, candidates

# -------------------------------
# Scoring functions
# -------------------------------
def content_scores(user_id, candidate_items):
    """Dot product between user vector and item embeddings (content-only)."""
    if user_id not in user_vecs:
        return None
    u = user_vecs[user_id]          # (dim,)
    scores = []
    for iid in candidate_items:
        if iid in row_map:
            i_vec = item_X[row_map[iid]]
            scores.append(float(np.dot(u, i_vec)))
        else:
            scores.append(-1e9)     # very low score if missing
    return np.array(scores)

def collab_scores(user_id, candidate_items):
    """ALS-based CF score: user_factors[u] dot item_factors[i]."""
    if user_id not in cf_user_id_to_idx:
        return None
    u_idx = cf_user_id_to_idx[user_id]
    u_vec = cf_user_factors[u_idx]    # (k,)
    scores = []
    for iid in candidate_items:
        idx = cf_item_id_to_idx.get(iid, None)
        if idx is None:
            scores.append(-1e9)
        else:
            scores.append(float(np.dot(u_vec, cf_item_factors[idx])))
    return np.array(scores)

def two_tower_scores(user_id, candidate_items):
    """Two-tower scores using learned user embedding and content item tower."""
    if user_id not in cf_user_id_to_idx:
        return None
    u_idx = cf_user_id_to_idx[user_id]
    # build item feature matrix for these candidates
    feats = []
    for iid in candidate_items:
        if iid in row_map:
            feats.append(item_X[row_map[iid]])
        else:
            feats.append(np.zeros(item_dim, dtype="float32"))
    feats = torch.tensor(np.stack(feats), dtype=torch.float32, device=device)
    u_batch = torch.full((len(candidate_items),), u_idx, dtype=torch.long, device=device)

    with torch.no_grad():
        s = two_tower(u_batch, feats).cpu().numpy()
    return s

# -------------------------------
# Metrics: Recall@K and NDCG@K
# -------------------------------
def recall_at_k(ranks, k):
    return np.mean([1.0 if r < k else 0.0 for r in ranks])

def ndcg_at_k(ranks, k):
    # only one relevant item per user (the test item)
    vals = []
    for r in ranks:
        if r < k:
            vals.append(1.0 / np.log2(r + 2))  # rank 0 -> log2(2)=1
        else:
            vals.append(0.0)
    return np.mean(vals)

# -------------------------------
# Main evaluation loop
# -------------------------------
def eval_models(n_users_eval=200, k=10):
    print(f"[eval] evaluating on {n_users_eval} users, K={k}")
    all_items = events["item_id"].unique()

    users = events["user_id"].value_counts()
    users = users[users >= 2].index.to_numpy()
    if len(users) == 0:
        print("[eval] no users with >=2 interactions")
        return

    users_eval = rng.choice(users, size=min(n_users_eval, len(users)), replace=False)

    ranks_content = []
    ranks_collab = []
    ranks_two_tower = []

    for uid in tqdm(users_eval):
        test_item, candidates = sample_candidates(uid, all_items, n_neg=99)
        if test_item is None:
            continue

        # content
        s_c = content_scores(uid, candidates)
        # collab
        s_cf = collab_scores(uid, candidates)
        # two-tower
        s_tt = two_tower_scores(uid, candidates)

        # Skip if any missing (e.g., user not in model)
        if s_c is None or s_cf is None or s_tt is None:
            continue

        def rank_of_test(scores):
            order = np.argsort(scores)[::-1]
            # position where candidates[order[pos]] == test_item
            idx = np.where(candidates[order] == test_item)[0]
            if len(idx) == 0:
                return 999  # not found
            return int(idx[0])

        ranks_content.append(rank_of_test(s_c))
        ranks_collab.append(rank_of_test(s_cf))
        ranks_two_tower.append(rank_of_test(s_tt))

    def summarize(name, ranks):
        if len(ranks) == 0:
            print(f"[eval] {name}: no valid users")
            return
        r = np.array(ranks)
        rec = recall_at_k(r, k)
        nd = ndcg_at_k(r, k)
        print(f"{name:12s}  Recall@{k}: {rec:.4f}   NDCG@{k}: {nd:.4f}   (n={len(r)})")

    print("\n========== H&M Model Comparison ==========")
    summarize("content", ranks_content)
    summarize("collab", ranks_collab)
    summarize("two_tower", ranks_two_tower)

if __name__ == "__main__":
    eval_models(n_users_eval=200, k=10)