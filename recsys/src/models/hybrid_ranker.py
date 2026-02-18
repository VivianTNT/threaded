# recsys/src/models/hybrid_ranker.py
"""
Unified Hybrid Ranker
Uses: Content (HM + RR) + Two-Tower (trained on both)
Generalizes to new users and new products.
"""

import numpy as np
import joblib
from pathlib import Path
import torch
import torch.nn as nn

DATA = Path("recsys/data")
ART = Path("recsys/artifacts")

RR_USER_OFFSET = 1_000_000_000
HM_MAX_ID = 3_000_000

# =====================================================
# Content-Based Two-Tower
# =====================================================

class ContentTwoTower(nn.Module):
    def __init__(self, feature_dim=384, embed_dim=128):
        super().__init__()
        self.user_tower = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, embed_dim),
            nn.ReLU(),
        )
        self.item_tower = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, embed_dim),
            nn.ReLU(),
        )
        self.scorer = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, u_vec, i_vec):
        u = self.user_tower(u_vec)
        i = self.item_tower(i_vec)
        x = torch.abs(u - i)
        return torch.sigmoid(self.scorer(x).squeeze(1))

# =====================================================
# Load All Artifacts
# =====================================================

artifacts = {"hm": {}, "rr": {}, "combined": {}}
device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

def _load_artifacts():
    for dataset in ["hm", "rr"]:
        faiss_path = ART / f"faiss_items_{dataset}.joblib"
        if faiss_path.exists():
            pack = joblib.load(faiss_path)
            artifacts[dataset]["item_X"] = pack["X"].astype("float32")
            artifacts[dataset]["row_map"] = pack["row_map"]
        else:
            artifacts[dataset]["item_X"] = None
            artifacts[dataset]["row_map"] = {}
        
        uv_path = ART / f"user_vectors_{dataset}.joblib"
        artifacts[dataset]["user_vecs"] = joblib.load(uv_path) if uv_path.exists() else {}
    
    # Combined FAISS (for lookup when item could be either)
    if (ART / "faiss_items.joblib").exists():
        pack = joblib.load(ART / "faiss_items.joblib")
        artifacts["combined"]["item_X"] = pack["X"].astype("float32")
        artifacts["combined"]["row_map"] = pack["row_map"]
    
    # Two-Tower: prefer combined (trained on HM+RR), else HM
    for name in ["content_two_tower_combined.pt", "content_two_tower_hm.pt", "content_two_tower_rr.pt"]:
        if (ART / name).exists():
            model = ContentTwoTower(feature_dim=384, embed_dim=128).to(device)
            model.load_state_dict(torch.load(ART / name, map_location=device))
            model.eval()
            artifacts["two_tower"] = model
            print(f"[hybrid] Loaded Two-Tower from {name}")
            return
    artifacts["two_tower"] = None

_load_artifacts()

def get_item_domain(item_id):
    return "hm" if item_id >= HM_MAX_ID else "rr"

def get_user_domain(user_id):
    return "rr" if user_id >= RR_USER_OFFSET else "hm"

# =====================================================
# Scoring: Known Users/Items (from training)
# =====================================================

def score_content(user_id, item_id):
    """Content: dot(user_vec, item_embedding)."""
    u_domain = get_user_domain(user_id)
    i_domain = get_item_domain(item_id)
    
    u_vecs = artifacts[u_domain].get("user_vecs", {})
    if user_id not in u_vecs:
        return None
    
    row_map = artifacts[i_domain].get("row_map", {})
    item_X = artifacts[i_domain].get("item_X")
    if item_id not in row_map or item_X is None:
        return None
    
    return float(np.dot(u_vecs[user_id], item_X[row_map[item_id]]))


def score_two_tower(user_id, item_id):
    """Two-Tower: neural score from content features."""
    model = artifacts.get("two_tower")
    if model is None:
        return None
    
    u_domain = get_user_domain(user_id)
    i_domain = get_item_domain(item_id)
    
    u_vecs = artifacts[u_domain].get("user_vecs", {})
    if user_id not in u_vecs:
        return None
    
    row_map = artifacts[i_domain].get("row_map", {})
    item_X = artifacts[i_domain].get("item_X")
    if item_id not in row_map or item_X is None:
        return None
    
    u = torch.tensor(u_vecs[user_id], dtype=torch.float32, device=device).unsqueeze(0)
    i = torch.tensor(item_X[row_map[item_id]], dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        return float(model(u, i).cpu().numpy()[0])


def score_hybrid(user_id, item_id, w_content=0.5, w_tt=0.5):
    """Content + Two-Tower for known users/items."""
    s1 = score_content(user_id, item_id)
    s2 = score_two_tower(user_id, item_id)
    
    parts, weights = [], []
    if s1 is not None:
        parts.append(s1)
        weights.append(w_content)
    if s2 is not None:
        parts.append(2 * (s2 - 0.5))
        weights.append(w_tt)
    
    if not parts:
        return None
    return sum(p * w for p, w in zip(parts, weights)) / sum(weights)


# =====================================================
# Scoring: New Users/Products (generalizable)
# =====================================================

def score_from_vectors(user_vec, item_vec, w_content=0.5, w_tt=0.5):
    """
    Score using raw vectors (for new users/products).
    user_vec: (384,) from avg of liked item embeddings
    item_vec: (384,) from embedding product title+description
    """
    s_content = float(np.dot(user_vec, item_vec))
    
    model = artifacts.get("two_tower")
    if model is None:
        return s_content
    
    u = torch.tensor(user_vec, dtype=torch.float32, device=device).unsqueeze(0)
    i = torch.tensor(item_vec, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        s_tt = float(model(u, i).cpu().numpy()[0])
    
    return w_content * s_content + w_tt * (2 * (s_tt - 0.5))


# =====================================================
# CLI Test
# =====================================================

if __name__ == "__main__":
    print("\n[Test] Known users/items...")
    
    hm_users = list(artifacts["hm"].get("user_vecs", {}).keys())
    hm_items = list(artifacts["hm"].get("row_map", {}).keys())
    
    if hm_users and hm_items:
        u, i = hm_users[0], hm_items[0]
        print(f"  HM user {u} → HM item {i}: hybrid={score_hybrid(u, i)}")
    
    rr_items = list(artifacts["rr"].get("row_map", {}).keys())
    rr_users = list(artifacts["rr"].get("user_vecs", {}).keys())
    
    if rr_users and rr_items:
        u, i = rr_users[0], rr_items[0]
        print(f"  RR user {u} → RR item {i}: hybrid={score_hybrid(u, i)}")
    
    if hm_users and rr_items:
        print(f"  HM user {hm_users[0]} → RR item {rr_items[0]}: hybrid={score_hybrid(hm_users[0], rr_items[0])}")
