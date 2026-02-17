# recsys/src/models/hybrid_ranker.py
"""
Unified Hybrid Ranker
Combines H&M and RetailRocket models to score any (user, item) pair.
Supports cross-domain recommendations (e.g. HM user -> RR item).
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from tqdm import tqdm
import argparse
import torch
import torch.nn as nn

DATA = Path("recsys/data")
ART = Path("recsys/artifacts")

# =====================================================
# Model Classes
# =====================================================

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

# =====================================================
# Global Artifact Storage
# =====================================================

artifacts = {
    "hm": {},
    "rr": {}
}

device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))

def load_dataset_artifacts(dataset):
    """Load all artifacts for a specific dataset."""
    print(f"[hybrid] Loading {dataset.upper()} artifacts...")
    
    # 1. FAISS (Item Embeddings)
    faiss_path = ART / f"faiss_items_{dataset}.joblib"
    if faiss_path.exists():
        pack = joblib.load(faiss_path)
        artifacts[dataset]["item_X"] = pack["X"].astype("float32")
        artifacts[dataset]["row_map"] = pack["row_map"]
    else:
        print(f"⚠️ Missing {faiss_path}")

    # 2. User Vectors (Content)
    uv_path = ART / f"user_vectors_{dataset}.joblib"
    if uv_path.exists():
        artifacts[dataset]["user_vecs"] = joblib.load(uv_path)
    else:
        print(f"⚠️ Missing {uv_path} (Content scoring will be disabled for {dataset} users)")
        artifacts[dataset]["user_vecs"] = {}

    # 3. ALS Model
    als_path = ART / f"als_{dataset}.joblib"
    if als_path.exists():
        als = joblib.load(als_path)
        artifacts[dataset]["als"] = {
            "user_factors": als["user_factors"].astype("float32"),
            "item_factors": als["item_factors"].astype("float32"),
            "user_map": als["user_id_to_idx"],
            "item_map": als["item_id_to_idx"],
        }
    else:
        print(f"⚠️ Missing {als_path}")

    # 4. Two-Tower Model & Mahout Vectors
    tt_path = ART / f"mahout_finetuned_{dataset}.pt"
    mahout_user_path = ART / f"user_vectors_mahout_{dataset}.joblib"
    mahout_item_path = ART / f"item_vectors_mahout_{dataset}.joblib"
    
    if tt_path.exists() and mahout_user_path.exists() and mahout_item_path.exists():
        # Load vectors
        mahout_user_vecs = joblib.load(mahout_user_path)
        mahout_item_vecs = joblib.load(mahout_item_path)
        
        artifacts[dataset]["mahout_user_vecs"] = mahout_user_vecs
        artifacts[dataset]["mahout_item_vecs"] = mahout_item_vecs
        
        # Load model
        sample_uid = next(iter(mahout_user_vecs))
        sample_iid = next(iter(mahout_item_vecs))
        
        user_dim = len(mahout_user_vecs[sample_uid])
        item_dim = len(mahout_item_vecs[sample_iid])
        
        model = MahoutTwoTower(user_dim, item_dim).to(device)
        model.load_state_dict(torch.load(tt_path, map_location=device))
        model.eval()
        artifacts[dataset]["two_tower"] = model
    else:
        print(f"⚠️ Missing Two-Tower artifacts for {dataset}")

# Load everything on import
load_dataset_artifacts("hm")
load_dataset_artifacts("rr")

# =====================================================
# Helper: Identify User/Item Domain
# =====================================================

HM_MAX_ID = 3_000_000

def get_domain(id_val):
    """Return 'hm' if ID >= 3M, else 'rr'."""
    return "hm" if id_val >= HM_MAX_ID else "rr"

# =====================================================
# Scoring Functions
# =====================================================

def score_content(user_id, item_id):
    """
    Content score: dot(user_content_vec, item_embedding).
    Works across domains (both use same embedding space).
    """
    u_domain = get_domain(user_id)
    i_domain = get_domain(item_id)
    
    # Get user vector
    u_vecs = artifacts[u_domain].get("user_vecs", {})
    if user_id not in u_vecs:
        return None
    u_vec = u_vecs[user_id]
    
    # Get item vector
    row_map = artifacts[i_domain].get("row_map", {})
    item_X = artifacts[i_domain].get("item_X")
    
    if item_id not in row_map:
        return None
        
    i_vec = item_X[row_map[item_id]]
    return float(np.dot(u_vec, i_vec))


def score_cf(user_id, item_id):
    """
    ALS score: dot(user_factor, item_factor).
    Strictly same-domain only.
    """
    u_domain = get_domain(user_id)
    i_domain = get_domain(item_id)
    
    if u_domain != i_domain:
        return None  # ALS dimensions don't match across domains
        
    als = artifacts[u_domain].get("als")
    if not als:
        return None
        
    if user_id not in als["user_map"]:
        return None
    if item_id not in als["item_map"]:
        return None
        
    u_idx = als["user_map"][user_id]
    i_idx = als["item_map"][item_id]
    
    u_vec = als["user_factors"][u_idx]
    i_vec = als["item_factors"][i_idx]
    
    return float(np.dot(u_vec, i_vec))


def score_two_tower(user_id, item_id):
    """
    Two-Tower score: Neural(user_mahout_vec, item_mahout_vec).
    Strictly same-domain only (requires Mahout vectors for both).
    """
    u_domain = get_domain(user_id)
    i_domain = get_domain(item_id)
    
    if u_domain != i_domain:
        return None
    
    model = artifacts[u_domain].get("two_tower")
    mahout_user_vecs = artifacts[u_domain].get("mahout_user_vecs")
    mahout_item_vecs = artifacts[u_domain].get("mahout_item_vecs")
    
    if model is None or mahout_user_vecs is None or mahout_item_vecs is None:
        return None
        
    if user_id not in mahout_user_vecs:
        return None
    if item_id not in mahout_item_vecs:
        return None
        
    u_vec = torch.tensor(mahout_user_vecs[user_id], dtype=torch.float32, device=device).unsqueeze(0)
    i_vec = torch.tensor(mahout_item_vecs[item_id], dtype=torch.float32, device=device).unsqueeze(0)
    
    with torch.no_grad():
        return float(model(u_vec, i_vec).cpu().numpy()[0])

# =====================================================
# Hybrid Score
# =====================================================

def score_hybrid(user_id, item_id, w_content=0.4, w_cf=0.4, w_tt=0.2):
    """
    Unified hybrid score.
    - Same domain: Uses all three.
    - Cross domain: Uses Content + Two-Tower (renormalizes weights).
    """
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
        # Map sigmoid [0,1] -> [-1,1] approx for consistency with dot products
        # or just keep as is. Let's keep [0,1] but scale it up slightly since dot products are often small
        parts.append(2 * (s3 - 0.5)) 
        weights.append(w_tt)
        
    if not parts:
        return None
        
    return sum(p * w for p, w in zip(parts, weights)) / sum(weights)

# =====================================================
# CLI Evaluation
# =====================================================

if __name__ == "__main__":
    # Simple test
    print("\n[Test] Scoring HM user 123 against HM item...")
    hm_item = 2010877501  # example HM item
    print(f"HM->HM: {score_hybrid(123, hm_item)}")
    
    print("\n[Test] Scoring HM user 123 against RR item...")
    rr_item = 10001  # example RR item
    print(f"HM->RR: {score_hybrid(123, rr_item)}")
