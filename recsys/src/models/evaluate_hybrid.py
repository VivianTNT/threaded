# recsys/src/models/evaluate_hybrid.py

import numpy as np
import pandas as pd
import joblib
import torch
from pathlib import Path
from tqdm import tqdm

from recsys.src.models.hybrid_ranker import (
    content_score,
    collab_score,
    two_tower_score,
    hybrid_score,
    get_content_candidates,
    get_collab_candidates,
)

# ==========================
# Load events for ground truth
# ==========================
BASE = Path("recsys")
DATA = BASE / "data"
ART = BASE / "artifacts"

print("[eval] loading events...")
# ideally use only SOME events to reduce compute
events = pd.read_parquet(DATA / "events.parquet")
# only positive interactions
events = events[events["event_type"] == "purchase"] if "event_type" in events else events
events = events[["user_id", "item_id"]].drop_duplicates()

# Build ground-truth dictionary: user_id → set of items
user_to_items = events.groupby("user_id")["item_id"].apply(set).to_dict()

# sample evaluation users
eval_users = list(user_to_items.keys())
np.random.shuffle(eval_users)
eval_users = eval_users[:200]   # evaluate on 200 users

print(f"[eval] evaluating on {len(eval_users)} users")

# ==========================
# Metric Functions
# ==========================

def recall_at_k(true_items, ranked_items, k=10):
    if not true_items:
        return np.nan
    ranked_topk = ranked_items[:k]
    hit_count = len([i for i in ranked_topk if i in true_items])
    return hit_count / len(true_items)

def ndcg_at_k(true_items, ranked_items, k=10):
    if not true_items:
        return np.nan
    dcg = 0.0
    for idx, item in enumerate(ranked_items[:k]):
        if item in true_items:
            dcg += 1 / np.log2(idx + 2)
    # ideal DCG
    ideal_hits = min(len(true_items), k)
    idcg = sum([1 / np.log2(i + 2) for i in range(ideal_hits)])
    return dcg / idcg if idcg > 0 else np.nan

# ==========================
# Evaluation Loop
# ==========================

results = {
    "content": {"recall": [], "ndcg": []},
    "collab": {"recall": [], "ndcg": []},
    "two_tower": {"recall": [], "ndcg": []},
    "hybrid": {"recall": [], "ndcg": []},
}

K = 10

for user in tqdm(eval_users, desc="Evaluating models"):

    true_items = user_to_items[user]

    # -----------------------
    # 1. Content-based retrieval
    # -----------------------
    cand = get_content_candidates(user, top_k=200)
    ranked = sorted(cand, key=lambda iid: content_score(user, iid), reverse=True)
    results["content"]["recall"].append(recall_at_k(true_items, ranked, K))
    results["content"]["ndcg"].append(ndcg_at_k(true_items, ranked, K))

    # -----------------------
    # 2. Collaborative filtering (ALS)
    # -----------------------
    cand = get_collab_candidates(user, top_k=200)
    ranked = sorted(cand, key=lambda iid: collab_score(user, iid), reverse=True)
    results["collab"]["recall"].append(recall_at_k(true_items, ranked, K))
    results["collab"]["ndcg"].append(ndcg_at_k(true_items, ranked, K))

    # -----------------------
    # 3. Neural Two-Tower Model
    # -----------------------
    combined = set(get_content_candidates(user, 200)) | set(get_collab_candidates(user, 200))
    ranked = sorted(list(combined), key=lambda iid: two_tower_score(user, iid), reverse=True)
    results["two_tower"]["recall"].append(recall_at_k(true_items, ranked, K))
    results["two_tower"]["ndcg"].append(ndcg_at_k(true_items, ranked, K))

    # -----------------------
    # 4. Hybrid Model
    # -----------------------
    ranked = sorted(
        list(combined),
        key=lambda iid: hybrid_score(user, iid, w_content=0.3, w_collab=0.3, w_two_tower=0.4),
        reverse=True
    )
    results["hybrid"]["recall"].append(recall_at_k(true_items, ranked, K))
    results["hybrid"]["ndcg"].append(ndcg_at_k(true_items, ranked, K))

# ==========================
# Summaries
# ==========================
print("\n================ Model Comparison ================")
rows = []
for model in ["content", "collab", "two_tower", "hybrid"]:
    recall_mean = np.nanmean(results[model]["recall"])
    ndcg_mean = np.nanmean(results[model]["ndcg"])
    rows.append((model, recall_mean, ndcg_mean))

df = pd.DataFrame(rows, columns=["model", "Recall@10", "NDCG@10"])
print(df)

df.to_csv(ART / "evaluation_summary.csv", index=False)
print(f"\nSaved metrics to {ART/'evaluation_summary.csv'}")