import joblib, pandas as pd, numpy as np
import faiss
from pathlib import Path

DATA = Path("recsys/data")
ART  = Path("recsys/artifacts")

P = joblib.load(ART / "faiss_items.joblib")
INDEX: faiss.Index = P["index"]
X = P["X"]
ITEM_IDS = P["item_ids"]
ROW_MAP = P["row_map"]

ITEMS = pd.read_parquet(DATA / "items.parquet").set_index("item_id")
USER_VECS = joblib.load(ART / "user_vectors.joblib")  # from your Week-2 job

def recommend_for_user(user_id: int, top_k: int = 10, domain: str = "HM"):
    # 1) get user vector
    if user_id not in USER_VECS:
        return []
    u = USER_VECS[user_id].astype("float32")[None, :]
    # 2) retrieve a larger pool (e.g., 1000)
    D, I = INDEX.search(u, 1000)
    ids = [int(ITEM_IDS[i]) for i in I[0]]
    # 3) filter by domain (e.g., 'HM') and drop items with missing metadata
    filtered = [iid for iid in ids if iid in ITEMS.index and ITEMS.at[iid, "source"] == domain]
    # 4) take top_k; return with metadata
    top = filtered[:top_k]
    cols = [c for c in ["title","brand","category_path","source"] if c in ITEMS.columns]
    out = ITEMS.loc[top, cols].reset_index().rename(columns={"index":"item_id"})
    return out