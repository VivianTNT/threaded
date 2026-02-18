"""Build RR user vectors (content-based) from events_rr + faiss_items_rr."""
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from tqdm import tqdm
import scipy.sparse as sp

DATA = Path("recsys/data")
ART = Path("recsys/artifacts")

events = pd.read_parquet(DATA / "events_rr.parquet")
faiss_pack = joblib.load(ART / "faiss_items_rr.joblib")
X = faiss_pack["X"]
row_map = faiss_pack["row_map"]

events = events[events["item_id"].isin(row_map.keys())].copy()
events["row_idx"] = events["item_id"].map(row_map)

tmin, tmax = events["timestamp"].min(), events["timestamp"].max()
events["weight"] = np.exp((events["timestamp"] - tmin) / (tmax - tmin + 1e-8) * 5)

user_idx, unique_users = pd.factorize(events["user_id"], sort=True)
R = sp.csr_matrix(
    (events["weight"].values.astype("float32"), (user_idx, events["row_idx"].values)),
    shape=(len(unique_users), X.shape[0])
)

user_embs = np.asarray(R @ X)
user_embs = user_embs / np.maximum(np.linalg.norm(user_embs, axis=1, keepdims=True), 1e-8)

user_vecs = {int(uid): user_embs[i] for i, uid in enumerate(unique_users)}
ART.mkdir(parents=True, exist_ok=True)
joblib.dump(user_vecs, ART / "user_vectors_rr.joblib")
print(f"[user_modeling_rr] Built {len(user_vecs):,} RR user vectors")
