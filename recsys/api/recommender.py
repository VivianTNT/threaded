from pathlib import Path
import joblib
import numpy as np
import faiss

BASE = Path("recsys")
ART = BASE / "artifacts"

# Load FAISS content-filtering model
faiss_pack = joblib.load(ART / "faiss_items_hm.joblib")
faiss_index = faiss_pack["index"]
item_ids = faiss_pack["item_ids"]
item_features = faiss_pack["X"]
row_map = faiss_pack["row_map"]

def faiss_search(vec: np.ndarray, k=20):
    vec = vec.reshape(1, -1).astype("float32")
    scores, indices = faiss_index.search(vec, k)
    return [
        {"item_id": int(item_ids[i]), "score": float(s)}
        for s, i in zip(scores[0], indices[0])
    ]