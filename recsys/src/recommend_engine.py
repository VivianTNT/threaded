"""
Unified recommendation engine: content (FAISS) and collab (ALS).

Strategy selection:
  - content: FAISS-based content filtering only
  - collab: ALS collaborative filtering (if artifact exists), else content
"""
import numpy as np
import joblib
import faiss
from pathlib import Path
from typing import Optional

ART = Path("recsys/artifacts")


def _load_faiss():
    pack = joblib.load(ART / "faiss_items_hm.joblib")
    return pack["index"], pack["item_ids"], pack["X"], pack["row_map"]


def _load_user_vectors():
    return joblib.load(ART / "user_vectors_hm.joblib")


def _load_als():
    path = ART / "als_hm.joblib"
    if not path.exists():
        return None
    return joblib.load(path)


# Lazy-loaded singletons
_faiss_pack = None
_user_vectors = None
_als_pack = None


def get_faiss():
    global _faiss_pack
    if _faiss_pack is None:
        _faiss_pack = _load_faiss()
    return _faiss_pack


def get_user_vectors():
    global _user_vectors
    if _user_vectors is None:
        _user_vectors = _load_user_vectors()
    return _user_vectors


def get_als():
    global _als_pack
    if _als_pack is None:
        _als_pack = _load_als()
    return _als_pack


def faiss_search(vec: np.ndarray, k: int = 20) -> list[dict]:
    """Content-based search via FAISS."""
    index, item_ids, _, _ = get_faiss()
    vec = vec.reshape(1, -1).astype("float32")
    scores, indices = index.search(vec, k)
    return [
        {"item_id": int(item_ids[i]), "score": float(s)}
        for s, i in zip(scores[0], indices[0])
    ]


def recommend_for_user(
    user_id: int,
    top_k: int = 20,
    strategy: str = "content",
) -> list[dict]:
    """
    Recommend items for a user.

    strategy: "content" (FAISS) or "collab" (ALS). If collab requested but
    ALS not available, falls back to content.
    """
    if strategy == "collab":
        als = get_als()
        if als is not None:
            return _recommend_collab(user_id, top_k, als)

    # content or fallback
    return _recommend_content(user_id, top_k)


def _recommend_content(user_id: int, top_k: int) -> list[dict]:
    """Content-based: user vector -> FAISS."""
    user_vectors = get_user_vectors()
    if user_id not in user_vectors:
        return []
    vec = user_vectors[user_id].astype("float32")
    return faiss_search(vec, k=top_k)


def _recommend_collab(user_id: int, top_k: int, als: dict) -> list[dict]:
    """ALS-based: user factors -> dot with all item factors -> top-K."""
    user_id_to_idx = als["user_id_to_idx"]
    item_factors = als["item_factors"]
    user_factors = als["user_factors"]
    idx_to_item_id = als["idx_to_item_id"]

    if user_id not in user_id_to_idx:
        return []

    u_idx = user_id_to_idx[user_id]
    u_vec = user_factors[u_idx].astype("float32")

    # Score all items
    scores = np.dot(item_factors, u_vec)
    top_indices = np.argsort(scores)[::-1][:top_k]

    return [
        {"item_id": int(idx_to_item_id[i]), "score": float(scores[i])}
        for i in top_indices
        if i in idx_to_item_id
    ]


def recommend_for_item(item_id: int, top_k: int = 20) -> list[dict]:
    """Item-to-item: content-based via FAISS."""
    _, _, item_X, row_map = get_faiss()
    if item_id not in row_map:
        return []
    idx = row_map[item_id]
    vec = item_X[idx].astype("float32")
    return faiss_search(vec, k=top_k)
