"""
Smoke test: load als_hm.joblib and return top-K item IDs for a known user.
Skips if als_hm.joblib does not exist.
"""
import pytest
import joblib
import numpy as np
from pathlib import Path

ART = Path("recsys/artifacts")
ALS_PATH = ART / "als_hm.joblib"


@pytest.fixture(scope="module")
def als_model():
    if not ALS_PATH.exists():
        pytest.skip(f"ALS artifact not found: {ALS_PATH}. Run import_factors first.")
    return joblib.load(ALS_PATH)


def test_collab_recommend_loads(als_model):
    """ALS artifact has required fields."""
    assert "user_factors" in als_model
    assert "item_factors" in als_model
    assert "user_id_to_idx" in als_model
    assert "item_id_to_idx" in als_model
    assert "idx_to_user_id" in als_model
    assert "idx_to_item_id" in als_model


def test_collab_recommend_top_k(als_model):
    """Returns top-K item IDs for a known user."""
    user_id_to_idx = als_model["user_id_to_idx"]
    idx_to_item_id = als_model["idx_to_item_id"]
    user_factors = als_model["user_factors"]
    item_factors = als_model["item_factors"]

    # Pick first known user
    if not user_id_to_idx:
        pytest.skip("No users in ALS model")
    user_id = next(iter(user_id_to_idx))
    u_idx = user_id_to_idx[user_id]
    u_vec = user_factors[u_idx].astype("float32")

    # Score all items
    scores = np.dot(item_factors, u_vec)
    top_k = 10
    top_indices = np.argsort(scores)[::-1][:top_k]

    # Map to item IDs
    item_ids = []
    for i in top_indices:
        if i in idx_to_item_id:
            item_ids.append(int(idx_to_item_id[i]))

    assert len(item_ids) > 0
    assert len(item_ids) <= top_k
    assert all(isinstance(x, int) for x in item_ids)
