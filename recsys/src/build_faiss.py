import pandas as pd, joblib, faiss
import numpy as np
from pathlib import Path
from recsys.src.embeddings import build_item_matrix

DATA = Path("recsys/data")
ART  = Path("recsys/artifacts")

def load_embedding_cache(path):
    if path.exists():
        cache = joblib.load(path)
        print(f"[build_faiss] loaded {len(cache['item_ids'])} cached embeddings from {path.name}")
        return cache["X"], cache["item_ids"]
    else:
        print(f"[build_faiss] no cache found at {path.name}")
        return None, []

def fuse_embeddings(image_embs, text_embs, w_image=0.7, w_text=0.3):
    # Fuse embeddings by weighted average, handling possible None inputs
    if image_embs is not None and text_embs is not None:
        return w_image * image_embs + w_text * text_embs
    elif image_embs is not None:
        return image_embs
    elif text_embs is not None:
        return text_embs
    else:
        return None

if __name__ == "__main__":
    items = pd.read_parquet(DATA / "items.parquet")

    # --- Load cached image and text embeddings if available ---
    image_cache_path = ART / "image_embeddings.joblib"
    text_cache_path = ART / "text_embeddings.joblib"

    old_image_X, old_image_ids = load_embedding_cache(image_cache_path)
    old_text_X, old_text_ids = load_embedding_cache(text_cache_path)

    # Determine common cached items (intersection)
    if old_image_ids and old_text_ids:
        cached_ids = list(set(old_image_ids) & set(old_text_ids))
        if cached_ids:
            # Align embeddings by cached_ids order
            image_idx_map = {i: idx for idx, i in enumerate(old_image_ids)}
            text_idx_map = {i: idx for idx, i in enumerate(old_text_ids)}
            old_image_X_aligned = np.array([old_image_X[image_idx_map[i]] for i in cached_ids], dtype=old_image_X.dtype)
            old_text_X_aligned = np.array([old_text_X[text_idx_map[i]] for i in cached_ids], dtype=old_text_X.dtype)
            old_fused_X = fuse_embeddings(old_image_X_aligned, old_text_X_aligned)
            old_item_ids = np.array(cached_ids)
        else:
            old_fused_X, old_item_ids = None, np.array([])
    elif old_image_ids:
        old_fused_X = old_image_X
        old_item_ids = np.array(old_image_ids)
    elif old_text_ids:
        old_fused_X = old_text_X
        old_item_ids = np.array(old_text_ids)
    else:
        old_fused_X, old_item_ids = None, np.array([])

    # --- Only embed new items ---
    new_items = items[~items["item_id"].isin(old_item_ids)]
    if len(new_items) > 0:
        print(f"[build_faiss] embedding {len(new_items)} new items ...")
        # Build new image and text embeddings
        new_image_X = build_item_matrix(new_items).astype("float16")
        # For text embeddings, assuming a similar function exists; if not, use zeros
        try:
            from recsys.src.text_embeddings import build_text_matrix
            new_text_X = build_text_matrix(new_items).astype("float16")
        except ImportError:
            print("[build_faiss] text embedding function not found, using zeros for text embeddings")
            new_text_X = np.zeros_like(new_image_X, dtype="float16")

        # Fuse new embeddings
        new_fused_X = fuse_embeddings(new_image_X, new_text_X)

        # Combine old and new fused embeddings
        if old_fused_X is not None:
            X = np.concatenate([old_fused_X, new_fused_X])
            item_ids = np.concatenate([old_item_ids, new_items["item_id"].to_numpy()])
        else:
            X = new_fused_X
            item_ids = new_items["item_id"].to_numpy()
    else:
        X, item_ids = old_fused_X, old_item_ids

    if X is None or len(X) == 0:
        print("[build_faiss] No embeddings available to index.")
        exit(0)

    # --- Build FAISS index (approximate, GPU optional) ---
    dim = X.shape[1]
    index = faiss.IndexHNSWFlat(dim, 32)
    index.hnsw.efConstruction = 100
    index.add(X)

    payload = {
        "index": index,
        "X": X,
        "item_ids": item_ids,
        "row_map": {int(i): idx for idx, i in enumerate(item_ids)},
    }

    ART.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, ART / "faiss_items.joblib")
    joblib.dump({"X": X, "item_ids": item_ids}, ART / "item_embs.joblib")
    print(f"[build_faiss] Indexed {X.shape[0]} items (dim={dim})")