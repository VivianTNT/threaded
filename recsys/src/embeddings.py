import numpy as np, pandas as pd, faiss
from sentence_transformers import SentenceTransformer

TXT = SentenceTransformer("all-MiniLM-L6-v2")  # fast & good

def build_item_matrix(items_df: pd.DataFrame):
    texts = (items_df["title"].fillna("") + " " + items_df["description"].fillna("")).tolist()
    X = TXT.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    X = X.astype("float32")
    faiss.normalize_L2(X)  # safety
    return X  # (N, D)

def build_faiss_index(X: np.ndarray):
    idx = faiss.IndexFlatIP(X.shape[1])      # cosine via normalized IP
    idx.add(X)
    return idx
