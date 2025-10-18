import json
import numpy as np, faiss
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
X = np.load("data/emb/items.npy")
items = json.load(open("data/emb/items.json"))
idx = faiss.read_index("data/emb/items.faiss")

def search(q, k=5):
    v = model.encode([q], normalize_embeddings=True).astype("float32")
    D, I = idx.search(v, k)
    return [(items[i]["name"], items[i]["price"], items[i]["url"]) for i in I[0]]

if __name__ == "__main__":
    for name, price, url in search("black blazer under 120", 5):
        print("-", name, price, url)
