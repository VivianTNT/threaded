import json, orjson
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss, numpy as np

ITEMS = Path("data/clean/items.jsonl")
MODEL = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def load_items():
    items, texts = [], []
    with open(ITEMS, "rb") as f:
        for line in f:
            obj = orjson.loads(line)
            items.append(obj)
            texts.append((obj.get("brand") or "") + " " + obj.get("name","") + " " + (obj.get("category") or ""))
    return items, texts

def main():
    items, texts = load_items()
    X = MODEL.encode(texts, normalize_embeddings=True)
    idx = faiss.IndexFlatIP(X.shape[1]); idx.add(X.astype("float32"))
    np.save("data/emb/items.npy", X)
    with open("data/emb/items.json","w") as f: json.dump(items, f)
    faiss.write_index(idx, "data/emb/items.faiss")
    print("Indexed:", len(items))

if __name__ == "__main__":
    Path("data/emb").mkdir(parents=True, exist_ok=True)
    main()
