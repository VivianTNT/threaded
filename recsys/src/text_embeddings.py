# recsys/src/text_embeddings.py

from transformers import CLIPProcessor, CLIPModel
import torch, joblib
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm

DATA, ART = Path("recsys/data"), Path("recsys/artifacts")

print("[text_embeddings] loading model ...")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

print("[text_embeddings] loading items parquet ...")
df = pd.read_parquet(DATA / "items.parquet")
print(f"[text_embeddings] loaded {len(df):,} items with columns: {list(df.columns)}")

# --- Detect usable text columns dynamically ---
text_cols = [c for c in df.columns if c.lower() in ["name", "title", "productdisplayname", "description"]]
brand_cols = [c for c in df.columns if c.lower() in ["brand", "brand_name", "manufacturer"]]

if not text_cols:
    raise ValueError("❌ Could not find any text columns (name/title/description) in items.parquet")

text_col = text_cols[0]
brand_col = brand_cols[0] if brand_cols else None

print(f"[text_embeddings] using text column: '{text_col}'")
if brand_col:
    print(f"[text_embeddings] using brand column: '{brand_col}'")
else:
    print("[text_embeddings] no brand column found; using only text")

texts = []
ids = []

for _, row in df.iterrows():
    name_part = str(row[text_col]) if pd.notna(row[text_col]) else ""
    brand_part = str(row[brand_col]) if brand_col and pd.notna(row[brand_col]) else ""
    text = f"{brand_part} {name_part}".strip()
    if not text:
        text = "generic product"
    texts.append(text)
    ids.append(row["item_id"])

batch_size = 256
text_embs = []

for i in tqdm(range(0, len(texts), batch_size), total=(len(texts) + batch_size - 1) // batch_size, desc="Encoding text embeddings"):
    batch_texts = texts[i:i+batch_size]
    inputs = processor(text=batch_texts, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        emb = model.get_text_features(**inputs).cpu().numpy()
    text_embs.append(emb)

X = np.vstack(text_embs).astype("float32")
joblib.dump({"X": X, "item_ids": ids}, ART / "text_embeddings.joblib")
print(f"✅ saved text embeddings for {len(ids):,} items (dim={X.shape[1]})")