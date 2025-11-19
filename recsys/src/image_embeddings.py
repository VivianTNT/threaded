# recsys/src/image_embeddings.py
import torch
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np
import joblib
from transformers import CLIPProcessor, CLIPModel

DATA = Path("recsys/data")
ART = Path("recsys/artifacts")
IMG_DIR = DATA / "images"

# Load the original H&M articles.csv
articles_path = DATA / "hm_raw" / "articles.csv"
articles = pd.read_csv(articles_path)

# Use article_id as the image key
print(f"[image_embeddings] Using {len(articles):,} H&M articles for image embeddings")

device = "mps" if torch.backends.mps.is_available() else (
    "cuda" if torch.cuda.is_available() else "cpu"
)
print(f"[image_embeddings] Loading CLIP model on {device} ...")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model.eval()

def embed_image(img_path):
    try:
        image = Image.open(img_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            emb = model.get_image_features(**inputs)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.cpu().numpy().flatten()
    except Exception:
        return None

embeddings, ids = [], []
print("[image_embeddings] Scanning for images...")

BATCH_SIZE = 16
CHECKPOINT_EVERY = 1000
checkpoint_path = ART / "image_embeddings_partial.joblib"

# Load any previous progress
if checkpoint_path.exists():
    saved = joblib.load(checkpoint_path)
    ids, embeddings = saved["ids"], saved["X_img"]
    print(f"[resume] Loaded {len(ids)} existing embeddings.")
else:
    ids, embeddings = [], []

processed_ids = set(ids)

batch_images, batch_ids = [], []

for _, row in tqdm(articles.iterrows(), total=len(articles)):
    article_id = str(row["article_id"]).zfill(10)
    if int(article_id) in processed_ids:
        continue  # skip already processed

    img_path = IMG_DIR / article_id[:3] / f"{article_id}.jpg"
    if img_path.exists():
        try:
            image = Image.open(img_path).convert("RGB")
            batch_images.append(image)
            batch_ids.append(int(article_id))
        except Exception:
            continue

        # When batch full → process
        if len(batch_images) == BATCH_SIZE:
            inputs = processor(images=batch_images, return_tensors="pt", padding=True).to(device)
            with torch.no_grad():
                batch_embs = model.get_image_features(**inputs)
                batch_embs = batch_embs / batch_embs.norm(dim=-1, keepdim=True)
            embeddings.extend(batch_embs.cpu().numpy())
            ids.extend(batch_ids)
            batch_images, batch_ids = [], []

            # Save checkpoint periodically
            if len(ids) % CHECKPOINT_EVERY == 0:
                joblib.dump({"ids": ids, "X_img": np.array(embeddings)}, checkpoint_path)
                print(f"[checkpoint] Saved {len(ids)} embeddings...")

# Process any leftovers
if batch_images:
    inputs = processor(images=batch_images, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        batch_embs = model.get_image_features(**inputs)
        batch_embs = batch_embs / batch_embs.norm(dim=-1, keepdim=True)
    embeddings.extend(batch_embs.cpu().numpy())
    ids.extend(batch_ids)

# Save final
joblib.dump({"ids": ids, "X_img": np.array(embeddings)}, ART / "image_embeddings.joblib")
print(f"[done] Saved {len(ids)} total embeddings to {ART / 'image_embeddings.joblib'}")