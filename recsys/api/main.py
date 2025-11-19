from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import joblib
from PIL import Image
import io
from pathlib import Path
import torch
from transformers import CLIPProcessor, CLIPModel
import faiss

# Paths
BASE = Path("recsys")
DATA = BASE / "data"
ART = BASE / "artifacts"

app = FastAPI()

# Allow local dev + frontend domain
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

############################################
# Load embeddings + FAISS
############################################

print("[api] Loading H&M FAISS index...")
faiss_pack = joblib.load(ART / "faiss_items_hm.joblib")
faiss_index = faiss_pack["index"]
item_ids = faiss_pack["item_ids"]
item_X = faiss_pack["X"]

row_map = faiss_pack["row_map"]  # item_id → row index


print("[api] Loading user_vectors_hm...")
user_vectors = joblib.load(ART / "user_vectors_hm.joblib")

############################################
# Load CLIP for image embedding
############################################

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[api] Loading CLIP on {device} ...")

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_model.eval()


############################################
# Utility: embed an uploaded image
############################################
def embed_image_file(file: UploadFile) -> np.ndarray:
    image = Image.open(io.BytesIO(file.file.read())).convert("RGB")
    inputs = clip_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        emb = clip_model.get_image_features(**inputs)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy().flatten().astype("float32")


############################################
# Utility: run FAISS search
############################################
def faiss_search(vec: np.ndarray, k: int = 20):
    vec = vec.reshape(1, -1)
    scores, indices = faiss_index.search(vec, k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        item_id = int(item_ids[idx])
        results.append({"item_id": item_id, "score": float(score)})
    return results


############################################
# ROUTES
############################################

@app.get("/health")
def health():
    return {"status": "ok"}


############################################
# 1. Embed image
############################################
@app.post("/embed/image")
async def embed_image(file: UploadFile = File(...)):
    vec = embed_image_file(file)
    return {"embedding": vec.tolist()}


############################################
# 2. Recommend from image(s)
############################################
class RecommendFromImagesRequest(BaseModel):
    images: list[str] = []  # base64 if frontend wants this (optional)


@app.post("/recommend/from_image")
async def recommend_from_image(file: UploadFile = File(...), top_k: int = 20):
    vec = embed_image_file(file)
    results = faiss_search(vec, k=top_k)
    return {"recommendations": results}


############################################
# 3. Recommend for user_id
############################################
@app.get("/recommend/user/{user_id}")
def recommend_user(user_id: int, top_k: int = 20):
    if user_id not in user_vectors:
        return {"recommendations": []}

    vec = user_vectors[user_id].astype("float32")
    results = faiss_search(vec, k=top_k)
    return {"recommendations": results}


############################################
# 4. Item-to-item similarity
############################################
@app.get("/recommend/item/{item_id}")
def recommend_item(item_id: int, top_k: int = 20):
    if item_id not in row_map:
        return {"recommendations": []}

    idx = row_map[item_id]
    vec = item_X[idx].astype("float32")
    results = faiss_search(vec, k=top_k)
    return {"recommendations": results}