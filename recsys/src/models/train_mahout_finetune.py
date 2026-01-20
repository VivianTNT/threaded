import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import joblib
from pathlib import Path
from tqdm import tqdm
import glob
import numpy as np

# Import from existing codebase
from recsys.src.models.ranking_model import RecSysDataset, TwoTowerModel, DATA, ART

def train_mahout_finetune(epochs=3, batch_size=512, lr=1e-3):
    print("[mahout_finetune] Loading Mahout vectors...")
    mahout_vecs_path = ART / "user_vectors_mahout.joblib"
    
    if not mahout_vecs_path.exists():
        raise FileNotFoundError(f"Could not find {mahout_vecs_path}. Did you run the Mahout import script?")
        
    user_vecs = joblib.load(mahout_vecs_path)
    
    # Check dimension of Mahout vectors
    sample_uid = next(iter(user_vecs))
    mahout_dim = len(user_vecs[sample_uid])
    print(f"[mahout_finetune] Mahout vector dim: {mahout_dim}")

    # Load FAISS items
    faiss_pack = joblib.load(ART / "faiss_items.joblib")
    item_ids = faiss_pack["item_ids"]
    item_X = faiss_pack["X"]
    # We still use CLIP item vectors for the item tower
    item_vecs = {int(i): item_X[idx] for idx, i in enumerate(item_ids)}
    item_dim = item_X.shape[1]

    # Modify Model to accept Mahout dimension for user tower
    class MahoutTwoTower(TwoTowerModel):
        def __init__(self, user_dim, item_dim):
            nn.Module.__init__(self)  # Initialize nn.Module directly
            
            # User Tower takes Mahout vectors
            self.user_tower = nn.Sequential(
                nn.Linear(user_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
            )
            # Item Tower takes CLIP vectors
            self.item_tower = nn.Sequential(
                nn.Linear(item_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
            )
            self.output = nn.Linear(64, 1)

    print("[mahout_finetune] Loading training data...")
    files = sorted(glob.glob(str(DATA / "train_pairs" / "train_pairs_part*.parquet")))
    df_list = [pd.read_parquet(f) for f in files]
    df = pd.concat(df_list, ignore_index=True)
    
    dataset = RecSysDataset(df, user_vecs, item_vecs)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MahoutTwoTower(user_dim=mahout_dim, item_dim=item_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    print(f"[mahout_finetune] Fine-tuning on {len(df):,} pairs...")
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        for u_vec, i_vec, label in tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}"):
            u_vec, i_vec, label = u_vec.to(device), i_vec.to(device), label.to(device)
            optimizer.zero_grad()
            pred = model(u_vec, i_vec)
            loss = criterion(pred, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} loss = {total_loss/len(loader):.4f}")

    out_path = ART / "mahout_finetuned_model.pt"
    torch.save(model.state_dict(), out_path)
    print(f"✅ Saved fine-tuned model to {out_path}")

if __name__ == "__main__":
    train_mahout_finetune()
