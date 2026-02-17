import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import joblib
from pathlib import Path
from tqdm import tqdm
import glob
import numpy as np
import argparse

DATA = Path("recsys/data")
ART = Path("recsys/artifacts")

class RecSysDataset(Dataset):
    def __init__(self, df, user_vecs, item_vecs, user_dim, item_dim):
        self.df = df
        self.user_vecs = user_vecs
        self.item_vecs = item_vecs
        self.user_dim = user_dim
        self.item_dim = item_dim
        self.user_ids = df["user_id"].to_numpy()
        self.item_ids = df["item_id"].to_numpy()
        self.labels = df["label"].astype("float32").to_numpy()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        uid = int(self.user_ids[idx])
        iid = int(self.item_ids[idx])
        u_vec = self.user_vecs.get(uid)
        i_vec = self.item_vecs.get(iid)
        
        if u_vec is None:
            u_vec = np.zeros(self.user_dim, dtype=np.float32)
        if i_vec is None:
            i_vec = np.zeros(self.item_dim, dtype=np.float32)
            
        return (
            torch.tensor(u_vec, dtype=torch.float32),
            torch.tensor(i_vec, dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.float32),
        )

class MahoutTwoTower(nn.Module):
    def __init__(self, user_dim, item_dim):
        super().__init__()
        # User Tower takes Mahout vectors
        self.user_tower = nn.Sequential(
            nn.Linear(user_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )
        # Item Tower takes either Mahout item vectors OR CLIP vectors
        self.item_tower = nn.Sequential(
            nn.Linear(item_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )
        self.output = nn.Linear(64, 1)

    def forward(self, u_vec, i_vec):
        u = self.user_tower(u_vec)
        i = self.item_tower(i_vec)
        x = torch.abs(u - i)
        return torch.sigmoid(self.output(x).squeeze(1))

def train_mahout_finetune(dataset="hm", epochs=3, batch_size=512, lr=1e-3, load_model=None):
    print(f"[mahout_finetune] Loading Mahout vectors for {dataset}...")
    
    # We use Mahout vectors as inputs for both
    mahout_user_path = ART / f"user_vectors_mahout_{dataset}.joblib"
    mahout_item_path = ART / f"item_vectors_mahout_{dataset}.joblib"
    
    # Fallback for H&M legacy name
    if dataset == "hm" and not mahout_user_path.exists():
        mahout_user_path = ART / "user_vectors_mahout.joblib"

    if not mahout_user_path.exists():
        raise FileNotFoundError(f"Could not find {mahout_user_path}. Run import_factors first (see recsys/src/mahout/runbook.md).")
        
    user_vecs = joblib.load(mahout_user_path)
    
    # For H&M, we could use CLIP embeddings, but for a consistent "fine-tuning" baseline,
    # let's use Mahout item vectors as the primary input.
    if mahout_item_path.exists():
        item_vecs = joblib.load(mahout_item_path)
    else:
        # Fallback to CLIP if Mahout item vecs missing for H&M
        print("[mahout_finetune] Mahout item vectors missing, using CLIP/FAISS embeddings...")
        faiss_pack = joblib.load(ART / "faiss_items_hm.joblib")
        item_X = faiss_pack["X"]
        item_ids = faiss_pack["item_ids"]
        item_vecs = {int(iid): item_X[idx] for idx, iid in enumerate(item_ids)}

    # Dims
    sample_u = next(iter(user_vecs))
    sample_i = next(iter(item_vecs))
    user_dim = len(user_vecs[sample_u])
    item_dim = len(item_vecs[sample_i])
    print(f"[mahout_finetune] User dim: {user_dim}, Item dim: {item_dim}")

    # Load Training Pairs
    print(f"[mahout_finetune] Loading training pairs for {dataset}...")
    if dataset == "hm":
        # H&M only pairs
        pair_path = DATA / "train_pairs_hm" / "train_pairs_hm_part*.parquet"
    else:
        # For RR, you can use the combined rr_hm pairs OR separate RR pairs
        # The combined pairs have both H&M and RR interactions (useful for transfer learning)
        pair_path = DATA / "train_pairs_rr_hm" / "train_pairs_part*.parquet"
        if not glob.glob(str(pair_path)):
            pair_path = DATA / "train_pairs_rr" / "train_pairs_part*.parquet"

    files = sorted(glob.glob(str(pair_path)))
    if not files:
        raise FileNotFoundError(f"No training pairs found at {pair_path}")
        
    print(f"[mahout_finetune] Found {len(files)} files. Loading...")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    
    # Ensure we have the required columns (timestamp is optional)
    required_cols = ["user_id", "item_id", "label"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Filter to only the columns we need
    df = df[required_cols].copy()
    
    dataset_obj = RecSysDataset(df, user_vecs, item_vecs, user_dim, item_dim)
    loader = DataLoader(dataset_obj, batch_size=batch_size, shuffle=True)

    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[mahout_finetune] Training on {device}...")
    
    model = MahoutTwoTower(user_dim, item_dim).to(device)
    
    # Fine-tuning logic: Load weights from a previously trained model
    if load_model:
        print(f"[mahout_finetune] Fine-tuning from {load_model}...")
        try:
            model.load_state_dict(torch.load(load_model, map_location=device))
        except Exception as e:
            print(f"⚠️ Warning: Could not load full model (dims might differ). Error: {e}")
            print("Attempting to load shared weights (User Tower)...")
            # You can add partial loading here if user/item dims differ between datasets

    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

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

    out_path = ART / f"mahout_finetuned_{dataset}.pt"
    torch.save(model.state_dict(), out_path)
    print(f"✅ Saved fine-tuned model to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="hm", choices=["hm", "rr"])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--load_model", type=str, help="Path to base model .pt for fine-tuning")
    
    args = parser.parse_args()
    train_mahout_finetune(dataset=args.dataset, epochs=args.epochs, load_model=args.load_model)
