"""
Feature-Based Two-Tower Training

Trains on CONTENT FEATURES instead of ALS vectors, making it generalizable
to new users and new products.

Architecture:
  - User Tower: Takes avg embedding of items user liked (384-dim)
  - Item Tower: Takes product embedding (384-dim)
  - Output: Interaction score [0,1]

This model works for:
  ✅ New users (build vector from their likes)
  ✅ New products (embed with same SentenceTransformer)
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from tqdm import tqdm
import glob
import argparse

DATA = Path("recsys/data")
ART = Path("recsys/artifacts")

# =====================================================
# Dataset Class
# =====================================================

class ContentTwoTowerDataset(Dataset):
    def __init__(self, df, item_embeddings, row_map, events_df):
        """
        df: training pairs with columns [user_id, item_id, label]
        item_embeddings: FAISS X matrix (N_items, 384)
        row_map: item_id -> row index
        events_df: full events to build user vectors
        """
        self.df = df
        self.item_embeddings = item_embeddings.astype("float32")
        self.row_map = row_map
        
        # Pre-compute user vectors from history
        print("[dataset] Building user vectors from interaction history...")
        self.user_vecs = self._build_user_vectors(events_df)
        
        # Filter to users we have vectors for
        valid_mask = df["user_id"].isin(self.user_vecs.keys())
        self.df = df[valid_mask].reset_index(drop=True)
        print(f"[dataset] Kept {len(self.df)} pairs with valid user vectors")
        
    def _build_user_vectors(self, events_df):
        """Build user vector = mean of items they've interacted with."""
        user_vecs = {}
        for uid, group in tqdm(events_df.groupby("user_id"), desc="Building user vecs"):
            item_ids = group["item_id"].values
            vecs = []
            for iid in item_ids:
                if iid in self.row_map:
                    vecs.append(self.item_embeddings[self.row_map[iid]])
            if vecs:
                vec = np.mean(vecs, axis=0)
                vec = vec / (np.linalg.norm(vec) + 1e-8)
                user_vecs[uid] = vec
        return user_vecs
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        uid = int(row["user_id"])
        iid = int(row["item_id"])
        label = float(row["label"])
        
        u_vec = self.user_vecs.get(uid)
        i_vec = self.item_embeddings[self.row_map[iid]] if iid in self.row_map else None
        
        if u_vec is None or i_vec is None:
            # Fallback to zeros (shouldn't happen after filtering)
            u_vec = np.zeros(384, dtype=np.float32)
            i_vec = np.zeros(384, dtype=np.float32)
        
        return (
            torch.tensor(u_vec, dtype=torch.float32),
            torch.tensor(i_vec, dtype=torch.float32),
            torch.tensor(label, dtype=torch.float32),
        )

# =====================================================
# Model Architecture
# =====================================================

class ContentTwoTower(nn.Module):
    def __init__(self, feature_dim=384, embed_dim=128):
        super().__init__()
        # Both towers take content embeddings (symmetric)
        self.user_tower = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, embed_dim),
            nn.ReLU(),
        )
        self.item_tower = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, embed_dim),
            nn.ReLU(),
        )
        self.scorer = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
    
    def forward(self, u_vec, i_vec):
        u = self.user_tower(u_vec)
        i = self.item_tower(i_vec)
        x = torch.abs(u - i)
        return torch.sigmoid(self.scorer(x).squeeze(1))

# =====================================================
# Training Loop
# =====================================================

def train(dataset="hm", epochs=3, batch_size=512, lr=1e-3, load_model=None):
    print(f"[train] Loading artifacts for {dataset}...")
    
    # Load FAISS embeddings (combined for "both")
    if dataset == "both":
        faiss_pack = joblib.load(ART / "faiss_items.joblib")
        events_hm = pd.read_parquet(DATA / "events_hm.parquet")[["user_id", "item_id"]]
        events_rr = pd.read_parquet(DATA / "events_rr.parquet")[["user_id", "item_id"]]
        events = pd.concat([events_hm, events_rr], ignore_index=True)
        pair_files_hm = sorted(glob.glob(str(DATA / "train_pairs_hm" / "*.parquet")))
        pair_files_rr = sorted(glob.glob(str(DATA / "train_pairs_rr" / "*.parquet")))
        pair_files = pair_files_hm + pair_files_rr
    else:
        faiss_pack = joblib.load(ART / f"faiss_items_{dataset}.joblib")
        events = pd.read_parquet(DATA / f"events_{dataset}.parquet")[["user_id", "item_id"]]
        pair_files = sorted(glob.glob(str(DATA / f"train_pairs_{dataset}" / "*.parquet")))
        if not pair_files:
            pair_files = sorted(glob.glob(str(DATA / "train_pairs_rr_hm" / "*.parquet")))
    
    item_X = faiss_pack["X"]
    row_map = faiss_pack["row_map"]
    
    if not pair_files:
        raise FileNotFoundError(f"No training pairs found. Run build_train_pairs_hm.py and build_train_pairs_rr.py first.")
    
    print(f"[train] Found {len(pair_files)} pair files, loading...")
    pairs = pd.concat([pd.read_parquet(f) for f in pair_files], ignore_index=True)
    
    # Create dataset
    dataset_obj = ContentTwoTowerDataset(pairs, item_X, row_map, events)
    loader = DataLoader(dataset_obj, batch_size=batch_size, shuffle=True, num_workers=0)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[train] Training on {device}...")
    
    model = ContentTwoTower(feature_dim=384, embed_dim=128).to(device)
    if load_model:
        path = Path(load_model) if not isinstance(load_model, Path) else load_model
        if path.exists():
            model.load_state_dict(torch.load(path, map_location=device))
            print(f"[train] Loaded pretrained weights from {path}")
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
        
        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1} loss = {avg_loss:.4f}")
    
    out_path = ART / f"content_two_tower_{dataset}.pt"
    if dataset == "both":
        out_path = ART / "content_two_tower_combined.pt"
    torch.save(model.state_dict(), out_path)
    print(f"✅ Saved model to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="hm", choices=["hm", "rr", "both"])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--load_model", type=str, help="Path to .pt to fine-tune from (e.g. content_two_tower_hm.pt)")
    
    args = parser.parse_args()
    train(dataset=args.dataset, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, load_model=args.load_model)
