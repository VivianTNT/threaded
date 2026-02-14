import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import joblib
from pathlib import Path
from tqdm import tqdm
import numpy as np
import glob

DATA = Path("recsys/data")
ART = Path("recsys/artifacts")

# -----------------------------
# Dataset
# -----------------------------
class RecSysDataset(Dataset):
    def __init__(self, df, user_vecs, item_vecs):
        self.df = df
        self.user_vecs = user_vecs
        self.item_vecs = item_vecs
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
        # Skip if missing (rare)
        if u_vec is None or i_vec is None:
            u_vec = np.zeros(384, dtype=np.float32)
            i_vec = np.zeros(384, dtype=np.float32)
        return (
            torch.tensor(u_vec, dtype=torch.float32),
            torch.tensor(i_vec, dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.float32),
        )

# -----------------------------
# Two-Tower Model
# -----------------------------
class TwoTowerModel(nn.Module):
    def __init__(self, dim=384):
        super().__init__()
        self.user_tower = nn.Sequential(
            nn.Linear(dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )
        self.item_tower = nn.Sequential(
            nn.Linear(dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )
        self.output = nn.Linear(64, 1)

    def forward(self, u_vec, i_vec):
        u_enc = self.user_tower(u_vec)
        i_enc = self.item_tower(i_vec)
        x = torch.abs(u_enc - i_enc)
        score = self.output(x).squeeze(1)
        return torch.sigmoid(score)

# -----------------------------
# Training Loop
# -----------------------------
def train_model(epochs=3, batch_size=512, lr=1e-3):
    print("[ranking_model] loading data ...")
    files = sorted(glob.glob(str(DATA / "train_pairs" / "train_pairs_part*.parquet")))

    # Load user/item vectors and faiss_pack before any usage
    user_vecs = joblib.load(ART / "user_vectors.joblib")
    faiss_pack = joblib.load(ART / "faiss_items.joblib")
    item_ids = faiss_pack["item_ids"]
    item_X = faiss_pack["X"]
    item_vecs = {int(i): item_X[idx] for idx, i in enumerate(item_ids)}

    # Concatenate all dataframes before creating the dataset
    df_list = [pd.read_parquet(f) for f in files]
    df = pd.concat(df_list, ignore_index=True)
    del df_list

    dataset = RecSysDataset(df, user_vecs, item_vecs)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TwoTowerModel(dim=item_X.shape[1]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    print(f"[ranking_model] training on {len(df):,} pairs ...")
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

    torch.save(model.state_dict(), ART / "two_tower_ranker.pt")
    print(f"✅ saved trained model to {ART / 'two_tower_ranker.pt'}")
    return model

if __name__ == "__main__":
    train_model(epochs=3, batch_size=512, lr=1e-3)