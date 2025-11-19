"""
Dataset loaders for different model types.
Provides utilities to load training data for various recommendation models.
"""
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Optional
import glob

DATA = Path("recsys/data")
ART = Path("recsys/artifacts")

class RecSysDataset(Dataset):
    """Dataset for two-tower and neural network models."""
    
    def __init__(self, df: pd.DataFrame, user_vecs: Dict, item_vecs: Dict):
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
        
        # Handle missing vectors
        if u_vec is None or i_vec is None:
            dim = 384  # Default embedding dimension
            u_vec = np.zeros(dim, dtype=np.float32)
            i_vec = np.zeros(dim, dtype=np.float32)
        
        import torch
        return (
            torch.tensor(u_vec, dtype=torch.float32),
            torch.tensor(i_vec, dtype=torch.float32),
            torch.tensor(self.labels[idx], dtype=torch.float32),
        )

class CollaborativeFilteringDataset:
    """Dataset loader for collaborative filtering (implicit feedback)."""
    
    @staticmethod
    def load_interactions():
        """Load user-item interactions for collaborative filtering."""
        files = sorted(glob.glob(str(DATA / "train_pairs" / "train_pairs_part*.parquet")))
        
        dfs = []
        for f in files:
            df = pd.read_parquet(f, columns=["user_id", "item_id", "label"])
            # Use only positives
            df = df[df["label"] == 1]
            dfs.append(df)
        
        return pd.concat(dfs, ignore_index=True)

class ContentFilteringDataset:
    """Dataset loader for content-based filtering."""
    
    @staticmethod
    def load_data():
        """Load data for content filtering."""
        files = sorted(glob.glob(str(DATA / "train_pairs" / "train_pairs_part*.parquet")))
        
        # Load all pairs (positives and negatives)
        dfs = []
        for f in files:
            df = pd.read_parquet(f)
            dfs.append(df)
        
        return pd.concat(dfs, ignore_index=True)

def load_embeddings():
    """Load user and item embeddings."""
    user_vecs = joblib.load(ART / "user_vectors.joblib")
    faiss_pack = joblib.load(ART / "faiss_items.joblib")
    item_ids = faiss_pack["item_ids"]
    item_X = faiss_pack["X"]
    item_vecs = {int(i): item_X[idx] for idx, i in enumerate(item_ids)}
    
    return user_vecs, item_vecs

def create_dataloader(
    chunk_files: Optional[list] = None,
    batch_size: int = 512,
    shuffle: bool = True,
    num_workers: int = 4
) -> DataLoader:
    """
    Create a DataLoader for training neural network models.
    
    Args:
        chunk_files: List of parquet files to load. If None, loads all chunks.
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
    
    Returns:
        DataLoader instance
    """
    if chunk_files is None:
        chunk_files = sorted(glob.glob(str(DATA / "train_pairs" / "train_pairs_part*.parquet")))
    
    # Load data
    dfs = [pd.read_parquet(f) for f in chunk_files]
    df = pd.concat(dfs, ignore_index=True)
    
    # Load embeddings
    user_vecs, item_vecs = load_embeddings()
    
    # Create dataset
    dataset = RecSysDataset(df, user_vecs, item_vecs)
    
    # Create dataloader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )
    
    return loader

def get_train_val_split(
    train_ratio: float = 0.8,
    random_state: int = 42
) -> tuple:
    """
    Split training pairs into train and validation sets.
    
    Args:
        train_ratio: Ratio of training data
        random_state: Random seed
    
    Returns:
        (train_df, val_df) tuple
    """
    files = sorted(glob.glob(str(DATA / "train_pairs" / "train_pairs_part*.parquet")))
    
    # Load all data
    dfs = [pd.read_parquet(f) for f in files]
    df = pd.concat(dfs, ignore_index=True)
    
    # Shuffle and split
    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    split_idx = int(len(df) * train_ratio)
    
    train_df = df.iloc[:split_idx]
    val_df = df.iloc[split_idx:]
    
    return train_df, val_df

