# recsys/src/models/collab_filtering.py

import pandas as pd
import numpy as np
from implicit.als import AlternatingLeastSquares
from scipy.sparse import coo_matrix
from pathlib import Path
import joblib

DATA = Path("recsys/data")
ART = Path("recsys/artifacts")

print("[collab_filtering] loading H&M events parquet...")

# Load clean H&M interactions only
events_path = DATA / "events.parquet"
df = pd.read_parquet(events_path, columns=["user_id", "item_id"])

# Filter out RetailRocket-style huge IDs (safety check)
df = df[df["item_id"] < 3000000]
df = df[df["user_id"] < 3000000]

df = df.drop_duplicates()
print(f"[collab_filtering] {len(df):,} valid H&M interactions loaded")

# Build sparse matrix
users = df["user_id"].astype("category")
items = df["item_id"].astype("category")

user_map = dict(enumerate(users.cat.categories))
item_map = dict(enumerate(items.cat.categories))

mat = coo_matrix(
    (np.ones(len(df), dtype=np.float32), (users.cat.codes, items.cat.codes)),
    shape=(len(users.cat.categories), len(items.cat.categories))
)

print("[collab_filtering] matrix shape:", mat.shape)

# Train ALS collaborative filter (implicit feedback)
print("[collab_filtering] training ALS model for H&M only...")
model = AlternatingLeastSquares(
    factors=64,
    regularization=0.05,
    iterations=15,
    use_gpu=False
)

model.fit(mat.T)

# Save factors
user_factors = model.user_factors
item_factors = model.item_factors

joblib.dump({
    "user_factors": user_factors,
    "item_factors": item_factors,
    "user_map": user_map,
    "item_map": item_map
}, ART / "hm_collab_model.joblib")

print("✅ saved H&M collaborative model to recsys/artifacts/hm_collab_model.joblib")