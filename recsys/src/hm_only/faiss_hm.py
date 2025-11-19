from pathlib import Path
import joblib
import numpy as np

try:
    import faiss
except ImportError:
    raise ImportError(
        "faiss not installed. Install with `pip install faiss-cpu` inside your venv."
    )

BASE = Path("recsys")
ART = BASE / "artifacts"

HM_MAX_ID = 3_000_000

in_path = ART / "faiss_items.joblib"
backup_path = ART / "faiss_items_backup.joblib"
out_path = ART / "faiss_items_hm.joblib"

print(f"[repair_faiss_hm] Loading {in_path} ...")
faiss_pack = joblib.load(in_path)

item_ids = np.array(faiss_pack["item_ids"])
X = np.array(faiss_pack["X"]).astype("float32")

print(f"[repair_faiss_hm] Original FAISS items: {len(item_ids)}")
print(f"[repair_faiss_hm] Original max item_id: {item_ids.max()}")

# Keep only H&M items
mask = item_ids < HM_MAX_ID
item_ids_hm = item_ids[mask]
X_hm = X[mask]

print(f"[repair_faiss_hm] H&M-only items: {len(item_ids_hm)}")
print(f"[repair_faiss_hm] H&M max item_id: {item_ids_hm.max()}")

dim = X_hm.shape[1]
print(f"[repair_faiss_hm] Embedding dim: {dim}")

# Rebuild FAISS index (using inner product; adjust if your original used L2)
index = faiss.IndexFlatIP(dim)
index.add(X_hm)

# Row map: item_id -> row index
row_map = {int(iid): idx for idx, iid in enumerate(item_ids_hm)}

# Backup original artifact once
if not backup_path.exists():
    print(f"[repair_faiss_hm] Backing up original FAISS pack to {backup_path}")
    joblib.dump(faiss_pack, backup_path)

new_pack = {
    "index": index,
    "item_ids": item_ids_hm,
    "X": X_hm,
    "row_map": row_map,
}

print(f"[repair_faiss_hm] Writing repaired FAISS pack to {out_path}")
joblib.dump(new_pack, out_path)

print("[repair_faiss_hm] Done. You can now point your code to faiss_items_hm.joblib.")