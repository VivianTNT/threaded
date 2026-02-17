"""
Build FAISS index for RetailRocket items only (text-only; RR has no images).
Reads from faiss_items.joblib and filters to item_ids < HM_MAX_ID.
"""
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

HM_MAX_ID = 3_000_000  # RR item_ids < 3M, HM >= 3M

in_path = ART / "faiss_items.joblib"
out_path = ART / "faiss_items_rr.joblib"

if not in_path.exists():
    raise FileNotFoundError(
        f"Run build_faiss first to create {in_path}. See recsys/PIPELINE_RUNBOOK.md"
    )

print(f"[faiss_rr] Loading {in_path} ...")
faiss_pack = joblib.load(in_path)

item_ids = np.array(faiss_pack["item_ids"])
X = np.array(faiss_pack["X"]).astype("float32")

print(f"[faiss_rr] Original FAISS items: {len(item_ids)}")

# Keep only RR items (small IDs)
mask = item_ids < HM_MAX_ID
item_ids_rr = item_ids[mask]
X_rr = X[mask]

print(f"[faiss_rr] RR-only items: {len(item_ids_rr)}")

if len(item_ids_rr) == 0:
    print("[faiss_rr] No RR items in faiss_items.joblib. Skipping.")
    exit(0)

dim = X_rr.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(X_rr)

row_map = {int(iid): idx for idx, iid in enumerate(item_ids_rr)}

new_pack = {
    "index": index,
    "item_ids": item_ids_rr,
    "X": X_rr,
    "row_map": row_map,
}

ART.mkdir(parents=True, exist_ok=True)
joblib.dump(new_pack, out_path)
print(f"[faiss_rr] Wrote {out_path}")
