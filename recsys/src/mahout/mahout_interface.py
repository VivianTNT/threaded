import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import csv

DATA = Path("recsys/data")
ART = Path("recsys/artifacts")
MAHOUT_DIR = DATA / "mahout"

def export_for_mahout():
    """
    Exports events to a CSV file format compatible with Apache Mahout.
    Format: userID,itemID,preference
    """
    print("[mahout] Loading events...")
    events = pd.read_parquet(DATA / "events_hm.parquet")
    
    # Filter valid items
    events = events[events["item_id"] < 3000000]
    events = events[events["user_id"] < 3000000]
    
    # Create simple interaction file: user, item, 1
    # Mahout often expects integers, so we ensure they are ints
    df = events[["user_id", "item_id"]].copy()
    df["preference"] = 1
    
    MAHOUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = MAHOUT_DIR / "interactions.csv"
    
    print(f"[mahout] Exporting {len(df):,} interactions to {out_path}...")
    df.to_csv(out_path, index=False, header=False)
    
    # Also save the mappings to interpret Mahout results later
    # (In this dataset IDs are already integers, so mapping is identity, 
    # but good practice to be explicit)
    unique_users = df["user_id"].unique()
    unique_items = df["item_id"].unique()
    
    mappings = {
        "user_map": {int(u): int(u) for u in unique_users},
        "item_map": {int(i): int(i) for i in unique_items}
    }
    joblib.dump(mappings, MAHOUT_DIR / "id_mappings.joblib")
    
    print("✅ Export complete.")
    print("\nNext Steps (Run Mahout externally):")
    print(f"1. Copy {out_path} to your Hadoop/Spark cluster or local Mahout setup.")
    print("2. Run Mahout ALS (e.g., using spark-itemsimilarity or parallelALS).")
    print("3. Export the resulting user and item factor matrices to CSV.")
    print("   - user_factors.csv (format: userId, feature1, feature2, ...)")
    print("   - item_factors.csv (format: itemId, feature1, feature2, ...)")
    print("4. Place those files in recsys/data/mahout/")


def import_mahout_results(user_factors_path, item_factors_path):
    """
    Imports Mahout output vectors and saves them as the 'collaborative filtering' model.
    This replaces the 'implicit' library model.
    """
    print(f"[mahout] Importing vectors from {user_factors_path} and {item_factors_path}...")
    
    # Load mappings to ensure we align IDs correctly
    mappings = joblib.load(MAHOUT_DIR / "id_mappings.joblib")
    
    # Helper to read vectors
    def read_vectors(path):
        vectors = {}
        with open(path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if not row: continue
                # Assuming format: ID, val1, val2, ...
                obj_id = int(row[0])
                vec = np.array([float(x) for x in row[1:]], dtype=np.float32)
                vectors[obj_id] = vec
        return vectors

    user_vecs_dict = read_vectors(user_factors_path)
    item_vecs_dict = read_vectors(item_factors_path)
    
    if not user_vecs_dict:
        raise ValueError("No user vectors found.")
        
    dim = len(next(iter(user_vecs_dict.values())))
    print(f"[mahout] Detected dimension: {dim}")
    
    # Convert to dense matrices (aligned with our internal mappings)
    # We need to map the Mahout IDs back to our sparse matrix indices if we were using 'implicit'
    # But since we are replacing 'implicit', we can structure the artifact directly.
    
    # Re-create the structure expected by 'recsys/src/models/hybrid_ranker.py'
    # It expects: "user_factors", "item_factors", "user_map", "item_map"
    # Where factors are arrays accessed by index, and map converts RealID -> Index
    
    sorted_users = sorted(user_vecs_dict.keys())
    sorted_items = sorted(item_vecs_dict.keys())
    
    user_map = {uid: idx for idx, uid in enumerate(sorted_users)}
    item_map = {iid: idx for idx, iid in enumerate(sorted_items)}
    
    user_factors = np.zeros((len(sorted_users), dim), dtype=np.float32)
    item_factors = np.zeros((len(sorted_items), dim), dtype=np.float32)
    
    for uid, vec in user_vecs_dict.items():
        user_factors[user_map[uid]] = vec
        
    for iid, vec in item_vecs_dict.items():
        item_factors[item_map[iid]] = vec
        
    # Save as the "hm_collab_model.joblib" effectively swapping out the backend
    payload = {
        "user_factors": user_factors,
        "item_factors": item_factors,
        "user_map": user_map,
        "item_map": item_map
    }
    
    out_path = ART / "hm_collab_model.joblib"
    joblib.dump(payload, out_path)
    print(f"✅ Saved Mahout vectors to {out_path}")
    print("The hybrid system will now use Mahout vectors for the 'collaborative' score.")
    
    # Also save as simple user vectors dictionary for Two-Tower fine-tuning
    user_vecs_export = {uid: vec for uid, vec in user_vecs_dict.items()}
    joblib.dump(user_vecs_export, ART / "user_vectors_mahout.joblib")
    print(f"✅ Saved Mahout user vectors to {ART / 'user_vectors_mahout.joblib'}")
    print("You can use this to fine-tune the Two-Tower model by using these vectors as input.")

if __name__ == "__main__":
    # Example usage:
    # 1. Export
    export_for_mahout()
    
    # 2. Import (Uncomment when files exist)
    # import_mahout_results(
    #     MAHOUT_DIR / "user_factors.csv", 
    #     MAHOUT_DIR / "item_factors.csv"
    # )
