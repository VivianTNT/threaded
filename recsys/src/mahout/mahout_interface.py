import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import csv

DATA = Path("recsys/data")
ART = Path("recsys/artifacts")
MAHOUT_DIR = DATA / "mahout"

def export_for_mahout(dataset="hm"):
    """
    Exports events to a CSV file format compatible with Apache Mahout.
    Format: userID,itemID,preference
    """
    print(f"[mahout] Loading events for {dataset}...")
    
    if dataset == "hm":
        events_path = DATA / "events_hm.parquet"
    else:
        # RetailRocket events
        events_path = DATA / "rr_raw" / "events.csv"
        
    if not events_path.exists():
        print(f"❌ Error: {events_path} not found.")
        return

    filtered_by_source = False
    
    if events_path.suffix == ".parquet":
        # We assume pyarrow/fastparquet is available in the environment running this
        print(f"[mahout] Reading parquet file: {events_path}...")
        try:
            # Read all columns first to check for source column
            events = pd.read_parquet(events_path)
            print(f"[mahout] Loaded {len(events):,} rows from parquet.")
            
            # Filter by source column if available (most reliable method)
            if "source" in events.columns:
                expected_source = "HM" if dataset == "hm" else "RR"
                before_filter = len(events)
                events = events[events["source"] == expected_source].copy()
                print(f"[mahout] Filtered by source='{expected_source}': {before_filter:,} -> {len(events):,} rows")
                filtered_by_source = True
            
            # Select only needed columns
            events = events[["user_id", "item_id"]].copy()
        except ImportError as e:
            print("❌ Error: pyarrow/fastparquet not found. Please install them.")
            print(f"   Error details: {e}")
            return
        except Exception as e:
            print(f"❌ Error reading parquet file: {e}")
            import traceback
            traceback.print_exc()
            return
    else:
        # CSV for RR
        events = pd.read_csv(events_path)
        # RR columns are timestamp, visitorid, event, itemid, transactionid
        if "visitorid" in events.columns:
            events = events.rename(columns={"visitorid": "user_id", "itemid": "item_id"})

    # Additional filtering if source column wasn't available
    # Note: H&M item IDs are actually large (2.1B-2.9B), RR IDs are smaller
    if not filtered_by_source:
        print(f"[mahout] Applying ID-based filtering (source column not available)...")
        if dataset == "hm":
            # H&M has large item IDs (>= 2B), so filter for large IDs
            before = len(events)
            events = events[(events["item_id"] >= 2000000000) | (events["user_id"] >= 2000000000)]
            print(f"[mahout] Filtered from {before:,} to {len(events):,} rows.")
        # RR filtering would be: events = events[(events["item_id"] < 3000000) & (events["user_id"] < 3000000)]
    
    # Create simple interaction file: user, item, 1
    print(f"[mahout] Preparing interaction pairs...")
    df = events[["user_id", "item_id"]].copy()
    df["preference"] = 1
    before_dedup = len(df)
    df = df.drop_duplicates()
    print(f"[mahout] Deduplicated from {before_dedup:,} to {len(df):,} pairs.")
    
    MAHOUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = MAHOUT_DIR / f"interactions_{dataset}.csv"
    
    print(f"[mahout] Exporting {len(df):,} interactions to {out_path}...")
    print(f"[mahout] This may take a minute for large files...")
    df.to_csv(out_path, index=False, header=False)
    print(f"[mahout] ✅ CSV export complete!")
    
    # Save mappings
    unique_users = df["user_id"].unique()
    unique_items = df["item_id"].unique()
    
    mappings = {
        "user_map": {int(u): idx for idx, u in enumerate(unique_users)},
        "item_map": {int(i): idx for idx, i in enumerate(unique_items)},
        "idx_to_user": {idx: int(u) for idx, u in enumerate(unique_users)},
        "idx_to_item": {idx: int(i) for idx, i in enumerate(unique_items)}
    }
    joblib.dump(mappings, MAHOUT_DIR / f"id_mappings_{dataset}.joblib")
    
    print(f"✅ Export for {dataset} complete.")


def import_mahout_results(dataset, user_factors_path, item_factors_path):
    """
    Imports Mahout output vectors and saves them as the 'collaborative filtering' model.
    """
    print(f"[mahout] Importing {dataset} vectors from {user_factors_path}...")
    
    def read_vectors(path):
        vectors = {}
        try:
            with open(path, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    if not row: continue
                    obj_id = int(row[0])
                    vec = np.array([float(x) for x in row[1:]], dtype=np.float32)
                    vectors[obj_id] = vec
        except Exception as e:
            print(f"❌ Error reading {path}: {e}")
            return None
        return vectors

    user_vecs_dict = read_vectors(user_factors_path)
    item_vecs_dict = read_vectors(item_factors_path)
    
    if not user_vecs_dict or not item_vecs_dict:
        return
        
    dim = len(next(iter(user_vecs_dict.values())))
    print(f"[mahout] Detected dimension: {dim}")
    
    # Use existing mapping if available, or create new from results
    map_path = MAHOUT_DIR / f"id_mappings_{dataset}.joblib"
    if map_path.exists():
        mappings = joblib.load(map_path)
        user_map = mappings["user_map"]
        item_map = mappings["item_map"]
    else:
        # Fallback: creation from results
        sorted_users = sorted(user_vecs_dict.keys())
        sorted_items = sorted(item_vecs_dict.keys())
        user_map = {uid: idx for idx, uid in enumerate(sorted_users)}
        item_map = {iid: idx for idx, iid in enumerate(sorted_items)}

    user_factors = np.zeros((len(user_map), dim), dtype=np.float32)
    item_factors = np.zeros((len(item_map), dim), dtype=np.float32)
    
    for uid, vec in user_vecs_dict.items():
        if uid in user_map:
            user_factors[user_map[uid]] = vec
        
    for iid, vec in item_vecs_dict.items():
        if iid in item_map:
            item_factors[item_map[iid]] = vec
        
    # Save for hybrid ranker
    payload = {
        "user_factors": user_factors,
        "item_factors": item_factors,
        "user_map": user_map,
        "item_map": item_map
    }
    
    # We save specific versions for H&M and RR
    out_path = ART / f"collab_model_{dataset}.joblib"
    joblib.dump(payload, out_path)
    
    # Legacy fallback for H&M
    if dataset == "hm":
        joblib.dump(payload, ART / "hm_collab_model.joblib")
        
    # Also save as dictionary for Two-Tower input
    user_dict_path = ART / f"user_vectors_mahout_{dataset}.joblib"
    item_dict_path = ART / f"item_vectors_mahout_{dataset}.joblib"
    
    joblib.dump(user_vecs_dict, user_dict_path)
    joblib.dump(item_vecs_dict, item_dict_path)
    
    print(f"✅ Saved {dataset} models and dicts to {ART}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="hm", choices=["hm", "rr"])
    parser.add_argument("--action", type=str, default="export", choices=["export", "import"])
    parser.add_argument("--user_vecs", type=str, help="Path to Mahout user factors CSV")
    parser.add_argument("--item_vecs", type=str, help="Path to Mahout item factors CSV")
    
    args = parser.parse_args()
    
    if args.action == "export":
        export_for_mahout(args.dataset)
    else:
        if not args.user_vecs or not args.item_vecs:
            print("❌ Please provide --user_vecs and --item_vecs for import.")
        else:
            import_mahout_results(args.dataset, args.user_vecs, args.item_vecs)
