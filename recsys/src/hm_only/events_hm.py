from pathlib import Path
import pandas as pd

BASE = Path("recsys")
DATA = BASE / "data"

HM_MAX_ID = 3_000_000  # threshold separating H&M from RetailRocket

in_path = DATA / "events.parquet"
backup_path = DATA / "events_original_backup.parquet"
out_path = DATA / "events_hm.parquet"  # overwrite in place after cleaning

print(f"[clean_events_hm] Loading {in_path} ...")
events = pd.read_parquet(in_path)

print(f"[clean_events_hm] Original shape: {events.shape}")
print(f"[clean_events_hm] Original max user_id: {events['user_id'].max()}")
print(f"[clean_events_hm] Original max item_id: {events['item_id'].max()}")

# Filter to H&M IDs only
hm_events = events[
    (events["user_id"] < HM_MAX_ID) &
    (events["item_id"] < HM_MAX_ID)
].copy()

print(f"[clean_events_hm] H&M-only shape: {hm_events.shape}")
print(f"[clean_events_hm] H&M max user_id: {hm_events['user_id'].max()}")
print(f"[clean_events_hm] H&M max item_id: {hm_events['item_id'].max()}")

# Backup original before overwriting
if not backup_path.exists():
    print(f"[clean_events_hm] Backing up original events to {backup_path}")
    events.to_parquet(backup_path, index=False)

print(f"[clean_events_hm] Writing cleaned events to {out_path}")
hm_events.to_parquet(out_path, index=False)

print("[clean_events_hm] Done.")