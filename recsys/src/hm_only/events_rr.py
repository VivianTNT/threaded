"""Extract RR-only events from events.parquet (creates events_rr.parquet)."""
from pathlib import Path
import pandas as pd

DATA = Path("recsys/data")
RR_USER_OFFSET = 1_000_000_000  # Offset RR user IDs to avoid collision with HM

in_path = DATA / "events.parquet"
out_path = DATA / "events_rr.parquet"

if not in_path.exists():
    raise FileNotFoundError(f"Run make_parquets first. Need {in_path}")

events = pd.read_parquet(in_path)

if "source" not in events.columns:
    raise ValueError("events.parquet needs 'source' column. Re-run make_parquets.")

rr = events[events["source"] == "RR"].copy()
rr["user_id"] = rr["user_id"] + RR_USER_OFFSET  # Avoid collision with HM user IDs

rr.to_parquet(out_path, index=False)
print(f"[events_rr] Wrote {len(rr):,} RR events to {out_path}")
