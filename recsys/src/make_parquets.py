import pandas as pd
from pathlib import Path
from recsys.src.hm_loader import load_hm_items, load_hm_events   # ← import new helper

RAW = Path("recsys/data/rr_raw")
OUT = Path("recsys/data")
OUT.mkdir(parents=True, exist_ok=True)

SMALL_TEST = False
NROWS = 300_000 if SMALL_TEST else None
CHUNK = 250_000

BASE_PROPS = {"name", "title", "brand", "categoryid", "price", "description"}
TOP_K_FREQ_PROPS = 30


# ---------------- RETAILROCKET LOADING ---------------- #

def load_category_map():
    c = pd.read_csv(RAW / "category_tree.csv")
    c.columns = [str(col).lower() for col in c.columns]
    if not {"categoryid", "parentid"}.issubset(set(c.columns)):
        raise ValueError(f"category_tree.csv must contain categoryid,parentid; found {list(c.columns)}")
    name_col = "categoryname" if "categoryname" in c.columns else None

    c["categoryid"] = c["categoryid"].astype("int64")
    c = c.set_index("categoryid")
    parent = c["parentid"].to_dict()
    names  = c[name_col].to_dict() if name_col else {}

    def label_for(cid: int) -> str:
        nm = names.get(cid)
        return nm.strip() if isinstance(nm, str) and nm.strip() else f"cat:{cid}"

    def path(cid):
        out = []
        while pd.notna(cid) and int(cid) in parent:
            cid = int(cid)
            out.append(label_for(cid))
            cid = parent.get(cid)
        return " > ".join(reversed(out)) if out else None

    return {int(cid): path(cid) for cid in c.index}


def load_rr_events():
    usecols = ["timestamp", "visitorid", "event", "itemid"]
    df = pd.read_csv(RAW / "events.csv", usecols=usecols, nrows=NROWS, low_memory=False)
    df = df.rename(columns={"visitorid": "user_id", "itemid": "item_id"})
    keep = {"view": "view", "addtocart": "cart", "transaction": "purchase"}
    df = df[df["event"].isin(keep)].copy()
    df["event"] = df["event"].map(keep)
    df["user_id"] = df["user_id"].astype("int64")
    df["item_id"] = df["item_id"].astype("int64")
    df["timestamp"] = df["timestamp"].astype("int64")
    print(f"[RetailRocket] events: {len(df):,}")
    return df[["user_id", "item_id", "event", "timestamp"]]


def top_frequent_properties(nrows=None, chunksize=CHUNK):
    vc = pd.Series(dtype="int64")
    for fname in ["item_properties_part1.csv", "item_properties_part2.csv"]:
        print(f"[props] counting in {fname} ...")
        for chunk in pd.read_csv(RAW / fname, usecols=["property"], nrows=nrows,
                                 chunksize=chunksize, low_memory=False):
            vc = vc.add(chunk["property"].value_counts(), fill_value=0).astype("int64")
    vc = vc.sort_values(ascending=False)
    extras = [p for p in vc.index.tolist() if p not in BASE_PROPS][:TOP_K_FREQ_PROPS]
    keep_set = set(BASE_PROPS) | set(extras)
    print(f"[props] keeping {len(keep_set)} props = BASE {len(BASE_PROPS)} + TOP {len(extras)}")
    return keep_set


def load_rr_items(keep_item_ids: set, props_keep: set):
    usecols = ["itemid", "timestamp", "property", "value"]
    latest = {}

    def maybe_update(iid, prop, ts, val):
        if prop not in props_keep or iid not in keep_item_ids:
            return
        key = (iid, prop)
        prev = latest.get(key)
        if (prev is None) or (ts >= prev[0]):
            latest[key] = (ts, val)

    for fname in ["item_properties_part1.csv", "item_properties_part2.csv"]:
        print(f"[RetailRocket] reading {fname} ...")
        for chunk in pd.read_csv(RAW / fname, usecols=usecols, chunksize=CHUNK, low_memory=False):
            if SMALL_TEST and len(latest) > 150_000:
                break
            chunk["itemid"] = chunk["itemid"].astype("int64")
            chunk["timestamp"] = chunk["timestamp"].astype("int64")
            for row in chunk.itertuples(index=False):
                maybe_update(row.itemid, row.property, row.timestamp, row.value)
        print(f"[RetailRocket] collected pairs: {len(latest):,}")

    recs = [(iid, prop, val) for (iid, prop), (_, val) in latest.items()]
    props = pd.DataFrame(recs, columns=["item_id", "property", "value"])
    items = props.pivot(index="item_id", columns="property", values="value").reset_index()

    cat_map = load_category_map()

    def _to_int_cat(x):
        if pd.isna(x): return None
        try: return int(float(x))
        except Exception: return None

    if "categoryid" in items.columns:
        items["category_path"] = items["categoryid"].apply(lambda v: cat_map.get(_to_int_cat(v)))
    else:
        items["category_path"] = None

    def mk_title(row):
        for k in ("name", "title", "brand"):
            if k in row.index:
                v = row[k]
                if isinstance(v, str) and v.strip():
                    return v.strip()
        cp = row.get("category_path")
        if isinstance(cp, str) and cp.strip():
            return f"Item {int(row['item_id'])} ({cp})"
        return f"Item {int(row['item_id'])}"

    items["title"] = items.apply(mk_title, axis=1).astype(str)

    desc_cols = [c for c in items.columns if c not in {"item_id", "title"}]
    items["description"] = items[desc_cols].apply(
        lambda r: " ".join(f"{c}:{str(r[c])}" for c in desc_cols if pd.notna(r[c]))[:500],
        axis=1,
    )

    desired = ["item_id", "title", "description", "category_path", "brand"]
    keep_cols = [c for c in desired if c in items.columns]
    items = items[keep_cols]
    items["image_path_or_url"] = ""
    items["item_id"] = items["item_id"].astype("int64")
    print(f"[RetailRocket] items: {len(items):,}")
    return items


# ---------------- MERGE WITH H&M ---------------- #

def merge_datasets(rr_items, rr_events, hm_items, hm_events):
    print("[merge] Combining RetailRocket + H&M datasets ...")
    items = pd.concat([rr_items, hm_items], ignore_index=True)
    events = pd.concat([rr_events, hm_events], ignore_index=True)
    print(f"[merge] Combined items: {len(items):,}, events: {len(events):,}")
    return items, events


# ---------------- MAIN ---------------- #

if __name__ == "__main__":
    print("[make_parquets] start")

    # 1️⃣ RetailRocket
    rr_events = load_rr_events()
    keep_ids = set(rr_events["item_id"].unique())
    props_keep = top_frequent_properties(nrows=None)
    rr_items = load_rr_items(keep_ids, props_keep)

    # 2️⃣ H&M
    hm_items = load_hm_items()
    hm_events = load_hm_events()
    
    rr_items["source"] = "RR"
    hm_items["source"] = "HM"
    rr_events["source"] = "RR"
    hm_events["source"] = "HM"
    # 3️⃣ Merge and Save
    items, events = merge_datasets(rr_items, rr_events, hm_items, hm_events)
    events.to_parquet(OUT / "events.parquet", index=False)
    items.to_parquet(OUT / "items.parquet", index=False)

    print("[make_parquets] wrote:", OUT / "events.parquet", "and", OUT / "items.parquet")
    print(events.head(3))
    print(items.head(3))