import pandas as pd
from pathlib import Path

RAW = Path("recsys/data/hm_raw")

def load_hm_items(nrows=None):
    """
    Converts H&M articles.csv → items DataFrame
    columns: item_id, title, description, category_path, brand, image_path_or_url
    """
    articles = pd.read_csv(RAW / "articles.csv", nrows=nrows, low_memory=False)
    # Create readable title and description
    articles["title"] = (
        articles["product_type_name"].fillna("") + " - " +
        articles["colour_group_name"].fillna("")
    ).str.strip()

    articles["description"] = (
        articles["product_group_name"].fillna("") + " " +
        articles["detail_desc"].fillna("")
    ).str.strip()

    articles["category_path"] = articles["index_group_name"] + " > " + articles["section_name"]
    articles["brand"] = "H&M"
    articles["item_id"] = articles["article_id"].astype(str).apply(lambda x: int("2" + x))  # offset IDs to avoid collision
    articles["image_path_or_url"] = (
        "https://www.kaggleusercontent.com/competitions/25749/images/" +
        articles["article_id"].astype(str) + ".jpg"
    )

    items = articles[[
        "item_id", "title", "description", "category_path", "brand", "image_path_or_url"
    ]]
    print(f"[hm_loader] Loaded {len(items):,} H&M items")
    return items

def load_hm_events(nrows=None):
    """
    Converts transactions_train.csv → events DataFrame
    columns: user_id, item_id, event, timestamp
    """
    tx = pd.read_csv(RAW / "transactions_train.csv", nrows=nrows, low_memory=False)
    tx["user_id"] = tx["customer_id"].astype("category").cat.codes.astype("int64")
    tx["item_id"] = tx["article_id"].astype(str).apply(lambda x: int("2" + x))
    tx["timestamp"] = pd.to_datetime(tx["t_dat"]).astype("int64") // 10**6
    tx["event"] = "purchase"

    events = tx[["user_id", "item_id", "event", "timestamp"]]
    print(f"[hm_loader] Loaded {len(events):,} H&M events")
    return events