# get_prod_urls.py
from typing import List, Optional

from .supabase_client import supabase


def fetch_brand_urls(limit: Optional[int] = None) -> List[str]:
    """
    Fetch product_url values from the `brands` table in Supabase.

    Only returns non-empty strings.
    """
    print("[SUPABASE] Fetching product_url list from brands table...")

    query = supabase.table("brands").select("product_url")
    if limit is not None:
        query = query.limit(limit)

    resp = query.execute()

    # supabase-py v2: resp.data; older versions: resp.get("data")
    data = getattr(resp, "data", None) or resp.get("data", [])

    urls: list[str] = []
    for row in data:
        url = (row.get("product_url") or "").strip()
        if url:
            urls.append(url)

    print(f"[SUPABASE] Retrieved {len(urls)} URLs from brands table.")
    return urls
