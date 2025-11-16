# scrape/upsert_product.py
from typing import Any, Dict
from .supabase_client import supabase
import logging
import json

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def upsert_product(product: Dict[str, Any]):
    """
    Insert or update a row in 'products' based on product_url.
    """

    if not product.get("product_url"):
        raise ValueError("product_url is required for upsert_product")

    def _as_str_or_none(val):
        return str(val) if val is not None else None

    domain = product.get("domain")
    # ensure domain is JSON-serializable
    if not isinstance(domain, (dict, list, str, int, float, bool, type(None))):
        domain = str(domain)

    record: Dict[str, Any] = {
        "product_url": _as_str_or_none(product.get("product_url")),
        "image_url": _as_str_or_none(product.get("image_url")),
        "name": product.get("name"),
        "price": product.get("price"),
        "currency": product.get("currency"),
        "domain": domain,
        "brand_name": product.get("brand_name"),
        "category": product.get("category"),
        "description": product.get("description"),

        "brand_id": None,   # Override default for now so no random UUID
    }

    # Log exactly what we're about to send
    logger.info("Upserting product record:\n%s", json.dumps(record, indent=2, default=str))

    try:
        response = (
            supabase
            .table("products")
            .upsert(record, on_conflict="product_url")
            .execute()
        )
    except Exception as e:
        logger.exception("Supabase upsert() raised an exception")
        return None

    # Supabase Python client v2 returns an object with .data and .error
    data = getattr(response, "data", None)
    error = getattr(response, "error", None)

    logger.info("Supabase raw response: %r", response)
    logger.info("Supabase response.data: %s", json.dumps(data, indent=2, default=str) if data else "None")

    if error:
        logger.error("Supabase response.error: %s", error)

    print(f"Upserted product: {record['product_url']} -> {record.get('name')}")
    return data
