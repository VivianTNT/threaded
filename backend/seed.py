#!/usr/bin/env python
"""
Seed script to create dummy users/products/events for local smoke tests.
"""
from __future__ import annotations

import argparse
import sys
import datetime as dt
from typing import Dict
from uuid import UUID, uuid5, NAMESPACE_DNS

from app.supabase_client import get_supabase


def _stable_uuid(name: str) -> UUID:
    return uuid5(NAMESPACE_DNS, f"threaded-seed-{name}")


def seed():
    client = get_supabase()
    user_id = _stable_uuid("user")
    product_id = _stable_uuid("product")

    product_url = "https://demo.brand.test/products/demo-item"
    brand = {
        # Schema uses product_url as PK; needs to match the products.product_url FK.
        "product_url": product_url,
        "name": "Threaded Demo Brand",
        "metadata": {"source": "seed"},
    }
    user = {"id": str(user_id), "handle": "demo_user", "metadata": {"source": "seed"}}
    product = {
        "id": str(product_id),
        "product_url": product_url,
        "image_url": "https://demo.brand.test/products/demo-item.jpg",
        "name": "Demo Jacket",
        "price": 120.0,
        "currency": "USD",
        "category": "outerwear",
        "description": "Lightweight demo jacket for seeding.",
        "domain": {"domain": "demo.brand.test"},
        "brand_name": brand["name"],
    }
    embeddings = [{
        "id": str(_stable_uuid("embedding")),
        "product_id": str(product_id),
        "text_embedding": [0.1, 0.2, 0.3, 0.4],
        "image_embedding": [0.4, 0.3, 0.2, 0.1],
    }]

    # Upsert core tables
    client.table("brands").upsert(brand, on_conflict="product_url").execute()
    client.table("users").upsert(user, on_conflict="id").execute()
    client.table("products").upsert(product, on_conflict="id").execute()

    for emb in embeddings:
        # Use the embedding PK (id) for upsert; product_id may not be unique in schema.
        client.table("product_embeddings").upsert(emb, on_conflict="id").execute()

    print("Seeded demo brand/user/product/events")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Seed dummy data into Supabase tables.")
    parser.parse_args()
    try:
        seed()
    except Exception as exc:
        print(
            "Seed failed:",
            exc,
            "\nCommon fixes:",
            "\n- Ensure .env has real SUPABASE_URL and SUPABASE_SERVICE_KEY (not placeholders)."
            "\n- Verify network access to Supabase.",
            file=sys.stderr,
        )
        sys.exit(1)
