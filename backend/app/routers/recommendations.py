from typing import Any, Dict, List, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query

from ..auth import _get_client, get_optional_user
from ..models import User

router = APIRouter()


def _dot(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _fetch_events_for_user(user_id: UUID) -> Dict[str, int]:
    client = _get_client()
    # Only pull product_id to avoid issues if the event column is a vector or custom type.
    resp = client.table("user_events").select("product_id").eq("user_id", str(user_id)).execute()
    data = getattr(resp, "data", None) or []
    counts: Dict[str, int] = {}
    for row in data:
        pid = row.get("product_id")
        if pid:
            counts[str(pid)] = counts.get(str(pid), 0) + 1
    return counts


def _attach_scores(rows: List[Dict[str, Any]], user_events: Dict[str, int]) -> List[Dict[str, Any]]:
    scored = []
    for row in rows:
        prod = row.get("products") or {}
        pid = str(prod.get("id") or row.get("product_id"))
        score = 1.0
        score += user_events.get(pid, 0) * 0.5
        row["score"] = score
        row["product"] = prod
        scored.append(row)
    scored.sort(key=lambda r: r.get("score", 0), reverse=True)
    return scored


def _fallback_recommendations() -> List[Dict[str, Any]]:
    prod = {
        "id": "seed-product",
        "name": "Demo Jacket",
        "price": 120.0,
        "currency": "USD",
        "product_url": "https://demo.brand.test/products/demo-item",    # TODO: replace with real URL
        "image_url": "https://demo.brand.test/products/demo-item.jpg",  # TODO: replace with real URL
        "brand_name": "Threaded Demo Brand",
        "category": "outerwear",
    }
    return [{"score": 1.0, "product": prod}]


@router.get("/recommendations")
def recommendations(
    limit: int = Query(default=10, ge=1, le=50),
    current_user: Optional[User] = Depends(get_optional_user),
):
    """
    Lightweight recommender: fetch embeddings, apply simple scoring, fallback to fixture.
    """
    client = _get_client()
    resp = (
        client.table("product_embeddings")
        .select("product_id,text_embedding,image_embedding,products(*)")
        .limit(limit * 3)
        .execute()
    )
    data = getattr(resp, "data", None) or []
    if not data:
        return _fallback_recommendations()

    events = _fetch_events_for_user(current_user.id) if current_user and current_user.id else {}
    scored = _attach_scores(data, events)
    return scored[:limit]


@router.get("/similar/{product_id}")
def similar(
    product_id: UUID,
    limit: int = Query(default=5, ge=1, le=20),
):
    """
    Find similar products by cosine similarity on text embeddings.
    """
    client = _get_client()
    resp = (
        client.table("product_embeddings")
        .select("product_id,text_embedding,products(*)")
        .execute()
    )
    rows = getattr(resp, "data", None) or []
    base = next((r for r in rows if str(r.get("product_id")) == str(product_id)), None)
    if not base:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Product embedding not found")
    base_vec = base.get("text_embedding") or []
    if not base_vec:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Base product missing text embedding")

    results = []
    for row in rows:
        if str(row.get("product_id")) == str(product_id):
            continue
        vec = row.get("text_embedding") or []
        if not vec:
            continue
        score = _dot(base_vec, vec)
        row["score"] = score
        row["product"] = row.get("products") or {}
        results.append(row)
    results.sort(key=lambda r: r.get("score", 0), reverse=True)
    return results[:limit]
