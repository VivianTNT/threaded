from datetime import datetime, timezone
from typing import List, Optional
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Query, status
from supabase import Client

from ..models import User
from ..supabase_client import get_supabase

router = APIRouter()


def _get_client() -> Client:
    try:
        return get_supabase()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Supabase client initialization failed",
        ) from exc


@router.get("/", response_model=List[User])
def list_users(limit: int = Query(default=20, ge=1, le=100)):
    client = _get_client()
    resp = client.table("users").select("*").limit(limit).execute()
    data = getattr(resp, "data", None)
    error = getattr(resp, "error", None)
    if error:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Supabase error: {error}",
        )
    return data or []


class TestUserCreate(User):
    handle: str = "demo_user"
    metadata: Optional[dict] = {"source": "test-create"}
    created_at: Optional[datetime] = None


@router.post("/test-create", response_model=User, status_code=status.HTTP_201_CREATED)
def create_test_user(request: TestUserCreate):
    """
    Creates a user row for smoke testing before auth is wired up.
    """
    client = _get_client()
    record = request.model_dump()
    record["id"] = record.get("id") or str(uuid4())
    record["created_at"] = record.get("created_at") or datetime.now(timezone.utc).isoformat()
    resp = client.table("users").upsert(record, on_conflict="handle").execute()
    data = getattr(resp, "data", None)
    error = getattr(resp, "error", None)
    if error:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Supabase error: {error}",
        )
    if not data:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create test user",
        )
    return data[0]
