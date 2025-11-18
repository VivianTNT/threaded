from typing import Optional

from fastapi import Header, HTTPException, status
from supabase import Client

from .supabase_client import get_supabase


def _get_client() -> Client:
    try:
        return get_supabase()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Supabase client initialization failed",
        ) from exc


def get_current_user(authorization: Optional[str] = Header(default=None)) -> dict:
    """
    Validate Supabase access token (Bearer) and return the auth user object.
    """
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing bearer token")
    token = authorization.split(" ", 1)[1]
    client = _get_client()
    try:
        user = client.auth.get_user(token).user
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token") from exc
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
    return user


def get_optional_user(authorization: Optional[str] = Header(default=None)) -> Optional[dict]:
    if not authorization:
        return None
    try:
        return get_current_user(authorization=authorization)
    except HTTPException:
        return None
