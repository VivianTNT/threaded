from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr

from ..auth import get_current_user, _get_client
from ..models import User

router = APIRouter()


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str
    handle: Optional[str] = None


class AuthResponse(BaseModel):
    access_token: str
    refresh_token: str | None = None
    token_type: str = "bearer"
    user: User


@router.post("/register", response_model=AuthResponse, status_code=status.HTTP_201_CREATED)
def register(payload: RegisterRequest):
    """
    Create a Supabase Auth user and mirror a profile row in `users`.
    """
    client = _get_client()
    # Create auth user (admin ensures immediate confirmation)
    auth_user = client.auth.admin.create_user(
        {
            "email": payload.email,
            "password": payload.password,
            "email_confirm": True,
        }
    )
    auth_id = auth_user.user.id  # type: ignore[attr-defined]
    handle = payload.handle or payload.email.split("@")[0]

    profile = {"id": auth_id, "handle": handle}
    profile_resp = client.table("users").upsert(profile, on_conflict="id").execute()
    error = getattr(profile_resp, "error", None)
    if error:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=f"Supabase error: {error}")
    user = User.model_validate(getattr(profile_resp, "data", [profile])[0])
    # Create a session for immediate use
    session = client.auth.sign_in_with_password({"email": payload.email, "password": payload.password})
    return AuthResponse(
        access_token=session.session.access_token,  # type: ignore[attr-defined]
        refresh_token=session.session.refresh_token,  # type: ignore[attr-defined]
        user=user,
    )


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


@router.post("/login", response_model=AuthResponse)
def login(payload: LoginRequest):
    client = _get_client()
    session = client.auth.sign_in_with_password({"email": payload.email, "password": payload.password})
    auth_user = session.user  # type: ignore[attr-defined]
    if auth_user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")
    # Fetch profile; create placeholder handle if none.
    resp = client.table("users").select("*").eq("id", auth_user.id).limit(1).execute()
    data = getattr(resp, "data", None) or []
    if not data:
        handle = payload.email.split("@")[0]
        client.table("users").upsert({"id": auth_user.id, "handle": handle}).execute()
        profile = {"id": auth_user.id, "handle": handle}
    else:
        profile = data[0]
    return AuthResponse(
        access_token=session.session.access_token,  # type: ignore[attr-defined]
        refresh_token=session.session.refresh_token,  # type: ignore[attr-defined]
        user=User.model_validate(profile),
    )


@router.get("/session", response_model=User)
def session(current_user: User = Depends(get_current_user)):
    return current_user
