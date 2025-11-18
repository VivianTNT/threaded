import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field, HttpUrl, ValidationError


class Settings(BaseModel):
    supabase_url: HttpUrl = Field(alias="SUPABASE_URL")
    supabase_key: str = Field(alias="SUPABASE_SERVICE_KEY")
    env: str = Field(default="local", alias="APP_ENV")


def _load_dotenv():
    # Load from repo root if present, otherwise rely on environment variables.
    root_env = Path(__file__).resolve().parents[2] / ".env"
    if root_env.exists():
        load_dotenv(root_env)
    else:
        load_dotenv()


@lru_cache()
def get_settings() -> Settings:
    _load_dotenv()
    try:
        return Settings(**os.environ)
    except ValidationError as exc:
        missing = [e["loc"][0] for e in exc.errors() if e["type"] == "missing"]
        detail = f"Missing required environment variables: {', '.join(missing)}"
        raise RuntimeError(detail) from exc
