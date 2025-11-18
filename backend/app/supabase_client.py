from functools import lru_cache

from supabase import Client, create_client

from .config import get_settings


@lru_cache()
def get_supabase() -> Client:
    settings = get_settings()
    url = str(settings.supabase_url)
    key = settings.supabase_key

    if "your-project.supabase.co" in url:
        raise RuntimeError(
            "SUPABASE_URL in .env is still the placeholder (your-project). "
            "Fill in your real project URL and service key."
        )
    return create_client(url, key)
