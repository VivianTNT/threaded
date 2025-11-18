from fastapi import FastAPI

from .config import get_settings
from .routers import auth, recommendations, users

settings = get_settings()
app = FastAPI(title="Threaded Backend", version="0.1.0")


@app.get("/health")
def healthcheck():
    return {"status": "ok", "env": settings.env}


app.include_router(auth.router, prefix="/auth", tags=["auth"])
app.include_router(users.router, prefix="/users", tags=["users"])
app.include_router(recommendations.router, tags=["recommendations"])
