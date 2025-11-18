# Backend (FastAPI + Supabase)

## Setup
1. Create a virtual env and install deps.
2. Update `.env` with `SUPABASE_URL` and `SUPABASE_SERVICE_KEY`.

## Run the API
```bash
uvicorn app.main:app --reload
```

## Seed dummy data
```bash
python seed.py
```

## Smoke test routes
- `GET /health` – service status
 - `GET /users?limit=10` – list users from Supabase
 - `POST /users/test-create` with body `{"handle": "demo_user"}` – create/upsert a test user
- `POST /auth/register` with `{"email": "user@example.com", "password": "secret", "handle": "user"}` – create auth user/profile and return Supabase access token
- `POST /auth/login` – obtain Supabase access/refresh tokens
- `GET /auth/session` – validate Supabase access token (Authorization: Bearer <access_token>)
- `GET /recommendations` – returns scored products (falls back to demo data if embeddings empty)
- `GET /similar/{product_id}` – naive text similarity using stored embeddings
