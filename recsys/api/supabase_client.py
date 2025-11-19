from supabase import create_client, Client

SUPABASE_URL = "YOUR_URL"
SUPABASE_KEY = "SERVICE_ROLE_KEY"

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def update_user_embedding(user_id: str, embedding):
    return supabase.table("user_profile").upsert({
        "id": user_id,
        "embedding": embedding
    }).execute()