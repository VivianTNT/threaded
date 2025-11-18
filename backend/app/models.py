from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from pydantic import BaseModel, HttpUrl


# --- public.brands ---
# PK is product_url (text); brand name lives here; optional metadata jsonb.
class Brand(BaseModel):
    product_url: HttpUrl  # PRIMARY KEY
    name: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    created_at: Optional[datetime] = None


# --- public.products ---
# FK (product_url) -> brands(product_url)
class Product(BaseModel):
    id: Optional[UUID] = None  # PRIMARY KEY
    product_url: Optional[HttpUrl] = None  # FK to brands.product_url
    image_url: Optional[HttpUrl] = None
    name: Optional[str] = None
    price: Optional[float] = None 
    currency: Optional[str] = None
    category: Optional[str] = None
    description: Optional[str] = None
    domain: Optional[Dict[str, Any]] = None  # jsonb
    brand_name: Optional[str] = None 
    created_at: Optional[datetime] = None


# --- public.product_embeddings ---
# PK id; FK product_id -> products(id); vector columns for image/text.
class ProductEmbedding(BaseModel):
    id: Optional[UUID] = None  # PRIMARY KEY
    product_id: Optional[UUID] = None  # FK to products.id
    image_embedding: Optional[List[float]] = None
    text_embedding: Optional[List[float]] = None 


# --- public.users ---
# Handle is UNIQUE with a default.
# Treat uploaded_images as a list of strings for now because is USER-DEFINED in DB.
class User(BaseModel):
    id: Optional[UUID] = None  # PRIMARY KEY
    handle: str = "new_user"
    metadata: Optional[Dict[str, Any]] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    uploaded_images: Optional[List[str]] = None  # adjust type if needed


# --- public.user_events ---
# PK id; FK user_id -> users(id); FK product_id -> products(id)
# Column names: event, context, created_at.
class UserEvent(BaseModel):
    id: Optional[UUID] = None  # PRIMARY KEY
    user_id: Optional[UUID] = None
    product_id: Optional[UUID] = None
    event: Optional[str] = None  # free-form event name; optional for now
    context: Optional[Dict[str, Any]] = None
    created_at: Optional[datetime] = None
