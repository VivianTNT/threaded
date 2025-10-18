from pydantic import BaseModel, HttpUrl, Field
from typing import Optional, List
from datetime import datetime

class Product(BaseModel):
    id: str                      # site-specific stable key or hashed URL
    url: HttpUrl
    domain: str                  # e.g., "zara.com"
    brand: Optional[str] = None
    name: str
    description: Optional[str] = None
    category: Optional[str] = None
    price: Optional[float] = None
    currency: Optional[str] = None
    image_url: Optional[HttpUrl] = None
    in_stock: Optional[bool] = None
    ts_crawled: datetime = Field(default_factory=datetime.utcnow)
    source: str = "scrape"       # or "snapshot"
