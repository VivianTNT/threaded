from urllib.parse import urlparse
from slugify import slugify
from .schema import Product
from typing import Dict

def make_id(url:str, name:str=None):
    dom = urlparse(url).netloc
    base = slugify((name or url)[0:80])
    return f"{dom}-{base}" if base else dom

def normalize(url:str, domain:str, raw:Dict) -> Product:
    # Ensure required fields
    name = raw.get("name") or "Unknown Product"
    prod = Product(
        id= make_id(url, name),
        url=url,
        domain=domain,
        brand = raw.get("brand"),
        name = name,
        description = raw.get("description"),
        category = raw.get("category"),
        price = raw.get("price"),
        currency = (raw.get("currency") or "USD") if raw.get("price") else None,
        image_url = raw.get("image_url"),
        in_stock = raw.get("in_stock")
    )
    return prod
