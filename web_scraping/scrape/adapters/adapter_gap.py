# adapter_gap.py
from bs4 import BeautifulSoup
from typing import Dict, Optional
import json

def _first(txt):
    return txt[0] if isinstance(txt, list) and txt else txt

def _parse_price(v: Optional[str]) -> Optional[float]:
    if not v: return None
    v = v.replace(",", "").strip()
    # Try plain number, or strip currency symbols like $€
    for ch in "$€£USD":
        v = v.replace(ch, "")
    try:
        return float(v)
    except Exception:
        return None

def _extract_jsonld(soup: BeautifulSoup) -> Dict:
    # Find the first JSON-LD script that looks like a Product
    for tag in soup.select('script[type="application/ld+json"]'):
        try:
            data = json.loads(tag.string or "")
        except Exception:
            continue
        # Some pages embed a list/graph
        candidates = []
        if isinstance(data, dict):
            candidates = [data] + data.get("@graph", [])
        elif isinstance(data, list):
            candidates = data
        for node in candidates:
            t = node.get("@type")
            if t == "Product" or (isinstance(t, list) and "Product" in t):
                name   = _first(node.get("name"))
                img    = _first(node.get("image"))
                offers = node.get("offers") or {}
                if isinstance(offers, list):
                    offers = offers[0] if offers else {}
                price  = _parse_price(_first(offers.get("price")))
                return {"name": name, "image_url": img, "price": price}
    return {}

def extract_gap(html: str) -> Dict:
    """
    Works for gap.com, oldnavy.gap.com, bananarepublic.gap.com, athleta.gap.com
    Prefers JSON-LD; falls back to visible selectors / meta tags.
    """
    soup = BeautifulSoup(html, "lxml")

    data = _extract_jsonld(soup)

    # Fallbacks if JSON-LD missing/partial
    if not data.get("name"):
        title = soup.select_one("h1, .product-title, .pdp-title, [data-automation-id='pdp-name']")
        if title:
            data["name"] = title.get_text(strip=True)
    if not data.get("image_url"):
        og = soup.select_one("meta[property='og:image'], meta[name='og:image']")
        if og and og.get("content"):
            data["image_url"] = og["content"]
        else:
            img = soup.select_one("img#main-product-image, .pdp-primary-image img, img[src]")
            if img:
                data["image_url"] = img.get("src")
    if data.get("price") is None:
        # Try common price containers or meta
        price_node = (
            soup.select_one("[itemprop='price'], meta[itemprop='price'], .price, .product-price, [data-automation-id='pdp-price']")
            or soup.select_one("meta[property='product:price:amount']")
        )
        if price_node:
            val = price_node.get("content") or price_node.get_text(strip=True)
            data["price"] = _parse_price(val)

    return data
