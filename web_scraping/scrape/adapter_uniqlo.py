from bs4 import BeautifulSoup
from typing import Dict

def extract_uniqlo(html: str) -> Dict:
    soup = BeautifulSoup(html, "lxml")
    name = soup.select_one("h1, .product-name")
    name = name.get_text(strip=True) if name else None
    img = soup.select_one("img[data-src], img[src]")
    image_url = img.get("data-src") or img.get("src") if img else None
    price_node = soup.select_one("[itemprop='price'], .price, .product-price")
    price = None
    if price_node:
        v = price_node.get("content") or price_node.get_text(strip=True)
        try: price = float(v.replace(",", "").strip("USD$€£"))
        except Exception: pass
    return {"name": name, "image_url": image_url, "price": price}
