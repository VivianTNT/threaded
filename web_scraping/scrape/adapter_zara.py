from bs4 import BeautifulSoup
from typing import Dict

def extract_zara(html: str) -> Dict:
    soup = BeautifulSoup(html, "lxml")
    # Zara often uses JSON-LD already; fall back to generic if needed.
    # Here we only override fields we know are better:
    name = soup.select_one("h1") or soup.select_one("[data-qa='product-name']")
    name = name.get_text(strip=True) if name else None
    img = soup.select_one("img[srcset], img[data-zoomimage], img[data-src]") or soup.select_one("img")
    image_url = img.get("data-zoomimage") or img.get("data-src") or img.get("src") if img else None
    # price
    price_node = soup.select_one("[itemprop='price'], .price__amount, .money-amount__main")
    price = None
    if price_node:
        v = price_node.get("content") or price_node.get_text(strip=True)
        try: price = float(v.replace(",", "").strip("USD$€£"))
        except Exception: pass
    return {"name": name, "image_url": image_url, "price": price}
