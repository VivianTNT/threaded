from bs4 import BeautifulSoup
import json, re
from typing import Optional, Dict

def parse_json_ld(soup: BeautifulSoup) -> Dict:
    out = {}
    for tag in soup.find_all("script", type=lambda t: t and "ld+json" in t):
        try:
            data = json.loads(tag.string or "{}")
            if isinstance(data, list):
                for d in data:
                    if d.get("@type") in ("Product","Offer","Organization"):
                        out.update(d)
            elif data.get("@type") in ("Product","Offer","Organization"):
                out.update(data)
        except Exception:
            continue
    return out

def extract_generic(html: str) -> Dict:
    soup = BeautifulSoup(html, "lxml")
    ld = parse_json_ld(soup)

    # Fallbacks
    name = ld.get("name") or (soup.find("h1") or soup.title).get_text(strip=True)
    desc = ld.get("description") or (soup.find("meta", {"name":"description"}) or {}).get("content")
    img = None
    if isinstance(ld.get("image"), list): img = ld["image"][0]
    elif isinstance(ld.get("image"), str): img = ld["image"]
    price, currency = None, None
    offers = ld.get("offers") or {}
    if isinstance(offers, dict):
        price = offers.get("price") or offers.get("lowPrice")
        currency = offers.get("priceCurrency")
    else:
        try: price = offers[0].get("price"); currency = offers[0].get("priceCurrency")
        except Exception: pass

    # Try common price selectors if missing
    if not price:
        for sel in ["[data-test*='price']", ".price", ".product-price", "[itemprop='price']"]:
            n = soup.select_one(sel)
            if n:
                text = n.get("content") or n.get_text(" ", strip=True)
                m = re.search(r"([0-9]+[.,][0-9]+|[0-9]+)", text)
                if m: price = float(m.group(1).replace(",", "")); break

    return {
        "name": name,
        "description": desc,
        "image_url": img,
        "price": float(price) if price else None,
        "currency": currency,
        "category": ld.get("category") or ld.get("aggregateRating",{}).get("itemReviewed"),
        "brand": (ld.get("brand") or {}).get("name") if isinstance(ld.get("brand"), dict) else ld.get("brand"),
    }
