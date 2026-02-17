import json
import re
from typing import Dict, Any, Optional
from bs4 import BeautifulSoup

def _safe_json_loads(text: str):
    """Best-effort JSON loader that ignores errors."""
    try:
        return json.loads(text)
    except Exception:
        return None


def _extract_from_ld_json(soup: BeautifulSoup) -> tuple[Optional[float], Optional[str]]:
    """
    Try to extract price & currency from JSON-LD script blocks.

    Many e-commerce sites embed a Product object like:
      {
        "@type": "Product",
        "offers": {
          "@type": "Offer",
          "price": "39.90",
          "priceCurrency": "USD"
        }
      }
    """
    scripts = soup.find_all("script", type="application/ld+json")
    for script in scripts:
        if not script.string:
            continue
        data = _safe_json_loads(script.string)
        if not data:
            continue

        # Normalize to a list
        if isinstance(data, dict):
            candidates = [data]
        elif isinstance(data, list):
            candidates = data
        else:
            continue

        for node in candidates:
            if not isinstance(node, dict):
                continue
            # Look for a Product object
            if node.get("@type") == "Product":
                offers = node.get("offers") or {}
                # offers could be a list or a dict
                if isinstance(offers, list) and offers:
                    offers = offers[0]
                if isinstance(offers, dict):
                    price_str = offers.get("price")
                    currency = offers.get("priceCurrency")
                    price = None
                    if isinstance(price_str, str):
                        try:
                            price = float(price_str)
                        except ValueError:
                            pass
                    return price, currency
    return None, None


def extract_uniqlo(html: str) -> Dict[str, Any]:
    soup = BeautifulSoup(html, "lxml")

    # --- NAME ---------------------------------------------------------------
    # Prefer og:title ("Barrel Jeans | UNIQLO US") and strip brand suffix.
    name = None
    og_title = soup.select_one("meta[property='og:title']")
    if og_title and og_title.get("content"):
        raw = og_title["content"].strip()
        # Strip things like " | UNIQLO US"
        name = raw.split("|")[0].strip()
    else:
        # Fallback: <title> tag
        title_tag = soup.find("title")
        if title_tag:
            raw = title_tag.get_text(strip=True)
            name = raw.split("|")[0].strip()

    # As a last resort, use the old behavior (this is what gave you "Outerwear")
    if not name:
        header = soup.select_one("h1, .product-name")
        if header:
            name = header.get_text(strip=True)

    # --- DESCRIPTION --------------------------------------------------------
    description = None
    meta_desc = soup.select_one("meta[name='description']")
    if meta_desc and meta_desc.get("content"):
        description = meta_desc["content"].strip()
    else:
        og_desc = soup.select_one("meta[property='og:description']")
        if og_desc and og_desc.get("content"):
            description = og_desc["content"].strip()

    # --- IMAGE --------------------------------------------------------------
    # For Uniqlo, og:image is the cleanest way to get the main PDP image.
    image_url = None
    og_img = soup.select_one("meta[property='og:image']")
    if og_img and og_img.get("content"):
        image_url = og_img["content"].strip()

    # Fallback: older behavior, but guarded properly
    if not image_url:
        img = soup.select_one("img[data-media-type='image'], img[data-src], img[src]")
        if img:
            image_url = img.get("data-src") or img.get("src")

    # --- PRICE & CURRENCY ---------------------------------------------------
    price: Optional[float] = None
    currency: Optional[str] = None

    # 1) Try JSON-LD Product offers
    price, currency = _extract_from_ld_json(soup)

    # 2) Fallback: generic price node in HTML (if Uniqlo ever exposes it there)
    if price is None:
        price_node = soup.select_one("[itemprop='price'], .price, .product-price")
        if price_node:
            v = price_node.get("content") or price_node.get_text(strip=True)
            if v:
                # Strip common currency symbols and thousands separators
                v_clean = re.sub(r"[^\d.,]", "", v)
                v_clean = v_clean.replace(",", "")
                try:
                    price = float(v_clean)
                except ValueError:
                    pass

    # 3) Fallback for currency: if we are on the US site, assume USD
    if currency is None:
        # You can pass in the URL separately, but if not, we try to infer from the HTML.
        og_url = soup.select_one("meta[property='og:url']")
        if og_url and "uniqlo.com/us/" in og_url.get("content", ""):
            currency = "USD"

    return {
        "name": name,
        "description": description,
        "image_url": image_url,
        "price": price,
        "currency": currency,
    }
