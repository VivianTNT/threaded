import json
import re
from typing import Any, Dict, Optional
from urllib.parse import urlparse

from bs4 import BeautifulSoup


CURRENCY_SYMBOLS = {
    "$": "USD",
    "£": "GBP",
    "€": "EUR",
    "¥": "JPY",
    "C$": "CAD",
    "A$": "AUD",
}


def _safe_json_loads(text: str):
    try:
        return json.loads(text)
    except Exception:
        return None


def _pick_product_node(data: Any) -> Optional[Dict[str, Any]]:
    """
    Given parsed JSON-LD data (dict or list), try to find a node
    that looks like a schema.org Product.
    """
    if isinstance(data, dict):
        # Some sites use @graph
        if "@graph" in data:
            for node in data["@graph"]:
                if isinstance(node, dict):
                    t = node.get("@type")
                    if t == "Product" or (isinstance(t, list) and "Product" in t):
                        return node
        # Or root itself is Product
        t = data.get("@type")
        if t == "Product" or (isinstance(t, list) and "Product" in t):
            return data

    if isinstance(data, list):
        for node in data:
            if not isinstance(node, dict):
                continue
            t = node.get("@type")
            if t == "Product" or (isinstance(t, list) and "Product" in t):
                return node

    return None


def _extract_from_ld_json(soup: BeautifulSoup) -> Dict[str, Any]:
    """
    Try to extract name, description, image, price, currency from JSON-LD Product.
    """
    result: Dict[str, Any] = {
        "name": None,
        "description": None,
        "image_url": None,
        "price": None,
        "currency": None,
    }

    scripts = soup.find_all("script", type="application/ld+json")
    for script in scripts:
        if not script.string:
            continue

        data = _safe_json_loads(script.string)
        if not data:
            continue

        prod = _pick_product_node(data)
        if not prod:
            continue

        # Basic fields
        if not result["name"]:
            name = prod.get("name")
            if isinstance(name, str):
                result["name"] = name.strip()

        if not result["description"]:
            desc = prod.get("description")
            if isinstance(desc, str):
                result["description"] = desc.strip()

        if not result["image_url"]:
            img = prod.get("image")
            if isinstance(img, list) and img:
                img = img[0]
            if isinstance(img, str):
                result["image_url"] = img.strip()

        # Offers (price + currency)
        offers = prod.get("offers")
        if offers:
            if isinstance(offers, list) and offers:
                offers = offers[0]
            if isinstance(offers, dict):
                price = offers.get("price")
                currency = offers.get("priceCurrency")
                if result["price"] is None and isinstance(price, (str, int, float)):
                    try:
                        result["price"] = float(price)
                    except (ValueError, TypeError):
                        pass
                if result["currency"] is None and isinstance(currency, str):
                    result["currency"] = currency.strip()

        # If we already filled the main stuff, we can stop early
        if (
            result["name"]
            and result["description"]
            and result["image_url"]
            and result["price"] is not None
        ):
            break

    return result


def _maybe_infer_currency_from_text(text: str) -> Optional[str]:
    for sym, cur in CURRENCY_SYMBOLS.items():
        if sym in text:
            return cur
    return None


def extract_generic(html: str, url: Optional[str] = None) -> Dict[str, Any]:
    soup = BeautifulSoup(html, "lxml")

    # ------------------------------------------------------------------ #
    # 1. First try JSON-LD Product
    # ------------------------------------------------------------------ #
    ld = _extract_from_ld_json(soup)
    name = ld["name"]
    description = ld["description"]
    image_url = ld["image_url"]
    price = ld["price"]
    currency = ld["currency"]

    # ------------------------------------------------------------------ #
    # 2. Open Graph + meta as next tier
    # ------------------------------------------------------------------ #
    # NAME
    if not name:
        og_title = soup.select_one("meta[property='og:title']")
        if og_title and og_title.get("content"):
            name = og_title["content"].strip()

    if not name:
        title_tag = soup.find("title")
        if title_tag and title_tag.string:
            name = title_tag.get_text(strip=True)

    # DESCRIPTION
    if not description:
        meta_desc = soup.select_one("meta[name='description']")
        if meta_desc and meta_desc.get("content"):
            description = meta_desc["content"].strip()
        else:
            og_desc = soup.select_one("meta[property='og:description']")
            if og_desc and og_desc.get("content"):
                description = og_desc["content"].strip()

    # IMAGE
    if not image_url:
        og_img = soup.select_one("meta[property='og:image'], meta[property='og:image:secure_url']")
        if og_img and og_img.get("content"):
            image_url = og_img["content"].strip()

    # ------------------------------------------------------------------ #
    # 3. HTML fallbacks for image & price
    # ------------------------------------------------------------------ #
    # IMAGE fallback: choose a "likely" product image (very heuristic)
    if not image_url:
        candidates = soup.find_all("img")
        best = None
        best_score = -1

        for img in candidates:
            src = img.get("data-src") or img.get("src")
            if not src:
                continue

            # crude heuristics: avoid sprites/icons, prefer bigger images
            src_lower = src.lower()
            if any(bad in src_lower for bad in ["sprite", "icon", "logo", "placeholder"]):
                continue

            width = img.get("width")
            height = img.get("height")
            try:
                w = int(width) if width else 0
                h = int(height) if height else 0
            except ValueError:
                w = h = 0

            score = w * h
            if score <= 0:
                score = 1  # at least count it

            if score > best_score:
                best_score = score
                best = src

        if best:
            image_url = best

    # PRICE fallback
    if price is None:
        price_selectors = [
            "[itemprop='price']",
            "[data-price]",
            ".price",
            ".product-price",
            ".current-price",
            ".sales",
        ]
        price_node = None
        for sel in price_selectors:
            price_node = soup.select_one(sel)
            if price_node:
                break

        if price_node:
            v = (
                price_node.get("content")
                or price_node.get("data-price")
                or price_node.get_text(strip=True)
            )
            if v:
                # Remove non-numeric (except ., ,)
                v_clean = re.sub(r"[^0-9.,]", "", v)
                v_clean = v_clean.replace(",", "")
                try:
                    price = float(v_clean)
                except ValueError:
                    price = None

            # Infer currency from full text if missing
            if not currency:
                full_text = price_node.get_text(" ", strip=True)
                currency = _maybe_infer_currency_from_text(full_text)

    # ------------------------------------------------------------------ #
    # 4. Last-resort currency guess based on URL / TLD
    # ------------------------------------------------------------------ #
    if not currency and url:
        host = urlparse(url).hostname or ""
        if host.endswith(".co.uk") or host.endswith(".uk"):
            currency = "GBP"
        elif host.endswith(".de") or host.endswith(".fr") or host.endswith(".eu"):
            currency = "EUR"
        elif host.endswith(".jp"):
            currency = "JPY"
        elif ".com" in host:
            currency = "USD"  # extremely rough default

    return {
        "name": name,
        "description": description,
        "image_url": image_url,
        "price": price,
        "currency": currency,
    }
