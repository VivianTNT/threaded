# robots.py
import urllib.robotparser as urp
import requests
from urllib.parse import urlsplit

UA = "RetailScrapePOC/1.0"
_cache = {}

def allowed(url: str, timeout=5) -> bool:
    parts = urlsplit(url)
    base = f"{parts.scheme}://{parts.netloc}"
    rp = _cache.get(base)
    if not rp:
        rp = urp.RobotFileParser()
        robots_url = f"{base}/robots.txt"
        try:
            r = requests.get(robots_url, timeout=timeout)
            r.raise_for_status()
            rp.parse(r.text.splitlines())
        except Exception:
            # If robots fetch fails, be conservative or permissive; choose one.
            # To avoid hangs, default to True here:
            rp.parse([])  # empty rules => allow all
        _cache[base] = rp
    return rp.can_fetch(UA, url)
