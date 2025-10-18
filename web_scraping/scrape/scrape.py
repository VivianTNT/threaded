import asyncio, orjson, tldextract, sys
from pathlib import Path
from bs4 import BeautifulSoup
from .robots import allowed
from .fetcher import fetch_html
from .cache import append_jsonl
from .parser_generic import extract_generic
from .normalizer import normalize
from . import adapter_zara, adapter_uniqlo

OUT = "items.jsonl"

ADAPTERS = {
    "zara.com": adapter_zara.extract_zara,
    "uniqlo.com": adapter_uniqlo.extract_uniqlo,
    # add more domains here...
}

def pick_adapter(domain:str):
    for d, fn in ADAPTERS.items():
        if domain.endswith(d): return fn
    return None

async def process(url: str):
    if not allowed(url): 
        print(f"[SKIP robots] {url}"); return None
    html = await fetch_html(url)
    domain = tldextract.extract(url).registered_domain
    adapter = pick_adapter(domain)
    raw = extract_generic(html)
    if adapter:
        raw_specific = adapter(html)
        raw.update({k:v for k,v in raw_specific.items() if v})
    prod = normalize(url, domain, raw)
    append_jsonl(OUT, orjson.loads(prod.model_dump_json()))
    return prod

async def main(urls_file="urls.txt"):
    urls = [u.strip() for u in Path(urls_file).read_text().splitlines() if u.strip()]
    for u in urls:
        try:
            p = await process(u)
            if p: print("OK:", p.id, "->", p.name, p.price)
        except Exception as e:
            print("ERR:", u, e)

if __name__ == "__main__":
    asyncio.run(main(sys.argv[1] if len(sys.argv)>1 else "urls.txt"))
