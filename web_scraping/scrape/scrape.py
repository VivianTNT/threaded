import asyncio, orjson, tldextract, sys
from pathlib import Path
from bs4 import BeautifulSoup
from .robots import allowed
from .fetcher import fetch_html, init_browser
from .cache import append_jsonl
from .normalizer import normalize
from .adapters import adapter_zara, adapter_uniqlo, adapter_gap, adapter_generic
from .upsert_product import upsert_product
from .get_prod_urls import fetch_brand_urls

OUT = "items.jsonl"

ADAPTERS = {
    "zara.com": adapter_zara.extract_zara,         # skipped by robots
    "uniqlo.com": adapter_uniqlo.extract_uniqlo,
    "gap.com": adapter_gap.extract_gap,
    "oldnavy.gap.com": adapter_gap.extract_gap,
    "bananarepublic.gap.com": adapter_gap.extract_gap,
    "athleta.gap.com": adapter_gap.extract_gap,
    # add more domains here...
}

def pick_adapter(domain:str):
    for d, fn in ADAPTERS.items():
        if domain.endswith(d): return fn
    return None

async def process(url: str, browser):
    if not allowed(url): 
        print(f"[SKIP robots] {url}")
        return None
    html = await fetch_html(url, browser)
    domain = tldextract.extract(url).registered_domain
    adapter = pick_adapter(domain)
    raw = adapter_generic.extract_generic(html, url=url)
    if adapter:
        raw_specific = adapter(html)
        raw.update({k:v for k,v in raw_specific.items() if v})
    prod = normalize(url, domain, raw)
    append_jsonl(OUT, orjson.loads(prod.model_dump_json()))

    product_dict = {
        "product_url": str(prod.url) if prod.url is not None else None,
        "image_url": str(prod.image_url) if prod.image_url is not None else None,
        "name": prod.name,
        "price": prod.price,
        "currency": prod.currency,
        "domain": prod.domain,
        "brand_name": prod.brand,
        "category": prod.category,
        "description": prod.description,
    }
    upsert_product(product_dict)
    return prod

async def main(limit: int | None = None):
    print("[INIT] Starting scrape run")

    # --- Load URLs from Supabase ---
    urls = fetch_brand_urls(limit=limit)
    if not urls:
        print("[INIT] No URLs returned from Supabase. Exiting.")
        return

    print(f"[INIT] Loaded {len(urls)} URLs from Supabase brands table")

    # --- Init browser / Playwright ---
    pw, browser = await init_browser()
    print("[INIT] Browser initialized, starting concurrent scraping...")

    sem = asyncio.Semaphore(3)  # limit concurrency to 3 pages
    print("[INIT] Concurrency limit set to 3 pages")

    async def safe_process(url: str):
        async with sem:
            print(f"[JOB] FETCH → {url}")
            try:
                p = await process(url, browser)
                if p:
                    print(f"[JOB] OK   → {p.id} | {p.domain} | {p.name} | {p.price}")
                else:
                    print(f"[JOB] SKIP → {url} (no product returned)")
            except Exception as e:
                print(f"[JOB] ERR  → {url} | {type(e).__name__}: {e}")

    await asyncio.gather(*(safe_process(u) for u in urls))

    print("[SHUTDOWN] Closing browser...")
    await browser.close()
    await pw.stop()
    print("[DONE] Scrape run finished.")


if __name__ == "__main__":
    # Optional CLI: pass an integer limit to only scrape N URLs from Supabase
    #   python -m scrape.scrape          -> scrape all URLs
    #   python -m scrape.scrape 50       -> scrape first 50 URLs
    arg_limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    asyncio.run(main(arg_limit))
