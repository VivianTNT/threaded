import asyncio, random
from playwright.async_api import async_playwright
from .cache import load_raw, save_raw

UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome Safari RetailScrapePOC/1.0"

async def fetch_html(url: str, use_cache=True, timeout_ms=20000, retries=2):
    if use_cache:
        cached = load_raw(url)
        if cached: return cached

    for attempt in range(retries + 1):
        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                ctx = await browser.new_context(user_agent=UA)
                page = await ctx.new_page()
                await page.goto(url, timeout=timeout_ms, wait_until="domcontentloaded")
                # Allow lazy JS to render price/name if needed
                await page.wait_for_timeout(500 + random.randint(0, 500))
                html = await page.content()
                await browser.close()
                save_raw(url, html)
                return html
        except Exception:
            if attempt == retries:
                raise
            await asyncio.sleep(1.0 + attempt)
