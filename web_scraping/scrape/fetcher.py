import asyncio, random
from playwright.async_api import async_playwright
from .cache import load_raw, save_raw

UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome Safari RetailScrapePOC/1.0"

async def init_browser():
    pw = await async_playwright().start()
    browser = await pw.chromium.launch(headless=True)
    return pw, browser

async def fetch_html(url: str, browser, use_cache=True, timeout_ms=20000, retries=2):
    if use_cache:
        cached = load_raw(url)
        if cached:
            return cached

    for attempt in range(retries + 1):
        try:
            ctx = await browser.new_context(user_agent=UA)
            page = await ctx.new_page()
            await page.goto(url, timeout=timeout_ms, wait_until="domcontentloaded")
            await page.wait_for_timeout(500 + random.randint(0, 500))
            html = await page.content()
            await ctx.close()
            save_raw(url, html)
            return html
        except Exception:
            if attempt == retries:
                raise
            await asyncio.sleep(1.0 + attempt)
