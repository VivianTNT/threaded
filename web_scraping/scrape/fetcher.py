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
            ctx = await browser.new_context(
                user_agent=UA,
                viewport={'width': 1280, 'height': 800},
                device_scale_factor=1,
            )
            # Basic steath script to hide that this is an automated browser
            await ctx.add_init_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")

            page = await ctx.new_page()
            
            # Navigate and wait for network to be mostly idle
            await page.goto(url, timeout=timeout_ms, wait_until="domcontentloaded")
            
            # For Uniqlo specifically, wait for swiper to initialize
            if "uniqlo.com" in url:
                try:
                    # Wait for the swiper container to appear
                    await page.wait_for_selector(".swiper-zoom-container img", timeout=5000)
                    # Give it a bit more time for images to load
                    await page.wait_for_timeout(1500)
                except:
                    # If swiper doesn't load, continue anyway
                    await page.wait_for_timeout(1000)
            else:
                await page.wait_for_timeout(500 + random.randint(0, 500))

            # Short random sleep to simulate human reading time
            await page.wait_for_timeout(2000 + random.randint(0, 1000))
            
            html = await page.content()
            await ctx.close()
            # save_raw(url, html)
            return html
        except Exception as e:
            if ctx:
                await ctx.close()

            print(f"  [Attempt {attempt+1}/{retries+1}] Failed: {e}")

            if attempt == retries:
                print(f"  [GIVE UP] Could not fetch {url}")
                return None
                # raise
            await asyncio.sleep(1.0 + attempt)
