import asyncio, random, os
from playwright.async_api import Playwright
from .cache import load_raw, save_raw

# Use common, modern User-Agent
UA = "Mozilla/5.0 (Windows NT 10.0; Win64 x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/118.0.0.0 Safari/537.36"

# Define a path for the persistent profile
PROFILE_DIR = os.path.join(os.getcwd(), "playwright_user_data")

async def init_browser(p: Playwright):
    """
    Initializes a PERSISTENT browser context using the provided stealthed playwright object.
    """
    print(f"[Fetcher] Initializing persistent browser profile at: {PROFILE_DIR}")
    print("[Fetcher] IMPORTANT: A browser window will open (headless=False).")
    print("[Fetcher] On the first run, Google will ask for a CAPTCHA.")
    print("[Fetcher] Please solve it manually in the browser window to proceed.")
    
    browser_context = await p.chromium.launch_persistent_context(
        PROFILE_DIR,
        headless=False,
        user_agent=UA,
        slow_mo=50, 
        viewport={'width': 1440, 'height': 900}
    )
    
    return browser_context

async def fetch_html(url: str, browser_context, use_cache=True, timeout_ms=20000, retries=2):
    """
    Fetches HTML using the provided persistent browser_context.
    """
    if use_cache:
        cached = load_raw(url)
        if cached:
            return cached

    page = None
    for attempt in range(retries + 1):
        try:
            page = await browser_context.new_page()

            print(f"  [Fetcher] Navigating to {url}...")
            await page.goto(url, timeout=timeout_ms, wait_until="domcontentloaded")
            
            # --- NEW BLOCK: HUMAN CAPTCHA INTERVENTION ---
            # Check the content of the page *after* it loads
            html_content_for_check = await page.content()
            
            # Use keywords from the debug_serp.html to detect the CAPTCHA
            if "Our systems have detected unusual traffic" in html_content_for_check or "g-recaptcha" in html_content_for_check:
                print("\n" + "="*60)
                print("  ** ** CAPTCHA DETECTED! SCRIPT IS PAUSED. ** **")
                print("  Please solve the CAPTCHA in the open browser window.")
                print("  The browser window will NOT close.")
                input("  >>> After solving, press [Enter] in this terminal to continue...")
                print("  ...Resuming script.")
                print("="*60 + "\n")
            

            # Now that the user has solved the CAPTCHA,
            # the 'page' object will have auto-navigated to the results.
            # We can now try to click the cookie consent button.
            try:
                reject_button = page.locator('button:has-text("Reject all")')
                await reject_button.click(timeout=2500)
                print("  [Fetcher] Handled cookie consent (Reject all).")
            except Exception:
                pass # No cookie page, that's fine

            await page.wait_for_timeout(500 + random.randint(0, 500))
            
            # Get the final page content (which should be the search results)
            html = await page.content()
            
            save_raw(url, html)
            
            await page.close()
            return html
            
        except Exception as e:
            print(f"  [Fetcher ERR] Attempt {attempt+1} failed: {e}")
            if page:
                await page.close()
            if attempt == retries:
                print(f"  [Fetcher FAIL] All retries failed for {url}")
                return None
            await asyncio.sleep(1.0 + attempt)

