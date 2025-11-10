import re, asyncio, time, random, os
from urllib.parse import urlparse
from typing import List, Dict, Optional
from bs4 import BeautifulSoup
from fake_useragent import UserAgent

from playwright.async_api import async_playwright, Playwright
from playwright_stealth import Stealth

try:
    from .schema import Product
    from .discovery_fetcher import init_browser, fetch_html 
    from .parser_generic import extract_generic as generic_parser
    
except ImportError as e:
    print(f"Discovery Agent failed to import core project modules: {e}")
    print("Please ensure you are running this from the 'threaded-main' directory, e.g.:")
    print("python -m web_scraping.scrape.discovery_agent")
    
    # Define placeholder structure
    class Product:
        def __init__(self, name=None, price=None, **kwargs):
            self.name = name; self.price = price
    
    # Update placeholders to match new signatures
    async def init_browser(p: Playwright): # Accepts 'p'
        raise ImportError("Failed to import real init_browser")
    async def fetch_html(url: str, browser_context, **kwargs) -> str: return ""
    def generic_parser(html: str) -> Product: return Product()

# Get the absolute path of this
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WEB_SCRAPING_DIR = os.path.dirname(SCRIPT_DIR)

# Define paths for the output files
DEBUG_FILE_PATH = os.path.join(WEB_SCRAPING_DIR, "debug_serp.html")
DOMAINS_FILE_PATH = os.path.join(WEB_SCRAPING_DIR, "new_domains.txt")

# ====================================================================
# Functions: get_base_domain, get_seed_urls, filter_irrelevant_domains,
# and run_scoping_test
# ====================================================================
def get_base_domain(url: str) -> Optional[str]:
    try:
        netloc = urlparse(url).netloc
        if not netloc: return None
        netloc = re.sub(r'^(www\.|m\.|shop\.)', '', netloc)
        parts = netloc.split('.')
        if len(parts) > 2 and parts[-2] not in ('co', 'com', 'org', 'net'):
            netloc = '.'.join(parts[-2:])
        elif len(parts) > 3 and parts[-3] in ('co', 'com', 'org', 'net'):
             netloc = '.'.join(parts[-3:])
        return netloc.lower()
    except Exception:
        return None

async def get_seed_urls(query: str, browser_context, max_results: int = 20) -> List[str]:
    """
    [PLANNER FUNCTION] Scrapes Google SERP using Playwright to find seed URLs.
    """
    
    search_url = f"https://www.google.com/search?q={query.replace(' ', '+')}&num={max_results}"
    print(f"\n[PLANNER] Executing live search: '{query}'")
    
    result_links = []
    html_content = "" # Initialize for debug logging
    try:
        # Fetch the search page
        # This fetches the results page (or pauses for CAPTCHA)
        html_content = await fetch_html(search_url, browser_context, use_cache=False)
        if not html_content:
             raise Exception("Failed to fetch HTML, content is None.")
        
        # Parse with BeautifulSoup
        soup = BeautifulSoup(html_content, "lxml")
        
        # SELECTOR STRATEGY
        # find all <a> (link) tags that have an <h3> (header) tag as a 
        # direct child. This is the main link for each result
        
        link_elements = soup.select('a:has(> h3)')
        
        for link_element in link_elements:
            url = link_element.get('href')
            
            # Clean Google's redirect URLs (e.g., /url?q=...)
            if url and url.startswith('/url?q='):
                # Get the part after '/url?q=' and before the first '&'
                url = url.split('&')[0].replace('/url?q=', '')
            
            # Final check
            if url and url.startswith('http') and not url.startswith('https://webcache.googleusercontent.com'):
                if url not in result_links: # Ensure uniqueness
                    result_links.append(url)
                
    except Exception as e:
        print(f"  ERROR during SERP scrape: {e}.")

    if not result_links:
        print("  WARNING: SERP scrape returned no results. Selectors may be broken.")
        # Save HTML for debugging
        try:
            with open(DEBUG_FILE_PATH, "w", encoding="utf-8") as f:
                f.write(html_content)
            print(f"  [Debug] Saved failed HTML to {DEBUG_FILE_PATH}. Open this file in a browser to see what the scraper saw.")
        except Exception as e:
            print(f"  [Debug] Could not save debug file: {e}")
    
    return result_links

def filter_irrelevant_domains(seed_urls: List[str]) -> Dict[str, str]:
    triage_queue: Dict[str, str] = {}
    
    for url in seed_urls:
        domain = get_base_domain(url)
        if not domain:
            continue
        if any(keyword in url.lower() for keyword in ['blog', 'support', 'faq', 'pinterest', 'twitter']):
            continue
        if domain in triage_queue:
            continue
        if not any(pattern in url.lower() for pattern in ['/product/', '/item/', '/shop/', '/sale']):
            continue
        triage_queue[domain] = url
    
    return triage_queue

async def run_scoping_test(url: str, domain: str, browser_context) -> bool:
    """
    [CRITIC FUNCTION] Performs an initial scrape using the project's fetcher.
    """
    print(f"  -> Critiquing {domain}...")
    try:
        # Fetch HTML content
        html_content = await fetch_html(url, browser_context)
        if not html_content:
            print("  FAIL: Failed to fetch HTML.")
            return False

        # Attempt generic parsing (which returns a dict)
        # pass the URL as well
        product_dict = generic_parser(html_content) 

        # Quality Check:
        # use dictionary .get() access
        # relax the test: for a category page, only require a 'name' 
        # to prove the site is parsable.
        product_name = product_dict.get('name')
        
        if product_dict and product_name:
            # don't need price for a simple scoping test.
            print(f"  PASS: Generic parser found critical data (Name: {product_name}).")
            return True
        else:
            print("  FAIL: Generic parser failed to extract critical product data (name).")
            return False

    except Exception as e:
        # This will catch any unexpected parser errors
        print(f"  ERROR during scoping test: {e}")
        return False

# ====================================================================
# CORE AGENT ORCHESTRATION LOGIC
# ====================================================================

async def run_discovery_cycle(search_queries: List[str]):
    """
    Orchestrates the entire discovery, triage, and scoping process
    using the recommended playwright-stealth pattern.
    """
    print("--- Starting Discovery Agent Cycle ---")
    all_seed_urls = []
    
    # Wrap the entire lifecycle in the Stealth context manager
    async with Stealth().use_async(async_playwright()) as p:
        
        browser_context = None
        try:
            # Pass the stealthed 'p' object to the init function
            browser_context = await init_browser(p)
        except Exception as e:
            print(f" FATAL: Could not initialize browser. {e}")
            return

        # Lead generation (planner)
        for query in search_queries:
            all_seed_urls.extend(await get_seed_urls(query, browser_context))
            await asyncio.sleep(random.randint(1, 3)) # Politeness delay

        # Triage and filter
        triage_queue = filter_irrelevant_domains(all_seed_urls)
        
        print(f"\n[TRIAGE] Identified {len(triage_queue)} unique retail-like domains for scoping.")

        # Scoping test (critic)
        scopable_domains: List[str] = []
        
        for domain, url in triage_queue.items():
            print(f"\nTesting Domain: {domain} (URL: {url})")
            if await run_scoping_test(url, domain, browser_context):
                scopable_domains.append(domain)
            
            await asyncio.sleep(random.randint(1, 2))

        # Stop the browser context
        if browser_context: 
            await browser_context.close()
        
        # 'pw.stop()' is handled automatically by the 'async with' block exiting
        print("\n[Fetcher] Browser context closed. Playwright stopped.")

    # Output logic is outside the 'async with' block
    if scopable_domains:
        try:
            with open(DOMAINS_FILE_PATH, 'w') as f:
                f.write('\n'.join(scopable_domains))
            print("\n--- DISCOVERY CYCLE COMPLETE ---")
            print(f"Successfully saved {len(scopable_domains)} new scopable domains to '{DOMAINS_FILE_PATH}'")
            print("New Scopable Domains Found and Ready for Full Scrape:")
            for domain in scopable_domains:
                print(f"-> {domain}")
        except IOError as e:
            print(f"\nERROR: Could not write to file {DOMAINS_FILE_PATH}. {e}")
    else:
        print("\n--- DISCOVERY CYCLE COMPLETE ---")
        print("No new scopable domains were identified in this cycle.")


if __name__ == '__main__':
    queries = [
        'fall coats online shopping',
        'online clothing stores',
    ]
    
    asyncio.run(run_discovery_cycle(queries))
