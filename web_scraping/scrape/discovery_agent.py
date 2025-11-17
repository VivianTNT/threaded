import json
import re, asyncio, time, random, os
from urllib.parse import urljoin, urlparse
from typing import List, Dict, Optional, Set
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
CATEGORY_URLS_FILE_PATH = os.path.join(WEB_SCRAPING_DIR, "new_category_urls.txt")
PRODUCT_URLS_FILE_PATH = os.path.join(WEB_SCRAPING_DIR, "new_product_urls.txt")

# Cap for testing purposes
MAX_PRODUCTS_TO_FIND = 100

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
        # if not any(pattern in url.lower() for pattern in ['/product/', '/item/', '/shop/', '/sale']):
        #     continue
        triage_queue[domain] = url
    
    return triage_queue

def find_category_links(html_content: str, domain: str, base_url: str) -> Set[str]:
    """
    Parses the HTML of a valid homepage or category page to find sub-links
    to product listing pages (e.g., "Women's", "Men's", "Sweaters").
    """
    soup = BeautifulSoup(html_content, "lxml")
    found_links: Set[str] = set()
    
    # Keywords that suggest a link is a product category
    KEYWORDS_POSITIVE = [
        'shop', 'women', 'men', 'clothing', 'apparel', 'sale', 'new', 
        'category', 'collection', 'browse', 'products', 'c/', 'cid=',
        'department='
    ]
    
    # Keywords that suggest a link is *not* a product category
    KEYWORDS_NEGATIVE = [
        'login', 'account', 'cart', 'careers', 'help', 'support', 'blog', 
        'about', 'company', 'investor', 'privacy', 'terms', 'store-locator',
        'find-a-store', 'gift-card'
    ]

    for a in soup.find_all('a', href=True):
        href = a['href']
        
        # 1. Check for negative keywords
        if any(kw in href.lower() for kw in KEYWORDS_NEGATIVE):
            continue
            
        # 2. Check for positive keywords
        if any(kw in href.lower() for kw in KEYWORDS_POSITIVE):
            
            # 3. Ensure it's an on-site link
            if href.startswith('/') or domain in href:
                # Build the absolute URL
                abs_url = urljoin(base_url, href)
                found_links.add(abs_url)

    # --- MODIFIED: Changed this print statement for clarity ---
    print(f"    -> Crawled page, found {len(found_links)} category links.")
    return found_links

def find_product_links(html_content: str, domain: str, base_url: str) -> Set[str]:
    """
    Parses the HTML of a *category* page to find links to specific *product* pages.
    """
    soup = BeautifulSoup(html_content, "lxml")
    found_links: Set[str] = set()
    
    # Keywords that suggest a link is a product
    KEYWORDS_POSITIVE = [
        '/product/', '/p/', '/item/', 'pid=', 'productid=', '.html'
        # Note: We look for a path structure, not just '.html' in general
    ]
    
    # Keywords that suggest a link is NOT a product (e.g., another category)
    KEYWORDS_NEGATIVE = [
        # Exclude category links
        'shop', 'women', 'men', 'clothing', 'apparel', 'sale', 'new', 
        'category', 'collection', 'browse', 'products', 'c/', 'cid=',
        'department=', 'filter=', 'sort=',
        # Exclude utility links
        'login', 'account', 'cart', 'careers', 'help', 'support', 'blog', 
        'about', 'company', 'investor', 'privacy', 'terms', 'store-locator',
        'find-a-store', 'gift-card'
    ]

    for a in soup.find_all('a', href=True):
        href = a['href']
        href_lower = href.lower()
        
        # 1. Check for negative keywords
        if any(kw in href_lower for kw in KEYWORDS_NEGATIVE):
            continue
            
        # 2. Check for positive keywords
        if any(kw in href_lower for kw in KEYWORDS_POSITIVE):
            
            # 3. Ensure it's an on-site link
            if href.startswith('/') or domain in href:
                # Build the absolute URL
                abs_url = urljoin(base_url, href)
                # Clean query params and anchors
                abs_url = abs_url.split('?')[0].split('#')[0] 
                found_links.add(abs_url)

    print(f"    -> Found {len(found_links)} potential *product* links.")
    return found_links

async def run_scoping_test(url: str, domain: str, browser_context) -> bool:
    """
    [CRITIC FUNCTION] Performs an initial scrape using the project's fetcher.
    """
    print(f"  -> Critiquing {domain}...")
    try:
        html_content = await fetch_html(url, browser_context)
        if not html_content:
            print("  ❌ FAIL: Failed to fetch HTML.")
            return None

        # 1. Get the name
        product_dict = generic_parser(html_content) 
        product_name = product_dict.get('name')

        # 2. Get the base_url for the link finder
        parts = urlparse(url)
        base_url = f"{parts.scheme}://{parts.netloc}"

        # 3. Check for category links
        category_links = find_category_links(html_content, domain, base_url)
        
        # Final Test: Must have a name AND at least one category link
        if product_name and len(category_links) > 0:
            print(f"  ✅ PASS: Found name '{product_name}' and {len(category_links)} category links.")
            # Return HTML and the links we just found (to avoid re-parsing)
            return (html_content, category_links) 
        else:
            print("  ❌ FAIL: Page lacked a name or 'retail-like' links (e.g., 'shop', 'women', 'men').")
            return None

    except Exception as e:
        print(f"  🚨 ERROR during scoping test: {e}")
        return None

# ====================================================================
# CORE AGENT ORCHESTRATION LOGIC
# ====================================================================

async def run_discovery_cycle(search_queries: List[str]):
    """
    Orchestrates the new two-phase discovery process:
    1. Find category URLs from homepages.
    2. Find product URLs from those category URLs.
    """
    print("--- Starting Discovery Agent Cycle ---")
    all_seed_urls = []
    
    async with Stealth().use_async(async_playwright()) as p:
        
        browser_context = None
        try:
            browser_context = await init_browser(p)
        except Exception as e:
            print(f"🚨 FATAL: Could not initialize browser. {e}")
            return

        # --- Phase 1: Find CATEGORY Links ---
        print("\n--- Phase 1: Finding Category Pages ---")
        
        for query in search_queries:
            all_seed_urls.extend(await get_seed_urls(query, browser_context))
            await asyncio.sleep(random.randint(1, 3)) 

        triage_queue = filter_irrelevant_domains(all_seed_urls)
        
        print(f"\n[TRIAGE] Identified {len(triage_queue)} unique retail-like domains for scoping.")

        all_category_links: Set[str] = set()
        for domain, url in triage_queue.items():
            print(f"\nTesting Domain: {domain} (URL: {url})")
            
            test_result = await run_scoping_test(url, domain, browser_context)
            if test_result:
                html_content, category_links = test_result # Unpack
                all_category_links.update(category_links)
            
            await asyncio.sleep(random.randint(1, 2))
        
        print(f"\n--- Phase 1 Complete: Found {len(all_category_links)} category URLs to crawl ---")

        # --- Phase 2: Find PRODUCT Links ---
        print("\n--- Phase 2: Finding Product Pages ---")
        all_product_links: Set[str] = set()

        for i, category_url in enumerate(all_category_links):
            if len(all_product_links) >= MAX_PRODUCTS_TO_FIND:
                print(f"\n[INFO] Reached product limit ({MAX_PRODUCTS_TO_FIND}). Stopping category crawl.")
                break
            
            print(f"\nCrawling Category URL {i+1}/{len(all_category_links)}: {category_url}")
            try:
                # Get domain/base for this specific URL
                parts = urlparse(category_url)
                domain = get_base_domain(category_url) # Use helper to get clean domain
                base_url = f"{parts.scheme}://{parts.netloc}"

                if not domain:
                    print("  ❌ FAIL: Could not parse domain.")
                    continue

                html_content = await fetch_html(category_url, browser_context)
                if not html_content:
                    print("  ❌ FAIL: Failed to fetch category page HTML.")
                    continue
                
                # Call the new function
                product_links = find_product_links(html_content, domain, base_url)
                all_product_links.update(product_links)

            except Exception as e:
                print(f"  🚨 ERROR crawling category page {category_url}: {e}")
            
            await asyncio.sleep(random.randint(1, 2)) # Politeness

        # --- End of Browser Operations ---
        if browser_context: 
            await browser_context.close()
        
        print("\n[Fetcher] Browser context closed. Playwright stopped.")

    # --- Save the FINAL list of PRODUCT URLs ---
    print(f"\n--- Phase 2 Complete: Found {len(all_product_links)} total product URLs ---")
    
    if all_product_links:
        try:
            with open(PRODUCT_URLS_FILE_PATH, 'w') as f:
                f.write('\n'.join(sorted(list(all_product_links))))
            
            print("\n--- DISCOVERY CYCLE COMPLETE ---")
            print(f"Successfully saved {len(all_product_links)} new product URLs to '{PRODUCT_URLS_FILE_PATH}'")
            print("Sample Product URLs Found:")
            for link in sorted(list(all_product_links))[:20]: # Print a sample
                print(f"-> {link}")
            if len(all_product_links) > 20:
                print(f"...and {len(all_product_links) - 20} more.")
                
        except IOError as e:
            print(f"\nERROR: Could not write to file {PRODUCT_URLS_FILE_PATH}. {e}")
            
    else:
        print("\n--- DISCOVERY CYCLE COMPLETE ---")
        print("No new product URLs were identified in this cycle.")

if __name__ == '__main__':
    queries = [
        # "gap clothing",
        'fall coats online shopping',
        # 'online clothing stores',
        "winter clothing shopping",
    ]
    
    asyncio.run(run_discovery_cycle(queries))
