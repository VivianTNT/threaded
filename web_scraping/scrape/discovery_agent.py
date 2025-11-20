import re, asyncio, time, random, os
from urllib.parse import urlparse, urljoin
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
    print("Please ensure you are running this from the 'threaded-main' directory.")
    
    class Product:
        def __init__(self, name=None, price=None, **kwargs):
            self.name = name; self.price = price
    async def init_browser(p: Playwright): raise ImportError("Failed to import real init_browser")
    async def fetch_html(url: str, browser_context, **kwargs) -> str: return ""
    def generic_parser(html: str) -> Product: return Product()

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WEB_SCRAPING_DIR = os.path.dirname(SCRIPT_DIR)
DEBUG_FILE_PATH = os.path.join(WEB_SCRAPING_DIR, "debug_serp.html")
RESULTS_FILE_PATH = os.path.join(WEB_SCRAPING_DIR, "new_product_urls.txt")

# Limit for testing purposes (prevent infinite crawling)
MAX_PRODUCTS_TO_FIND = 800

# Domains to Explicitly Skip
SKIP_DOMAINS = {
    # Marketplaces
    'amazon.com', 'ebay.com', 'etsy.com', 'walmart.com', 'target.com',
    'aliexpress.com', 'temu.com', 'macys.com', 'ajio.com'
    # Social Media
    'pinterest.com', 'instagram.com', 'facebook.com', 'tiktok.com', 
    'youtube.com', 'twitter.com', 'reddit.com',
    # Info / News
    'wikipedia.org', 'nytimes.com', 'vogue.com', 'elle.com', 'harpersbazaar.com',
    'medium.com', 'linkedin.com',
    # Miscellaneous
    'mackage.com', 'sunandski.com', 'beginningboutique.com', 'bloomingdales.com',
    'krsaddleshop.com', 'boohoo.com', 'express.com', 'taylorstitch.com', 'hm.com',
    'arctix.com'
}

# ====================================================================
# CORE UTILS: Domain Extraction & Search
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
    search_url = f"https://www.google.com/search?q={query.replace(' ', '+')}&num={max_results}"
    print(f"\n[PLANNER] Executing live search: '{query}'")
    
    result_links = []
    try:
        html_content = await fetch_html(search_url, browser_context, use_cache=False)
        if not html_content: raise Exception("Failed to fetch HTML.")
        
        soup = BeautifulSoup(html_content, "lxml")
        # Find links that look like results (Anchor tags with H3 headers)
        link_elements = soup.select('a:has(> h3)')
        
        for link_element in link_elements:
            url = link_element.get('href')
            if url and url.startswith('/url?q='):
                url = url.split('&')[0].replace('/url?q=', '')
            if url and url.startswith('http') and not url.startswith('https://webcache.googleusercontent.com'):
                if url not in result_links: result_links.append(url)
                
    except Exception as e:
        print(f"  ERROR during SERP scrape: {e}.")

    if not result_links:
        print("  WARNING: SERP scrape returned no results.")
        with open(DEBUG_FILE_PATH, "w", encoding="utf-8") as f: f.write(html_content or "")
    
    return result_links

def filter_irrelevant_domains(seed_urls: List[str]) -> Dict[str, str]:
    """
    Triages domains.
    """
    triage_queue: Dict[str, str] = {}
    
    for url in seed_urls:
        domain = get_base_domain(url)
        if not domain: continue
        
        # Check against explicit skip list
        if domain in SKIP_DOMAINS:
            continue

        # Filter out obvious non-retail keywords in the URL itself
        if any(keyword in url.lower() for keyword in ['blog', 'support', 'faq', 'career']):
            continue
            
        if domain in triage_queue:
            continue
        
        triage_queue[domain] = url
    
    return triage_queue

# ====================================================================
# DEEP CRAWLING LOGIC: Categories & Products
# ====================================================================

def find_category_links(html_content: str, domain: str, base_url: str) -> Set[str]:
    """
    Finds links to 'Men', 'Women', 'Sale', etc.
    """
    soup = BeautifulSoup(html_content, "lxml")
    found_links: Set[str] = set()
    
    KEYWORDS_POSITIVE = ['shop', 'women', 'men', 'clothing', 'apparel', 'sale', 'new', 'collection', 'browse', 'department']
    KEYWORDS_NEGATIVE = ['login', 'account', 'cart', 'careers', 'help', 'support', 'blog', 'privacy', 'terms']

    for a in soup.find_all('a', href=True):
        href = a['href'].lower()
        # Skip negatives
        if any(kw in href for kw in KEYWORDS_NEGATIVE):
            continue
        # Must contain positive keywords
        if any(kw in href for kw in KEYWORDS_POSITIVE):
            # Must be on-site
            if a['href'].startswith('/') or domain in a['href']:
                abs_url = urljoin(base_url, a['href'])
                found_links.add(abs_url)
    return found_links

def find_product_links(html_content: str, domain: str, base_url: str) -> Set[str]:
    """
    Finds product pages using Regex and Keywords.
    Captures SKUs like 'TB246-LCL236G.html' or '/p/12345'.
    """
    soup = BeautifulSoup(html_content, "lxml")
    found_links: Set[str] = set()

    # 1. Strong Indicators (Keywords)
    KEYWORDS = [
        '/product/', '/products', '/p/', 
        '/item/', '/shop/product',
        'pid=', 'productid=']
    
    # 2. Pattern Matching (Regex)
    # Matches:
    #  - /12345.html (Numeric ID at end)
    #  - /TB246-LCL236G.html (Alphanumeric at end)
    #  - /p/12345 (Standard path)
    PRODUCT_PATTERNS = [
        re.compile(r"\/p\/\d+"),                      # /p/12345
        re.compile(r"\/product\/.+"),                 # /product/name
        re.compile(r"-[0-9]{5,}(\.html)?$"),          # ...-123456.html
        re.compile(r"\/[A-Z0-9]{4,}-[A-Z0-9]{4,}(\.html)?$"), # /SKU-CODE.html
        re.compile(r"\/[a-z0-9-]+\/[A-Z0-9-]{5,}(\.html)?$")  # /name/SKU.html
    ]

    NEG_KEYWORDS = ['login', 'account', 'cart', 'help', 'blog', 'facebook', 'twitter', 'policy', 'search']

    for a in soup.find_all('a', href=True):
        raw_href = a['href']
        href = raw_href.lower()

        # Filter irrelevant
        if any(kw in href for kw in NEG_KEYWORDS): continue
        
        is_product = False
        
        # Check Keywords
        if any(kw in href for kw in KEYWORDS):
            is_product = True
        
        # Check Regex if keywords failed
        if not is_product:
            # Strip query params for regex check
            path = raw_href.split('?')[0]
            for pattern in PRODUCT_PATTERNS:
                if pattern.search(path):
                    is_product = True
                    break

        if is_product:
            if raw_href.startswith('/') or domain in raw_href:
                abs_url = urljoin(base_url, raw_href)
                # Clean URL (remove query params)
                clean_url = abs_url.split('?')[0].split('#')[0]
                found_links.add(clean_url)

    return found_links

async def run_scoping_test(url: str, domain: str, browser_context) -> Optional[tuple[str, Set[str]]]:
    """
    [CRITIC FUNCTION]
    Validates a domain by checking if it has a Name AND Category links.
    Returns (html_content, set_of_category_links) if valid.
    """
    print(f"  -> Critiquing {domain}...")
    try:
        html_content = await fetch_html(url, browser_context)
        if not html_content: return None

        # 1. Check Name (basic validity)
        product_dict = generic_parser(html_content) 
        product_name = product_dict.get('name')
        
        # 2. Check for Category Links (retail signal)
        parts = urlparse(url)
        base_url = f"{parts.scheme}://{parts.netloc}"
        category_links = find_category_links(html_content, domain, base_url)
        
        if product_name and len(category_links) > 0:
            print(f"  PASS: Found name '{product_name}' and {len(category_links)} category links.")
            return (html_content, category_links)
        else:
            print("  FAIL: Site lacked retail signals (Name + Category links).")
            return None

    except Exception as e:
        print(f"  ERROR during scoping test: {e}")
        return None

# ====================================================================
# RESULTS HANDLING (DB Integration Point)
# ====================================================================

def save_results(product_urls: Set[str]):
    """
    Handles saving the discovered product URLs.
    """
    if not product_urls: return
    
    # TODO: FUTURE SUPABASE INTEGRATION
    
    try:
        with open(RESULTS_FILE_PATH, 'w') as f:
            f.write('\n'.join(sorted(list(product_urls))))
        print(f"\n[SUCCESS] Saved {len(product_urls)} product URLs to '{RESULTS_FILE_PATH}'")
    except IOError as e:
        print(f"\n[ERROR] Could not write file: {e}")

# ====================================================================
# MAIN CYCLE
# ====================================================================

async def run_discovery_cycle(search_queries: List[str]):
    print("--- Starting Discovery Agent Cycle ---")
    all_seed_urls = []
    
    async with Stealth().use_async(async_playwright()) as p:
        try:
            browser_context = await init_browser(p)
        except Exception as e:
            print(f" FATAL: {e}"); return

        # Discovery (search & scope)
        for query in search_queries:
            all_seed_urls.extend(await get_seed_urls(query, browser_context))
            await asyncio.sleep(random.randint(1, 3))

        triage_queue = filter_irrelevant_domains(all_seed_urls)
        print(f"\n[TRIAGE] Identified {len(triage_queue)} unique domains for scoping (after skip list).")

        all_category_links = set()
        
        for domain, url in triage_queue.items():
            result = await run_scoping_test(url, domain, browser_context)
            if result:
                _, cats = result
                all_category_links.update(cats)
            await asyncio.sleep(1)

        # Harvest (crawl categories for products)
        print(f"\n--- Phase 2: Crawling {len(all_category_links)} Categories for Products ---")
        all_product_links = set()

        for i, cat_url in enumerate(all_category_links):
            if len(all_product_links) >= MAX_PRODUCTS_TO_FIND:
                print("  [Limit Reached] Stopping crawl.")
                break
            
            print(f"  Crawling {i+1}/{len(all_category_links)}: {cat_url}")
            try:
                parts = urlparse(cat_url)
                domain = get_base_domain(cat_url)
                if not domain: continue
                
                html = await fetch_html(cat_url, browser_context)
                if html:
                    prods = find_product_links(html, domain, f"{parts.scheme}://{parts.netloc}")
                    new_prods = prods - all_product_links
                    if new_prods:
                        print(f"    -> Found {len(new_prods)} new products.")
                        all_product_links.update(new_prods)
            except Exception as e:
                print(f"    Error: {e}")
            
            await asyncio.sleep(random.randint(1, 2))

        if browser_context: await browser_context.close()

    # Save final results
    save_results(all_product_links)
    print("\n--- DISCOVERY CYCLE COMPLETE ---")

if __name__ == '__main__':
    queries = [
        # "gap clothing",
        'men coats online shopping',
        # 'online clothing stores',
        "winter clothing shopping",
    ]

    asyncio.run(run_discovery_cycle(queries))
