import urllib.robotparser as rp
from urllib.parse import urlparse

def allowed(url: str, ua="RetailScrapePOC/1.0"):
    p = urlparse(url)
    robots = f"{p.scheme}://{p.netloc}/robots.txt"
    r = rp.RobotFileParser()
    try:
        r.set_url(robots); r.read()
        return r.can_fetch(ua, url)
    except Exception:
        return True  # fail-open for demo; you can change to False if you prefer
