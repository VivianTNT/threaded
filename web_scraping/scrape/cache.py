from pathlib import Path
import orjson, hashlib

RAW = Path("data/raw_html")
CLEAN = Path("data/clean")
RAW.mkdir(parents=True, exist_ok=True)
CLEAN.mkdir(parents=True, exist_ok=True)

def key_for(url: str) -> str:
    return hashlib.sha1(url.encode()).hexdigest()

def load_raw(url: str):
    p = RAW / f"{key_for(url)}.html"
    return p.read_text(encoding="utf-8") if p.exists() else None

def save_raw(url: str, html: str):
    (RAW / f"{key_for(url)}.html").write_text(html, encoding="utf-8")

def append_jsonl(filename: str, obj: dict):
    with open(CLEAN / filename, "ab") as f:
        f.write(orjson.dumps(obj) + b"\n")
