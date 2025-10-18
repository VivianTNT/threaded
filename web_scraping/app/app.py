import streamlit as st, orjson, json, numpy as np, faiss
from sentence_transformers import SentenceTransformer
from pathlib import Path
from urllib.parse import urlparse
from scrape.scrape import process as scrape_one
import asyncio

st.set_page_config(page_title="Retail Scrape POC", layout="centered")
st.title("Retail Scraping & Retrieval (POC)")

# Load index if available
idx, items, model = None, None, None
if Path("data/emb/items.faiss").exists():
    idx = faiss.read_index("data/emb/items.faiss")
    items = json.load(open("data/emb/items.json"))
    X = np.load("data/emb/items.npy")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

tab1, tab2 = st.tabs(["Parse URL", "Search"])

with tab1:
    url = st.text_input("Paste a product URL")
    if st.button("Parse"):
        with st.spinner("Scraping..."):
            prod = asyncio.run(scrape_one(url))
        if prod:
            st.json(orjson.loads(prod.model_dump_json()))
            if prod.image_url:
                st.image(prod.image_url, width=250)

with tab2:
    q = st.text_input("Query (e.g., 'black blazer under 120')")
    k = st.slider("Top-K", 1, 10, 5)
    if st.button("Search"):
        if idx is None:
            st.warning("Index not built yet. Run retrieval/build_index.py first.")
        else:
            v = model.encode([q], normalize_embeddings=True).astype("float32")
            D, I = idx.search(v, k)
            for i in I[0]:
                it = items[i]
                st.write(f"**{it['name']}** — {it.get('price')} {it.get('currency') or ''}")
                if it.get("image_url"): st.image(it["image_url"], width=200)
                st.write(it["url"])
                st.divider()
