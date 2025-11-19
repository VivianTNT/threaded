# recsys/api/example_usage_backend.py
import requests

API = "http://localhost:8000"

print(requests.get(f"{API}/recommend/item/12345").json())