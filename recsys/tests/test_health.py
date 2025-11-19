import requests

API = "http://localhost:8000"

print(requests.get(f"{API}/health").json())
