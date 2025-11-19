import requests

API = "http://localhost:8000"

with open("recsys/shirt.jpg", "rb") as f:
    files = {"file": f}
    r = requests.post(f"{API}/embed/image", files=files)

print(r.json())