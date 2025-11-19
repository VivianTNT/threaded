import requests

API = "http://localhost:8000"

item_id = 466800  # pick any H&M item in your dataset
r = requests.get(f"{API}/recommend/item/{item_id}")

print("\nSimilar items:")
for x in r.json()["recommendations"]:
    print(x)