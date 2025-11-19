import requests

API = "http://localhost:8000"

with open("recsys/shirt.jpg", "rb") as f:
    files = {"file": f}
    r = requests.post(f"{API}/recommend/from_image", files=files)

print("\nTop-20 Recommendations:")
for x in r.json()["recommendations"]:
    print(x)