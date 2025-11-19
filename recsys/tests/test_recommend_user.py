import requests

API = "http://localhost:8000"
USER = "user_123"

r = requests.get(f"{API}/recommend/user/{USER}")
print("\nUser recommendations:")
for x in r.json()["recommendations"]:
    print(x)