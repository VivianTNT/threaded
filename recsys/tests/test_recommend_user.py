import requests

API = "http://localhost:8000"
USER_ID = 123  # API expects int

r = requests.get(f"{API}/recommend/user/{USER_ID}")
print("\nUser recommendations:")
for x in r.json()["recommendations"]:
    print(x)