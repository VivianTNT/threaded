import requests

API = "http://localhost:8000"
USER = "user_123"

files = [
    ("files", open("recsys/shirt.jpg", "rb")),
]

r = requests.post(f"{API}/user/coldstart/{USER}", files=files)
print("Cold start embedding created:")
print(r.json())