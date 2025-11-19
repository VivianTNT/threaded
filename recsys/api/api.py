import requests

class RecommenderAPI:
    def __init__(self, base_url="http://localhost:8000"):
        self.base = base_url

    def recommend_user(self, user_id, top_k=20):
        r = requests.get(f"{self.base}/recommend/user/{user_id}", params={"top_k": top_k})
        return r.json()

    def recommend_item(self, item_id, top_k=20):
        r = requests.get(f"{self.base}/recommend/item/{item_id}", params={"top_k": top_k})
        return r.json()

    def recommend_from_image(self, file_path, top_k=20):
        with open(file_path, "rb") as f:
            r = requests.post(
                f"{self.base}/recommend/from_image",
                files={"file": f},
                params={"top_k": top_k}
            )
        return r.json()