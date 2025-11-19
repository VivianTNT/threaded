"""
Content-based filtering model.
Uses item embeddings to find similar items based on content similarity.
For cold start: can recommend to new users based on their first interaction.
"""
import numpy as np
import pandas as pd
import joblib
import faiss
from pathlib import Path
from typing import List, Tuple, Dict

DATA = Path("recsys/data")
ART = Path("recsys/artifacts")

class ContentFiltering:
    """
    Content-based filtering using item embeddings.
    Recommends items similar to items the user has interacted with.
    """
    
    def __init__(self):
        """Load item embeddings and build similarity index."""
        print("[content_filtering] loading item embeddings...")
        
        # Load FAISS index with item embeddings
        faiss_pack = joblib.load(ART / "faiss_items.joblib")
        self.index = faiss_pack["index"]
        self.item_ids = faiss_pack["item_ids"]
        self.item_embeddings = faiss_pack["X"]
        self.row_map = faiss_pack["row_map"]
        
        # Load events to get user-item interactions
        print("[content_filtering] loading events...")
        self.events = pd.read_parquet(DATA / "events.parquet")
        
        # Build user interaction history
        self.user_items = {}
        for _, row in self.events.iterrows():
            uid = int(row["user_id"])
            iid = int(row["item_id"])
            if uid not in self.user_items:
                self.user_items[uid] = []
            self.user_items[uid].append(iid)
        
        print(f"[content_filtering] loaded {len(self.user_items):,} users")
    
    def recommend_for_user(
        self,
        user_id: int,
        top_k: int = 10,
        exclude_interacted: bool = True,
        n_candidates: int = 1000
    ) -> List[Tuple[int, float]]:
        """
        Recommend items for a user based on content similarity.
        
        Args:
            user_id: User ID
            top_k: Number of recommendations
            exclude_interacted: Whether to exclude items user already interacted with
            n_candidates: Number of candidates to retrieve before filtering
        
        Returns:
            List of (item_id, score) tuples
        """
        if user_id not in self.user_items:
            # Cold start: return empty or popular items
            return []
        
        # Get user's interacted items
        interacted_items = set(self.user_items[user_id])
        
        # Aggregate user's item embeddings (average)
        item_embs = []
        for iid in interacted_items:
            if iid in self.row_map:
                idx = self.row_map[iid]
                item_embs.append(self.item_embeddings[idx])
        
        if not item_embs:
            return []
        
        # Average user's item embeddings to get user profile
        user_profile = np.mean(item_embs, axis=0).astype("float32")
        user_profile = user_profile / (np.linalg.norm(user_profile) + 1e-8)  # Normalize
        user_profile = user_profile.reshape(1, -1)
        
        # Search for similar items
        scores, indices = self.index.search(user_profile, n_candidates)
        
        recommendations = []
        for score, idx in zip(scores[0], indices[0]):
            item_id = int(self.item_ids[idx])
            
            if exclude_interacted and item_id in interacted_items:
                continue
            
            recommendations.append((item_id, float(score)))
            
            if len(recommendations) >= top_k:
                break
        
        return recommendations
    
    def predict_score(self, user_id: int, item_id: int) -> float:
        """
        Predict relevance score for a user-item pair.
        
        Args:
            user_id: User ID
            item_id: Item ID
        
        Returns:
            Relevance score
        """
        if user_id not in self.user_items:
            return 0.0
        
        if item_id not in self.row_map:
            return 0.0
        
        # Get user's interacted items
        interacted_items = self.user_items[user_id]
        
        # Get item embedding
        item_idx = self.row_map[item_id]
        item_emb = self.item_embeddings[item_idx]
        
        # Find maximum similarity with user's interacted items
        max_sim = 0.0
        for iid in interacted_items:
            if iid in self.row_map:
                other_idx = self.row_map[iid]
                other_emb = self.item_embeddings[other_idx]
                
                # Cosine similarity
                sim = np.dot(item_emb, other_emb) / (
                    np.linalg.norm(item_emb) * np.linalg.norm(other_emb) + 1e-8
                )
                max_sim = max(max_sim, sim)
        
        return float(max_sim)
    
    def predict_batch(
        self,
        user_ids: np.ndarray,
        item_ids: np.ndarray
    ) -> np.ndarray:
        """
        Predict scores for a batch of user-item pairs.
        
        Args:
            user_ids: Array of user IDs
            item_ids: Array of item IDs
        
        Returns:
            Array of predicted scores
        """
        scores = []
        for uid, iid in zip(user_ids, item_ids):
            scores.append(self.predict_score(int(uid), int(iid)))
        return np.array(scores)

def train_content_filtering():
    """Train and save content filtering model."""
    model = ContentFiltering()
    
    # Save model
    ART.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "user_items": model.user_items,
        "row_map": model.row_map,
    }, ART / "content_model.joblib")
    
    print("✅ saved content filtering model to recsys/artifacts/content_model.joblib")
    return model

if __name__ == "__main__":
    train_content_filtering()


"""
Content‑based filtering model (H&M‑only version).
Uses faiss_items_hm.joblib instead of faiss_items.joblib.
Assumes events_hm.parquet is used (RetailRocket removed).
"""

import numpy as np
import pandas as pd
import joblib
import faiss
from pathlib import Path
from typing import List, Tuple, Dict

DATA = Path("recsys/data")
ART = Path("recsys/artifacts")


class ContentFilteringHM:
    """
    Content-based filtering for H&M only.
    Uses the repaired FAISS pack (faiss_items_hm.joblib).
    """

    def __init__(self):
        print("[content_hm] loading FAISS H&M embeddings...")

        faiss_pack = joblib.load(ART / "faiss_items_hm.joblib")
        self.index = faiss_pack["index"]
        self.item_ids = faiss_pack["item_ids"]
        self.item_embeddings = faiss_pack["X"]
        self.row_map = faiss_pack["row_map"]

        print("[content_hm] loading events_hm...")
        events = pd.read_parquet(DATA / "events_hm.parquet")

        # Build user→items dictionary
        self.user_items: Dict[int, list] = {}
        for uid, iid in zip(events["user_id"], events["item_id"]):
            uid = int(uid)
            iid = int(iid)
            if uid not in self.user_items:
                self.user_items[uid] = []
            self.user_items[uid].append(iid)

        print(f"[content_hm] loaded users: {len(self.user_items):,}")

    def recommend_for_user(
        self,
        user_id: int,
        top_k: int = 10,
        exclude_interacted: bool = True,
        n_candidates: int = 1000,
    ) -> List[Tuple[int, float]]:

        if user_id not in self.user_items:
            return []

        interacted = set(self.user_items[user_id])

        # Aggregate item embeddings for the user's profile
        embs = []
        for iid in interacted:
            if iid in self.row_map:
                embs.append(self.item_embeddings[self.row_map[iid]])

        if not embs:
            return []

        user_profile = np.mean(embs, axis=0).astype("float32")
        user_profile /= (np.linalg.norm(user_profile) + 1e-8)
        user_profile = user_profile.reshape(1, -1)

        scores, idxs = self.index.search(user_profile, n_candidates)

        out = []
        for s, idx in zip(scores[0], idxs[0]):
            iid = int(self.item_ids[idx])
            if exclude_interacted and iid in interacted:
                continue
            out.append((iid, float(s)))
            if len(out) >= top_k:
                break
        return out

    def predict_score(self, user_id: int, item_id: int) -> float:
        if user_id not in self.user_items:
            return 0.0
        if item_id not in self.row_map:
            return 0.0

        item_emb = self.item_embeddings[self.row_map[item_id]]
        interacted = self.user_items[user_id]

        best = 0.0
        for iid in interacted:
            if iid in self.row_map:
                e = self.item_embeddings[self.row_map[iid]]
                sim = np.dot(item_emb, e) / (
                    np.linalg.norm(item_emb) * np.linalg.norm(e) + 1e-8
                )
                best = max(best, sim)
        return float(best)


def train_content_hm():
    model = ContentFilteringHM()
    joblib.dump(
        {"user_items": model.user_items, "row_map": model.row_map},
        ART / "content_model_hm.joblib",
    )
    print("✅ saved H&M content model to recsys/artifacts/content_model_hm.joblib")
    return model


if __name__ == "__main__":
    train_content_hm()