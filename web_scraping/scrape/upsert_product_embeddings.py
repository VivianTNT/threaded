from __future__ import annotations

from io import BytesIO
from typing import Any, Dict, List, Optional
import logging
import json
from pathlib import Path

import requests
import torch
from PIL import Image, UnidentifiedImageError
from sentence_transformers import SentenceTransformer
from transformers import CLIPModel, CLIPProcessor

from .supabase_client import supabase


logger = logging.getLogger(__name__)


class ProductEmbeddingsUpserter:
    """
    Builds embeddings from `products` and upserts to `product_embeddings`.

    Expected DB setup:
    - `product_embeddings.product_id` has a UNIQUE constraint
    - `product_embeddings.text_embedding` is pgvector with dimension 384
      (for all-MiniLM-L6-v2)
    - `product_embeddings.image_embedding` is pgvector with dimension 512
      (for clip-ViT-B-32)
    """

    def __init__(
        self,
        text_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        image_model_name: str = "openai/clip-vit-base-patch32",
        image_timeout_seconds: int = 15,
        skipped_log_path: str = "web_scraping/scrape/logs/skipped_product_embeddings.jsonl",
    ) -> None:
        self.text_model = SentenceTransformer(text_model_name)
        self.device = (
            "mps"
            if torch.backends.mps.is_available()
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.image_model = CLIPModel.from_pretrained(image_model_name).to(self.device)
        self.image_processor = CLIPProcessor.from_pretrained(image_model_name)
        self.image_model.eval()
        self.image_timeout_seconds = image_timeout_seconds
        self.skipped_log_path = Path(skipped_log_path)
        self.skipped_log_path.parent.mkdir(parents=True, exist_ok=True)

    def fetch_products(self, batch_size: int = 500) -> List[Dict[str, Any]]:
        """
        Fetch products containing fields needed to build embeddings.
        """
        all_products: List[Dict[str, Any]] = []
        start = 0

        while True:
            end = start + batch_size - 1
            response = (
                supabase.table("products")
                .select("id,image_url,description")
                .order("id")
                .range(start, end)
                .execute()
            )
            batch = getattr(response, "data", None) or []
            if not batch:
                break
            all_products.extend(batch)
            if len(batch) < batch_size:
                break
            start += batch_size

        logger.info("Fetched %d products from table 'products'.", len(all_products))
        return all_products

    def _embed_text(self, text: Optional[str]) -> Optional[List[float]]:
        if not text:
            return None
        vector = self.text_model.encode([text], normalize_embeddings=True)[0]
        return vector.astype("float32").tolist()

    def _download_image(self, image_url: str) -> Optional[Image.Image]:
        try:
            response = requests.get(image_url, timeout=self.image_timeout_seconds)
            response.raise_for_status()
            return Image.open(BytesIO(response.content)).convert("RGB")
        except (
            requests.RequestException,
            UnidentifiedImageError,
            OSError,
            ValueError,
        ) as exc:
            logger.warning("Failed to load image for %s: %s", image_url, exc)
            return None

    def _embed_image(self, image_url: Optional[str]) -> Optional[List[float]]:
        if not image_url:
            return None
        image = self._download_image(image_url)
        if image is None:
            return None
        with torch.no_grad():
            inputs = self.image_processor(images=image, return_tensors="pt").to(
                self.device
            )
            image_out = self.image_model.get_image_features(**inputs)
            # Some transformers builds return a model output object instead of a tensor.
            if isinstance(image_out, torch.Tensor):
                vector = image_out
            elif hasattr(image_out, "image_embeds") and image_out.image_embeds is not None:
                vector = image_out.image_embeds
            elif hasattr(image_out, "pooler_output") and image_out.pooler_output is not None:
                vector = image_out.pooler_output
                if hasattr(self.image_model, "visual_projection"):
                    proj = self.image_model.visual_projection
                    # Only project when dimensions match the projection input.
                    if (
                        hasattr(proj, "in_features")
                        and vector.shape[-1] == proj.in_features
                    ):
                        vector = proj(vector)
            else:
                raise TypeError(
                    f"Unsupported CLIP output type from get_image_features: {type(image_out)}"
                )
            vector = vector / vector.norm(dim=-1, keepdim=True)
        return vector.cpu().numpy().flatten().astype("float32").tolist()

    def build_embedding_record(self, product: Dict[str, Any]) -> Dict[str, Any]:
        product_id = product.get("id")
        if not product_id:
            raise ValueError("Product row is missing `id`.")

        return {
            "product_id": str(product_id),
            "text_embedding": self._embed_text(product.get("description")),
            "image_embedding": self._embed_image(product.get("image_url")),
        }

    def _append_skipped_log(
        self,
        product: Dict[str, Any],
        *,
        missing_text_embedding: bool,
        missing_image_embedding: bool,
    ) -> None:
        row = {
            "product_id": str(product.get("id")),
            "missing_text_embedding": missing_text_embedding,
            "missing_image_embedding": missing_image_embedding,
            "has_description": bool(product.get("description")),
            "has_image_url": bool(product.get("image_url")),
        }
        with self.skipped_log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

    def upsert_embeddings(self, products: List[Dict[str, Any]]) -> int:
        records: List[Dict[str, Any]] = []
        # Start fresh each run so the file reflects the current run only.
        self.skipped_log_path.write_text("", encoding="utf-8")

        for product in products:
            record = self.build_embedding_record(product)

            missing_text = record["text_embedding"] is None
            missing_image = record["image_embedding"] is None

            # Skip rows where either embedding is missing and log them for review.
            if missing_text or missing_image:
                self._append_skipped_log(
                    product,
                    missing_text_embedding=missing_text,
                    missing_image_embedding=missing_image,
                )
                continue
            records.append(record)

        if not records:
            logger.info("No embedding records to upsert.")
            return 0

        (
            supabase.table("product_embeddings")
            .upsert(records, on_conflict="product_id")
            .execute()
        )
        logger.info(
            "Upserted %d product embedding rows. Skipped rows logged at %s",
            len(records),
            self.skipped_log_path,
        )
        return len(records)

    def run(self, batch_size: int = 500) -> int:
        products = self.fetch_products(batch_size=batch_size)
        return self.upsert_embeddings(products)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    count = ProductEmbeddingsUpserter().run()
    print(f"Upserted {count} product embedding rows.")
