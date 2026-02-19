from __future__ import annotations

from io import BytesIO
from typing import Any, Dict, List, Optional
import logging
import json
from pathlib import Path

import requests
import torch
from PIL import Image, UnidentifiedImageError
from transformers import CLIPModel, CLIPProcessor

from .supabase_client import supabase


logger = logging.getLogger(__name__)


class ProductEmbeddingsUpserter:
    """
    Builds embeddings from `products` and upserts to `product_embeddings`.

    Expected DB setup:
    - `product_embeddings.product_id` has a UNIQUE constraint
    - `product_embeddings.text_embedding` is pgvector with dimension 512
      (for clip-vit-base-patch32 text features)
    - `product_embeddings.image_embedding` is pgvector with dimension 512
      (for clip-ViT-B-32)
    """

    def __init__(
        self,
        clip_model_name: str = "openai/clip-vit-base-patch32",
        image_timeout_seconds: int = 15,
        skipped_log_path: str = "web_scraping/scrape/logs/skipped_product_embeddings.jsonl",
    ) -> None:
        self.device = (
            "mps"
            if torch.backends.mps.is_available()
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.clip_model = CLIPModel.from_pretrained(clip_model_name).to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
        self.clip_model.eval()
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
                .select("id,image_url,description,name,brand_name")
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

    def _build_text_input(self, product: Dict[str, Any]) -> str:
        brand_part = str(product.get("brand_name") or "").strip()
        text_source = product.get("name") or product.get("description") or ""
        text_part = str(text_source).strip()
        text = f"{brand_part} {text_part}".strip()
        return text if text else "generic product"

    def _embed_text(self, product: Dict[str, Any]) -> Optional[List[float]]:
        text = self._build_text_input(product)
        with torch.no_grad():
            inputs = self.clip_processor(
                text=[text],
                return_tensors="pt",
                truncation=True,
                padding=True,
            ).to(self.device)
            text_out = self.clip_model.get_text_features(**inputs)
            vector = self._coerce_clip_vector(text_out, modality="text")
        return vector.cpu().numpy().flatten().astype("float32").tolist()

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
            inputs = self.clip_processor(images=image, return_tensors="pt").to(
                self.device
            )
            image_out = self.clip_model.get_image_features(**inputs)
            vector = self._coerce_clip_vector(image_out, modality="image")
            vector = vector / vector.norm(dim=-1, keepdim=True)
        return vector.cpu().numpy().flatten().astype("float32").tolist()

    def _coerce_clip_vector(self, output: Any, modality: str) -> torch.Tensor:
        """
        Coerce CLIP feature outputs to a tensor across transformers version differences.
        """
        if isinstance(output, torch.Tensor):
            return output

        embeds_attr = "text_embeds" if modality == "text" else "image_embeds"
        projection_attr = "text_projection" if modality == "text" else "visual_projection"

        embeds = getattr(output, embeds_attr, None)
        if isinstance(embeds, torch.Tensor):
            return embeds

        pooled = getattr(output, "pooler_output", None)
        if isinstance(pooled, torch.Tensor):
            proj = getattr(self.clip_model, projection_attr, None)
            if (
                proj is not None
                and hasattr(proj, "in_features")
                and pooled.shape[-1] == proj.in_features
            ):
                return proj(pooled)
            return pooled

        raise TypeError(
            f"Unsupported CLIP output type for {modality}: {type(output)}"
        )

    def build_embedding_record(self, product: Dict[str, Any]) -> Dict[str, Any]:
        product_id = product.get("id")
        if not product_id:
            raise ValueError("Product row is missing `id`.")

        return {
            "product_id": str(product_id),
            "text_embedding": self._embed_text(product),
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
