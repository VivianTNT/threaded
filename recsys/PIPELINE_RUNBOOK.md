# Full Recommendation Pipeline Runbook

End-to-end commands and scripts for the hybrid recommendation system.

**Architecture overview (generalizable to new users/products):**
1. **Content filtering** – FAISS indices: H&M (image+text), RR (text-only)
2. **Two-Tower** – Feature-based, trained on HM + RR interactions
3. **Hybrid** – Content + Two-Tower (no ALS; ALS is ID-based and doesn't generalize)

---

## Prerequisites

```bash
cd /Users/megdy/senior_design
pip install -e .  # or your venv
```

---

## Phase 1: Data Preparation

```bash
# 1. Merge H&M + RetailRocket into events.parquet, items.parquet
python -m recsys.src.make_parquets

# 2. Extract H&M-only events
python -m recsys.src.hm_only.events_hm

# 3. Extract RR-only events (for RR user vectors, combined Two-Tower)
python -m recsys.src.hm_only.events_rr
```

**Outputs:** `events.parquet`, `items.parquet`, `events_hm.parquet`, `events_rr.parquet`

---

## Phase 2: ALS (Collaborative Filtering) – H&M Only, No Images

```bash
# 1. Export interactions for Mahout/Spark ALS
python -m recsys.src.mahout.export_interactions --dataset hm

# 2. Train ALS (Colab or local Spark)
# Option A – Colab: upload interactions_hm.csv, run train_als_spark_colab.ipynb
# Option B – Local:
python -m recsys.src.mahout.train_als_spark \
  --input recsys/data/mahout/interactions_hm.csv \
  --output recsys/data/mahout/als_output_hm \
  --rank 32 --iters 5

# 3. Import factors → artifacts
python -m recsys.src.mahout.import_factors --dataset hm \
  --user_factors recsys/data/mahout/als_output_hm/user_factors.csv \
  --item_factors recsys/data/mahout/als_output_hm/item_factors.csv
```

**Outputs:** `user_vectors_mahout_hm.joblib`, `item_vectors_mahout_hm.joblib`, `als_hm.joblib`, `collab_model_hm.joblib`

---

## Phase 3: Content Filtering (FAISS)

### 3a. H&M – Image + Text (CLIP)

```bash
# 1. Image embeddings (H&M images only; uses hm_images/)
python -m recsys.src.image_embeddings

# 2. Text embeddings (all items from items.parquet)
python -m recsys.src.text_embeddings

# 3. Build combined FAISS index (fuses image + text for items with both)
python -m recsys.src.build_faiss

# 4. Filter to H&M-only FAISS index
python -m recsys.src.hm_only.faiss_hm
```

**Outputs:** `faiss_items.joblib` → `faiss_items_hm.joblib`

### 3b. RR – Text Only (no images)

```bash
python -m recsys.src.hm_only.faiss_rr
```

**Outputs:** `faiss_items_rr.joblib`

### 3c. User Vectors (Content)

```bash
# H&M user vectors
python -m recsys.src.user_modeling

# RR user vectors
python -m recsys.src.user_modeling_rr
```

**Outputs:** `user_vectors_hm.joblib`, `user_vectors_rr.joblib`

---

## Phase 4: Two-Tower (Feature-Based, Generalizable)

```bash
# 1. Build training pairs
python -m recsys.src.hm_only.build_train_pairs_hm
python -m recsys.src.hm_only.build_train_pairs_rr

# 2. Train on BOTH HM + RR (uses H&M images/text + RR interactions)
python -m recsys.src.models.train_content_two_tower --dataset both --epochs 3
```

**Outputs:** `content_two_tower_combined.pt` (generalizes to new users and new products)

---

## Phase 5: Hybrid (Weighted Sum)

The hybrid ranker combines:
- **Content** – FAISS similarity (user vector vs item embedding)
- **Collab** – ALS dot product (user_vectors_mahout × item_vectors_mahout)
- **Two-Tower** – Mahout-finetuned neural score

```bash
# Run evaluation
python -m recsys.src.models.hybrid_ranker
```

**Default weights:** `w_content=0.4`, `w_cf=0.4`, `w_tt=0.2` (configurable in `score_hybrid`)

---

## Phase 6: API

```bash
uvicorn recsys.api.main:app --reload --host 0.0.0.0 --port 8000
```

```bash
# Content-only
curl "http://localhost:8000/recommend/user/123?top_k=10&strategy=content"

# Collab (ALS)
curl "http://localhost:8000/recommend/user/123?top_k=10&strategy=collab"

# Hybrid (when hybrid_ranker artifacts exist)
curl "http://localhost:8000/recommend/user/123?top_k=10&strategy=hybrid"
```

---

## Quick Reference: Script → Artifact

| Script | Creates |
|--------|---------|
| `make_parquets` | events.parquet, items.parquet |
| `events_hm` | events_hm.parquet |
| `export_interactions` | interactions_hm.csv, id_mappings_hm.joblib |
| `train_als_spark` | user_factors.csv, item_factors.csv |
| `import_factors` | als_hm.joblib, user_vectors_mahout_hm.joblib, item_vectors_mahout_hm.joblib |
| `image_embeddings` | image_embeddings.joblib |
| `text_embeddings` | text_embeddings.joblib |
| `build_faiss` | faiss_items.joblib |
| `faiss_hm` | faiss_items_hm.joblib |
| `faiss_rr` | faiss_items_rr.joblib |
| `user_modeling` | user_vectors_hm.joblib |
| `build_train_pairs_hm` | train_pairs_hm/*.parquet |
| `train_mahout_finetune` | mahout_finetuned_hm.pt |

---

## Final Necessary Files (Keep These)

These are required at runtime by the API, `recommend_engine`, and `hybrid_ranker`. Do not delete.

### `recsys/artifacts/`

| File | Used For |
|------|----------|
| `faiss_items_hm.joblib` | Content strategy, hybrid content score |
| `faiss_items_rr.joblib` | RR content (if used) |
| `user_vectors_hm.joblib` | Content user vectors |
| `als_hm.joblib` | Collab strategy |
| `collab_model_hm.joblib` or `hm_collab_model.joblib` | Hybrid CF score |
| `user_vectors_mahout_hm.joblib` | Two-Tower user input |
| `item_vectors_mahout_hm.joblib` | Two-Tower item input |
| `mahout_finetuned_hm.pt` | Hybrid Two-Tower score |

### `recsys/data/`

| File | Used For |
|------|----------|
| `events_hm.parquet` | Hybrid evaluation, user_modeling |
| `items.parquet` | Item metadata (if API serves it) |

---

## Intermediate Files (Don't Delete)

Outputs from pipeline steps that are inputs to later steps. Deleting these forces a full re-run of the pipeline.

### `recsys/artifacts/`

| File | Input To |
|------|----------|
| `image_embeddings.joblib` | build_faiss |
| `text_embeddings.joblib` | build_faiss |
| `faiss_items.joblib` | faiss_hm, faiss_rr |

### `recsys/data/`

| File | Input To |
|------|----------|
| `events.parquet` | events_hm |
| `items.parquet` | build_faiss, text_embeddings |
| `mahout/interactions_hm.csv` | train_als_spark |
| `mahout/id_mappings_hm.joblib` | import_factors |
| `mahout/als_output_hm/user_factors.csv` | import_factors |
| `mahout/als_output_hm/item_factors.csv` | import_factors |
| `train_pairs_hm/*.parquet` | train_mahout_finetune |

---

## Safe to Delete

These can be removed without breaking the pipeline. Regenerate by re-running the relevant step if needed.

| File | Notes |
|------|-------|
| `recsys/artifacts/faiss_items_backup.joblib` | One-time backup from faiss_hm |
| `recsys/data/events_original_backup.parquet` | One-time backup from events_hm |
| `recsys/artifacts/image_embeddings_partial.joblib` | Checkpoint; image_embeddings will resume or restart |
| `recsys/artifacts/two_tower_hm_v2_final.pt` | Legacy; only if using Mahout Two-Tower path |
| `recsys/artifacts/two_tower_v2_*.joblib` | Legacy ID maps; only if using Mahout path |

---

## ID Scheme

- **H&M items:** `item_id = int("2" + article_id)` → large IDs (e.g. 2_010_877_501_5)
- **RR items:** raw `itemid` from events → small IDs (e.g. 355908)
- **Threshold:** `HM_MAX_ID = 3_000_000` – RR &lt; 3M, H&M ≥ 3M
