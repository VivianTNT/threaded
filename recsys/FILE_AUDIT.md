# File & Artifact Audit

What exists, why it's there, and whether it's still relevant.

---

## 1. Script → Artifact Creation Mapping

| Script | Creates | Used By |
|--------|---------|---------|
| **make_parquets.py** | `events.parquet`, `items.parquet` | events_hm, build_faiss, text_embeddings |
| **hm_only/events_hm.py** | `events_hm.parquet` | user_modeling, hybrid_ranker, build_train_pairs_hm, eval_hm_models |
| **build_faiss.py** | `faiss_items.joblib` | faiss_hm |
| **hm_only/faiss_hm.py** | `faiss_items_hm.joblib` | API, recommend_engine, hybrid_ranker, user_modeling |
| **user_modeling.py** | `user_vectors_hm.joblib` | API, recommend_engine, hybrid_ranker |
| **export_interactions.py** | `interactions_*.csv`, `id_mappings_*.joblib` | train_als_spark, import_factors |
| **train_als_spark.py** (or Colab) | `user_factors.csv`, `item_factors.csv` | import_factors |
| **import_factors.py** | `als_*.joblib`, `collab_model_*.joblib`, `hm_collab_model.joblib`, `user_vectors_mahout_*.joblib`, `item_vectors_mahout_*.joblib` | recommend_engine (collab), hybrid_ranker, train_mahout_finetune |
| **hm_only/build_train_pairs_hm.py** | `train_pairs_hm/*.parquet` | train_mahout_finetune |
| **train_mahout_finetune.py** | `mahout_finetuned_*.pt` | hybrid_ranker |
| **text_embeddings.py** | `text_embeddings.joblib` | build_faiss (cache) |
| **image_embeddings.py** | `image_embeddings.joblib` | build_faiss (cache) |

---

## 2. ARTIFACTS (`recsys/artifacts/`)

| Artifact | Created By | Used By |
|----------|------------|---------|
| **faiss_items_hm.joblib** | faiss_hm.py | API, recommend_engine, hybrid_ranker, user_modeling |
| **user_vectors_hm.joblib** | user_modeling.py | API, recommend_engine, hybrid_ranker |
| **als_hm.joblib** / **als_rr.joblib** | import_factors.py | recommend_engine (strategy=collab) |
| **collab_model_hm.joblib** / **hm_collab_model.joblib** | import_factors.py | hybrid_ranker |
| **user_vectors_mahout_*.joblib** / **item_vectors_mahout_*.joblib** | import_factors.py | train_mahout_finetune |
| **mahout_finetuned_*.pt** | train_mahout_finetune.py | hybrid_ranker |
| **faiss_items.joblib** | build_faiss.py | faiss_hm.py |
| **image_embeddings.joblib** / **text_embeddings.joblib** | text_embeddings.py, image_embeddings.py | build_faiss.py (cache) |
| **faiss_items_backup.joblib** | faiss_hm.py | (safety backup) |
| **two_tower_hm_v2_final.pt** / **two_tower_v2_*.joblib** | (legacy, manual) | hybrid_ranker fallback |

---

## 3. DATA (`recsys/data/`)

### 3a. Raw Data (source of truth)

| Path | Used By |
|------|---------|
| **hm_raw/** | export_interactions, hm_loader, image_embeddings |
| **rr_raw/** | export_interactions, make_parquets |

### 3b. Derived Data

| Path | Created By | Used By |
|------|------------|---------|
| **events.parquet** | make_parquets | events_hm |
| **items.parquet** | make_parquets | build_faiss, text_embeddings |
| **events_hm.parquet** | events_hm.py | user_modeling, hybrid_ranker, build_train_pairs_hm, eval_hm_models |
| **events_original_backup.parquet** | events_hm.py | (safety backup) |
| **train_pairs_hm/** | build_train_pairs_hm | train_mahout_finetune |
| **train_pairs_rr_hm/** / **train_pairs_rr/** | (manual/other) | train_mahout_finetune (RR) |
| **mahout/interactions_*.csv** | export_interactions | train_als_spark |
| **mahout/id_mappings_*.joblib** | export_interactions | import_factors |
| **mahout/als_output_*/** | train_als_spark | import_factors (input) |

---

## 4. Pipeline Summary

### Content pipeline (API needs this)

```
hm_raw + rr_raw
  → make_parquets → events.parquet, items.parquet
  → events_hm.py → events_hm.parquet
  → build_faiss.py (items.parquet) → faiss_items.joblib
  → faiss_hm.py → faiss_items_hm.joblib
  → user_modeling.py (events_hm + faiss_items_hm) → user_vectors_hm.joblib
```

### ALS pipeline (optional collab)

```
hm_raw / rr_raw
  → export_interactions → interactions_*.csv, id_mappings_*.joblib
  → train_als_spark (Colab) → user_factors.csv, item_factors.csv
  → import_factors → als_*.joblib, user_vectors_mahout_*, etc.
```

### Two-Tower pipeline (optional hybrid)

```
train_pairs_hm (from build_train_pairs_hm + events_hm)
  + user_vectors_mahout_hm, item_vectors_mahout_hm
  → train_mahout_finetune → mahout_finetuned_hm.pt
```

---

## 5. Files Kept (and why)

| File | Creates / Purpose |
|------|-------------------|
| **make_parquets.py** | events.parquet, items.parquet – content pipeline start |
| **hm_only/events_hm.py** | events_hm.parquet – H&M-only events |
| **hm_only/faiss_hm.py** | faiss_items_hm.joblib – API dependency |
| **hm_only/build_train_pairs_hm.py** | train_pairs_hm/ – Two-Tower training |
| **build_faiss.py** | faiss_items.joblib – input to faiss_hm |
| **user_modeling.py** | user_vectors_hm.joblib – API dependency |
| **hm_loader.py** | Loads hm_raw – used by make_parquets |
| **embeddings.py** | Text embeddings – used by build_faiss |
| **text_embeddings.py** | text_embeddings.joblib cache – used by build_faiss |
| **image_embeddings.py** | image_embeddings.joblib cache – used by build_faiss |
| **eval_hm_models.py** | Evaluation script – optional |
| **mahout/** (export, train, import, runbook) | ALS pipeline |
| **models/hybrid_ranker.py** | Hybrid scoring – optional for strategy=hybrid |
| **models/train_mahout_finetune.py** | Two-Tower training – optional |

---

## 6. Files Deleted (cleanup applied)

| File | Reason |
|------|--------|
| **dataset_loaders.py** | Loaded train_pairs/ (mixed) – not used by current pipeline |
| **build_training_pairs.py** | Created train_pairs/ – superseded by build_train_pairs_hm |
| **content_filtering.py** | Created content_model*.joblib – not used by API |
| **test_coldstart_user.py** | Tested non-existent /user/coldstart endpoint |
| **artifacts/evaluation_summary.csv** | Output from deleted evaluate_hybrid |

---

## 7. Fixes Applied

| Fix | Change |
|-----|--------|
| **make_parquets.py** | `RAW = recsys/data/rr_raw` (was data/raw) |
| **test_recommend_user.py** | `USER_ID = 123` (int, was "user_123" string) |
| **build_faiss.py** | Removed item_embs.joblib write (unused) |
