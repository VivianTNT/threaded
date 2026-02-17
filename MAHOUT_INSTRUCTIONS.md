# Apache Mahout / ALS Integration Guide

This project supports **Spark MLlib ALS** (Mahout-compatible) for collaborative filtering on H&M and RetailRocket.

## Quick Start

See **`recsys/src/mahout/runbook.md`** for step-by-step instructions.

## Pipeline Overview

1. **Export** – `python -m recsys.src.mahout.export_interactions --dataset hm` (or `rr`)
2. **Train** – Spark ALS on Colab or local (see runbook)
3. **Import** – `python -m recsys.src.mahout.import_factors --dataset hm --user_factors ... --item_factors ...`
4. **API** – Uses content (FAISS) by default; collab when `als_hm.joblib` exists

## Fine-tune with Neural Two-Tower

After importing ALS factors:

```bash
python -m recsys.src.models.train_mahout_finetune --dataset hm --epochs 5
```

## Summary for Presentation

- **Baseline**: Spark MLlib ALS (Collaborative Filtering)
- **Fine-tuning**: Neural Two-Tower Model
- **Transfer Learning**: Initialize RetailRocket model with H&M weights
