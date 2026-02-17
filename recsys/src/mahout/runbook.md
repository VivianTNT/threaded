# ALS Pipeline Runbook

Step-by-step instructions to export → train → import → run API.

## Prerequisites

- Python 3.9+ with `recsys` package
- For local training: PySpark (`pip install pyspark`)
- For Colab training: Google account

---

## Step 1: Export Interactions

From project root:

```bash
# H&M (hm_raw)
python -m recsys.src.mahout.export_interactions --dataset hm

# RetailRocket (rr_raw)
python -m recsys.src.mahout.export_interactions --dataset rr
```

**Output:**
- `recsys/data/mahout/interactions_hm.csv` (or `interactions_rr.csv`)
- `recsys/data/mahout/id_mappings_hm.joblib` (or `id_mappings_rr.joblib`)

**Check progress:** Script prints row counts. For large H&M data, use `--nrows 100000` for a quick test.

---

## Step 2: Train ALS (Colab or VM)

### Option A: Google Colab (recommended when laptop may sleep)

1. Open [Google Colab](https://colab.research.google.com)
2. Upload `interactions_hm.csv` to `/content/`
3. Open `recsys/src/mahout/train_als_spark_colab.ipynb` (or copy cells into a new notebook)
4. Set `INPUT_CSV = "/content/interactions_hm.csv"`
5. Run all cells
6. Download `user_factors.csv` and `item_factors.csv` from `/content/als_output/`

### Option B: Local / VM with nohup

```bash
# Run in background (survives terminal close if using nohup)
nohup python -m recsys.src.mahout.train_als_spark \
  --input recsys/data/mahout/interactions_hm.csv \
  --output recsys/data/mahout/als_output_hm \
  --rank 32 --iters 5 \
  > als_train.log 2>&1 &

# Check progress
tail -f als_train.log
```

### Option C: screen (for long runs)

```bash
screen -S als_train
python -m recsys.src.mahout.train_als_spark \
  --input recsys/data/mahout/interactions_hm.csv \
  --output recsys/data/mahout/als_output_hm \
  --rank 32 --iters 5

# Detach: Ctrl+A, D
# Reattach: screen -r als_train
```

**Output:** `user_factors.csv` and `item_factors.csv` in the output directory.

---

## Step 3: Import Factors

Place the downloaded CSVs in `recsys/data/mahout/` (or pass full paths):

```bash
python -m recsys.src.mahout.import_factors --dataset hm \
  --user_factors recsys/data/mahout/als_output_hm/user_factors.csv \
  --item_factors recsys/data/mahout/als_output_hm/item_factors.csv
```

**Output:**
- `recsys/artifacts/als_hm.joblib`
- `recsys/artifacts/collab_model_hm.joblib` (legacy)
- `recsys/artifacts/hm_collab_model.joblib` (legacy, for H&M)

---

## Step 4: Run API

```bash
uvicorn recsys.api.main:app --reload --host 0.0.0.0 --port 8000
```

The API starts without retraining. It uses:
- **Content-only** (FAISS) if no ALS artifact exists
- **Content + Collab** if `als_hm.joblib` exists

---

## Step 5: Test Endpoints

```bash
# Health
curl http://localhost:8000/health

# User recommendations (content)
curl "http://localhost:8000/recommend/user/123?top_k=10&strategy=content"

# User recommendations (collab, when als_hm.joblib exists)
curl "http://localhost:8000/recommend/user/123?top_k=10&strategy=collab"

# Item-to-item
curl "http://localhost:8000/recommend/item/2023456789?top_k=10"
```

---

## Troubleshooting

| Issue | Fix |
|------|-----|
| `FileNotFoundError: H&M transactions` | Ensure `recsys/data/hm_raw/transactions_train.csv` exists |
| `FileNotFoundError: id_mappings` | Run `export_interactions` before `import_factors` |
| Spark OOM | Reduce `--rank` (e.g. 16), `--iters` (e.g. 3), or use `--max-per-user` / `--max-per-item` in export |
| Colab session timeout | Use smaller `--nrows` in export for quick test, or Colab Pro for longer runs |

---

## Spark UI (when running locally)

If Spark is running, the UI is typically at `http://localhost:4040` (driver). Check logs for the actual URL.
