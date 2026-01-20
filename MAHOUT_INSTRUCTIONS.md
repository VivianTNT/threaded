# Apache Mahout Integration Guide

This project supports using **Apache Mahout** as the baseline collaborative filtering model. The resulting vectors can be imported to replace the default `implicit` ALS implementation.

## Overview

1. **Export** interaction data from this project to CSV.
2. **Train** using Apache Mahout (externally).
3. **Import** the resulting vectors back into this project.
4. **Fine-tune** (Optional): Use these vectors to initialize the Neural Two-Tower model.

## Step 1: Export Data

Run the export script to generate the CSV file Mahout needs:

```bash
python -m recsys.src.mahout.mahout_interface
```

This will create:
- `recsys/data/mahout/interactions.csv` (Format: `userID,itemID,preference`)

## Step 2: Run Apache Mahout

Assuming you have Apache Mahout (and Spark/Hadoop) installed, run the ALS factorization.

*Example (using Mahout Spark Shell or CLI):*

```bash
# This is a conceptual command; adapt to your Mahout version (0.13+)
mahout spark-itemsimilarity \
    --input interactions.csv \
    --output output_dir \
    --master local[4]
```

Or if using the **ALSWR** (Alternating Least Squares with Weighted Regularization) algorithm:

```bash
mahout parallelALS \
    --input interactions.csv \
    --output output_als \
    --numFeatures 64 \
    --lambda 0.05 \
    --numIterations 15
```

## Step 3: Format Output

You need to convert Mahout's output (often SequenceFiles) to simple CSV files:
- `user_factors.csv`: `userID, feature1, feature2, ...`
- `item_factors.csv`: `itemID, feature1, feature2, ...`

Place these files in `recsys/data/mahout/`.

## Step 4: Import Results

Edit `recsys/src/mahout/mahout_interface.py` to uncomment the import lines, or run:

```python
from recsys.src.mahout.mahout_interface import import_mahout_results
from pathlib import Path

DATA = Path("recsys/data/mahout")
import_mahout_results(
    DATA / "user_factors.csv",
    DATA / "item_factors.csv"
)
```

## Result

The file `recsys/artifacts/hm_collab_model.joblib` will be updated with your Mahout vectors. The existing Hybrid Ranker will automatically use these new vectors for scoring.
