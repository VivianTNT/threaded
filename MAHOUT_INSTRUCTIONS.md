# Apache Mahout Integration Guide (H&M and RetailRocket)

This project supports using **Apache Mahout** as the baseline collaborative filtering model for both H&M and RetailRocket.

## Step 1: Export Data

Run the export script for each dataset:

```bash
# Export H&M
python -m recsys.src.mahout.mahout_interface --dataset hm --action export

# Export RetailRocket
python -m recsys.src.mahout.mahout_interface --dataset rr --action export
```

This creates `recsys/data/mahout/interactions_hm.csv` and `interactions_rr.csv`.

## Step 2: Run Apache Mahout (Google Colab Recommended)

Mahout requires Java and Spark. It is easiest to run this on **Google Colab**.

1. Upload your `interactions_*.csv` to Colab.
2. Install Mahout/Spark in Colab.
3. Run the ALS factorization:

```bash
# Example for H&M
mahout parallelALS \
    --input interactions_hm.csv \
    --output output_hm \
    --numFeatures 64 \
    --lambda 0.05 \
    --numIterations 15
```

4. Download the resulting `user_factors.csv` and `item_factors.csv` from the output directory.

## Step 3: Import Results

Place the downloaded CSVs in `recsys/data/mahout/` and run:

```bash
# Import H&M
python -m recsys.src.mahout.mahout_interface --dataset hm --action import --user_vecs recsys/data/mahout/user_factors_hm.csv --item_vecs recsys/data/mahout/item_factors_hm.csv

# Import RetailRocket
python -m recsys.src.mahout.mahout_interface --dataset rr --action import --user_vecs recsys/data/mahout/user_factors_rr.csv --item_vecs recsys/data/mahout/item_factors_rr.csv
```

## Step 4: Fine-tune with Neural Two-Tower

Now use the Mahout vectors as the baseline for your PyTorch model.

### 1. Train the H&M Base Model
This model learns from fashion-specific interactions.
```bash
python -m recsys.src.models.train_mahout_finetune --dataset hm --epochs 5
```

### 2. Fine-tune using RetailRocket
Load the H&M weights and refine them on the RetailRocket interaction patterns.
```bash
python -m recsys.src.models.train_mahout_finetune --dataset rr --epochs 3 --load_model recsys/artifacts/mahout_finetuned_hm.pt
```

## Summary for Presentation

- **Baseline**: Apache Mahout (Collaborative Filtering). It learns latent preferences from raw interaction matrices.
- **Fine-tuning**: Neural Two-Tower Model. It takes the linear Mahout vectors as input and applies non-linear transformations (Deep Learning) to capture more complex user-item relationships.
- **Transfer Learning**: We initialize the RetailRocket model with H&M weights. This "transfers" the knowledge learned from the feature-rich H&M dataset (which has images/text) to the RetailRocket domain.
