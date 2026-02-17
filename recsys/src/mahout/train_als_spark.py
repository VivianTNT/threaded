#!/usr/bin/env python3
"""
Train ALS using Spark MLlib (Mahout-compatible objective).

Runs on Spark locally or in Colab. Writes user_factors and item_factors to CSV
for import_factors.py to convert to als_*.joblib.

Optimized for fast runtime: deduped input, caps, fewer iterations.
Use nohup/screen for long runs when laptop may sleep.
"""
import argparse
import sys
from pathlib import Path

# Spark session setup
def get_spark_session(master="local[*]", app_name="ALS-Train"):
    try:
        from pyspark.sql import SparkSession
    except ImportError:
        print("❌ PySpark not installed. Run: pip install pyspark")
        sys.exit(1)

    spark = (
        SparkSession.builder
        .master(master)
        .appName(app_name)
        .config("spark.driver.memory", "4g")
        .config("spark.sql.shuffle.partitions", "64")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


def train_als(
    input_csv: str,
    output_dir: str,
    rank: int = 32,
    max_iter: int = 5,
    reg_param: float = 0.1,
    seed: int = 42,
) -> None:
    """Train ALS and write user_factors.csv, item_factors.csv to output_dir."""
    from pyspark.ml.recommendation import ALS

    spark = get_spark_session()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[ALS] Loading {input_csv}...")
    df = spark.read.csv(
        str(input_csv),
        header=True,
        inferSchema=True,
    )
    # Expect columns: user, item, value
    df = df.selectExpr("user as userId", "item as itemId", "value as rating")
    n = df.count()
    print(f"[ALS] Loaded {n:,} interactions")

    print(f"[ALS] Training (rank={rank}, maxIter={max_iter}, regParam={reg_param})...")
    als = ALS(
        rank=rank,
        maxIter=max_iter,
        regParam=reg_param,
        userCol="userId",
        itemCol="itemId",
        ratingCol="rating",
        seed=seed,
        coldStartStrategy="drop",
    )
    model = als.fit(df)

    # Explode features array to columns (id, f0, f1, ..., f_{rank-1}) for import_factors
    from pyspark.sql import functions as F

    def explode_factors(df):
        # df has id, features (array)
        for i in range(rank):
            df = df.withColumn(f"f{i}", F.col("features")[i])
        cols = ["id"] + [f"f{i}" for i in range(rank)]
        return df.select(cols)

    # Spark writes CSV to a directory; coalesce to single file and rename
    import shutil

    def write_factors_df(df, name: str) -> None:
        exploded = explode_factors(df)
        tmp_dir = output_dir / f"{name}_tmp"
        out_file = output_dir / f"{name}.csv"
        exploded.coalesce(1).write.mode("overwrite").option("header", "true").csv(str(tmp_dir))
        part_files = list(tmp_dir.glob("part-*.csv"))
        if part_files:
            shutil.move(str(part_files[0]), str(out_file))
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print("[ALS] Writing user_factors.csv...")
    write_factors_df(model.userFactors, "user_factors")

    print("[ALS] Writing item_factors.csv...")
    write_factors_df(model.itemFactors, "item_factors")

    print(f"[ALS] Done. Factors in {output_dir}")
    spark.stop()


def main():
    parser = argparse.ArgumentParser(description="Train ALS with Spark MLlib")
    parser.add_argument("--input", type=str, required=True, help="Path to interactions CSV")
    parser.add_argument("--output", type=str, required=True, help="Output directory for factors")
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--reg", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_als(
        input_csv=args.input,
        output_dir=args.output,
        rank=args.rank,
        max_iter=args.iters,
        reg_param=args.reg,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
