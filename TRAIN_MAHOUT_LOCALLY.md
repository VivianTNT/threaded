# Training Mahout Locally (Background Process)

## Prerequisites Installation

First, install Java and Spark:

```bash
# Install Java (if not already installed)
brew install openjdk@11

# Set JAVA_HOME
export JAVA_HOME=$(/usr/libexec/java_home -v 11)

# Install Spark
brew install apache-spark

# Download Mahout (or use Maven)
# Option 1: Download pre-built Mahout
wget https://archive.apache.org/dist/mahout/0.13.0/apache-mahout-distribution-0.13.0.tar.gz
tar -xzf apache-mahout-distribution-0.13.0.tar.gz
export MAHOUT_HOME=$(pwd)/apache-mahout-distribution-0.13.0

# Option 2: Use Maven (recommended for latest)
# Mahout is now part of Spark MLlib, so you can use Spark directly
```

## Training Commands (Background Process)

### Method 1: Using `nohup` (Simple)

```bash
# Train H&M model in background
nohup spark-submit \
  --class org.apache.mahout.sparkbindings.blas.ABtDotted \
  --master local[*] \
  --driver-memory 4g \
  --executor-memory 4g \
  $MAHOUT_HOME/mahout-spark-0.13.0.jar \
  --input recsys/data/mahout/interactions_hm.csv \
  --output recsys/data/mahout/output_hm \
  --numFeatures 64 \
  --lambda 0.05 \
  --numIterations 15 \
  > mahout_hm_training.log 2>&1 &

# Save the process ID
echo $! > mahout_hm.pid
```

### Method 2: Using `tmux` (Recommended - Better for Monitoring)

```bash
# Install tmux if needed
brew install tmux

# Start a new tmux session
tmux new -s mahout_training

# Inside tmux, run the training:
spark-submit \
  --class org.apache.mahout.sparkbindings.blas.ABtDotted \
  --master local[*] \
  --driver-memory 4g \
  --executor-memory 4g \
  $MAHOUT_HOME/mahout-spark-0.13.0.jar \
  --input recsys/data/mahout/interactions_hm.csv \
  --output recsys/data/mahout/output_hm \
  --numFeatures 64 \
  --lambda 0.05 \
  --numIterations 15

# Detach from tmux: Press Ctrl+B, then D
# Reattach later: tmux attach -t mahout_training
```

## Monitoring Progress

### If using nohup:
```bash
# Watch the log file in real-time
tail -f mahout_hm_training.log

# Check if process is still running
ps -p $(cat mahout_hm.pid)

# Check resource usage
top -pid $(cat mahout_hm.pid)
```

### If using tmux:
```bash
# List all tmux sessions
tmux ls

# Reattach to see progress
tmux attach -t mahout_training

# Detach again: Ctrl+B, then D
```

## Alternative: Use Python `implicit` Library (Easier, Same Algorithm)

Since Mahout setup is complex, here's an easier alternative that does the same ALS algorithm:

```bash
# Install implicit
pip install implicit

# Train in background with nohup
nohup python3 -c "
from implicit.als import AlternatingLeastSquares
import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix
import joblib
from pathlib import Path

DATA = Path('recsys/data/mahout')
ART = Path('recsys/artifacts')

print('Loading interactions_hm.csv...')
df = pd.read_csv(DATA / 'interactions_hm.csv', header=None, names=['user_id', 'item_id', 'pref'])

# Build sparse matrix
users = df['user_id'].astype('category')
items = df['item_id'].astype('category')
mat = coo_matrix(
    (df['pref'].values, (users.cat.codes, items.cat.codes)),
    shape=(len(users.cat.categories), len(items.cat.categories))
)

print(f'Matrix shape: {mat.shape}')
print('Training ALS (Mahout-compatible)...')

model = AlternatingLeastSquares(
    factors=64,
    regularization=0.05,
    iterations=15,
    use_gpu=False
)
model.fit(mat.T)

# Save in Mahout-compatible format
user_map = {i: uid for i, uid in enumerate(users.cat.categories)}
item_map = {i: iid for i, iid in enumerate(items.cat.categories)}

joblib.dump({
    'user_factors': model.user_factors,
    'item_factors': model.item_factors,
    'user_map': user_map,
    'item_map': item_map
}, ART / 'hm_collab_model.joblib')

print('✅ Training complete!')
" > mahout_hm_training.log 2>&1 &

echo $! > mahout_hm.pid
```

## Check Training Status

```bash
# See latest log entries
tail -20 mahout_hm_training.log

# Follow log in real-time
tail -f mahout_hm_training.log

# Check if still running
ps aux | grep -E "(spark|mahout|python)" | grep -v grep
```

## After Training Completes

The output will be in:
- `recsys/data/mahout/output_hm/` (if using Spark/Mahout)
- Or `recsys/artifacts/hm_collab_model.joblib` (if using implicit)

Then import using:
```bash
python3 -m recsys.src.mahout.mahout_interface --dataset hm --action import --user_vecs <path_to_user_factors> --item_vecs <path_to_item_factors>
```
