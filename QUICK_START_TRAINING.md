# Quick Start: Train Mahout Locally

## Step 1: Install Required Library

```bash
pip3 install implicit
```

## Step 2: Train H&M Model (Background Process)

### Option A: Using `nohup` (Simple)

```bash
# Start training in background
nohup python3 -m recsys.src.mahout.train_als_local --dataset hm > mahout_hm.log 2>&1 &

# Save process ID
echo $! > mahout_hm.pid

# Monitor progress
tail -f mahout_hm.log

# Check if still running
ps -p $(cat mahout_hm.pid)
```

### Option B: Using `screen` (Better for monitoring)

```bash
# Install screen if needed
brew install screen

# Start screen session
screen -S mahout

# Inside screen, run:
python3 -m recsys.src.mahout.train_als_local --dataset hm

# Detach: Press Ctrl+A, then D
# Reattach: screen -r mahout
```

## Step 3: Train RetailRocket Model (After H&M finishes)

```bash
# Same process for RR
nohup python3 -m recsys.src.mahout.train_als_local --dataset rr > mahout_rr.log 2>&1 &
echo $! > mahout_rr.pid
tail -f mahout_rr.log
```

## Monitor Progress

```bash
# Watch log file
tail -f mahout_hm.log

# Check process status
ps aux | grep train_als_local | grep -v grep

# Check file sizes (models are being saved)
ls -lh recsys/artifacts/*mahout*
```

## Expected Output

When training completes, you'll see:
- `recsys/artifacts/collab_model_hm.joblib` (or `hm_collab_model.joblib`)
- `recsys/artifacts/user_vectors_mahout_hm.joblib`
- `recsys/artifacts/item_vectors_mahout_hm.joblib`

## Time Estimate

- H&M (2.1M interactions): ~15-30 minutes
- RR (similar size): ~15-30 minutes

## Troubleshooting

If training fails:
```bash
# Check the log
cat mahout_hm.log

# Make sure implicit is installed
python3 -c "import implicit; print('OK')"
```
