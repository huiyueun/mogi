# Jinyong Experiments

This folder contains Jinyong's local experiment helpers and outputs.

## 1. Make 5-Fold Split

Creates a 5-fold assignment file from the 10,000 training samples.
Each fold is used as validation once, so every run is 8,000 train / 2,000 valid.

```bash
python3 jinyong/make_fold_split.py \
  --root data \
  --folds 5 \
  --seed 42 \
  --out jinyong/fold_assignments_5fold.csv
```

Output:

```text
jinyong/fold_assignments_5fold.csv
```

## 2. Train OOF Predictions

Run from the project root.

```bash
python3 huiyu/experiments/goh30_oof_lite.py \
  --root data \
  --folds 5 \
  --epochs 3 \
  --device cuda \
  --out-dir jinyong/hit5_oof_5fold
```

For a faster smoke test:

```bash
python3 huiyu/experiments/goh30_oof_lite.py \
  --root data \
  --folds 5 \
  --epochs 1 \
  --device cuda \
  --sample-size 1000 \
  --out-dir jinyong/hit5_oof_smoke
```

## 3. Evaluate Multi-Laser Hit

Evaluates whether any of the candidate predictions falls within 1 cm.

```bash
python3 jinyong/evaluate_multi_laser_hit.py \
  --root data \
  --oof-dir jinyong/hit5_oof_5fold \
  --order oof_ode oof_gru oof_kalman oof_cv oof_h
```

Output:

```text
jinyong/hit5_oof_5fold/multi_laser_hit_report.csv
```
