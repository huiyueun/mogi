# Huiyu Mosquito Experiments

Personal experiment workspace for the mosquito trajectory project.

## Current Best

| submission | Public | Private |
| --- | ---: | ---: |
| `outputs/second_place_expert_blends/confirmed_ode_heavy_800_second200.csv` | 0.7034 | 0.7042 |

Best confirmed blend:

- 80% GOH30 ODE-heavy:
  - GRU 20%
  - ODE 60%
  - HyperPhysics 20%
- 20% restored 2nd-place prediction

## Layout

- `huiyu/experiments/`
  - Personal experiment scripts.
  - OOF-lite, GOH30 component prediction, MoE searches, TCN/Transformer-lite tests, and 2nd-place blending.
- `huiyu/docs/experiment_progress.md`
  - Full experiment log, submitted scores, conclusions, and next candidates.
- `huiyu/submissions/`
  - Local copy of key submission artifacts.
  - CSV files are ignored by git to avoid cluttering the shared repo.

## Key Commands

Restore and blend the 2nd-place expert:

```bash
.venv-cu128/bin/python huiyu/experiments/blend_second_place_expert.py \
  --out-dir outputs/second_place_expert_blends
```

Generate GOH30 component predictions and ODE-heavy candidates:

```bash
.venv-cu128/bin/python huiyu/experiments/goh30_component_submissions.py
```

Run OOF-lite with TCN:

```bash
.venv-cu128/bin/python huiyu/experiments/goh30_oof_lite.py \
  --folds 3 \
  --epochs 10 \
  --device cuda \
  --include-tcn \
  --out-dir outputs/goh30_oof_tcn10_cuda
```

## Important Conclusions

- ODE-heavy GOH30 transferred to LB: Private 0.7033.
- H-heavy/KMeans MoE/learned stacking did not transfer.
- TCN and Transformer-lite improved OOF but did not improve LB.
- The strongest improvement came from blending an LB-verified 2nd-place expert with GOH30 ODE-heavy.
