# Codex Instructions: Mosquito Trajectory Noise Removal

## Context

We are working with a mosquito trajectory prediction dataset.

Each sample contains a 3D trajectory over 400 ms:

- 11 timesteps
- timestep interval: 40 ms
- coordinates: `x`, `y`, `z`
- time range: `-400 ms` to `0 ms`
- target: future 3D coordinate after the observed trajectory

There are no explicit missing values:

- no NaN
- no empty cells
- no inf / -inf
- no duplicated trajectories
- all samples have exactly 11 timesteps

However, some trajectories may contain implicit missing-like noise or sensor artifacts, such as:

- repeated coordinates
- near-zero movement steps
- one-point spikes
- abnormally large step distance
- abnormally large acceleration
- abnormally large jerk
- target coordinates inconsistent with recent motion

Implement two noise-removal modes:

1. `criteria_a_conservative`
2. `criteria_b_strict`

The goal is to remove suspicious training samples before model training.

For test data, do **not** remove rows. Instead, generate quality flags and optionally create cleaned/interpolated trajectories for inference-time blending.

---

# Task 1: Implement Trajectory Quality Feature Extraction

Create a reusable function:

```python
def compute_trajectory_quality_features(df, has_target: bool = False) -> pd.DataFrame:
    ...
```

The function should compute trajectory quality features for each sample.

Assume each sample has 11 coordinate points:

```text
x_0, y_0, z_0
x_1, y_1, z_1
...
x_10, y_10, z_10
```

or an equivalent long-format representation.

If the actual dataset uses different column names, adapt the parser, but keep the final quality-feature names identical.

Use:

```python
dt = 0.04
```

because each timestep is 40 ms apart.

For each sample, compute:

```python
pos = trajectory coordinates with shape (11, 3)
dpos = np.diff(pos, axis=0)
step_dist = np.linalg.norm(dpos, axis=1)
vel = dpos / dt
acc = np.diff(vel, axis=0) / dt
jerk = np.diff(acc, axis=0) / dt
```

Required output columns:

```text
id
min_step
max_step
median_step
max_step_over_median
zero_step_count
tiny_step_lt_1e4
max_acc
max_jerk
max_interp_resid
max_interp_resid_over_med_step
straightness
```

Definitions:

```python
min_step = step_dist.min()
max_step = step_dist.max()
median_step = np.median(step_dist)
max_step_over_median = max_step / (median_step + 1e-8)
zero_step_count = np.sum(step_dist == 0)
tiny_step_lt_1e4 = min_step < 1e-4
max_acc = np.linalg.norm(acc, axis=1).max()
max_jerk = np.linalg.norm(jerk, axis=1).max()
```

For `straightness`:

```python
displacement = np.linalg.norm(pos[-1] - pos[0])
path_length = np.sum(step_dist)
straightness = displacement / (path_length + 1e-8)
```

For one-point interpolation residual:

```python
interp_resids = []

for t in range(1, 10):
    expected = 0.5 * (pos[t - 1] + pos[t + 1])
    resid = np.linalg.norm(pos[t] - expected)
    interp_resids.append(resid)

max_interp_resid = max(interp_resids)
max_interp_resid_over_med_step = max_interp_resid / (median_step + 1e-8)
```

---

# Task 2: Add Train-Only Target Consistency Features

If `has_target=True`, compute target consistency features.

Assume the target coordinate is:

```python
target = np.array([target_x, target_y, target_z])
```

or equivalent column names.

Use the last observed position and the last observed velocity:

```python
last_pos = pos[-1]
last_vel = (pos[-1] - pos[-2]) / dt
cv80_pred = last_pos + 0.08 * last_vel
```

Required train-only columns:

```text
target_dist_from_last
target_minus_cv80
future_v_change_norm
```

Definitions:

```python
target_dist_from_last = np.linalg.norm(target - last_pos)
target_minus_cv80 = np.linalg.norm(target - cv80_pred)

future_vel = (target - last_pos) / 0.08
future_v_change_norm = np.linalg.norm(future_vel - last_vel)
```

For test data, these columns should either be omitted or filled with `np.nan`.

---

# Task 3: Implement Criteria A - Conservative Noise Removal

Implement a function:

```python
def apply_criteria_a_conservative(quality_df: pd.DataFrame, has_target: bool = False) -> pd.Series:
    ...
```

This function should return a boolean mask where `True` means the sample should be removed from the training set.

Criteria A is conservative. It should remove only highly suspicious samples.

Use the following thresholds:

```python
criteria_a = (
    (quality_df["zero_step_count"] > 0) |
    (quality_df["min_step"] < 1e-4) |
    (quality_df["max_step"] > 0.05398218) |
    (quality_df["max_acc"] > 54.68043) |
    (quality_df["max_jerk"] > 1528.03563) |
    (quality_df["max_interp_resid"] > 0.04374434)
)
```

If `has_target=True`, also add train-label consistency checks:

```python
criteria_a = criteria_a | (
    (quality_df["target_minus_cv80"] > 0.14535968) |
    (quality_df["future_v_change_norm"] > 1.81699603)
)
```

Expected behavior:

- Apply this removal only to the training set.
- Do not remove test rows.
- Save the removed training sample IDs to:

```text
removed_train_ids_criteria_a_conservative.csv
```

- Save the remaining cleaned training data to:

```text
train_cleaned_criteria_a_conservative.csv
```

- Save the quality report with flags to:

```text
train_quality_report_criteria_a_conservative.csv
```

Recommended quality flag column:

```text
remove_criteria_a_conservative
```

---

# Task 4: Implement Criteria B - Strict Noise Removal

Implement a function:

```python
def apply_criteria_b_strict(quality_df: pd.DataFrame, has_target: bool = False) -> pd.Series:
    ...
```

This function should return a boolean mask where `True` means the sample should be removed from the training set.

Criteria B includes all Criteria A rules and adds more aggressive spike rules.

First compute Criteria A:

```python
criteria_a_mask = apply_criteria_a_conservative(quality_df, has_target=has_target)
```

Then add strict rules:

```python
criteria_b = criteria_a_mask | (
    (quality_df["max_interp_resid_over_med_step"] > 1.5) |
    (quality_df["max_step_over_median"] > 3.0)
)
```

Expected behavior:

- Apply this removal only to the training set.
- Do not remove test rows.
- Save the removed training sample IDs to:

```text
removed_train_ids_criteria_b_strict.csv
```

- Save the remaining cleaned training data to:

```text
train_cleaned_criteria_b_strict.csv
```

- Save the quality report with flags to:

```text
train_quality_report_criteria_b_strict.csv
```

Recommended quality flag column:

```text
remove_criteria_b_strict
```

---

# Task 5: Test Data Handling

For test data:

Do not remove any rows because every test sample needs a prediction.

Instead, compute the same quality features and flags:

```text
remove_criteria_a_conservative
remove_criteria_b_strict
```

Save:

```text
test_quality_report_with_noise_flags.csv
```

These flags can be used later for:

- special inference handling
- interpolation-based cleaning
- blending original prediction and cleaned-trajectory prediction
- increasing physics-based extrapolation weight for suspicious samples

---

# Task 6: Optional Spike Correction for Test-Time Inference

Implement an optional trajectory cleaning function:

```python
def clean_single_point_spikes(pos: np.ndarray, threshold: float = 0.04374434) -> np.ndarray:
    ...
```

Logic:

```python
cleaned = pos.copy()

for t in range(1, 10):
    expected = 0.5 * (pos[t - 1] + pos[t + 1])
    resid = np.linalg.norm(pos[t] - expected)

    if resid > threshold:
        cleaned[t] = expected

return cleaned
```

Use this only for inference-time experiment.

Do not overwrite the original dataset.

Recommended approach:

1. Train model on original or cleaned training data.
2. Predict using original test trajectories.
3. Predict using cleaned test trajectories.
4. Blend predictions for suspicious test samples.

Example:

```python
final_pred = original_pred.copy()

suspicious = test_quality_df["remove_criteria_b_strict"].values

final_pred[suspicious] = (
    0.5 * original_pred[suspicious] +
    0.5 * cleaned_pred[suspicious]
)
```

The blend ratio should be tuned by cross-validation.

---

# Task 7: Validation Protocol

Evaluate three settings using the same cross-validation split:

```text
1. no removal
2. criteria_a_conservative removal
3. criteria_b_strict removal
```

For each setting, report:

```text
number of removed samples
R-Hit@1cm
mean Euclidean distance
median Euclidean distance
```

Do not assume strict removal is always better.

Expected behavior from a quick Ridge baseline:

```text
no removal: slightly lower
criteria_a_conservative: slightly better
criteria_b_strict: slightly better or similar
```

But final performance must be validated using the actual competition model.

---

# Task 8: Important Warnings

Do not remove samples only because they have:

```text
large turning angle
low straightness
large target distance from last observed point
```

These may be real mosquito motion patterns, not missing values.

Avoid using these as removal rules:

```python
turn_gt_120_count > 0
straightness < 0.132
target_dist_from_last > 0.107898
```

These features may be useful as model inputs, but they should not be used as direct removal conditions.

---

# Final Deliverables

Please implement:

```text
quality feature extraction
criteria A conservative removal
criteria B strict removal
test quality flagging
optional spike correction
validation comparison
```

Expected output files:

```text
train_quality_report_criteria_a_conservative.csv
train_quality_report_criteria_b_strict.csv
test_quality_report_with_noise_flags.csv

removed_train_ids_criteria_a_conservative.csv
removed_train_ids_criteria_b_strict.csv

train_cleaned_criteria_a_conservative.csv
train_cleaned_criteria_b_strict.csv

noise_removal_validation_results.csv
```

Keep the code modular and reusable.

The removal mode should be configurable:

```python
noise_removal_mode = None
noise_removal_mode = "criteria_a_conservative"
noise_removal_mode = "criteria_b_strict"
```
