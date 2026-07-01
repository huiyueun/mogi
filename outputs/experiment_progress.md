# Mosquito Trajectory Experiment Progress

Date: 2026-07-01

## 0. Executive Summary

Current best confirmed submission:

| submission | Public | Private |
| --- | ---: | ---: |
| `outputs/second_place_expert_blends/confirmed_ode_heavy_800_second200.csv` | 0.7034 | 0.7042 |

Best confirmed pipeline:

- Start from GOH30 component predictions.
- Use ODE-heavy GOH30 blend:
  - GRU 20%
  - ODE 60%
  - HyperPhysics 20%
- Restore the 2nd-place prediction from `best_solve/_[private 2nd] 코드 공유.ipynb`.
- Final blend:
  - 80% ODE-heavy GOH30
  - 20% 2nd-place prediction

Main lessons:

- ODE-heavy was the only GOH30-internal change that transferred positively to LB.
- H-heavy, KMeans MoE, learned stacking, regime weighting, cluster feature injection, TCN, and Transformer-lite did not improve LB.
- The strongest improvement came from prediction blending with another LB-verified expert, not from additional local OOF tuning.

## 1. Baseline Reproduction

Base code:

- `best_solve/[Private_LB 1st] 코드 공유.ipynb`

Model structure:

- GOH30 = GRU 10 seeds + Neural ODE 10 seeds + HyperPhysics 10 seeds
- Original ensemble is equal-weighted over all 30 models.
- Architecture-level original weight:
  - GRU 33.3%
  - ODE 33.3%
  - HyperPhysics 33.3%

Generated base artifacts:

- `models_goh30/phaseG_full_0.pt` ... `phaseG_full_9.pt`
- `models_goh30/phaseODE_full_0.pt` ... `phaseODE_full_9.pt`
- `models_goh30/phaseH_full_0.pt` ... `phaseH_full_9.pt`
- `submission_GOH30.csv`

Committed:

- Commit `b9c6bd1 Add GOH30 base submission`
- Included only `submission_GOH30.csv`.
- Model weights and experiment outputs were intentionally left untracked.

Submitted baseline score:

| submission | Public | Private |
| --- | ---: | ---: |
| `submission_GOH30.csv` | 0.7020 | 0.7025 |

## 2. Constant Velocity / Kalman Baseline Experiment

Script:

- `experiments/turn_phase_residual_experiment.py`

Idea:

- Start from constant velocity:
  - `pred = last + 2 * (last - prev)`
- Add Kalman CV, residual models, turn/phase features, and regime gates.

OOF results:

| candidate | OOF hit |
| --- | ---: |
| constant velocity | 0.5788 |
| Kalman CV | 0.5964 |
| CV residual + turn/phase | 0.5969 |
| Kalman residual + turn/phase | 0.5985 |
| blend CV/Kalman residual | 0.6025 |
| blend + high_speed Kalman gate | 0.6124 |

Conclusion:

- Regime-based gating helped the weak CV baseline.
- The same Kalman high-speed idea did not meaningfully improve full GOH30.

## 3. Kalman Postprocessing on GOH30

Script:

- `experiments/goh30_regime_submissions.py`

Generated candidates:

- `outputs/goh30_regime_submissions/case00_goh30_original.csv`
- `case01_high_speed_kalman_005.csv`
- `case02_high_speed_kalman_010.csv`
- `case03_high_speed_kalman_015.csv`
- `case04_high_speed_kalman_020.csv`
- etc.

Submitted scores:

| submission | Public | Private |
| --- | ---: | ---: |
| original GOH30 | 0.7020 | 0.7025 |
| high_speed Kalman 5% | 0.7022 | 0.7026 |
| high_speed Kalman 10% | 0.7020 | 0.7021 |
| high_speed Kalman 15% | 0.7016 | 0.7018 |
| high_speed Kalman 20% | 0.7018 | 0.7019 |

Conclusion:

- Kalman blend has only a very weak signal.
- 5% was slightly better, but the effect is too small to rely on.
- Higher Kalman weights clearly hurt.

## 4. OOF-lite Infrastructure

Goal:

- Avoid using daily submissions as the primary experiment loop.
- Build a local OOF-lite validation pipeline to test ensemble and MoE ideas before submission.

Main script:

- `experiments/goh30_oof_lite.py`

Features:

- 3-fold OOF-lite
- GRU OOF prediction
- ODE OOF prediction
- Optional HyperPhysics OOF prediction via `--include-h`
- CV/Kalman OOF prediction
- Regime score report

GPU environment:

- Created `.venv-cu128`
- Installed PyTorch `2.11.0+cu128`
- Confirmed RTX 5070 support:
  - GPU capability `(12, 0)`
  - `sm_120` appears in `torch.cuda.get_arch_list()`

## 5. GRU/ODE OOF-lite Results

Command:

```bash
python experiments/goh30_oof_lite.py \
  --folds 3 \
  --epochs 10 \
  --device cuda \
  --out-dir outputs/goh30_oof_lite_gru_ode10_cuda
```

Key OOF results:

| candidate | OOF hit |
| --- | ---: |
| GRU | 0.6263 |
| ODE | 0.6316 |
| GRU/ODE 50:50 | 0.6298 |
| best global GRU/ODE weight, GRU15/ODE85 | 0.6319 |

Conclusion:

- ODE is stronger than GRU in this OOF-lite setup.
- ODE-heavy weighting is better than 50:50.

## 6. Full GOH30 Component Predictions

Script:

- `experiments/goh30_component_submissions.py`

Generated component predictions:

- `outputs/goh30_component_submissions/pred_gru.npy`
- `pred_ode.npy`
- `pred_h.npy`
- `pred_equal.npy`

Generated ODE-heavy submissions:

- `case01_ode_heavy_g25_o50_h25.csv`
- `case02_ode_heavy_g20_o60_h20.csv`
- `case03_ode_heavy_g15_o65_h20.csv`
- `case04_ode_heavy_g15_o70_h15.csv`
- etc.

Submitted score:

| submission | Public | Private |
| --- | ---: | ---: |
| `case02_ode_heavy_g20_o60_h20.csv` | 0.7018 | 0.7033 |

Conclusion:

- ODE-heavy signal transferred to Private LB.
- Public decreased slightly, but Private improved by +0.0008 over original.

## 7. PPT Ideas Implemented

Proposal ideas:

- Scene clustering
- Noise criteria
- Mixture of Experts
- Expert-specific weighting

Implemented so far:

| PPT idea | Implementation |
| --- | --- |
| Scene clustering | KMeans over trajectory features |
| Noise criteria | high_speed, high_noise, high_acc, hard_turn, vertical_change regimes |
| Mixture of Experts | GRU/ODE/H architecture weighting |
| Expert gating | Cluster-specific GRU/ODE/H weights |
| Stability check | Foldwise OOF and multi-seed OOF comparison |

## 8. Cluster MoE with GRU/ODE

Scripts:

- `experiments/search_cluster_moe.py`
- `experiments/make_cluster_moe_submissions.py`

GRU/ODE-only OOF-lite results:

| candidate | OOF hit |
| --- | ---: |
| ODE | 0.6316 |
| global GRU15/ODE85 | 0.6319 |
| KMeans5 cluster MoE | 0.6324 |
| KMeans6 cluster MoE | 0.6325 |
| KMeans8 cluster MoE | 0.6327 |

Conclusion:

- Cluster-based weighting improved over global ODE-heavy.
- This supported the PPT Scene clustering + MoE hypothesis.

## 9. H-included OOF-lite

Command:

```bash
python experiments/goh30_oof_lite.py \
  --folds 3 \
  --epochs 10 \
  --include-h \
  --h-epochs 6 \
  --device cuda \
  --out-dir outputs/goh30_oof_lite_gru_ode_h10_6_cuda
```

Key OOF results:

| candidate | OOF hit |
| --- | ---: |
| H only | 0.6533 |
| GOH-lite equal | 0.6432 |
| G20/O60/H20 | 0.6400 |
| G15/O65/H20 | 0.6403 |
| best H-heavy G10/O05/H85 | 0.6544 |

Foldwise for H-heavy:

| candidate | fold1 | fold2 | fold3 | all |
| --- | ---: | ---: | ---: | ---: |
| G10/O05/H85 | 0.6557 | 0.6646 | 0.6430 | 0.6544 |
| H only | 0.6548 | 0.6646 | 0.6406 | 0.6533 |
| GOH-lite equal | 0.6419 | 0.6526 | 0.6352 | 0.6432 |

Conclusion:

- H is much stronger in OOF-lite than GRU/ODE.
- H-heavy ensembles are much better than equal-weight GOH-lite.

## 10. H-included Cluster MoE

Script:

- `experiments/search_full_moe_from_oof.py`

Seed 42 OOF:

| candidate | OOF hit |
| --- | ---: |
| equal | 0.6432 |
| H only | 0.6533 |
| H-heavy G10/O05/H85 | 0.6544 |
| KMeans5 GOH MoE | 0.6563 |
| KMeans6 GOH MoE | 0.6565 |
| KMeans8 GOH MoE | 0.6570 |

Seed 42 foldwise:

| candidate | fold1 | fold2 | fold3 | all |
| --- | ---: | ---: | ---: | ---: |
| KMeans8 GOH MoE | 0.6563 | 0.6685 | 0.6463 | 0.6570 |
| KMeans6 GOH MoE | 0.6557 | 0.6676 | 0.6463 | 0.6565 |
| KMeans5 GOH MoE | 0.6554 | 0.6685 | 0.6451 | 0.6563 |
| H-heavy G10/O05/H85 | 0.6557 | 0.6646 | 0.6430 | 0.6544 |
| equal | 0.6419 | 0.6526 | 0.6352 | 0.6432 |

Seed 777 OOF:

| candidate | OOF hit |
| --- | ---: |
| equal | 0.6437 |
| H only | 0.6528 |
| H-heavy G10/O05/H85 | 0.6535 |
| KMeans5 GOH MoE | 0.6559 |
| KMeans6 GOH MoE | 0.6562 |
| KMeans8 GOH MoE | 0.6572 |

Seed 777 foldwise:

| candidate | fold1 | fold2 | fold3 | all |
| --- | ---: | ---: | ---: | ---: |
| KMeans8 GOH MoE | 0.6548 | 0.6661 | 0.6508 | 0.6572 |
| KMeans6 GOH MoE | 0.6548 | 0.6649 | 0.6490 | 0.6562 |
| KMeans5 GOH MoE | 0.6533 | 0.6655 | 0.6490 | 0.6559 |
| H-heavy G10/O05/H85 | 0.6515 | 0.6610 | 0.6481 | 0.6535 |
| equal | 0.6437 | not listed above | not listed above | 0.6437 |

Conclusion:

- KMeans8 GOH MoE is the strongest local candidate across two OOF seeds.
- The improvement over equal is large and fold-stable.
- This is currently the strongest PPT-aligned method.

## 11. Stability / Movement Report

Script:

- `experiments/stabilize_moe_candidates.py`

Generated stable candidates:

- `outputs/goh30_stable_moe_submissions/avg_seed42_seed777_k8.csv`
- `avg_seed777_k5_k6_k8.csv`
- `avg_seed42_k5_k6_k8.csv`
- `avg_all_k5_k6_k8_both_seeds.csv`
- `blend_seed777_k8_90_equal_10.csv`
- `blend_seed777_k8_80_equal_20.csv`
- `blend_avg_k8_90_equal_10.csv`

Seed42 vs seed777 K8 difference:

| comparison | mean shift | p90 shift | >1cm count |
| --- | ---: | ---: | ---: |
| seed42 K8 vs seed777 K8 | 0.000225 | 0.000555 | 0 |

Candidate shift vs original equal:

| candidate | mean shift | p90 shift | >1cm count |
| --- | ---: | ---: | ---: |
| seed777 K8 | 0.001200 | 0.002340 | 28 |
| avg seed42/777 K8 | 0.001236 | 0.002373 | 26 |
| blend seed777 K8 90% + equal 10% | 0.001080 | 0.002106 | 17 |
| blend seed777 K8 80% + equal 20% | 0.000960 | 0.001872 | 9 |

Conclusion:

- K8 MoE is seed-stable.
- The candidate does not move predictions aggressively relative to equal GOH30.
- Averaging seed42 and seed777 K8 is a reasonable stability-oriented candidate.

## 12. Learned OOF Stacking

Script:

- `experiments/learned_oof_stacking.py`

Goal:

- Learn per-sample GRU/ODE/H weights from OOF predictions and trajectory features.
- Validate the meta model with an additional internal stack-fold split, not by fitting directly on all OOF labels.
- Compare seed42 and seed777 OOF runs for stability.

Implemented stackers:

- Logistic regression classifier over the best expert label.
- HistGradientBoosting classifier over the best expert label.
- Predicted class probabilities are used as GRU/ODE/H weights.
- Weights are blended with stable priors such as global H-heavy and H85.

Command:

```bash
.venv-cu128/bin/python experiments/learned_oof_stacking.py \
  --oof-dirs \
    outputs/goh30_oof_lite_gru_ode_h10_6_cuda \
    outputs/goh30_oof_lite_gru_ode_h10_6_seed777_cuda \
  --out-dir outputs/goh30_learned_stacking
```

Best learned stacking stability:

| candidate | mean OOF hit | std | min | max |
| --- | ---: | ---: | ---: | ---: |
| HGB global prior alpha 25% | 0.65485 | 0.00035 | 0.6546 | 0.6551 |
| HGB H85 prior alpha 50% | 0.65460 | 0.00014 | 0.6545 | 0.6547 |
| HGB H75 prior alpha 25% | 0.65445 | 0.00021 | 0.6543 | 0.6546 |
| global H-heavy best | 0.65440 | 0.00000 | 0.6544 | 0.6544 |

Comparison against existing best:

| method | seed42 OOF | seed777 OOF | conclusion |
| --- | ---: | ---: | --- |
| global H-heavy | 0.6544 | 0.6544 | strong simple baseline |
| learned stacking best | 0.6551 | 0.6546 | small, stable gain over H-heavy |
| KMeans8 GOH MoE | 0.6570 | 0.6572 | still stronger than learned stacking |

Generated learned stacking candidates:

- `outputs/goh30_learned_stacking/avg_seeds_hgb_global_a025.csv`
- `outputs/goh30_learned_stacking/avg_seeds_hgb_h85_a050.csv`
- `outputs/goh30_learned_stacking/avg_seeds_hgb_h75_a025.csv`

Shift report:

| candidate | mean shift vs equal | p90 shift | >1cm count |
| --- | ---: | ---: | ---: |
| avg HGB global alpha25 | 0.001040 | 0.001998 | 28 |
| avg HGB H85 alpha50 | 0.000941 | 0.001795 | 31 |
| avg HGB H75 alpha25 | 0.000943 | 0.001814 | 20 |

Conclusion:

- Learned stacking is valid and stable, but the gain is smaller than cluster MoE.
- It should not replace K8 cluster GOH MoE as the main submission candidate yet.
- It can be kept as a fallback or used later as an additional feature/candidate inside a larger ensemble.

## 13. Current Best Submission Candidates

Known LB-confirmed improvement:

1. `outputs/goh30_component_submissions/case02_ode_heavy_g20_o60_h20.csv`
   - Public 0.7018
   - Private 0.7033

Strongest local OOF candidate:

1. `outputs/goh30_full_moe_seed777_submissions/k8_cluster_goh_moe.csv`
   - Seed777 local OOF 0.6572

Most stable local candidate:

1. `outputs/goh30_stable_moe_submissions/avg_seed42_seed777_k8.csv`
   - K8 is strong in both OOF seeds.
   - Seed42/777 K8 predictions are very close.
   - Submitted score later showed this did not transfer:
     - Public 0.6966
     - Private 0.6965
   - This invalidates K8/H-heavy MoE as a submission-priority method.

More conservative stable candidate:

1. `outputs/goh30_stable_moe_submissions/blend_avg_k8_90_equal_10.csv`
   - Slightly closer to original equal GOH30.

Recommendation if only one future submission is allowed:

- Do not submit K8/H-heavy MoE again.
- The strongest LB-confirmed candidate remains `outputs/goh30_component_submissions/case02_ode_heavy_g20_o60_h20.csv`.

Recommendation if two future submissions are allowed:

1. Re-test only conservative ODE-heavy or near-original blends.
2. Avoid H-heavy/K8 cluster candidates unless a better validation scheme confirms them.

Learned stacking fallback:

1. `outputs/goh30_learned_stacking/avg_seeds_hgb_global_a025.csv`
   - Stable, but weaker than K8 cluster MoE in local OOF.

## 14. Implemented Scripts

Core OOF / prediction:

- `experiments/goh30_oof_lite.py`
- `experiments/goh30_component_submissions.py`
- `experiments/train_tcn_submissions.py`

Search / MoE:

- `experiments/search_oof_blends.py`
- `experiments/search_cluster_moe.py`
- `experiments/search_goh_lite_weights.py`
- `experiments/search_full_moe_from_oof.py`
- `experiments/learned_oof_stacking.py`

Submission generation:

- `experiments/make_cluster_moe_submissions.py`
- `experiments/make_h_heavy_submissions.py`
- `experiments/stabilize_moe_candidates.py`
- `experiments/stabilize_tcn_candidates.py`
- `experiments/blend_second_place_expert.py`

Reporting:

- `experiments/foldwise_oof_report.py`

## 15. Remaining Ideas

Still not implemented:

- Full 5-fold GOH30 OOF.
- H 12-epoch OOF-lite to more closely match full H training.
- Clean standalone runner that executes GOH30 + 2nd-place expert + final blend end to end.
- Optional: recreate 3rd/9th/12th-place predictions as additional LB-verified experts.

## 16. TCN Expert

Motivation:

- Previous PPT-style postprocessing and cluster gating did not transfer to LB.
- Add a new expert with a different inductive bias instead of reweighting H/cluster heavily.
- TCN reads short local temporal patterns with dilated Conv1D blocks.

OOF-lite command:

```bash
.venv-cu128/bin/python experiments/goh30_oof_lite.py \
  --folds 3 \
  --epochs 10 \
  --device cuda \
  --include-tcn \
  --out-dir outputs/goh30_oof_tcn10_cuda
```

OOF-lite results:

| candidate | OOF hit |
| --- | ---: |
| TCN | 0.6425 |
| ODE 90% + TCN 10% | 0.6332 |
| ODE 95% + TCN 5% | 0.6331 |
| G15/O75/TCN10 | 0.6331 |
| ODE-heavy no H | 0.6319 |
| ODE | 0.6316 |

Conclusion:

- TCN is the first new expert with a clear positive OOF signal.
- It should be blended conservatively with the LB-confirmed ODE-heavy candidate.

Full TCN training:

```bash
.venv-cu128/bin/python experiments/train_tcn_submissions.py \
  --device cuda \
  --epochs 30 \
  --seeds 5 \
  --models-dir models_tcn_e30_s5 \
  --out-dir outputs/goh30_tcn_e30_s5_submissions

.venv-cu128/bin/python experiments/train_tcn_submissions.py \
  --device cuda \
  --epochs 30 \
  --seeds 5 \
  --seed-offset 4000 \
  --models-dir models_tcn_e30_s5_seed4000 \
  --out-dir outputs/goh30_tcn_e30_s5_seed4000_submissions
```

Stabilized TCN candidates:

Script:

- `experiments/stabilize_tcn_candidates.py`

Generated:

- `outputs/goh30_tcn_stable_submissions/confirmed_ode_heavy_975_tcn025.csv`
- `outputs/goh30_tcn_stable_submissions/confirmed_ode_heavy_950_tcn050.csv`
- `outputs/goh30_tcn_stable_submissions/confirmed_ode_heavy_900_tcn100.csv`
- `outputs/goh30_tcn_stable_submissions/confirmed_ode_heavy_850_tcn150.csv`

Shift vs LB-confirmed ODE-heavy:

| candidate | mean shift | p90 shift | max shift | >1cm count |
| --- | ---: | ---: | ---: | ---: |
| ODE-heavy 97.5% + avg TCN 2.5% | 0.000028 | 0.000053 | 0.001067 | 0 |
| ODE-heavy 95% + avg TCN 5% | 0.000057 | 0.000107 | 0.002134 | 0 |
| ODE-heavy 90% + avg TCN 10% | 0.000114 | 0.000214 | 0.004268 | 0 |
| ODE-heavy 85% + avg TCN 15% | 0.000170 | 0.000320 | 0.006402 | 0 |

Current TCN submission recommendation:

1. `outputs/goh30_tcn_stable_submissions/confirmed_ode_heavy_900_tcn100.csv`
2. More conservative: `outputs/goh30_tcn_stable_submissions/confirmed_ode_heavy_950_tcn050.csv`

Submitted TCN result:

| submission | Public | Private |
| --- | ---: | ---: |
| `confirmed_ode_heavy_900_tcn100.csv` | 0.7022 | 0.7030 |

Conclusion:

- TCN did not collapse like K8/H-heavy, but it also did not improve over the LB-confirmed ODE-heavy result.
- Current best confirmed result remains `case02_ode_heavy_g20_o60_h20.csv` at Private 0.7033.
- Do not spend more limited submissions on TCN 10%.

## 17. 2nd Place Expert Blend

Script:

- `experiments/blend_second_place_expert.py`

Source:

- `best_solve/_[private 2nd] 코드 공유.ipynb`
- The notebook contains compressed base and physics test predictions.
- Restored final 2nd-place prediction:
  - Public 0.7022
  - Private 0.7031

Generated:

- `outputs/second_place_expert_blends/second_place_restored.csv`
- `outputs/second_place_expert_blends/confirmed_ode_heavy_975_second025.csv`
- `outputs/second_place_expert_blends/confirmed_ode_heavy_950_second050.csv`
- `outputs/second_place_expert_blends/confirmed_ode_heavy_900_second100.csv`
- `outputs/second_place_expert_blends/confirmed_ode_heavy_850_second150.csv`
- `outputs/second_place_expert_blends/confirmed_ode_heavy_800_second200.csv`

Shift vs LB-confirmed ODE-heavy:

| candidate | mean shift | p90 shift | max shift | >1cm count |
| --- | ---: | ---: | ---: | ---: |
| 2nd-place restored | 0.001675 | 0.003132 | 0.017793 | 74 |
| ODE-heavy 97.5% + 2nd 2.5% | 0.000042 | 0.000078 | 0.000445 | 0 |
| ODE-heavy 95% + 2nd 5% | 0.000084 | 0.000157 | 0.000890 | 0 |
| ODE-heavy 90% + 2nd 10% | 0.000168 | 0.000313 | 0.001779 | 0 |
| ODE-heavy 85% + 2nd 15% | 0.000251 | 0.000470 | 0.002669 | 0 |
| ODE-heavy 80% + 2nd 20% | 0.000335 | 0.000627 | 0.003559 | 0 |

Conclusion:

- This is more credible than TCN/Transformer because the 2nd-place expert is LB-verified.
- However, it is slightly lower than our best confirmed ODE-heavy result, so use only as a small diversification blend.
- Submitted 20% blend:

| submission | Public | Private |
| --- | ---: | ---: |
| `confirmed_ode_heavy_800_second200.csv` | 0.7034 | 0.7042 |

Current confirmed best:

- `outputs/second_place_expert_blends/confirmed_ode_heavy_800_second200.csv`
- Public 0.7034 / Private 0.7042

Next exploration candidates if submissions remain:

1. `outputs/second_place_expert_blends/confirmed_ode_heavy_750_second250.csv`
2. `outputs/second_place_expert_blends/confirmed_ode_heavy_700_second300.csv`

Avoid jumping straight to 40-50% unless submissions are cheap, because 2nd-place restored is slightly weaker than the ODE-heavy base.
