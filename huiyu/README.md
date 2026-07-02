# Huiyu 모기 궤적 실험 정리

이 폴더는 `/home/huiyu/Workspace/summer` 레포 안에서 진행한 개인 실험 작업물을 팀 공용 파일과 분리해 모아둔 공간이다.

## 현재 최고 결과

| 제출 파일 | Public | Private |
| --- | ---: | ---: |
| `outputs/learned_second_gating_grid060/fixed_050_second_final.csv` | 0.7052 | 0.7053 |

현재 최고 조합:

- 50%: GOH30 ODE-heavy
  - GRU 20%
  - ODE 60%
  - HyperPhysics 20%
- 50%: 2등 코드 최종 예측 복원값
  - second base 60%
  - second phys 40%

전체 비율로 풀면:

- 1등 GRU 10%
- 1등 ODE/RK 30%
- 1등 HyperPhysics 10%
- 2등 base 30%
- 2등 phys 20%

## 폴더 구조

- `huiyu/experiments/`
  - 개인 실험 스크립트 모음
  - OOF-lite, GOH30 컴포넌트 예측, MoE 탐색, TCN/Transformer-lite, 2등 expert blend 포함
- `huiyu/docs/`
  - `01_experiment_progress.md`: 전체 실험 흐름, 제출 점수, 실패/성공 결론 정리
  - `02_top_expert_comparison_view_guide.md`: 1등/2등 예측 차이 그림과 표를 보는 순서
  - `03_top_expert_cause_analysis.md`: 1등/2등 예측 차이가 생긴 원인 가설
  - `04_second_component_analysis.md`: 2등 base/phys/final 구성요소 분해 분석
  - `05_final_ensemble_conclusion.md`: 왜 1등 앙상블에 2등 앙상블을 더했는지 최종 결론
  - `06_learned_moe_gating_plan.md`: 궤적별 MoE 비율을 학습하는 후속 실험 계획과 실행 명령
  - `07_blend_ratio_behavior_analysis.md`: 50:50 블렌드가 좋아진 이유와 비율별 행동 변화 분석
  - `08_final_presentation_summary.md`: 최종 발표자료용 통합 요약
- `huiyu/figures/`
  - 발표자료와 문서에서 참조하는 분석 그림
- `huiyu/submissions/`
  - 주요 제출 CSV 로컬 복사본
  - 팀 레포가 무거워지지 않도록 CSV는 git 추적에서 제외

## 주요 실행 명령

2등 코드 예측 복원 및 blend 생성:

```bash
.venv-cu128/bin/python huiyu/experiments/blend_second_place_expert.py \
  --out-dir outputs/second_place_expert_blends
```

GOH30 컴포넌트 예측 및 ODE-heavy 후보 생성:

```bash
.venv-cu128/bin/python huiyu/experiments/goh30_component_submissions.py
```

TCN OOF-lite 실행:

```bash
.venv-cu128/bin/python huiyu/experiments/goh30_oof_lite.py \
  --folds 3 \
  --epochs 10 \
  --device cuda \
  --include-tcn \
  --out-dir outputs/goh30_oof_tcn10_cuda
```

## 핵심 결론

- GOH30 내부 수정 중 실제 LB로 개선이 전이된 것은 ODE-heavy뿐이었다.
- H-heavy, KMeans MoE, learned stacking, regime weighting, cluster feature injection은 LB 개선으로 이어지지 않았다.
- TCN과 Transformer-lite는 OOF에서는 좋아졌지만 LB에서는 개선되지 않았다.
- 가장 큰 개선은 LB에서 검증된 2등 코드 예측을 GOH30 ODE-heavy와 50:50으로 섞은 prediction blending에서 나왔다.
- 2등 내부에서 phys 비중을 줄인 후보가 실패했기 때문에, 회전물리 expert가 최종 개선에 실제로 기여한 것으로 해석한다.
