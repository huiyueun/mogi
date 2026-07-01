# Huiyu 모기 궤적 실험 정리

이 폴더는 `/home/huiyu/Workspace/summer` 레포 안에서 진행한 개인 실험 작업물을 팀 공용 파일과 분리해 모아둔 공간이다.

## 현재 최고 결과

| 제출 파일 | Public | Private |
| --- | ---: | ---: |
| `outputs/second_place_expert_blends/confirmed_ode_heavy_800_second200.csv` | 0.7034 | 0.7042 |

현재 최고 조합:

- 80%: GOH30 ODE-heavy
  - GRU 20%
  - ODE 60%
  - HyperPhysics 20%
- 20%: 2등 코드 최종 예측 복원값

## 폴더 구조

- `huiyu/experiments/`
  - 개인 실험 스크립트 모음
  - OOF-lite, GOH30 컴포넌트 예측, MoE 탐색, TCN/Transformer-lite, 2등 expert blend 포함
- `huiyu/docs/experiment_progress.md`
  - 전체 실험 흐름, 제출 점수, 실패/성공 결론 정리
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
- 가장 큰 개선은 LB에서 검증된 2등 코드 예측을 GOH30 ODE-heavy와 섞은 prediction blending에서 나왔다.
