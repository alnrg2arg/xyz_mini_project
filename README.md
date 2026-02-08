# Physical AI VLM Mini-Project (FetchReach)

Physical AI 면접을 위해 **VLM을 실제 파이프라인에 연결**한 미니 프로젝트입니다.
핵심은 정책 학습이 아니라 **데이터 정제/평가 레이어에 VLM을 붙여** 반복 속도를 높이는 것입니다.

## 내가 한 일 (요약)
- 시뮬 데이터를 수집하고 GT(progress/성공)를 구성
- 프레임 기반 VLM 태깅 + judge를 **structured outputs로 안정화**
- VLM 출력(1200 JSON, 에러 0)을 정량 평가 및 리포트로 연결
- ROI 적용 전/후 성능을 ablation으로 비교

## 파이프라인 개요

```
[시뮬 데이터 수집] → [GT/키프레임] → [VLM 추론] → [평가/리포트]
     Day 1                 Day 2           Day 3         Day 4
```

## 디렉토리 구조 (핵심)

```
xyz/
├── physai_vlm/
│   ├── __init__.py
│   └── taxonomy.py                # Reach taxonomy + schemas
├── collect_fetch_dataset_v2.py     # FetchReach 데이터 수집
├── build_vlm_inputs.py             # keyframes/labels/requests 생성
├── run_vlm_requests.py             # VLM 실행 (structured outputs + retry)
├── evaluate_judge.py               # progress MAE/RMSE
├── evaluate_success_auc.py         # p_success AUC
├── evaluate_success_pr.py          # PR table
├── report_taxonomy.py              # failure taxonomy 리포트
├── report_confusion_grid.py        # TP/FP/TN/FN 리포트
├── extract_uncertain_samples.py    # 불확실성 상위 샘플
├── roi_center_crop.py              # ROI 프레임 생성
├── compare_roi_effect.py           # ROI ablation 리포트
├── dataset_v2/
│   ├── episodes.jsonl
│   ├── frames/                      # 원본 프레임 (git 제외)
│   ├── frames_roi/                  # ROI 프레임 (git 제외)
│   └── vlm/
│       ├── keyframes.jsonl
│       ├── labels_gt.jsonl
│       ├── vlm_requests.jsonl
│       ├── vlm_outputs.jsonl
│       └── vlm_outputs_roi.jsonl
└── results/
    ├── summary.md
    ├── summary_with_images.md
    ├── confusion_grid_report.md
    ├── failure_taxonomy_report.md
    ├── roi_ablation.md
    └── uncertain_samples.jsonl
```

## 핵심 증거 (면접용)
- **VLM 실제 호출**: `dataset_v2/vlm/vlm_outputs.jsonl` (1200 lines)
- **스키마 검증 0 에러**: `python validate_vlm_outputs.py`
- **평가/리포트 연동**: `evaluate_judge.py`, `report_*`, `summary_with_images.md`

## 결과 요약
- **progress MAE / RMSE**: 0.2927 / 0.3555  
- **p_success AUC**: 0.4655  
- **ROI ablation (Δ)**: AUC +0.0219, MAE +0.0213  
  (baseline/ROI 상세: `results/roi_ablation.md`)

## 예시 이미지 (TP/FP/TN)
![TP](results/assets/TP.png)
![FP](results/assets/FP.png)
![TN](results/assets/TN.png)

## 설치 및 실행

### 환경 설정
```bash
conda create -n physai_vlm python=3.10 -y
conda activate physai_vlm
pip install -r requirements.txt
```

### Day 1: 데이터 수집
```bash
python collect_fetch_dataset_v2.py
```

### Day 2: 입력 생성
```bash
python build_vlm_inputs.py
```

### Day 3: VLM 실행 (structured outputs)
```bash
export OPENAI_API_KEY="your-key"
export VLM_MODEL="gpt-4o-mini"
python run_vlm_requests.py
python validate_vlm_outputs.py
```

### Day 4: 평가/리포트
```bash
python evaluate_judge.py
python evaluate_success_auc.py
python evaluate_success_pr.py
python report_taxonomy.py
python report_confusion_grid.py
python extract_uncertain_samples.py
python summarize_results.py
python summarize_results_with_images.py
```

## Reach Taxonomy

| Type | 설명 |
|------|------|
| goal_mismatch | 목표 방향/목표 위치 불일치 |
| slow_progress | 진행이 정체됨 |
| oscillation | 진동/불안정 |
| occlusion | 시야 가림 |
| unknown | 미분류 |

Stage enum: `approach | reach | align | idle | unknown`

## 면접 포인트 (요약)

- "정책 학습이 아니라 데이터/평가 레이어에 VLM을 적용"
- "VLM 출력(1200 JSON, 에러 0)으로 정량 평가까지 연결"
- "ROI ablation으로 비용/성능 트레이드오프 제시"

## 제출용 파일 (면접관용)
- `results/summary_with_images.md`
- `results/confusion_grid_report.md`
- `results/roi_ablation.md`
