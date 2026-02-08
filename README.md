# Physical AI VLM Mini-Project

<<<<<<< HEAD
VLM 기반 로봇 데이터 분석 프로젝트 (FetchReach)

## 프로젝트 목표

1. **VLM 로그 자동 태깅**: 프레임 기반 stage/failure_type 자동 분류
2. **Success/Reward Judge**: VLM 기반 성공 확률/진행도 판정
3. **ROI ablation**: ROI 적용 전/후 성능 비교
=======
XYZ 면접 준비를 위한 VLM 기반 로봇 데이터 분석 프로젝트

## 프로젝트 목표

1. **VLM 로그 자동 태깅**: 로봇 시연 영상을 VLM으로 분석하여 stage/failure type 자동 분류
2. **Success/Reward Judge**: VLM 기반 성공 여부 및 진행도 판정기 구현
>>>>>>> 4ce5fca (Deliver VLM pipeline package)

## 파이프라인 개요

```
<<<<<<< HEAD
[시뮬 데이터 수집] → [GT/키프레임] → [VLM 추론] → [평가/리포트]
     Day 1                 Day 2           Day 3         Day 4
=======
[시뮬 데이터 수집] → [프레임 추출] → [VLM 추론] → [분석/리포트]
     Day 1              Day 2          Day 3-4        Day 5-6
>>>>>>> 4ce5fca (Deliver VLM pipeline package)
```

## 디렉토리 구조

```
xyz/
<<<<<<< HEAD
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
=======
├── collect_sim_dataset.py   # 시뮬 데이터 수집 스크립트
├── vlm_tagger.py            # VLM 자동 태깅 (Day 3)
├── reward_judge.py          # Success/Reward 판정기 (Day 4)
├── analyze_results.py       # 분석 및 리포트 생성 (Day 5)
├── dataset/
│   ├── episodes.jsonl       # 에피소드 메타데이터
│   └── frames/              # 프레임 이미지들
│       └── E0000/
│           └── 000000.png
├── results/
│   ├── tags.jsonl           # VLM 태깅 결과
│   └── judge_scores.jsonl   # Reward judge 결과
└── reports/
    └── failure_analysis.md  # 실패 분석 리포트
>>>>>>> 4ce5fca (Deliver VLM pipeline package)
```

## 설치 및 실행

### 환경 설정
```bash
conda create -n physai_vlm python=3.10 -y
conda activate physai_vlm
pip install -r requirements.txt
```

### Day 1: 데이터 수집
```bash
<<<<<<< HEAD
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
=======
python collect_sim_dataset.py
```

## Failure Taxonomy (8가지)

| Type | 설명 |
|------|------|
| grasp_miss | 집기 실패 (정렬 실패) |
| slip_drop | 미끄러져 떨어짐 |
| collision | 장애물/테이블 충돌 |
| occlusion | 시야 가림 |
| wrong_target | 잘못된 물체/위치 |
| unstable_contact | 불안정한 접촉 |
| out_of_workspace | 리치 범위 초과 |
| unknown | 미분류 (triage 대상) |
>>>>>>> 4ce5fca (Deliver VLM pipeline package)

## 면접 포인트

- "정책 학습이 아니라 데이터/평가 레이어에 VLM을 적용"
<<<<<<< HEAD
- "VLM 출력(1200 JSON, 에러 0)으로 정량 평가까지 연결"
- "ROI ablation으로 비용/성능 트레이드오프 제시"
=======
- "실패 케이스 분류로 데이터 정제 룰 자동 생성"
- "RL reward 설계 비용을 VLM으로 절감"
>>>>>>> 4ce5fca (Deliver VLM pipeline package)
