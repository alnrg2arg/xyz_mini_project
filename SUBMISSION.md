## 제출용 요약 (면접관용)

이 프로젝트는 **VLM을 실제 파이프라인에 연결**하여
데이터 정제/평가/리포팅까지 자동화한 미니 프로젝트입니다.

### 핵심 증거
- VLM 실제 호출 결과: `dataset_v2/vlm/vlm_outputs.jsonl` (1200 lines)
- 스키마 검증: `validate_vlm_outputs.py` → errors=0

### 주요 지표
- progress MAE / RMSE: **0.2927 / 0.3555**
- p_success AUC: **0.4655**
- ROI ablation: AUC **+0.0219** (상세: `results/roi_ablation.md`)

### 면접관이 바로 보면 좋은 파일 3개
1. `results/summary_with_images.md`
2. `results/confusion_grid_report.md`
3. `results/roi_ablation.md`

### 한 줄 설명
“VLM을 정책이 아니라 **데이터 정제/평가 레이어**에 붙여
현장 로그의 품질과 반복 속도를 높이는 파이프라인을 구현했습니다.”
