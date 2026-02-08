## Delivery Package

This package contains the end-to-end VLM evaluation pipeline and the final reports.
Large raw frames are excluded to keep the repo lightweight.

### Included outputs (evidence)
- `dataset_v2/vlm/vlm_outputs.jsonl` (1200 structured outputs, tagging 600 / judge 600)
- `dataset_v2/vlm/vlm_outputs_roi.jsonl` (ROI run, 1200 outputs)
- `results/summary.md`
- `results/summary_with_images.md` (image links expect local frames)
- `results/confusion_grid_report.md`
- `results/failure_taxonomy_report.md`
- `results/roi_ablation.md`
- `results/uncertain_samples.jsonl`

### Core scripts
- `run_vlm_requests.py` (Structured outputs + retry)
- `build_vlm_inputs.py`
- `collect_fetch_dataset_v2.py`
- `evaluate_judge.py`, `evaluate_success_auc.py`, `evaluate_success_pr.py`
- `report_taxonomy.py`, `report_confusion_grid.py`, `summarize_results*.py`
- `compare_roi_effect.py`, `extract_uncertain_samples.py`, `roi_center_crop.py`

### Notes
- Raw frames are excluded via `.gitignore`.
- If you want image-embedded reports to render, keep local frames under:
  - `dataset_v2/frames/`
  - `dataset_v2/frames_roi/`

### Quick reproduction
```bash
conda activate physai_vlm
python run_vlm_requests.py
python validate_vlm_outputs.py
python evaluate_judge.py
python evaluate_success_auc.py
python evaluate_success_pr.py
python report_taxonomy.py
python report_confusion_grid.py
python extract_uncertain_samples.py
python summarize_results.py
python summarize_results_with_images.py
```
