#!/usr/bin/env python3
"""
Compare baseline vs ROI VLM outputs and write roi_ablation.md.
"""

import json
import os
import numpy as np


BASE_OUTPUTS = os.getenv("BASE_OUTPUTS", "dataset_v2/vlm/vlm_outputs.jsonl")
ROI_OUTPUTS = os.getenv("ROI_OUTPUTS", "dataset_v2/vlm/vlm_outputs_roi.jsonl")
LABELS_PATH = os.getenv("LABELS_PATH", "dataset_v2/vlm/labels_gt.jsonl")
OUT_MD = os.getenv("OUT_MD", "results/roi_ablation.md")


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def compute_auc(y_true, y_score):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_score = np.asarray(y_score, dtype=np.float64)
    pos = y_true == 1
    neg = y_true == 0
    n_pos = pos.sum()
    n_neg = neg.sum()
    if n_pos == 0 or n_neg == 0:
        return None
    order = np.argsort(y_score)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(y_score) + 1)
    sum_ranks_pos = ranks[pos].sum()
    auc = (sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


def metrics(outputs_path, labels):
    preds = []
    success = []
    for r in load_jsonl(outputs_path):
        if r.get("type") != "judge":
            continue
        key = (r["episode_id"], r["t"])
        if key not in labels:
            continue
        out = r.get("output", {})
        if not isinstance(out, dict):
            continue
        p = out.get("p_success", None)
        prog = out.get("progress", None)
        if prog is None or p is None:
            continue
        gt = labels[key]["progress_gt"]
        gt_s = 1 if labels[key].get("is_success_episode_so_far") else 0
        preds.append((gt, float(prog)))
        success.append((gt_s, float(p)))

    if not preds:
        return None

    gts = np.array([x[0] for x in preds], dtype=float)
    prs = np.array([x[1] for x in preds], dtype=float)
    mae = float(np.mean(np.abs(gts - prs)))
    rmse = float(np.sqrt(np.mean((gts - prs) ** 2)))

    y_true = np.array([x[0] for x in success], dtype=int)
    y_score = np.array([x[1] for x in success], dtype=float)
    auc = compute_auc(y_true, y_score)

    best_f1 = 0.0
    best_th = 0.5
    for th in [0.1 * i for i in range(1, 10)]:
        pred = (y_score >= th).astype(int)
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        fn = int(((pred == 0) & (y_true == 1)).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        if f1 > best_f1:
            best_f1 = f1
            best_th = th

    return {"mae": mae, "rmse": rmse, "auc": auc, "best_f1": best_f1, "best_th": best_th}


def main():
    if not os.path.exists(BASE_OUTPUTS) or not os.path.exists(ROI_OUTPUTS):
        print("[WARN] Missing outputs for baseline or ROI.")
        return

    labels = {}
    for r in load_jsonl(LABELS_PATH):
        key = (r["episode_id"], r["t"])
        labels[key] = r

    base = metrics(BASE_OUTPUTS, labels)
    roi = metrics(ROI_OUTPUTS, labels)
    if not base or not roi:
        print("[WARN] Not enough data to compare.")
        return

    lines = []
    lines.append("# ROI Ablation Report\n")
    lines.append("| Metric | Baseline | ROI | Δ (ROI - Base) |")
    lines.append("|---|---:|---:|---:|")
    for k in ["mae", "rmse", "auc", "best_f1"]:
        b = base[k]
        r = roi[k]
        delta = r - b if (b is not None and r is not None) else "n/a"
        if isinstance(delta, float):
            lines.append(f"| {k} | {b:.4f} | {r:.4f} | {delta:.4f} |")
        else:
            lines.append(f"| {k} | {b} | {r} | {delta} |")
    lines.append("")
    lines.append(f"- best_f1 threshold (baseline): {base['best_th']:.1f}")
    lines.append(f"- best_f1 threshold (roi): {roi['best_th']:.1f}")

    os.makedirs("results", exist_ok=True)
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("[DONE] wrote", OUT_MD)


if __name__ == "__main__":
    main()
