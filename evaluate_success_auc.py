#!/usr/bin/env python3
"""
Evaluate p_success vs is_success for VLM judge outputs.

Requires:
  dataset_v2/vlm/labels_gt.jsonl
  dataset_v2/vlm/vlm_outputs.jsonl
"""

import json
import numpy as np


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def compute_auc(y_true, y_score):
    """
    Compute ROC AUC with rank-based method (no sklearn).
    """
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


def main():
    labels = {}
    for r in load_jsonl("dataset_v2/vlm/labels_gt.jsonl"):
        key = (r["episode_id"], r["t"], r["image_path"])
        labels[key] = r

    y_true = []
    y_score = []
    for r in load_jsonl("dataset_v2/vlm/vlm_outputs.jsonl"):
        if r.get("type") != "judge":
            continue
        key = (r["episode_id"], r["t"], r["image_path"])
        if key not in labels:
            continue
        out = r.get("output", {})
        if not isinstance(out, dict):
            continue
        p = out.get("p_success", None)
        if p is None:
            continue
        gt = labels[key]["is_success_episode_so_far"]
        y_true.append(1 if gt else 0)
        y_score.append(float(p))

    if not y_true:
        print("[WARN] No judge predictions found.")
        return

    auc = compute_auc(y_true, y_score)
    pos = sum(y_true)
    neg = len(y_true) - pos
    if auc is None:
        print(f"[WARN] AUC undefined (pos={pos}, neg={neg}).")
        return
    print(f"[Judge] p_success AUC={auc:.4f} (pos={pos}, neg={neg}, N={len(y_true)})")


if __name__ == "__main__":
    main()
