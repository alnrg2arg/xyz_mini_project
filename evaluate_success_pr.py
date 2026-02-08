#!/usr/bin/env python3
"""
Compute precision/recall at multiple thresholds for p_success.
"""

import json
import os
import numpy as np


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def main():
    outputs_path = os.getenv("OUTPUTS_PATH", "dataset_v2/vlm/vlm_outputs.jsonl")
    labels = {}
    for r in load_jsonl("dataset_v2/vlm/labels_gt.jsonl"):
        key = (r["episode_id"], r["t"], r["image_path"])
        labels[key] = r

    y_true = []
    y_score = []
    for r in load_jsonl(outputs_path):
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

    y_true = np.array(y_true, dtype=int)
    y_score = np.array(y_score, dtype=float)

    thresholds = [0.1 * i for i in range(1, 10)]
    print("threshold,precision,recall,positives")
    for th in thresholds:
        pred = (y_score >= th).astype(int)
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        fn = int(((pred == 0) & (y_true == 1)).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        positives = int(pred.sum())
        print(f"{th:.1f},{precision:.3f},{recall:.3f},{positives}")


if __name__ == "__main__":
    main()
