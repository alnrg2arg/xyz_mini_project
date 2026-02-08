#!/usr/bin/env python3
"""
Evaluate VLM judge outputs against GT progress labels.

Expected input:
  dataset_v2/vlm/labels_gt.jsonl
  dataset_v2/vlm/vlm_outputs.jsonl

Each line in vlm_outputs.jsonl should be:
{
  "type": "judge",
  "episode_id": "E0001",
  "t": 10,
  "image_path": "...png",
  "output": {
     "p_success": 0.5,
     "progress": 0.3,
     "uncertainty": 0.2
  }
}
"""

import json
import numpy as np


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def main():
    labels = {}
    for r in load_jsonl("dataset_v2/vlm/labels_gt.jsonl"):
        key = (r["episode_id"], r["t"], r["image_path"])
        labels[key] = r

    preds = []
    for r in load_jsonl("dataset_v2/vlm/vlm_outputs.jsonl"):
        if r.get("type") != "judge":
            continue
        key = (r["episode_id"], r["t"], r["image_path"])
        if key not in labels:
            continue
        gt = labels[key]["progress_gt"]
        out = r.get("output", {})
        pr = out.get("progress", None)
        if pr is None:
            continue
        preds.append((gt, float(pr)))

    if not preds:
        print("[WARN] No judge predictions found.")
        return

    gts = np.array([x[0] for x in preds], dtype=float)
    prs = np.array([x[1] for x in preds], dtype=float)

    mae = np.mean(np.abs(gts - prs))
    rmse = np.sqrt(np.mean((gts - prs) ** 2))
    print(f"[Judge] progress MAE={mae:.4f}, RMSE={rmse:.4f}, N={len(preds)}")


if __name__ == "__main__":
    main()
