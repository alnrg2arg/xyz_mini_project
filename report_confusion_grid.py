#!/usr/bin/env python3
"""
Generate 2x2 grid report (TP/FP/TN/FN) with example frames.
"""

import json
import os
from collections import defaultdict


LABELS_PATH = "dataset_v2/vlm/labels_gt.jsonl"
OUTPUTS_PATH = "dataset_v2/vlm/vlm_outputs.jsonl"
OUT_MD = "results/confusion_grid_report.md"
THRESHOLD = 0.5
MAX_EXAMPLES_PER_CELL = 6


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def main():
    if not os.path.exists(LABELS_PATH) or not os.path.exists(OUTPUTS_PATH):
        print("[WARN] Missing labels or outputs.")
        return

    labels = {}
    for r in load_jsonl(LABELS_PATH):
        key = (r["episode_id"], r["t"], r["image_path"])
        labels[key] = r

    cells = defaultdict(list)  # cell -> list of records

    for r in load_jsonl(OUTPUTS_PATH):
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
        try:
            p = float(p)
        except Exception:
            continue

        gt = 1 if labels[key].get("is_success_episode_so_far") else 0
        pred = 1 if p >= THRESHOLD else 0
        if gt == 1 and pred == 1:
            cell = "TP"
        elif gt == 0 and pred == 1:
            cell = "FP"
        elif gt == 0 and pred == 0:
            cell = "TN"
        else:
            cell = "FN"

        if len(cells[cell]) < MAX_EXAMPLES_PER_CELL:
            cells[cell].append({
                "episode_id": r.get("episode_id"),
                "t": r.get("t"),
                "image_path": r.get("image_path"),
                "p_success": p,
            })

    os.makedirs("results", exist_ok=True)
    lines = []
    lines.append("# Confusion Grid Report\n")
    lines.append(f"- threshold: {THRESHOLD}\n")

    for cell in ["TP", "FP", "TN", "FN"]:
        lines.append(f"## {cell}\n")
        for ex in cells.get(cell, []):
            rel = os.path.relpath(ex["image_path"], start=os.path.dirname(OUT_MD))
            lines.append(f"- {ex['episode_id']} t={ex['t']} p={ex['p_success']:.2f}")
            lines.append(f"  - ![]({rel})")
        lines.append("")

    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("[DONE] wrote", OUT_MD)


if __name__ == "__main__":
    main()
