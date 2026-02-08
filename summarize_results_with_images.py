#!/usr/bin/env python3
"""
Generate a summary report with thumbnail examples.
Requires: dataset_v2/vlm/vlm_outputs.jsonl
"""

import json
import os
from collections import defaultdict


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def main():
    out_path = os.getenv("OUT_MD", "results/summary_with_images.md")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    outputs_path = os.getenv("OUTPUTS_PATH", "dataset_v2/vlm/vlm_outputs.jsonl")
    if not os.path.exists(outputs_path):
        print("[WARN] No VLM outputs found:", outputs_path)
        return

    examples = defaultdict(list)  # failure_type -> list of image paths

    for r in load_jsonl(outputs_path):
        if r.get("type") != "tagging":
            continue
        out = r.get("output", {})
        if not isinstance(out, dict):
            continue
        ft = out.get("failure_type", "unknown")
        img = r.get("image_path")
        if img and len(examples[ft]) < 3:
            examples[ft].append(img)

    lines = []
    lines.append("# Summary with Examples\n")
    lines.append("## Failure Type Examples\n")

    for ft, imgs in examples.items():
        lines.append(f"\n### {ft}\n")
        for p in imgs:
            rel = os.path.relpath(p, start=os.path.dirname(out_path))
            lines.append(f"![{ft}]({rel})")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("[DONE] wrote", out_path)


if __name__ == "__main__":
    main()
