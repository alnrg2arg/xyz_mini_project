#!/usr/bin/env python3
"""
Extract top-N most uncertain judge samples for human audit.
"""

import json
import os
from heapq import nlargest


IN_PATH = os.getenv("OUTPUTS_PATH", "dataset_v2/vlm/vlm_outputs.jsonl")
OUT_PATH = os.getenv("OUT_PATH", "results/uncertain_samples.jsonl")
TOP_N = 50


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def main():
    if not os.path.exists(IN_PATH):
        print("[WARN] No VLM outputs found:", IN_PATH)
        return

    candidates = []
    for r in load_jsonl(IN_PATH):
        if r.get("type") != "judge":
            continue
        out = r.get("output", {})
        if not isinstance(out, dict):
            continue
        unc = out.get("uncertainty", None)
        if unc is None:
            continue
        try:
            unc = float(unc)
        except Exception:
            continue
        candidates.append((unc, r))

    top = nlargest(TOP_N, candidates, key=lambda x: x[0])

    os.makedirs(os.path.dirname(OUT_PATH) or ".", exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for unc, r in top:
            rec = {
                "uncertainty": unc,
                "episode_id": r.get("episode_id"),
                "t": r.get("t"),
                "image_path": r.get("image_path"),
                "output": r.get("output"),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[DONE] wrote {OUT_PATH} (top {len(top)})")


if __name__ == "__main__":
    main()
