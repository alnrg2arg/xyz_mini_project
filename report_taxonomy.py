#!/usr/bin/env python3
"""
Generate failure taxonomy report from VLM outputs.
"""

import json
from collections import Counter, defaultdict
import os

IN_PATH = "dataset_v2/vlm/vlm_outputs.jsonl"
OUT_MD = "results/failure_taxonomy_report.md"


def main():
    counts = Counter()
    stage_counts = Counter()
    examples = defaultdict(list)

    with open(IN_PATH, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("type") != "tagging":
                continue
            out = r.get("output", {})
            if not isinstance(out, dict):
                continue
            ft = out.get("failure_type", "unknown")
            st = out.get("stage", "unknown")
            notes = out.get("notes", out.get("raw_text", "")) or ""

            counts[ft] += 1
            stage_counts[st] += 1
            if len(examples[ft]) < 5:
                examples[ft].append((r["episode_id"], r["t"], r["image_path"], notes))

    lines = []
    lines.append("# Failure Taxonomy Report\n")
    lines.append("## Failure type frequency\n")
    for ft, c in counts.most_common():
        lines.append(f"- **{ft}**: {c}")

    lines.append("\n## Stage frequency\n")
    for st, c in stage_counts.most_common():
        lines.append(f"- **{st}**: {c}")

    lines.append("\n## Examples (up to 5 each)\n")
    for ft, exs in examples.items():
        lines.append(f"\n### {ft}\n")
        for (ep, t, path, notes) in exs:
            lines.append(f"- {ep} t={t} | {path}")
            if notes:
                lines.append(f"  - note: {notes}")

    os.makedirs("results", exist_ok=True)
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("[DONE] wrote", OUT_MD)


if __name__ == "__main__":
    main()
