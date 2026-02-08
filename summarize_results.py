#!/usr/bin/env python3
"""
Generate a one-page summary report for the mini-project.
"""

import json
import os
from collections import Counter


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def main():
    os.makedirs("results", exist_ok=True)
    out_path = os.getenv("OUT_MD", "results/summary.md")

    # Dataset stats
    dataset_path = os.getenv("DATASET_PATH", "dataset_v2/episodes.jsonl")
    frames_dir = "dataset_v2/frames"
    env_id = None
    eps = set()
    success_eps = set()
    for r in load_jsonl(dataset_path):
        env_id = env_id or r.get("env_id")
        eps.add(r["episode_id"])
        if r.get("is_success_episode_so_far"):
            success_eps.add(r["episode_id"])

    # VLM outputs stats
    outputs_path = os.getenv("OUTPUTS_PATH", "dataset_v2/vlm/vlm_outputs.jsonl")
    tag_counts = Counter()
    stage_counts = Counter()
    judge_count = 0

    if os.path.exists(outputs_path):
        for r in load_jsonl(outputs_path):
            if r.get("type") == "tagging":
                out = r.get("output", {})
                if isinstance(out, dict):
                    tag_counts[out.get("failure_type", "unknown")] += 1
                    stage_counts[out.get("stage", "unknown")] += 1
            elif r.get("type") == "judge":
                judge_count += 1

    lines = []
    lines.append("# Mini Project Summary\n")
    lines.append("## Dataset\n")
    lines.append(f"- env_id: {env_id}")
    lines.append(f"- episodes: {len(eps)}")
    lines.append(f"- success episodes: {len(success_eps)}")
    if os.path.isdir(frames_dir):
        frame_count = sum(
            len(files) for _, _, files in os.walk(frames_dir) if files
        )
        lines.append(f"- frames: {frame_count}")

    lines.append("\n## VLM Outputs\n")
    if os.path.exists(outputs_path):
        lines.append(f"- judge outputs: {judge_count}")
        if tag_counts:
            lines.append("- top failure types:")
            for k, v in tag_counts.most_common(5):
                lines.append(f"  - {k}: {v}")
        if stage_counts:
            lines.append("- top stages:")
            for k, v in stage_counts.most_common(5):
                lines.append(f"  - {k}: {v}")
    else:
        lines.append("- no VLM outputs found yet")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("[DONE] wrote", out_path)


if __name__ == "__main__":
    main()
