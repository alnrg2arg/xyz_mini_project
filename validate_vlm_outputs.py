#!/usr/bin/env python3
"""
Validate VLM outputs against expected schema.
"""

import json
import os


def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def main():
    path = os.getenv("OUTPUTS_PATH", "dataset_v2/vlm/vlm_outputs.jsonl")
    required_common = {"type", "episode_id", "t", "image_path", "output"}
    required_tagging = {"stage", "failure_type"}
    required_judge = {"p_success", "progress", "uncertainty"}

    total = 0
    ok = 0
    errors = 0
    type_counts = {"tagging": 0, "judge": 0}

    for r in load_jsonl(path):
        total += 1
        missing = required_common - set(r.keys())
        if missing:
            errors += 1
            continue

        rtype = r.get("type")
        out = r.get("output", {})
        if not isinstance(out, dict):
            errors += 1
            continue

        if rtype == "tagging":
            type_counts["tagging"] += 1
            if not required_tagging.issubset(out.keys()):
                errors += 1
                continue
        elif rtype == "judge":
            type_counts["judge"] += 1
            if not required_judge.issubset(out.keys()):
                errors += 1
                continue
        else:
            errors += 1
            continue

        ok += 1

    print(f"[VALIDATE] total={total}, ok={ok}, errors={errors}")
    print(f"[VALIDATE] tagging={type_counts['tagging']}, judge={type_counts['judge']}")


if __name__ == "__main__":
    main()
