#!/usr/bin/env python3
"""
Day 2 - Build VLM inputs:
  - keyframes.jsonl
  - labels_gt.jsonl
  - vlm_requests.jsonl
"""

import os
import json
import glob
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from collections import defaultdict

import numpy as np

from physai_vlm.taxonomy import make_prompt_tagging, make_prompt_judge


@dataclass
class Cfg:
    data_dir: str = "dataset_v2"
    frames_dir: str = "dataset_v2/frames"
    out_dir: str = "dataset_v2/vlm"
    keyframes_per_episode: int = 10


def load_records(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def stage_heuristic(
    dist_grip_to_obj: Optional[float],
    dist_obj_to_goal: Optional[float],
    is_success: bool,
) -> str:
    """
    FetchReach용 stage GT 휴리스틱.

    NOTE:
    - FetchReach-v4에서 achieved_goal은 보통 gripper 위치(=현재 상태),
      desired_goal은 목표 위치이므로 dist_obj_to_goal ~= dist(gripper, goal).
    - dist_grip_to_obj는 Reach에서는 의미가 약할 수 있어(0에 수렴하는 경우가 흔함).
    """
    if is_success:
        return "idle"
    if dist_obj_to_goal is None:
        return "unknown"

    # 거리 기반으로만 stage를 구분 (간단/설명가능)
    if dist_obj_to_goal > 0.15:
        return "approach"
    if dist_obj_to_goal > 0.05:
        return "reach"
    if dist_obj_to_goal > 0.01:
        return "align"
    return "idle"


def progress_from_dist(dist_obj_to_goal: Optional[float]) -> float:
    if dist_obj_to_goal is None:
        return 0.0
    max_dist = 0.30
    p = 1.0 - min(dist_obj_to_goal / max_dist, 1.0)
    return float(np.clip(p, 0.0, 1.0))


def pick_keyframes(frame_paths: List[str], k: int) -> List[str]:
    if len(frame_paths) == 0:
        return []
    frame_paths = sorted(frame_paths)
    if len(frame_paths) <= k:
        return frame_paths
    idx = np.linspace(0, len(frame_paths) - 1, k).round().astype(int)
    return [frame_paths[i] for i in idx]


def main():
    cfg = Cfg(
        data_dir=os.getenv("DATA_DIR", "dataset_v2"),
        frames_dir=os.getenv("FRAMES_DIR", "dataset_v2/frames"),
        out_dir=os.getenv("OUT_DIR", "dataset_v2/vlm"),
        keyframes_per_episode=int(os.getenv("KEYFRAMES_PER_EPISODE", "10")),
    )
    os.makedirs(cfg.out_dir, exist_ok=True)

    meta_path = os.path.join(cfg.data_dir, "episodes.jsonl")
    by_ep: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for r in load_records(meta_path):
        by_ep[r["episode_id"]].append(r)

    keyframes_out = os.path.join(cfg.out_dir, "keyframes.jsonl")
    labels_out = os.path.join(cfg.out_dir, "labels_gt.jsonl")
    req_out = os.path.join(cfg.out_dir, "vlm_requests.jsonl")

    with open(keyframes_out, "w", encoding="utf-8") as fk, \
         open(labels_out, "w", encoding="utf-8") as fl, \
         open(req_out, "w", encoding="utf-8") as fr:

        for ep, steps in by_ep.items():
            steps = sorted(steps, key=lambda x: x["t"])
            frame_paths = sorted(glob.glob(os.path.join(cfg.frames_dir, ep, "*.png")))
            kf_paths = pick_keyframes(frame_paths, cfg.keyframes_per_episode)

            fk.write(json.dumps({"episode_id": ep, "keyframes": kf_paths}, ensure_ascii=False) + "\n")

            step_by_t = {s["t"]: s for s in steps}
            frame_ts = [int(os.path.splitext(os.path.basename(p))[0]) for p in kf_paths]
            ep_T = int(steps[-1]["t"]) if steps else 0

            for p, t in zip(kf_paths, frame_ts):
                s = step_by_t.get(t, None)
                if s is None:
                    s = min(steps, key=lambda x: abs(x["t"] - t))

                is_s = bool(s.get("is_success_episode_so_far", False))
                dist_g2o = s.get("dist_grip_to_obj", None)
                dist_o2g = s.get("dist_obj_to_goal", None)

                gt = {
                    "episode_id": ep,
                    "t": int(s["t"]),
                    "image_path": p,
                    "is_success_episode_so_far": is_s,
                    "dist_grip_to_obj": dist_g2o,
                    "dist_obj_to_goal": dist_o2g,
                    "progress_gt": progress_from_dist(dist_o2g),
                    "stage_gt": stage_heuristic(dist_g2o, dist_o2g, is_s),
                }
                fl.write(json.dumps(gt, ensure_ascii=False) + "\n")

                # Build a tiny "clip" context for tagging (prev/current/next keyframe)
                # This helps the model detect slow_progress / oscillation better than 1 frame.
                try:
                    idx = kf_paths.index(p)
                except ValueError:
                    idx = 0
                clip_paths = []
                for j in [max(0, idx - 1), idx, min(len(kf_paths) - 1, idx + 1)]:
                    clip_paths.append(kf_paths[j])

                context = f"Context: timestep {int(s['t'])} / {ep_T}."
                tag_prompt = make_prompt_tagging() + "\n" + context
                judge_prompt = make_prompt_judge() + "\n" + context

                fr.write(json.dumps({
                    "type": "tagging",
                    "episode_id": ep,
                    "t": int(s["t"]),
                    "image_path": p,
                    "image_paths": clip_paths,
                    "prompt": tag_prompt,
                }, ensure_ascii=False) + "\n")

                fr.write(json.dumps({
                    "type": "judge",
                    "episode_id": ep,
                    "t": int(s["t"]),
                    "image_path": p,
                    "prompt": judge_prompt,
                }, ensure_ascii=False) + "\n")

    print("[DONE] Wrote:")
    print(" -", keyframes_out)
    print(" -", labels_out)
    print(" -", req_out)


if __name__ == "__main__":
    main()
