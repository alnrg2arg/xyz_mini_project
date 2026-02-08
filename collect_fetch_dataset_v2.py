#!/usr/bin/env python3
"""
Day 2 - FetchPickAndPlace dataset collection

Goals:
1) Mix scripted + random episodes to get some successes.
2) Log minimal observation parts for progress labels.
3) Save frames for VLM tagging/judge.
"""

import os
import json
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple

import numpy as np
import gymnasium as gym
import imageio.v2 as imageio
from tqdm import tqdm

try:
    import gymnasium_robotics
    HAS_ROBOTICS = True
except ImportError:
    HAS_ROBOTICS = False


@dataclass
class Cfg:
    out_dir: str = "dataset_v2"
    env_id: str = "FetchReach-v4"
    num_episodes: int = 60
    max_steps: int = 200
    fps_sample: int = 2
    seed: int = 0
    scripted_ratio: float = 0.8
    render_mode: str = "rgb_array"


def ensure_dirs(cfg: Cfg) -> str:
    os.makedirs(cfg.out_dir, exist_ok=True)
    frames_root = os.path.join(cfg.out_dir, "frames")
    os.makedirs(frames_root, exist_ok=True)
    return frames_root


def to_uint8_rgb(frame: np.ndarray) -> Optional[np.ndarray]:
    if frame is None:
        return None
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    if frame.shape[-1] == 4:
        frame = frame[..., :3]
    return frame


def extract_is_success(info: Dict[str, Any]) -> Optional[bool]:
    if "is_success" in info:
        v = info["is_success"]
        if isinstance(v, (bool, np.bool_)):
            return bool(v)
        if isinstance(v, (int, float, np.integer, np.floating)):
            return bool(v > 0.5)
    return None


def get_obs_parts(obs: Any) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    if isinstance(obs, dict):
        ag = obs.get("achieved_goal", None)
        dg = obs.get("desired_goal", None)
        o = obs.get("observation", None)
        return o, ag, dg
    return None, None, None


def scripted_policy(obs: Any, phase_state: Dict[str, Any]) -> np.ndarray:
    """
    Simple heuristic controller for FetchPickAndPlace:
    - phase 0: move gripper to object
    - phase 1: close gripper
    - phase 2: move to goal
    - phase 3: open gripper
    """
    o, ag, dg = get_obs_parts(obs)
    action = np.zeros(4, dtype=np.float32)

    if o is None or ag is None or dg is None:
        action[:3] = np.random.uniform(-0.05, 0.05, size=3)
        action[3] = -1.0
        return action

    gripper_pos = o[:3].copy()
    obj_pos = ag.copy()
    goal_pos = dg.copy()

    phase = phase_state.get("phase", 0)

    def move_towards(target, gain=8.0, clip=0.20):
        delta = target - gripper_pos
        delta = np.clip(gain * delta, -clip, clip)
        return delta

    approach_above = obj_pos.copy()
    approach_above[2] += 0.06
    approach_down = obj_pos.copy()
    approach_down[2] += 0.005

    dist_xy = np.linalg.norm((obj_pos - gripper_pos)[:2])
    dist_obj = np.linalg.norm(obj_pos - gripper_pos)
    dist_obj_to_goal = np.linalg.norm(obj_pos - goal_pos)

    if phase == 0:
        action[:3] = move_towards(approach_above)
        action[3] = 1.0
        if dist_xy < 0.03 and abs(gripper_pos[2] - approach_above[2]) < 0.02:
            phase_state["phase"] = 1
            phase_state["phase_t"] = 0
    elif phase == 1:
        action[:3] = move_towards(approach_down)
        action[3] = 1.0
        if dist_obj < 0.02:
            phase_state["phase"] = 2
            phase_state["phase_t"] = 0
    elif phase == 2:
        action[:3] = 0.0
        action[3] = -1.0
        phase_state["phase_t"] = phase_state.get("phase_t", 0) + 1
        if phase_state["phase_t"] >= 10:
            phase_state["phase"] = 3
            phase_state["phase_t"] = 0
    elif phase == 3:
        lift_target = gripper_pos.copy()
        lift_target[2] = obj_pos[2] + 0.10
        action[:3] = move_towards(lift_target)
        action[3] = -1.0
        if gripper_pos[2] > obj_pos[2] + 0.08:
            phase_state["phase"] = 4
            phase_state["phase_t"] = 0
    elif phase == 4:
        goal_target = goal_pos.copy()
        goal_target[2] += 0.05
        action[:3] = move_towards(goal_target)
        action[3] = -1.0
        if dist_obj_to_goal < 0.05:
            phase_state["phase"] = 5
            phase_state["phase_t"] = 0
    else:
        action[:3] = 0.0
        action[3] = 1.0
        phase_state["phase_t"] = phase_state.get("phase_t", 0) + 1

    return action


def main():
    cfg = Cfg()

    if not HAS_ROBOTICS:
        raise ImportError("gymnasium-robotics is required: pip install gymnasium-robotics")

    gymnasium_robotics.register_robotics_envs()
    frames_root = ensure_dirs(cfg)
    meta_path = os.path.join(cfg.out_dir, "episodes.jsonl")

    env = gym.make(cfg.env_id, render_mode=cfg.render_mode)
    rng = np.random.default_rng(cfg.seed)

    stats = {"success_eps": 0, "total_eps": 0, "frames": 0, "scripted_eps": 0}

    with open(meta_path, "w", encoding="utf-8") as f:
        for ep in tqdm(range(cfg.num_episodes), desc="episodes"):
            episode_id = f"E{ep:04d}"
            ep_dir = os.path.join(frames_root, episode_id)
            os.makedirs(ep_dir, exist_ok=True)

            obs, info = env.reset(seed=cfg.seed + ep)
            use_scripted = bool(rng.random() < cfg.scripted_ratio)
            if use_scripted:
                stats["scripted_eps"] += 1
            phase_state = {"phase": 0, "phase_t": 0}
            success_ep = False

            for t in range(cfg.max_steps):
                if use_scripted:
                    action = scripted_policy(obs, phase_state)
                else:
                    action = env.action_space.sample()

                obs, reward, terminated, truncated, info = env.step(action)
                done = bool(terminated or truncated)

                is_success = extract_is_success(info)
                if is_success is True:
                    success_ep = True

                o, ag, dg = get_obs_parts(obs)
                dist_obj_to_goal = None
                dist_grip_to_obj = None
                if o is not None and ag is not None and dg is not None:
                    gripper_pos = o[:3]
                    obj_pos = ag
                    goal_pos = dg
                    dist_obj_to_goal = float(np.linalg.norm(obj_pos - goal_pos))
                    dist_grip_to_obj = float(np.linalg.norm(gripper_pos - obj_pos))

                frame = env.render()
                if (t % cfg.fps_sample) == 0 and frame is not None:
                    frame = to_uint8_rgb(np.asarray(frame))
                    imageio.imwrite(os.path.join(ep_dir, f"{t:06d}.png"), frame)
                    stats["frames"] += 1

                rec = {
                    "env_id": cfg.env_id,
                    "episode_id": episode_id,
                    "t": t,
                    "scripted": use_scripted,
                    "action": action.tolist(),
                    "reward": float(reward) if reward is not None else None,
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "done": done,
                    "is_success_step": is_success,
                    "is_success_episode_so_far": success_ep,
                    "dist_obj_to_goal": dist_obj_to_goal,
                    "dist_grip_to_obj": dist_grip_to_obj,
                    "achieved_goal": ag.tolist() if isinstance(ag, np.ndarray) else None,
                    "desired_goal": dg.tolist() if isinstance(dg, np.ndarray) else None,
                }
                f.write(json.dumps(rec) + "\n")

                if done:
                    break

            stats["total_eps"] += 1
            if success_ep:
                stats["success_eps"] += 1

    env.close()

    print("[DONE] Wrote:", meta_path)
    print(f"Total episodes: {stats['total_eps']}")
    print(f"Scripted episodes: {stats['scripted_eps']}")
    print(f"Success episodes: {stats['success_eps']}")
    print(f"Frames saved: {stats['frames']}")


if __name__ == "__main__":
    main()
