#!/usr/bin/env python3
"""
Physical AI VLM Mini-Project - Day 1
시뮬레이션 데이터 수집 파이프라인

목표: ManiSkill 환경에서 에피소드를 생성하고, 프레임 + 메타데이터를 저장

Usage:
    python collect_sim_dataset.py
    python collect_sim_dataset.py --num_episodes 50 --env_id "PickCube-v1"
"""

import os
import json
import argparse
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List

import numpy as np
from tqdm import tqdm

# Optional imports - will check availability
try:
    import gymnasium as gym
    HAS_GYM = True
except ImportError:
    HAS_GYM = False

try:
    import imageio.v2 as imageio
    HAS_IMAGEIO = True
except ImportError:
    try:
        import imageio
        HAS_IMAGEIO = True
    except ImportError:
        HAS_IMAGEIO = False

try:
    import mani_skill.envs  # ManiSkill 3.x
    HAS_MANISKILL = True
    MANISKILL_VERSION = 3
except ImportError:
    try:
        import mani_skill2.envs  # ManiSkill 2.x
        HAS_MANISKILL = True
        MANISKILL_VERSION = 2
    except ImportError:
        HAS_MANISKILL = False
        MANISKILL_VERSION = 0


# -----------------------
# Config
# -----------------------
@dataclass
class CollectConfig:
    """데이터 수집 설정"""
    out_dir: str = "dataset"
    num_episodes: int = 30
    max_steps: int = 200
    fps_sample: int = 1  # save every N steps (1 = every step)
    seed: int = 42
    env_id: Optional[str] = None  # None이면 자동 탐색
    render_mode: str = "rgb_array"
    obs_mode: str = "rgbd"  # ManiSkill: "state", "rgbd", "pointcloud"
    control_mode: Optional[str] = None
    image_size: int = 256  # 렌더링 이미지 크기


def check_dependencies() -> List[str]:
    """필요한 패키지 설치 여부 확인"""
    missing = []
    if not HAS_GYM:
        missing.append("gymnasium")
    if not HAS_IMAGEIO:
        missing.append("imageio")
    if not HAS_MANISKILL:
        missing.append("mani-skill (or mani-skill2)")
    return missing


def find_maniskill_env_id(prefer_keywords=("Pick", "Cube", "Place", "Lift", "Push")) -> Optional[str]:
    """
    ManiSkill 환경 ID 자동 탐색
    간단한 manipulation task 우선 선택
    """
    if not HAS_GYM:
        return None
    
    all_envs = list(gym.registry.keys())
    
    # ManiSkill 관련 환경만 필터링
    maniskill_envs = [e for e in all_envs if any(kw in e for kw in ["ManiSkill", "Pick", "Push", "Lift", "Place", "Cube"])]
    
    if not maniskill_envs:
        # Fallback: 일반적인 Gymnasium 환경
        fallback_envs = ["FetchPickAndPlace-v2", "FetchPush-v2", "FetchReach-v2"]
        for env in fallback_envs:
            if env in all_envs:
                return env
        return None
    
    # 선호 키워드 기반 점수 계산
    scored = []
    for env_id in maniskill_envs:
        score = sum(1 for kw in prefer_keywords if kw.lower() in env_id.lower())
        # v1이 보통 더 안정적
        if "-v1" in env_id:
            score += 0.5
        scored.append((score, env_id))
    
    scored.sort(reverse=True)
    return scored[0][1] if scored else maniskill_envs[0]


def safe_make_env(cfg: CollectConfig):
    """
    환경 생성 (ManiSkill 버전 자동 감지)
    """
    env_id = cfg.env_id or find_maniskill_env_id()
    
    if env_id is None:
        raise RuntimeError(
            "사용 가능한 환경을 찾을 수 없습니다.\n"
            "ManiSkill 설치를 확인해주세요: pip install mani-skill"
        )
    
    print(f"[INFO] 환경 ID: {env_id}")
    print(f"[INFO] ManiSkill 버전: {MANISKILL_VERSION if HAS_MANISKILL else 'N/A'}")
    
    # 기본 kwargs
    make_kwargs = {"render_mode": cfg.render_mode}
    
    # ManiSkill 전용 kwargs
    if HAS_MANISKILL and ("ManiSkill" in env_id or "Pick" in env_id or "Push" in env_id):
        if MANISKILL_VERSION >= 3:
            make_kwargs["obs_mode"] = cfg.obs_mode
            if cfg.control_mode:
                make_kwargs["control_mode"] = cfg.control_mode
        else:  # ManiSkill 2.x
            make_kwargs["obs_mode"] = cfg.obs_mode if cfg.obs_mode != "rgbd" else "rgb"
    
    # 환경 생성 시도
    try:
        env = gym.make(env_id, **make_kwargs)
        print(f"[INFO] 환경 생성 성공: {make_kwargs}")
    except TypeError as e:
        print(f"[WARN] kwargs 오류: {e}")
        print("[WARN] 기본 설정으로 재시도...")
        env = gym.make(env_id, render_mode=cfg.render_mode)
    
    return env_id, env


def ensure_dirs(cfg: CollectConfig) -> str:
    """필요한 디렉토리 생성"""
    frames_dir = os.path.join(cfg.out_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(cfg.out_dir, exist_ok=True)
    return frames_dir


def to_uint8_rgb(frame: np.ndarray) -> Optional[np.ndarray]:
    """프레임을 uint8 RGB로 변환"""
    if frame is None:
        return None
    
    frame = np.asarray(frame)
    
    # 정규화된 float (0~1) -> uint8
    if frame.dtype in [np.float32, np.float64]:
        if frame.max() <= 1.0:
            frame = (frame * 255).astype(np.uint8)
        else:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
    elif frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    
    # RGBA -> RGB
    if len(frame.shape) == 3 and frame.shape[-1] == 4:
        frame = frame[..., :3]
    
    return frame


def extract_image_from_obs(obs) -> Optional[np.ndarray]:
    """
    관측값에서 이미지 추출
    ManiSkill은 다양한 형태로 이미지를 제공
    """
    if isinstance(obs, np.ndarray) and len(obs.shape) == 3:
        return obs
    
    if isinstance(obs, dict):
        # ManiSkill 3.x 스타일
        if "sensor_data" in obs:
            sensor_data = obs["sensor_data"]
            for cam_name in ["base_camera", "hand_camera", "camera"]:
                if cam_name in sensor_data:
                    cam_data = sensor_data[cam_name]
                    if "rgb" in cam_data:
                        return cam_data["rgb"]
                    if "rgba" in cam_data:
                        return cam_data["rgba"]
        
        # ManiSkill 2.x 스타일
        for key in ["image", "rgb", "rgbd", "color", "pixels", "camera"]:
            if key in obs:
                img = obs[key]
                if isinstance(img, np.ndarray):
                    return img
                if isinstance(img, dict):
                    for sub_key in ["rgb", "color", "base_camera"]:
                        if sub_key in img:
                            return img[sub_key]
    
    return None


def extract_success(info: Dict[str, Any]) -> Optional[bool]:
    """
    info에서 성공 여부 추출
    다양한 환경에서 다른 키를 사용
    """
    success_keys = ["success", "is_success", "task_success", "episode_success", "achieved"]
    
    for key in success_keys:
        if key in info:
            val = info[key]
            if isinstance(val, (bool, np.bool_)):
                return bool(val)
            if isinstance(val, (int, float, np.integer, np.floating)):
                return bool(val > 0.5)
    
    return None


def jsonable(x) -> Any:
    """값을 JSON 직렬화 가능하게 변환"""
    if x is None:
        return None
    if isinstance(x, (bool, int, float, str)):
        return x
    if isinstance(x, (np.bool_, np.integer)):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, np.ndarray):
        if x.size <= 20:
            return x.tolist()
        return {"shape": list(x.shape), "dtype": str(x.dtype), "mean": float(x.mean())}
    return str(x)


def collect_dataset(cfg: CollectConfig):
    """
    메인 데이터 수집 함수
    """
    print("=" * 60)
    print("Physical AI VLM Mini-Project - Day 1")
    print("시뮬레이션 데이터 수집")
    print("=" * 60)
    
    # 의존성 체크
    missing = check_dependencies()
    if missing:
        print(f"\n[ERROR] 필요한 패키지가 설치되지 않았습니다: {missing}")
        print("\n설치 방법:")
        print("  conda create -n physai_vlm python=3.10 -y")
        print("  conda activate physai_vlm")
        print("  pip install -r requirements.txt")
        return False
    
    # 디렉토리 생성
    frames_root = ensure_dirs(cfg)
    print(f"\n[INFO] 출력 디렉토리: {cfg.out_dir}")
    print(f"[INFO] 프레임 저장: {frames_root}")
    
    # 환경 생성
    try:
        env_id, env = safe_make_env(cfg)
    except Exception as e:
        print(f"\n[ERROR] 환경 생성 실패: {e}")
        print("\n대안:")
        print("  1. ManiSkill 재설치: pip install mani-skill")
        print("  2. 다른 환경 지정: python collect_sim_dataset.py --env_id 'CartPole-v1'")
        return False
    
    # 초기 리셋
    try:
        obs, info = env.reset(seed=cfg.seed)
        print(f"[INFO] 환경 초기화 성공")
    except Exception as e:
        print(f"[WARN] seed 리셋 실패: {e}")
        obs, info = env.reset()
    
    # 메타데이터 파일
    meta_path = os.path.join(cfg.out_dir, "episodes.jsonl")
    
    # 통계 추적
    stats = {
        "total_steps": 0,
        "total_episodes": 0,
        "success_episodes": 0,
        "frames_saved": 0,
    }
    
    print(f"\n[INFO] 데이터 수집 시작: {cfg.num_episodes} 에피소드")
    print("-" * 40)
    
    with open(meta_path, "w", encoding="utf-8") as f:
        for ep in tqdm(range(cfg.num_episodes), desc="에피소드 수집"):
            episode_id = f"E{ep:04d}"
            ep_dir = os.path.join(frames_root, episode_id)
            os.makedirs(ep_dir, exist_ok=True)
            
            # 에피소드 리셋
            try:
                obs, info = env.reset(seed=cfg.seed + ep)
            except:
                obs, info = env.reset()
            
            ep_success_final = None
            ep_total_reward = 0.0
            
            for t in range(cfg.max_steps):
                # 랜덤 액션 (Day 1에서는 파이프라인 테스트용)
                action = env.action_space.sample()
                
                try:
                    obs, reward, terminated, truncated, info = env.step(action)
                except Exception as e:
                    print(f"\n[WARN] Step 오류 (ep={ep}, t={t}): {e}")
                    break
                
                done = bool(terminated or truncated)
                success = extract_success(info)
                if success is not None:
                    ep_success_final = success
                
                ep_total_reward += float(reward) if reward else 0.0
                
                # 프레임 렌더링 및 저장
                frame = None
                if (t % cfg.fps_sample) == 0:
                    # 방법 1: render() 사용
                    try:
                        frame = env.render()
                    except:
                        pass
                    
                    # 방법 2: 관측값에서 추출
                    if frame is None:
                        frame = extract_image_from_obs(obs)
                    
                    if frame is not None:
                        frame = to_uint8_rgb(frame)
                        if frame is not None and frame.size > 0:
                            frame_path = os.path.join(ep_dir, f"{t:06d}.png")
                            try:
                                imageio.imwrite(frame_path, frame)
                                stats["frames_saved"] += 1
                            except Exception as e:
                                print(f"\n[WARN] 프레임 저장 실패: {e}")
                
                # 메타데이터 기록
                record = {
                    "env_id": env_id,
                    "episode_id": episode_id,
                    "t": t,
                    "reward": float(reward) if reward is not None else 0.0,
                    "cumulative_reward": ep_total_reward,
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "done": done,
                    "success_step": success,
                    "success_episode": ep_success_final,
                    "action_shape": list(np.asarray(action).shape) if action is not None else None,
                }
                
                # info에서 유용한 정보 추출
                if isinstance(info, dict):
                    for key in ["elapsed_steps", "success", "is_success", "distance", "achieved_goal"]:
                        if key in info:
                            record[f"info_{key}"] = jsonable(info[key])
                
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                stats["total_steps"] += 1
                
                if done:
                    break
            
            stats["total_episodes"] += 1
            if ep_success_final:
                stats["success_episodes"] += 1
    
    env.close()
    
    # 결과 출력
    print("\n" + "=" * 60)
    print("데이터 수집 완료!")
    print("=" * 60)
    print(f"  - 총 에피소드: {stats['total_episodes']}")
    print(f"  - 성공 에피소드: {stats['success_episodes']}")
    print(f"  - 총 스텝: {stats['total_steps']}")
    print(f"  - 저장된 프레임: {stats['frames_saved']}")
    print(f"\n출력 파일:")
    print(f"  - 메타데이터: {meta_path}")
    print(f"  - 프레임: {frames_root}/E0000/...png")
    
    return True


def parse_args():
    parser = argparse.ArgumentParser(description="Physical AI 시뮬 데이터 수집")
    parser.add_argument("--num_episodes", type=int, default=30, help="수집할 에피소드 수")
    parser.add_argument("--max_steps", type=int, default=200, help="에피소드당 최대 스텝")
    parser.add_argument("--env_id", type=str, default=None, help="환경 ID (예: PickCube-v1)")
    parser.add_argument("--out_dir", type=str, default="dataset", help="출력 디렉토리")
    parser.add_argument("--fps_sample", type=int, default=1, help="프레임 샘플링 간격")
    parser.add_argument("--seed", type=int, default=42, help="랜덤 시드")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    cfg = CollectConfig(
        out_dir=args.out_dir,
        num_episodes=args.num_episodes,
        max_steps=args.max_steps,
        env_id=args.env_id,
        fps_sample=args.fps_sample,
        seed=args.seed,
    )
    
    success = collect_dataset(cfg)
    exit(0 if success else 1)
