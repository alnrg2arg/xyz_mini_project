#!/usr/bin/env python3
"""
Physical AI VLM Mini-Project - Day 2-3
VLM 기반 로봇 로그 자동 태깅 + Success/Reward Judge

사용법:
    python vlm_tagger.py --input dataset/frames --output results/tags.jsonl
    python vlm_tagger.py --mode judge --input dataset/frames --output results/judge.jsonl
"""

import os
import json
import base64
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from tqdm import tqdm

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    from PIL import Image
    import io
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


# =============================================================================
# Taxonomy 정의 (면접에서 설명할 8가지 실패 유형)
# =============================================================================
STAGE_TAXONOMY = [
    "approach",      # 물체에 접근 중
    "grasp",         # 집기 시도 중
    "lift",          # 들어올리는 중
    "transport",     # 이동 중
    "place",         # 내려놓는 중
    "retreat",       # 후퇴/원위치 복귀
    "idle",          # 대기 상태
    "unknown",       # 분류 불가
]

FAILURE_TAXONOMY = [
    "grasp_miss",        # 집기 실패 (정렬 실패)
    "slip_drop",         # 미끄러져 떨어짐
    "collision",         # 장애물/테이블 충돌
    "occlusion",         # 시야 가림
    "wrong_target",      # 잘못된 물체/위치
    "unstable_contact",  # 불안정한 접촉
    "out_of_workspace",  # 리치 범위 초과
    "success",           # 성공 (실패 아님)
    "unknown",           # 미분류
]


# =============================================================================
# 데이터 구조
# =============================================================================
@dataclass
class StageTag:
    """단계 태깅 결과"""
    episode_id: str
    frame_idx: int
    stage: str
    stage_confidence: float
    failure_type: str
    failure_confidence: float
    notes: str


@dataclass
class JudgeResult:
    """Success/Reward Judge 결과"""
    episode_id: str
    frame_idx: int
    p_success: float      # 성공 확률 (0~1)
    progress: float       # 진행도 (0~1)
    reward: float         # RL reward (-1~+1)
    uncertainty: float    # 불확실성 (0~1)
    notes: str


# =============================================================================
# VLM 프롬프트 템플릿
# =============================================================================
STAGE_TAGGING_PROMPT = """당신은 로봇 조작 영상을 분석하는 전문가입니다.

이 프레임을 보고 다음을 JSON 형식으로 출력하세요:

1. stage: 현재 로봇의 단계
   - approach: 물체에 접근 중
   - grasp: 집기 시도 중  
   - lift: 들어올리는 중
   - transport: 이동 중
   - place: 내려놓는 중
   - retreat: 후퇴/원위치
   - idle: 대기 상태
   - unknown: 분류 불가

2. failure_type: 실패 유형 (정상이면 "success")
   - grasp_miss: 집기 실패
   - slip_drop: 미끄러져 떨어짐
   - collision: 충돌 발생
   - occlusion: 시야 가림
   - wrong_target: 잘못된 대상
   - unstable_contact: 불안정한 접촉
   - out_of_workspace: 작업 범위 초과
   - success: 정상 진행 중
   - unknown: 판단 불가

3. confidence: 각 판단의 확신도 (0~1)

4. notes: 관측 가능한 근거 (한 문장)

출력 형식 (JSON만 출력):
{
  "stage": "approach",
  "stage_confidence": 0.85,
  "failure_type": "success",
  "failure_confidence": 0.90,
  "notes": "로봇 팔이 물체를 향해 이동 중"
}
"""

REWARD_JUDGE_PROMPT = """당신은 로봇 조작의 성공 여부를 판정하는 전문가입니다.

이 프레임을 보고 다음을 JSON 형식으로 출력하세요:

1. p_success: 이 에피소드가 성공할 확률 (0~1)
2. progress: 현재까지의 진행도 (0~1)
   - 0: 시작 상태
   - 0.3: 물체 접근 완료
   - 0.5: 물체 집기 완료
   - 0.7: 물체 들어올림
   - 0.9: 목표 위치 근처
   - 1.0: 완전 성공
3. uncertainty: 판단의 불확실성 (0~1, 가려짐/모호함이 있으면 높게)
4. notes: 판단 근거 (한 문장)

출력 형식 (JSON만 출력):
{
  "p_success": 0.7,
  "progress": 0.5,
  "uncertainty": 0.2,
  "notes": "물체를 성공적으로 집었으나 아직 들어올리지 않음"
}
"""


# =============================================================================
# 이미지 유틸리티
# =============================================================================
def encode_image_base64(image_path: str) -> str:
    """이미지를 base64로 인코딩"""
    with open(image_path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")


def load_and_resize_image(image_path: str, max_size: int = 512) -> str:
    """이미지 로드 후 리사이즈하여 base64 인코딩"""
    if not HAS_PIL:
        return encode_image_base64(image_path)
    
    img = Image.open(image_path)
    
    # 리사이즈 (API 비용 절감)
    if max(img.size) > max_size:
        ratio = max_size / max(img.size)
        new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
        img = img.resize(new_size, Image.LANCZOS)
    
    # base64로 변환
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.standard_b64encode(buffer.getvalue()).decode("utf-8")


# =============================================================================
# VLM 추론 클래스
# =============================================================================
class VLMTagger:
    """VLM 기반 로봇 로그 태거"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        if not HAS_OPENAI:
            raise ImportError("openai 패키지가 필요합니다: pip install openai")
        
        self.client = OpenAI(api_key=api_key)
        self.model = model
    
    def tag_frame(self, image_path: str) -> StageTag:
        """단일 프레임 태깅"""
        image_b64 = load_and_resize_image(image_path)
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": STAGE_TAGGING_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_b64}",
                                "detail": "low"  # 비용 절감
                            }
                        }
                    ]
                }
            ],
            max_tokens=300,
            temperature=0.1,  # 일관성을 위해 낮게
        )
        
        # JSON 파싱
        try:
            result_text = response.choices[0].message.content
            # JSON 블록 추출
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0]
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0]
            
            result = json.loads(result_text.strip())
        except (json.JSONDecodeError, IndexError):
            result = {
                "stage": "unknown",
                "stage_confidence": 0.0,
                "failure_type": "unknown",
                "failure_confidence": 0.0,
                "notes": "JSON 파싱 실패"
            }
        
        # episode_id와 frame_idx는 호출 측에서 설정
        return StageTag(
            episode_id="",
            frame_idx=0,
            stage=result.get("stage", "unknown"),
            stage_confidence=result.get("stage_confidence", 0.0),
            failure_type=result.get("failure_type", "unknown"),
            failure_confidence=result.get("failure_confidence", 0.0),
            notes=result.get("notes", "")
        )
    
    def judge_frame(self, image_path: str) -> JudgeResult:
        """단일 프레임에 대한 성공/보상 판정"""
        image_b64 = load_and_resize_image(image_path)
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": REWARD_JUDGE_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{image_b64}",
                                "detail": "low"
                            }
                        }
                    ]
                }
            ],
            max_tokens=200,
            temperature=0.1,
        )
        
        try:
            result_text = response.choices[0].message.content
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0]
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0]
            
            result = json.loads(result_text.strip())
        except (json.JSONDecodeError, IndexError):
            result = {
                "p_success": 0.5,
                "progress": 0.0,
                "uncertainty": 1.0,
                "notes": "JSON 파싱 실패"
            }
        
        # reward 계산: progress 기반 + 불확실성 페널티
        progress = result.get("progress", 0.0)
        uncertainty = result.get("uncertainty", 0.5)
        reward = (2 * progress - 1) * (1 - 0.3 * uncertainty)
        
        return JudgeResult(
            episode_id="",
            frame_idx=0,
            p_success=result.get("p_success", 0.5),
            progress=progress,
            reward=round(reward, 3),
            uncertainty=uncertainty,
            notes=result.get("notes", "")
        )


# =============================================================================
# 배치 처리
# =============================================================================
def process_episode_frames(
    tagger: VLMTagger,
    episode_dir: str,
    mode: str = "tag",
    sample_every: int = 5
) -> List[Dict]:
    """에피소드 프레임들을 처리"""
    results = []
    episode_id = Path(episode_dir).name
    
    frame_files = sorted([
        f for f in os.listdir(episode_dir)
        if f.endswith(('.png', '.jpg', '.jpeg'))
    ])
    
    # 샘플링 (API 비용 절감)
    sampled_frames = frame_files[::sample_every]
    
    for frame_file in tqdm(sampled_frames, desc=f"  {episode_id}", leave=False):
        frame_path = os.path.join(episode_dir, frame_file)
        frame_idx = int(Path(frame_file).stem)
        
        try:
            if mode == "tag":
                result = tagger.tag_frame(frame_path)
            else:  # judge
                result = tagger.judge_frame(frame_path)
            
            result.episode_id = episode_id
            result.frame_idx = frame_idx
            results.append(asdict(result))
        except Exception as e:
            print(f"\n[WARN] {episode_id}/{frame_file} 처리 실패: {e}")
    
    return results


def run_batch_processing(
    input_dir: str,
    output_path: str,
    mode: str = "tag",
    api_key: Optional[str] = None,
    sample_every: int = 5,
    max_episodes: Optional[int] = None
):
    """전체 배치 처리"""
    print("=" * 60)
    print(f"VLM {'태깅' if mode == 'tag' else 'Judge'} 시작")
    print("=" * 60)
    
    # API 키 확인
    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("\n[ERROR] OPENAI_API_KEY가 설정되지 않았습니다.")
        print("설정 방법:")
        print("  export OPENAI_API_KEY='your-api-key'")
        print("  또는")
        print("  python vlm_tagger.py --api_key 'your-api-key'")
        return
    
    tagger = VLMTagger(api_key=api_key)
    
    # 에피소드 디렉토리 목록
    episode_dirs = sorted([
        os.path.join(input_dir, d)
        for d in os.listdir(input_dir)
        if os.path.isdir(os.path.join(input_dir, d))
    ])
    
    if max_episodes:
        episode_dirs = episode_dirs[:max_episodes]
    
    print(f"\n처리할 에피소드: {len(episode_dirs)}개")
    print(f"샘플링 간격: {sample_every} 프레임마다 1개")
    print(f"출력 파일: {output_path}")
    print("-" * 40)
    
    # 결과 저장
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    all_results = []
    for episode_dir in tqdm(episode_dirs, desc="에피소드"):
        results = process_episode_frames(tagger, episode_dir, mode, sample_every)
        all_results.extend(results)
    
    # JSONL로 저장
    with open(output_path, "w", encoding="utf-8") as f:
        for result in all_results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    
    print("\n" + "=" * 60)
    print("처리 완료!")
    print("=" * 60)
    print(f"총 처리 프레임: {len(all_results)}개")
    print(f"출력 파일: {output_path}")


# =============================================================================
# 로컬 테스트 (API 없이)
# =============================================================================
def run_local_demo(input_dir: str):
    """API 없이 로컬 데모 실행"""
    print("=" * 60)
    print("VLM 태거 로컬 데모 (API 없이)")
    print("=" * 60)
    
    episode_dirs = sorted([
        os.path.join(input_dir, d)
        for d in os.listdir(input_dir)
        if os.path.isdir(os.path.join(input_dir, d))
    ])[:1]  # 첫 번째 에피소드만
    
    if not episode_dirs:
        print("[ERROR] 프레임이 없습니다.")
        return
    
    episode_dir = episode_dirs[0]
    frames = sorted([f for f in os.listdir(episode_dir) if f.endswith('.png')])[:3]
    
    print(f"\n에피소드: {os.path.basename(episode_dir)}")
    print(f"샘플 프레임: {frames}")
    
    print("\n[데모] VLM API를 사용하면 각 프레임에 대해 다음과 같은 결과를 얻습니다:\n")
    
    demo_results = [
        {
            "episode_id": "E0000",
            "frame_idx": 0,
            "stage": "approach",
            "stage_confidence": 0.85,
            "failure_type": "success",
            "failure_confidence": 0.92,
            "notes": "로봇 팔이 물체를 향해 이동 중"
        },
        {
            "episode_id": "E0000",
            "frame_idx": 10,
            "stage": "grasp",
            "stage_confidence": 0.78,
            "failure_type": "success",
            "failure_confidence": 0.80,
            "notes": "그리퍼가 물체에 접근하여 집기 시도 중"
        },
        {
            "episode_id": "E0000",
            "frame_idx": 20,
            "stage": "transport",
            "stage_confidence": 0.90,
            "failure_type": "success",
            "failure_confidence": 0.88,
            "notes": "물체를 들어올려 이동 중"
        }
    ]
    
    for result in demo_results:
        print(json.dumps(result, indent=2, ensure_ascii=False))
        print()
    
    print("실제 VLM API 사용:")
    print("  export OPENAI_API_KEY='your-key'")
    print("  python vlm_tagger.py --input dataset/frames --output results/tags.jsonl")


# =============================================================================
# Main
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="VLM 기반 로봇 로그 태깅")
    parser.add_argument("--input", type=str, default="dataset/frames",
                       help="입력 프레임 디렉토리")
    parser.add_argument("--output", type=str, default="results/tags.jsonl",
                       help="출력 JSONL 파일")
    parser.add_argument("--mode", type=str, choices=["tag", "judge"], default="tag",
                       help="모드: tag(태깅) 또는 judge(성공판정)")
    parser.add_argument("--api_key", type=str, default=None,
                       help="OpenAI API 키")
    parser.add_argument("--sample_every", type=int, default=5,
                       help="프레임 샘플링 간격")
    parser.add_argument("--max_episodes", type=int, default=None,
                       help="최대 처리 에피소드 수")
    parser.add_argument("--demo", action="store_true",
                       help="API 없이 데모 실행")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    if args.demo:
        run_local_demo(args.input)
    else:
        run_batch_processing(
            input_dir=args.input,
            output_path=args.output,
            mode=args.mode,
            api_key=args.api_key,
            sample_every=args.sample_every,
            max_episodes=args.max_episodes
        )
