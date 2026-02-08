# Day 1 완료 - 환경 설정 및 데이터 수집 파이프라인

## 설치된 환경

```
Conda 환경: physai_vlm (Python 3.10.19)
```

### 핵심 패키지
- numpy: 2.2.6
- gymnasium: 0.29.1
- mani-skill: 3.0.0b22
- mujoco: 3.4.0
- sapien: 3.0.2
- torch: 2.10.0

### VLM 관련 패키지
- openai: 2.16.0
- Pillow: 12.1.0
- pandas: 2.3.3
- matplotlib: 3.10.8

## Day 1 성과

### 데이터 수집 테스트 결과
- 환경: Pusher-v4 (MuJoCo)
- 에피소드: 5개
- 총 프레임: 250개 (480x480 RGB)
- 메타데이터: `dataset/episodes.jsonl`

### 파일 구조
```
xyz/
├── collect_sim_dataset.py   ✅ 동작 확인
├── verify_install.py        ✅ 동작 확인
├── requirements.txt
├── README.md
├── DAY1_COMPLETE.md         (현재 파일)
├── dataset/
│   ├── episodes.jsonl       ✅ 생성됨 (250 records)
│   └── frames/
│       ├── E0000/           ✅ 50 프레임
│       ├── E0001/           ✅ 50 프레임
│       ├── E0002/           ✅ 50 프레임
│       ├── E0003/           ✅ 50 프레임
│       └── E0004/           ✅ 50 프레임
├── results/
└── reports/
```

## 환경 활성화 방법

```bash
conda activate physai_vlm
```

## 알려진 이슈

### macOS Vulkan 호환성
- ManiSkill 3.x의 ManiSkill 환경(PickCube-v1 등)은 Vulkan 렌더링이 필요
- macOS는 Metal을 사용하므로 ManiSkill 환경 직접 사용 불가
- **대안**: MuJoCo 환경 (Pusher-v4, FetchPickAndPlace-v2 등) 사용

### 사용 가능한 환경
1. **MuJoCo 환경 (macOS 지원)**:
   - Pusher-v4: 물체 밀기 태스크
   - FetchPickAndPlace-v2: Pick-and-place 태스크
   
2. **ManiSkill 환경 (Linux GPU 필요)**:
   - PickCube-v1
   - PushCube-v1
   - 등 26개 환경

## Day 2 목표

### VLM 자동 태깅 구현
1. 프레임별 stage 분류 (approach/grasp/lift/place/idle)
2. 실패 유형 분류 (8가지 taxonomy)
3. 데이터 정제 룰 생성

### Success/Reward Judge 구현
1. VLM 기반 성공 여부 판정
2. 진행도(progress) 점수 산출
3. RL reward 변환

## 실행 명령어

### 더 많은 데이터 수집
```bash
conda activate physai_vlm
cd /Users/gingerb/xyz
python collect_sim_dataset.py --num_episodes 30 --max_steps 100 --env_id "Pusher-v4"
```

### FetchPickAndPlace 환경으로 수집
```bash
pip install "gymnasium[robotics]"
python collect_sim_dataset.py --num_episodes 30 --env_id "FetchPickAndPlace-v2"
```
