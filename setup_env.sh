#!/bin/bash
# Physical AI VLM Mini-Project 환경 설정 스크립트
# Usage: bash setup_env.sh

set -e

echo "=========================================="
echo "Physical AI VLM 프로젝트 환경 설정"
echo "=========================================="

# Conda 환경 이름
ENV_NAME="physai_vlm"

# Conda 확인
if ! command -v conda &> /dev/null; then
    echo "[ERROR] Conda가 설치되어 있지 않습니다."
    echo "Anaconda/Miniconda를 설치해주세요: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

# 기존 환경 확인
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "[INFO] 기존 환경 '${ENV_NAME}'이 존재합니다."
    read -p "삭제하고 새로 만들까요? (y/n): " answer
    if [ "$answer" = "y" ]; then
        echo "[INFO] 기존 환경 삭제 중..."
        conda env remove -n ${ENV_NAME} -y
    else
        echo "[INFO] 기존 환경을 사용합니다."
        echo "활성화: conda activate ${ENV_NAME}"
        exit 0
    fi
fi

echo ""
echo "[Step 1/4] Conda 환경 생성 (Python 3.10)..."
conda create -n ${ENV_NAME} python=3.10 -y

echo ""
echo "[Step 2/4] 환경 활성화..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ${ENV_NAME}

echo ""
echo "[Step 3/4] 핵심 패키지 설치..."
pip install --upgrade pip

# 기본 패키지 먼저 설치
pip install numpy>=1.24.0 gymnasium>=0.29.0 tqdm imageio opencv-python

# ManiSkill 설치 시도 (실패해도 계속 진행)
echo ""
echo "[Step 4/4] ManiSkill 설치..."
pip install mani-skill || {
    echo "[WARN] ManiSkill 3.x 설치 실패, 2.x 시도..."
    pip install mani-skill2 || {
        echo "[WARN] ManiSkill 설치 실패. 기본 Gymnasium 환경만 사용합니다."
    }
}

# VLM 관련 패키지 (Day 2+)
pip install openai Pillow pandas matplotlib seaborn

echo ""
echo "=========================================="
echo "설치 완료!"
echo "=========================================="
echo ""
echo "다음 단계:"
echo "  1. 환경 활성화: conda activate ${ENV_NAME}"
echo "  2. 데이터 수집: python collect_sim_dataset.py"
echo ""
