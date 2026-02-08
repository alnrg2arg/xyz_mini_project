#!/usr/bin/env python3
"""
설치 검증 스크립트
환경이 제대로 설정되었는지 확인
"""

import sys

def check_package(name, import_name=None):
    """패키지 설치 확인"""
    import_name = import_name or name
    try:
        module = __import__(import_name)
        version = getattr(module, '__version__', 'N/A')
        print(f"  ✓ {name}: {version}")
        return True
    except ImportError:
        print(f"  ✗ {name}: 설치되지 않음")
        return False

def check_maniskill():
    """ManiSkill 환경 확인"""
    try:
        import gymnasium as gym
        
        # ManiSkill 3.x
        try:
            import mani_skill.envs
            print("  ✓ ManiSkill 3.x 감지됨")
            return True
        except ImportError:
            pass
        
        # ManiSkill 2.x
        try:
            import mani_skill2.envs
            print("  ✓ ManiSkill 2.x 감지됨")
            return True
        except ImportError:
            pass
        
        print("  ✗ ManiSkill이 설치되지 않음")
        return False
    except ImportError:
        print("  ✗ Gymnasium이 설치되지 않음")
        return False

def list_available_envs():
    """사용 가능한 환경 목록"""
    try:
        import gymnasium as gym
        
        all_envs = list(gym.registry.keys())
        
        # ManiSkill 관련 환경
        maniskill_envs = [e for e in all_envs if any(kw in e for kw in ["ManiSkill", "Pick", "Push", "Lift", "Place"])]
        
        print("\n사용 가능한 ManiSkill 관련 환경:")
        if maniskill_envs:
            for env in sorted(maniskill_envs)[:10]:
                print(f"    - {env}")
            if len(maniskill_envs) > 10:
                print(f"    ... 외 {len(maniskill_envs) - 10}개")
        else:
            print("    (ManiSkill 환경 없음)")
            print("\n대안으로 사용 가능한 환경:")
            fallback = ["CartPole-v1", "MountainCar-v0", "Pendulum-v1"]
            for env in fallback:
                if env in all_envs:
                    print(f"    - {env}")
        
        return len(maniskill_envs) > 0
    except ImportError:
        return False

def main():
    print("=" * 50)
    print("Physical AI VLM 프로젝트 - 환경 검증")
    print("=" * 50)
    
    print("\n[1] Python 버전")
    print(f"  Python: {sys.version}")
    
    print("\n[2] 핵심 패키지")
    essential = [
        ("numpy", "numpy"),
        ("gymnasium", "gymnasium"),
        ("imageio", "imageio"),
        ("opencv-python", "cv2"),
        ("tqdm", "tqdm"),
    ]
    
    essential_ok = all(check_package(name, imp) for name, imp in essential)
    
    print("\n[3] ManiSkill")
    maniskill_ok = check_maniskill()
    
    print("\n[4] VLM 관련 패키지 (Day 2+)")
    vlm_packages = [
        ("openai", "openai"),
        ("Pillow", "PIL"),
        ("pandas", "pandas"),
        ("matplotlib", "matplotlib"),
    ]
    vlm_ok = all(check_package(name, imp) for name, imp in vlm_packages)
    
    print("\n[5] 환경 목록")
    envs_ok = list_available_envs()
    
    print("\n" + "=" * 50)
    if essential_ok:
        print("✓ 핵심 패키지: OK")
    else:
        print("✗ 핵심 패키지: 일부 누락")
    
    if maniskill_ok:
        print("✓ ManiSkill: OK")
    else:
        print("✗ ManiSkill: 설치 필요 (pip install mani-skill)")
    
    if vlm_ok:
        print("✓ VLM 패키지: OK")
    else:
        print("△ VLM 패키지: 일부 누락 (Day 2에 필요)")
    
    print("=" * 50)
    
    if essential_ok:
        print("\n다음 단계: python collect_sim_dataset.py")
        if not maniskill_ok:
            print("(ManiSkill 없이도 기본 환경으로 테스트 가능)")
    else:
        print("\n설치가 필요합니다:")
        print("  bash setup_env.sh")
        print("  또는")
        print("  pip install -r requirements.txt")
    
    return essential_ok

if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
