#!/usr/bin/env python3
"""
ELMo 학습을 위한 데이터 준비 파이프라인

이 스크립트는 다음 단계를 수행합니다:
1. 데이터셋 다운로드
2. 데이터 전처리
3. 전처리된 데이터 검증

사용법:
    python prepare_data.py [--force-download] [--force-preprocess] [--validate-only]
"""

import argparse
import sys
import os

# 현재 디렉토리를 Python 경로에 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset_download import main as download_main
from preprocess_for_elmo import main as preprocess_main
from data_loader import validate_preprocessed_data, sample_data


def parse_args():
    """명령행 인자 파싱"""
    parser = argparse.ArgumentParser(
        description="ELMo 학습을 위한 데이터 준비 파이프라인",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
    # 전체 파이프라인 실행
    python prepare_data.py
    
    # 강제로 다시 다운로드
    python prepare_data.py --force-download
    
    # 강제로 다시 전처리
    python prepare_data.py --force-preprocess
    
    # 검증만 수행
    python prepare_data.py --validate-only
        """
    )
    
    parser.add_argument(
        '--force-download',
        action='store_true',
        help='기존 다운로드를 무시하고 강제로 다시 다운로드'
    )
    
    parser.add_argument(
        '--force-preprocess',
        action='store_true', 
        help='기존 전처리 결과를 무시하고 강제로 다시 전처리'
    )
    
    parser.add_argument(
        '--validate-only',
        action='store_true',
        help='다운로드/전처리 없이 검증만 수행'
    )
    
    parser.add_argument(
        '--sample',
        action='store_true',
        help='검증 후 데이터 샘플 출력'
    )
    
    return parser.parse_args()


def clean_cache_if_needed(force_download: bool, force_preprocess: bool):
    """필요시 캐시 정리"""
    from preprocess_for_elmo import OUTPUT_DIR, CACHE_DIR
    
    if force_download and os.path.exists(CACHE_DIR):
        print(f"🗑️ 기존 다운로드 캐시 삭제: {CACHE_DIR}")
        import shutil
        shutil.rmtree(CACHE_DIR)
    
    if force_preprocess and os.path.exists(OUTPUT_DIR):
        print(f"🗑️ 기존 전처리 결과 삭제: {OUTPUT_DIR}")
        import shutil
        shutil.rmtree(OUTPUT_DIR)


def main():
    """메인 함수"""
    args = parse_args()
    
    print("🚀 ELMo 데이터 준비 파이프라인 시작")
    print("=" * 50)
    
    # 필요시 캐시 정리
    if args.force_download or args.force_preprocess:
        clean_cache_if_needed(args.force_download, args.force_preprocess)
    
    # 검증만 수행하는 경우
    if args.validate_only:
        from preprocess_for_elmo import OUTPUT_DIR
        
        print("🔍 데이터 검증만 수행합니다...")
        if validate_preprocessed_data(OUTPUT_DIR):
            if args.sample:
                sample_data(OUTPUT_DIR)
        return
    
    # 1단계: 데이터셋 다운로드
    print("\n📥 1단계: 데이터셋 다운로드")
    print("-" * 30)
    try:
        download_main()
        print("✅ 다운로드 단계 완료")
    except Exception as e:
        print(f"❌ 다운로드 실패: {e}")
        return
    
    # 2단계: 데이터 전처리
    print("\n⚙️ 2단계: 데이터 전처리")
    print("-" * 30)
    try:
        preprocess_main()
        print("✅ 전처리 단계 완료")
    except Exception as e:
        print(f"❌ 전처리 실패: {e}")
        return
    
    # 3단계: 데이터 검증
    print("\n🔍 3단계: 데이터 검증")
    print("-" * 30)
    from preprocess_for_elmo import OUTPUT_DIR
    
    if validate_preprocessed_data(OUTPUT_DIR):
        print("✅ 검증 단계 완료")
        
        if args.sample:
            print("\n📋 데이터 샘플:")
            print("-" * 30)
            sample_data(OUTPUT_DIR)
    else:
        print("❌ 검증 실패")
        return
    
    print("\n🎉 모든 단계가 성공적으로 완료되었습니다!")
    print("=" * 50)
    print("이제 학습을 시작할 수 있습니다.")
    print(f"전처리된 데이터 위치: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()