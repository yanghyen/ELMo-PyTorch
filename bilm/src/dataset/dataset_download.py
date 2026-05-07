import json
import os
import time
from datasets import load_dataset

# =====================
# Config
# =====================
DATASET_NAME = "lsb/enwiki20230101"
SAVE_DIR = "data/pretrain/raw/huggingface_cache"
DOWNLOAD_INFO_FILE = os.path.join(SAVE_DIR, "download_info.json")


def check_dataset_exists():
    """데이터셋이 이미 다운로드되었는지 확인"""
    if not os.path.exists(DOWNLOAD_INFO_FILE):
        return False
    
    try:
        with open(DOWNLOAD_INFO_FILE, 'r') as f:
            info = json.load(f)
        
        # 기본 검증
        if info.get('dataset_name') != DATASET_NAME:
            return False
            
        # 캐시 디렉토리 확인
        if not os.path.exists(SAVE_DIR) or not os.listdir(SAVE_DIR):
            return False
        
        print(f"✅ 데이터셋이 이미 다운로드되어 있습니다:")
        print(f"   - 데이터셋: {info['dataset_name']}")
        print(f"   - 다운로드 시간: {info['download_time']}")
        print(f"   - 캐시 위치: {SAVE_DIR}")
        
        return True
        
    except (json.JSONDecodeError, KeyError) as e:
        print(f"⚠️ 다운로드 정보 파일에 문제가 있습니다: {e}")
        return False


def download_dataset():
    """데이터셋 다운로드"""
    print(f"🚀 데이터셋 다운로드 시작...")
    print(f"   - 데이터셋: {DATASET_NAME}")
    print(f"   - 저장 위치: {SAVE_DIR}")
    
    start_time = time.time()
    
    try:
        # 디렉토리 생성
        os.makedirs(SAVE_DIR, exist_ok=True)
        
        # 데이터셋 다운로드
        ds = load_dataset(
            DATASET_NAME,
            cache_dir=SAVE_DIR,
            trust_remote_code=True  # 필요한 경우
        )
        
        elapsed_time = time.time() - start_time
        
        # 다운로드 정보 저장
        download_info = {
            'dataset_name': DATASET_NAME,
            'download_time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'elapsed_seconds': elapsed_time,
            'cache_dir': SAVE_DIR,
            'splits': list(ds.keys()) if hasattr(ds, 'keys') else ['train']
        }
        
        with open(DOWNLOAD_INFO_FILE, 'w') as f:
            json.dump(download_info, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 다운로드 완료! (소요 시간: {elapsed_time/60:.1f}분)")
        print(f"📊 사용 가능한 분할: {download_info['splits']}")
        
        return True
        
    except Exception as e:
        print(f"❌ 다운로드 실패: {e}")
        return False


def verify_dataset():
    """다운로드된 데이터셋 검증"""
    print("🔍 데이터셋 검증 중...")
    
    try:
        # 데이터셋 로드 테스트
        ds = load_dataset(
            DATASET_NAME,
            cache_dir=SAVE_DIR,
            streaming=True
        )
        
        # 첫 번째 샘플 확인
        sample = next(iter(ds['train']))
        
        if 'text' not in sample:
            print("❌ 데이터셋에 'text' 필드가 없습니다.")
            return False
        
        text_length = len(sample['text'])
        print(f"✅ 데이터셋 검증 완료!")
        print(f"   - 첫 번째 샘플 텍스트 길이: {text_length:,} 문자")
        print(f"   - 샘플 미리보기: {sample['text'][:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ 데이터셋 검증 실패: {e}")
        return False


def main():
    """메인 함수"""
    # 이미 다운로드되었는지 확인
    if check_dataset_exists():
        print("🔄 기존 다운로드를 사용합니다. 다시 다운로드하려면 캐시 디렉토리를 삭제하세요.")
        
        # 검증만 수행
        if not verify_dataset():
            print("⚠️ 기존 데이터셋에 문제가 있습니다. 다시 다운로드를 권장합니다.")
        return
    
    # 새로 다운로드
    if download_dataset():
        verify_dataset()
    else:
        print("❌ 다운로드에 실패했습니다.")


if __name__ == "__main__":
    main()

