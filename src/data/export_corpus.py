from datasets import load_dataset
import os

# 현재 스크립트의 디렉토리를 기준으로 상대 경로 계산
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR    = os.path.dirname(SCRIPT_DIR)
ROOT_DIR   = os.path.dirname(SRC_DIR)

# 1. Hugging Face 데이터셋 캐시의 최상위 경로를 지정합니다.
CACHE_DIR = os.path.join(ROOT_DIR, "data/pretrain/raw/huggingface_cache")

# 2. 최종 코퍼스 파일이 저장될 경로를 지정합니다.
OUTPUT_PATH = os.path.join(ROOT_DIR, "data/pretrain/raw/elmo_corpus.txt")

# 3. 데이터셋 로드 (캐시된 파일을 자동으로 사용합니다)
print("📘 캐시된 데이터셋 로드 중...")
try:
    # ds는 DatasetDict 객체입니다. (예: {'train': Dataset(...)} )
    ds = load_dataset("lsb/enwiki20230101", cache_dir=CACHE_DIR)
except Exception as e:
    print(f"❌ 데이터셋 로드 오류. CACHE_DIR을 확인해주세요: {e}")
    ds = load_dataset("lsb/enwiki20230101")

ds_train = ds['train']
total_docs = len(ds_train)

half_docs = total_docs // 2

ds_to_process = ds_train[:half_docs]['text'] 

print(f"문서를 {OUTPUT_PATH} 파일로 순차적으로 내보내는 중...")

write_count = 0
with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
    for i, text in enumerate(ds_to_process): 
        stripped_text = text.strip()
        
        if stripped_text:
            f.write(stripped_text)
            f.write('\n\n') 
            write_count += 1
        
        if (i + 1) % 100000 == 0:
            print(f"    - {i+1:,}번째 문서까지 처리 완료...")

print("\n---")
print(f"코퍼스 파일 생성 완료: {OUTPUT_PATH}")
print(f"Total documents exported: {write_count:,}")
os.system(f"ls -lh {OUTPUT_PATH}") 