# ELMo 데이터셋 준비 도구

이 디렉토리는 ELMo 모델 학습을 위한 데이터셋 다운로드, 전처리, 검증 도구를 제공합니다.

## 🚀 빠른 시작

### 전체 파이프라인 실행
```bash
cd /home/ssai/Workspace/ELMo_repo/bilm-tf/bilm/src/dataset
python prepare_data.py
```

이 명령어는 다음을 자동으로 수행합니다:
1. 📥 Wikipedia 데이터셋 다운로드 (lsb/enwiki20230101)
2. ⚙️ 텍스트 전처리 및 토큰화
3. 🔍 전처리된 데이터 검증

## 📁 파일 구조

```
dataset/
├── prepare_data.py          # 🎯 메인 파이프라인 스크립트
├── dataset_download.py      # 📥 데이터셋 다운로드
├── preprocess_for_elmo.py   # ⚙️ 데이터 전처리
├── data_loader.py          # 📊 데이터 로딩 및 검증
├── data.py                 # 🔧 기존 데이터 처리 유틸리티
└── README.md               # 📖 이 파일
```

## 🛠️ 개별 스크립트 사용법

### 1. 데이터셋 다운로드
```bash
python dataset_download.py
```

**특징:**
- ✅ 중복 다운로드 방지
- 📊 다운로드 정보 저장
- 🔍 데이터셋 무결성 검증

### 2. 데이터 전처리
```bash
python preprocess_for_elmo.py
```

**특징:**
- ✅ 중복 실행 방지
- 💾 체크포인트 기능 (중단 시 재개 가능)
- 📈 실시간 진행 상황 표시
- 🗂️ 샤드 단위 데이터 분할

**설정 (preprocess_for_elmo.py 상단):**
```python
MIN_TOKENS = 5          # 최소 토큰 수
MAX_TOKENS = 50         # 최대 토큰 수  
VOCAB_SIZE = 50000      # 어휘 크기
SHARD_SIZE = 100000     # 샤드당 문장 수
```

### 3. 데이터 검증
```bash
python data_loader.py data/pretrain/elmo validate
```

### 4. 데이터 샘플 확인
```bash
python data_loader.py data/pretrain/elmo sample
```

## 📊 출력 데이터 구조

전처리 완료 후 다음과 같은 구조로 데이터가 저장됩니다:

```
data/pretrain/elmo/
├── train/
│   ├── train_00000.txt
│   ├── train_00001.txt
│   └── ...
├── valid/
│   └── valid_00000.txt (비어있을 수 있음)
├── test/
│   └── test_00000.txt (비어있을 수 있음)
├── vocab.txt              # 어휘 사전
└── preprocess_meta.json   # 메타데이터
```

### 메타데이터 예시
```json
{
  "dataset_name": "lsb/enwiki20230101",
  "num_sentences": 1500000,
  "num_train": 1500000,
  "num_valid": 0,
  "num_test": 0,
  "vocab_size": 50000,
  "train_prefix": "data/pretrain/elmo/train/train_*.txt",
  "vocab_file": "data/pretrain/elmo/vocab.txt"
}
```

## 🔧 고급 사용법

### 강제 재실행
```bash
# 강제로 다시 다운로드
python prepare_data.py --force-download

# 강제로 다시 전처리  
python prepare_data.py --force-preprocess

# 둘 다
python prepare_data.py --force-download --force-preprocess
```

### 검증만 수행
```bash
python prepare_data.py --validate-only
```

### 샘플 데이터 확인
```bash
python prepare_data.py --validate-only --sample
```

## 🔄 중단된 작업 재개

전처리 중 작업이 중단되더라도 자동으로 체크포인트에서 재개됩니다:

```bash
# 중단된 지점부터 자동 재개
python preprocess_for_elmo.py
```

## 📈 성능 최적화

### 메모리 사용량 최적화
- 스트리밍 방식으로 대용량 데이터셋 처리
- 샤드 단위로 데이터 분할하여 메모리 효율성 확보

### 처리 속도 최적화  
- 정규표현식 기반 빠른 토큰화
- 간단한 문장 분할 (NLTK 대신)
- 어휘 구축 시 샘플링 옵션

## 🚨 문제 해결

### 다운로드 실패
```bash
# 캐시 디렉토리 삭제 후 재시도
rm -rf data/pretrain/raw/huggingface_cache
python dataset_download.py
```

### 전처리 실패
```bash
# 출력 디렉토리 삭제 후 재시도
rm -rf data/pretrain/elmo
python preprocess_for_elmo.py
```

### 메모리 부족
`preprocess_for_elmo.py`에서 다음 설정을 조정:
```python
SHARD_SIZE = 50000      # 샤드 크기 줄이기
VOCAB_SAMPLE_RATE = 0.1 # 어휘 샘플링 비율 줄이기
```

## 🔗 학습에서 사용하기

전처리된 데이터를 학습에서 사용하는 예시:

```python
from data_loader import ELMoDataLoader

# 데이터 로더 생성
loader = ELMoDataLoader('data/pretrain/elmo')

# 통계 확인
stats = loader.get_stats()
print(f"어휘 크기: {stats['vocab_size']}")
print(f"학습 문장 수: {stats['train_sentences']}")

# 문장 순차 로딩
for sentence in loader.load_sentences('train', max_files=1):
    print(sentence)
    break

# 토큰화된 문장 로딩
for tokens in loader.load_tokenized_sentences('train', max_files=1):
    print(tokens)
    break

# 인덱스로 변환된 문장 로딩
for indices in loader.load_indexed_sentences('train', max_files=1):
    print(indices)
    break
```

## ⚙️ 설정 변경

주요 설정은 각 스크립트 상단에서 변경할 수 있습니다:

**dataset_download.py:**
```python
DATASET_NAME = "lsb/enwiki20230101"  # 다른 데이터셋 사용 시 변경
SAVE_DIR = "data/pretrain/raw/huggingface_cache"
```

**preprocess_for_elmo.py:**
```python
MIN_TOKENS = 5          # 최소 토큰 수
MAX_TOKENS = 50         # 최대 토큰 수
VOCAB_SIZE = 50000      # 어휘 크기
SHARD_SIZE = 100000     # 샤드 크기
LOWERCASE = False       # 소문자 변환 여부
```

이제 학습할 때마다 전처리를 다시 하지 않고, 미리 준비된 데이터를 효율적으로 사용할 수 있습니다! 🎉