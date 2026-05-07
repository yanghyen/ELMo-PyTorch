import json
import os
import random
import re
import time
from collections import Counter
from typing import Iterable, List

from datasets import load_dataset

# =====================
# Config
# =====================
DATASET_NAME = "lsb/enwiki20230101"
DATASET_SPLIT = "train"
CACHE_DIR = "data/pretrain/raw/huggingface_cache"
OUTPUT_DIR = "data/pretrain/elmo"

MIN_TOKENS = 5
MAX_TOKENS = 50
VOCAB_SIZE = 50000
SHARD_SIZE = 100000

VALID_RATIO = 0.01
TEST_RATIO = 0.0
SEED = 42

LOWERCASE = False
# VOCAB_SAMPLE_RATE = 0.1  # 🔥 vocab 만들 때만 일부 샘플링
VOCAB_SAMPLE_RATE = 1.0

# 문장 후보 품질 (숫자 비율·반복)
MAX_NUMERIC_TOKEN_RATIO = 0.4
MIN_TOKENS_FOR_QUALITY_FILTERS = 8
MIN_UNIQUE_TOKEN_RATIO = 0.25
MAX_SAME_TOKEN_RUN_RATIO = 0.45

TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
_SENT_BOUNDARY = re.compile(r"(?<=[.!?])\s+")

# 위키 표 / 인포박스 / 정렬용 마크업 줄 (자연어 문장이 아닌 경우)
_WIKI_TABLE_START = re.compile(r"^\s*(\{\||\|\}|\|-)", re.UNICODE)
_WIKI_INFOBOX_PIPE = re.compile(
    r"^\s*\|\s*[\w][\w\s\-]{0,80}?\s*=",
    re.UNICODE,
)


# =====================
# Utils
# =====================
def looks_like_wiki_table_or_markup_line(line: str) -> bool:
    """
    MediaWiki 표·틀·HTML 잔재에 가까운 줄이면 True (학습용 문장에서 제외).
    보수적으로: 명확한 마크업 토큰이 많을 때만 건너뜀.
    """
    s = line.strip()
    if not s:
        return True
    sl = s.lower()

    if _WIKI_TABLE_START.match(s):
        return True

    # 표 셀 / 스타일 / 정렬
    markup_hits = (
        "bgcolor",
        "colspan",
        "rowspan",
        "wikitable",
        "sortable",
        "data-sort-value",
        "data-sort-",
        "vertical-align",
        "text-align:",
        "font-size:",
        "align=",
        "align =",
        "valign=",
        "style=",
        "class=",
        "id=",
        "|-",  # 일부 덤프에서 공백 없이 붙는 경우
    )
    if any(h in sl for h in markup_hits):
        return True

    # HTML 주석·태그 덩어리
    if "<!--" in s or "-->" in s:
        return True
    if s.count("<") >= 2 and s.count(">") >= 2:
        return True

    # 표 한 행: 맨 앞이 | 이고 셀 구분자가 많음
    if s.startswith("|"):
        if s.count("|") >= 2:
            return True
        if _WIKI_INFOBOX_PIPE.match(s):
            return True

    # 숫자·파이프·구두점만 있는 짧은 표 인덱스 행 등
    if s.count("|") >= 3 and re.match(r"^[\d|\s.,:;–—\-]+$", s):
        return True

    return False


def tokenize(text: str, lowercase: bool = True) -> List[str]:
    if lowercase:
        text = text.lower()
    return TOKEN_RE.findall(text)


def better_sentence_split(text: str) -> List[str]:
    """
    줄바꿈으로 블록을 나눈 뒤, 각 블록을 . ! ? 뒤 공백 기준으로 문장 후보 분리.
    (한 줄 = 한 문장 가정보다 위키 문단에 맞음. 약어 U.S. 등은 오분할 가능.)
    """
    out: List[str] = []
    for block in text.split("\n"):
        block = block.strip()
        if not block:
            continue
        for piece in _SENT_BOUNDARY.split(block):
            s = piece.strip()
            if s:
                out.append(s)
    return out


def has_url(raw_sent: str) -> bool:
    sl = raw_sent.lower()
    return "http://" in sl or "https://" in sl or "www." in sl


def numeric_token_ratio(toks: List[str]) -> float:
    if not toks:
        return 0.0
    n = sum(1 for t in toks if t.isdigit())
    return n / len(toks)


def unique_token_ratio(toks: List[str]) -> float:
    if not toks:
        return 1.0
    return len(set(toks)) / len(toks)


def longest_same_token_run_ratio(toks: List[str]) -> float:
    """같은 토큰이 연속으로 나오는 최대 길이 / 전체 길이."""
    if not toks:
        return 0.0
    best = cur = 1
    for i in range(1, len(toks)):
        if toks[i] == toks[i - 1]:
            cur += 1
            best = max(best, cur)
        else:
            cur = 1
    return best / len(toks)


def should_skip_low_quality_sentence(toks: List[str]) -> bool:
    """숫자 위주·토큰 반복이 심한 문장 제외."""
    if len(toks) < MIN_TOKENS_FOR_QUALITY_FILTERS:
        return False
    if numeric_token_ratio(toks) > MAX_NUMERIC_TOKEN_RATIO:
        return True
    if unique_token_ratio(toks) < MIN_UNIQUE_TOKEN_RATIO:
        return True
    if longest_same_token_run_ratio(toks) > MAX_SAME_TOKEN_RUN_RATIO:
        return True
    return False


# =====================
# Shard Writer
# =====================
class ShardWriter:
    def __init__(self, output_dir, prefix, shard_size):
        self.output_dir = output_dir
        self.prefix = prefix
        self.shard_size = shard_size

        os.makedirs(output_dir, exist_ok=True)

        self.buffer = []
        self.shard_idx = 0
        self.paths = []

    def add(self, line: str):
        self.buffer.append(line)
        if len(self.buffer) >= self.shard_size:
            self.flush()

    def flush(self):
        if not self.buffer:
            return

        path = os.path.join(
            self.output_dir,
            f"{self.prefix}_{self.shard_idx:05d}.txt"
        )

        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(self.buffer))
            f.write("\n")

        self.paths.append(path)
        self.buffer = []
        self.shard_idx += 1

    def close(self):
        self.flush()
        return self.paths


# =====================
# Utils for checkpointing
# =====================
def check_preprocessing_completed(output_dir):
    """전처리가 이미 완료되었는지 확인"""
    meta_path = os.path.join(output_dir, "preprocess_meta.json")
    vocab_path = os.path.join(output_dir, "vocab.txt")
    
    if not (os.path.exists(meta_path) and os.path.exists(vocab_path)):
        return False
    
    try:
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        
        # 필수 필드 확인
        required_fields = ['num_sentences', 'vocab_size', 'train_prefix', 'vocab_file']
        if not all(field in meta for field in required_fields):
            return False
        
        # 실제 파일들이 존재하는지 확인
        train_dir = os.path.join(output_dir, "train")
        if not os.path.exists(train_dir) or not os.listdir(train_dir):
            return False
            
        print(f"✅ 전처리가 이미 완료되어 있습니다:")
        print(f"   - 총 문장 수: {meta['num_sentences']:,}")
        print(f"   - 어휘 크기: {meta['vocab_size']:,}")
        print(f"   - 학습 데이터: {train_dir}")
        print(f"   - 어휘 파일: {vocab_path}")
        return True
        
    except (json.JSONDecodeError, KeyError) as e:
        print(f"⚠️ 메타데이터 파일에 문제가 있습니다: {e}")
        return False


def save_checkpoint(output_dir, processed_count, token_counter, shard_writers):
    """진행 상황을 체크포인트로 저장"""
    checkpoint_path = os.path.join(output_dir, "checkpoint.json")
    vocab_checkpoint_path = os.path.join(output_dir, "vocab_checkpoint.json")
    
    checkpoint = {
        'processed_count': processed_count,
        'timestamp': time.time(),
        'vocab_counter_size': len(token_counter),
        'shard_indices': {
            'train': shard_writers[0].shard_idx,
            'valid': shard_writers[1].shard_idx,
            'test': shard_writers[2].shard_idx
        }
    }
    
    with open(checkpoint_path, 'w') as f:
        json.dump(checkpoint, f, indent=2)
    
    # 어휘 카운터 저장 (큰 파일이므로 별도 저장)
    vocab_data = dict(token_counter.most_common())
    with open(vocab_checkpoint_path, 'w') as f:
        json.dump(vocab_data, f)


def load_checkpoint(output_dir):
    """체크포인트에서 진행 상황 복원"""
    checkpoint_path = os.path.join(output_dir, "checkpoint.json")
    vocab_checkpoint_path = os.path.join(output_dir, "vocab_checkpoint.json")
    
    if not (os.path.exists(checkpoint_path) and os.path.exists(vocab_checkpoint_path)):
        return None, None
    
    try:
        with open(checkpoint_path, 'r') as f:
            checkpoint = json.load(f)
        
        with open(vocab_checkpoint_path, 'r') as f:
            vocab_data = json.load(f)
        
        token_counter = Counter(vocab_data)
        
        print(f"📂 체크포인트 발견! 이전 진행 상황:")
        print(f"   - 처리된 문장: {checkpoint['processed_count']:,}")
        print(f"   - 어휘 크기: {checkpoint['vocab_counter_size']:,}")
        
        return checkpoint, token_counter
        
    except (json.JSONDecodeError, KeyError) as e:
        print(f"⚠️ 체크포인트 로드 실패: {e}")
        return None, None


# =====================
# Main
# =====================
def main():
    random.seed(SEED)
    
    # 이미 전처리가 완료되었는지 확인
    if check_preprocessing_completed(OUTPUT_DIR):
        print("🔄 기존 전처리 결과를 사용합니다. 다시 전처리하려면 출력 디렉토리를 삭제하세요.")
        return

    # 체크포인트에서 복원 시도
    checkpoint, token_counter = load_checkpoint(OUTPUT_DIR)
    
    dataset = load_dataset(
        DATASET_NAME,
        split=DATASET_SPLIT,
        cache_dir=CACHE_DIR,
        streaming=True,  # 🔥 핵심
    )

    train_writer = ShardWriter(os.path.join(OUTPUT_DIR, "train"), "train", SHARD_SIZE)
    valid_writer = ShardWriter(os.path.join(OUTPUT_DIR, "valid"), "valid", SHARD_SIZE)
    test_writer = ShardWriter(os.path.join(OUTPUT_DIR, "test"), "test", SHARD_SIZE)

    if token_counter is None:
        token_counter = Counter()

    n_total = 0
    n_train = 0
    n_valid = 0
    n_test = 0
    
    # 체크포인트가 있으면 해당 지점부터 재개
    skip_count = 0
    if checkpoint:
        skip_count = checkpoint['processed_count']
        # 샤드 인덱스 복원
        if 'shard_indices' in checkpoint:
            train_writer.shard_idx = checkpoint['shard_indices']['train']
            valid_writer.shard_idx = checkpoint['shard_indices']['valid']  
            test_writer.shard_idx = checkpoint['shard_indices']['test']
        print(f"🔄 {skip_count:,}개 문장부터 재개합니다...")
    else:
        print("🆕 새로운 전처리를 시작합니다...")

    print(f"🚀 전처리 시작...")
    print(f"   - 출력 디렉토리: {OUTPUT_DIR}")
    print(f"   - 최소/최대 토큰 수: {MIN_TOKENS}-{MAX_TOKENS}")
    print(f"   - 어휘 크기: {VOCAB_SIZE:,}")
    print(f"   - 샤드 크기: {SHARD_SIZE:,}")
    
    start_time = time.time()
    last_checkpoint = time.time()

    current_processed = 0
    
    for record in dataset:
        text = record.get("text", "")
        if not text:
            continue

        sentences = better_sentence_split(text)

        for raw_sent in sentences:
            # 체크포인트 이전 데이터는 건너뛰기
            if current_processed < skip_count:
                current_processed += 1
                continue

            if looks_like_wiki_table_or_markup_line(raw_sent):
                continue

            if has_url(raw_sent):
                continue

            toks = tokenize(raw_sent, LOWERCASE)

            if not (MIN_TOKENS <= len(toks) <= MAX_TOKENS):
                continue

            if should_skip_low_quality_sentence(toks):
                continue

            sentence = " ".join(toks)

            # =====================
            # split (online)
            # =====================
            r = random.random()

            if r < TEST_RATIO:
                test_writer.add(sentence)
                n_test += 1
            elif r < TEST_RATIO + VALID_RATIO:
                valid_writer.add(sentence)
                n_valid += 1
            else:
                train_writer.add(sentence)
                n_train += 1

            n_total += 1

            # =====================
            # vocab sampling
            # =====================
            if random.random() < VOCAB_SAMPLE_RATE:
                token_counter.update(toks)
            
            # 진행 상황 출력 (10,000개마다)
            if n_total % 10000 == 0 and n_total > 0:
                elapsed = time.time() - start_time
                rate = n_total / elapsed
                print(f"📊 처리된 문장: {n_total:,} ({rate:.1f} 문장/초)")
                
                # 체크포인트 저장 (60초마다)
                if time.time() - last_checkpoint > 60:
                    save_checkpoint(OUTPUT_DIR, n_total, token_counter, 
                                  [train_writer, valid_writer, test_writer])
                    last_checkpoint = time.time()

    # flush shards
    train_files = train_writer.close()
    valid_files = valid_writer.close()
    test_files = test_writer.close()

    # =====================
    # vocab 생성
    # =====================
    vocab_path = os.path.join(OUTPUT_DIR, "vocab.txt")

    special_tokens = ["<S>", "</S>", "<UNK>"]

    most_common = [
        tok for tok, _ in token_counter.most_common(
            max(0, VOCAB_SIZE - len(special_tokens))
        )
    ]

    vocab_tokens = special_tokens + [
        t for t in most_common if t not in special_tokens
    ]

    with open(vocab_path, "w", encoding="utf-8") as f:
        f.write("\n".join(vocab_tokens))
        f.write("\n")

    # =====================
    # meta 저장
    # =====================
    meta = {
        "dataset_name": DATASET_NAME,
        "num_sentences": n_total,
        "num_train": n_train,
        "num_valid": n_valid,
        "num_test": n_test,
        "vocab_size": len(vocab_tokens),
        "train_prefix": os.path.join(OUTPUT_DIR, "train", "train_*.txt"),
        "valid_prefix": os.path.join(OUTPUT_DIR, "valid", "valid_*.txt"),
        "test_prefix": os.path.join(OUTPUT_DIR, "test", "test_*.txt"),
        "vocab_file": vocab_path,
    }

    with open(os.path.join(OUTPUT_DIR, "preprocess_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # 체크포인트 파일들 삭제
    checkpoint_path = os.path.join(OUTPUT_DIR, "checkpoint.json")
    vocab_checkpoint_path = os.path.join(OUTPUT_DIR, "vocab_checkpoint.json")
    
    for path in [checkpoint_path, vocab_checkpoint_path]:
        if os.path.exists(path):
            os.remove(path)
    
    elapsed_time = time.time() - start_time
    print(f"✅ 전처리 완료! (소요 시간: {elapsed_time/60:.1f}분)")
    print(f"📈 처리 속도: {n_total/(elapsed_time):.1f} 문장/초")
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()