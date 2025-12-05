import os
import pickle
import re
from typing import Generator, List, Tuple, Dict
from collections import Counter

import numpy as np
import nltk
try:
    nltk.download("stopwords", quiet=True)
    nltk.download("punkt", quiet=True) 
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
except ImportError:
    print("NLTK is not installed. Using simple split() for tokenization.")
    def word_tokenize(text):
        return re.findall(r"\b\w+\b", text) 
    stopwords = set()

# -----------------------------
# 경로 설정
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(SCRIPT_DIR)
ROOT_DIR = os.path.dirname(SRC_DIR)

CORPUS_PATH = os.path.join(ROOT_DIR, "data/pretrain/raw/elmo_corpus.txt")
TOKENIZED_TRAIN_PATH = os.path.join(ROOT_DIR, "data/pretrain/tokenized_corpus.txt")
TOKEN_INDICES_PATH = os.path.join(ROOT_DIR, "data/pretrain/token_indices.npy")

# ELMo 특수 토큰
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
BOS_TOKEN = "<S>"
EOS_TOKEN = "</S>"

STOPWORDS = set(stopwords.words('english')) if 'stopwords' in locals() and stopwords else set()

# -----------------------------
def preprocess_tokens(tokens: list):
    """토큰 리스트 전체 전처리"""
    # clean_token 제거: 토큰을 그대로 반환
    return [t for t in tokens if t]

# -----------------------------
def preprocess_text(text: str) -> list:
    """단일 문서 텍스트에 대해 전처리 및 토큰화를 수행합니다."""

    text = re.sub(r'==\s*(References|External links|See also|Notes|Sources)\s*==.*', '', text, flags=re.DOTALL | re.IGNORECASE)
    
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    tokens = word_tokenize(text)

    return preprocess_tokens(tokens)

# -----------------------------
def process_corpus_and_stream(path=CORPUS_PATH) -> Generator[List[str], None, None]:
    """
    원본 파일을 한 줄씩 읽어 문서('\n\n'으로 구분)를 재구성하고, 
    전처리 및 토큰화된 토큰 리스트(문장/문맥 단위)를 순차적으로 yield
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Corpus file not found: {path}.")
    
    print(f"Starting streaming process from {path}.")
    
    doc_buffer = []
    doc_count = 0
    
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            
            if line:
                doc_buffer.append(line)
                
            if not line and doc_buffer:
                doce_text = " ".join(doc_buffer)
                
                tokens = preprocess_text(doce_text)
                
                if tokens:
                    yield tokens 
                    doc_count += 1
                    
                doc_buffer = []
                
                if doc_count % 100000 == 0 and doc_count > 0:
                    print(f"Processed {doc_count:,} documents so far...")
                    
        if doc_buffer:
            doce_text = " ".join(doc_buffer)
            tokens = preprocess_text(doce_text)
            if tokens:
                yield tokens 
                doc_count += 1
    print(f"\nProcessing complete. Total documents processd: {doc_count:,}")


def build_vocab_stream(
    file_path: str,
    min_count: int = 1
) -> Tuple[List[str], Dict[str, int], Dict[int, str], Dict[str, int]]:
    """
    토큰화된 파일을 스트리밍 방식으로 읽어 vocab을 구축합니다.
    
    Args:
        file_path: 토큰화된 텍스트 파일 경로 (한 줄에 공백으로 구분된 토큰들)
        min_count: vocab에 포함될 최소 빈도수
    
    Returns:
        vocab: 단어 리스트 (특수 토큰 포함)
        word2idx: 단어 -> 인덱스 매핑
        idx2word: 인덱스 -> 단어 매핑
        word_freq: 단어 빈도 딕셔너리
    """
    print(f"Building vocabulary from {file_path}...")
    
    word_counter = Counter()
    total_lines = 0
    
    # 1. 단어 빈도 계산
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            tokens = line.strip().split()
            word_counter.update(tokens)
            total_lines += 1
            if total_lines % 100000 == 0:
                print(f"  Processed {total_lines:,} lines...")
    
    print(f"Total unique tokens (before filtering): {len(word_counter):,}")
    
    # 2. min_count 이상인 단어만 선택
    filtered_words = {word: count for word, count in word_counter.items() 
                     if count >= min_count}
    print(f"Tokens with count >= {min_count}: {len(filtered_words):,}")
    
    # 3. 특수 토큰 추가 (ELMo 학습에 필요)
    # 순서: PAD(0), UNK(1), BOS(2), EOS(3), 일반 단어들(4~)
    vocab = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]
    vocab.extend(sorted(filtered_words.keys()))
    
    # 4. word2idx, idx2word 생성
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for word, idx in word2idx.items()}
    
    # 5. word_freq 생성 (특수 토큰 제외하고 실제 단어만)
    word_freq = {word: count for word, count in word_counter.items() 
                if word in word2idx and word not in [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]}
    
    print(f"✅ Vocabulary built: {len(vocab):,} tokens (including special tokens)")
    print(f"   Special tokens: {PAD_TOKEN}(0), {UNK_TOKEN}(1), {BOS_TOKEN}(2), {EOS_TOKEN}(3)")
    
    return vocab, word2idx, idx2word, word_freq


def save_token_indices_to_binary(
    token_stream: Generator[List[str], None, None],
    word2idx: dict,
    save_path=TOKEN_INDICES_PATH
):
    """
    토큰 스트림을 인덱스로 변환하여 바이너리 파일로 저장합니다.
    vocab에 없는 토큰은 UNK 토큰으로 변환합니다.
    """
    print(f"Indexing corpus and saving to {save_path}...")

    all_indices = []
    total_tokens_count = 0
    unk_count = 0
    
    unk_idx = word2idx.get(UNK_TOKEN, 1)
    
    for tokens in token_stream:
        indices = []
        for token in tokens:
            if token in word2idx:
                indices.append(word2idx[token])
            else:
                indices.append(unk_idx)
                unk_count += 1
        all_indices.extend(indices)
        
        total_tokens_count += len(indices)
        if total_tokens_count % 50000000 == 0 and total_tokens_count > 0:
            print(f"Tokens indexed so far: {total_tokens_count:,} (UNK: {unk_count:,})")
    
    token_indices_array = np.array(all_indices, dtype=np.int32)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, token_indices_array)
    
    print(f"\n✅ Corpus indexing complete.")
    print(f"   Total indices: {len(token_indices_array):,}")
    print(f"   UNK tokens: {unk_count:,} ({unk_count/len(token_indices_array)*100:.2f}%)")
    print(f"   Saved to {save_path}")

if __name__ == "__main__":
    
    # ----------------------------- 1. Vocab 구축 및 임시 파일 생성 (NS/HS 공통) -----------------------------
    try:
        # A. 원본 코퍼스를 읽어 토큰화된 내용을 임시 파일에 저장 (Vocab 구축용)
        print(f"Saving temporary tokenized corpus to {TOKENIZED_TRAIN_PATH} for vocab building...")
        total_temp_tokens = 0
        os.makedirs(os.path.dirname(TOKENIZED_TRAIN_PATH), exist_ok=True)
        with open(TOKENIZED_TRAIN_PATH, "w", encoding="utf-8") as f:
            temp_stream = process_corpus_and_stream(CORPUS_PATH)
            for tokens in temp_stream:
                f.write(" ".join(tokens) + "\n")
                total_temp_tokens += len(tokens)
        print(f"Temporary tokenized file created. Total tokens: {total_temp_tokens}")
        
        # B. 임시 파일로 Vocab 구축
        VOCAB_MIN_COUNT = 3 # config 값을 가정
        vocab, word2idx, idx2word, word_freq = build_vocab_stream(
            TOKENIZED_TRAIN_PATH,
            min_count=VOCAB_MIN_COUNT
        )
        
        # ----------------------------- 2. Vocab 파일 저장 (NS/HS 공통) -----------------------------
        # Vocab 파일 저장: train.py가 로드할 수 있도록 저장합니다.
        vocab_data = {
            "vocab": vocab, 
            "word2idx": word2idx, 
            "idx2word": idx2word, 
            "word_freq": word_freq,
            "vocab_size": len(vocab),
            "special_tokens": {
                "PAD": PAD_TOKEN,
                "UNK": UNK_TOKEN,
                "BOS": BOS_TOKEN,
                "EOS": EOS_TOKEN
            }
        }
        vocab_filename = "vocab.pkl"
        vocab_path = os.path.join(ROOT_DIR, "data/pretrain", vocab_filename)
        os.makedirs(os.path.dirname(vocab_path), exist_ok=True)
        
        with open(vocab_path, "wb") as f:
            pickle.dump(vocab_data, f)
        print(f"✅ Final Vocab saved to {vocab_path}")
        print(f"   Vocab size: {len(vocab):,}")
        print(f"   Word frequency stats: min={min(word_freq.values()) if word_freq else 0}, "
              f"max={max(word_freq.values()) if word_freq else 0}")
        
        # ----------------------------- 3. 학습 인덱스 생성 및 저장 -----------------------------
        # Vocab 구축을 위해 사용한 스트림은 소진되었으므로, 새 스트림 생성
        print("\n" + "="*60)
        print("Starting token indexing for training...")
        print("="*60)
        final_token_stream = process_corpus_and_stream(CORPUS_PATH) 
        save_token_indices_to_binary(final_token_stream, word2idx, TOKEN_INDICES_PATH)
        
        # ----------------------------- 4. 임시 파일 삭제 (유지) -----------------------------
        if os.path.exists(TOKENIZED_TRAIN_PATH):
            os.remove(TOKENIZED_TRAIN_PATH) # 👈 이 파일은 이제 필요 없으므로 삭제
            print(f"🧹 Removed temporary file: {TOKENIZED_TRAIN_PATH}")
            
    except FileNotFoundError as e:
        print(f"오류: {e}")
    except Exception as e:
        print(f"치명적 오류: {e}")