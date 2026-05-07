"""
전처리된 ELMo 데이터를 로딩하고 검증하는 유틸리티
"""

import json
import os
import glob
import random
from typing import List, Iterator, Tuple, Optional
from collections import Counter


class ELMoDataLoader:
    """전처리된 ELMo 데이터를 로딩하는 클래스"""
    
    def __init__(self, data_dir: str):
        """
        Args:
            data_dir: 전처리된 데이터가 저장된 디렉토리
        """
        self.data_dir = data_dir
        self.meta_path = os.path.join(data_dir, "preprocess_meta.json")
        self.vocab_path = os.path.join(data_dir, "vocab.txt")
        
        # 메타데이터 로드
        self.meta = self._load_meta()
        
        # 어휘 사전 로드
        self.vocab = self._load_vocab()
        self.word_to_id = {word: i for i, word in enumerate(self.vocab)}
        self.id_to_word = {i: word for i, word in enumerate(self.vocab)}
        
    def _load_meta(self) -> dict:
        """메타데이터 로드"""
        if not os.path.exists(self.meta_path):
            raise FileNotFoundError(f"메타데이터 파일을 찾을 수 없습니다: {self.meta_path}")
        
        with open(self.meta_path, 'r') as f:
            return json.load(f)
    
    def _load_vocab(self) -> List[str]:
        """어휘 사전 로드"""
        if not os.path.exists(self.vocab_path):
            raise FileNotFoundError(f"어휘 파일을 찾을 수 없습니다: {self.vocab_path}")
        
        with open(self.vocab_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    
    def get_data_files(self, split: str) -> List[str]:
        """특정 분할의 데이터 파일 목록 반환"""
        split_dir = os.path.join(self.data_dir, split)
        if not os.path.exists(split_dir):
            return []
        
        pattern = os.path.join(split_dir, f"{split}_*.txt")
        files = sorted(glob.glob(pattern))
        return files
    
    def load_sentences(self, split: str, max_files: Optional[int] = None) -> Iterator[str]:
        """문장들을 순차적으로 로드"""
        files = self.get_data_files(split)
        
        if max_files:
            files = files[:max_files]
        
        for file_path in files:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        yield line
    
    def load_tokenized_sentences(self, split: str, max_files: Optional[int] = None) -> Iterator[List[str]]:
        """토큰화된 문장들을 순차적으로 로드"""
        for sentence in self.load_sentences(split, max_files):
            yield sentence.split()
    
    def load_indexed_sentences(self, split: str, max_files: Optional[int] = None) -> Iterator[List[int]]:
        """인덱스로 변환된 문장들을 순차적으로 로드"""
        unk_id = self.word_to_id.get('<UNK>', 0)
        
        for tokens in self.load_tokenized_sentences(split, max_files):
            indices = [self.word_to_id.get(token, unk_id) for token in tokens]
            yield indices
    
    def get_vocab_size(self) -> int:
        """어휘 크기 반환"""
        return len(self.vocab)
    
    def get_stats(self) -> dict:
        """데이터셋 통계 반환"""
        return {
            'dataset_name': self.meta.get('dataset_name', 'Unknown'),
            'total_sentences': self.meta.get('num_sentences', 0),
            'train_sentences': self.meta.get('num_train', 0),
            'valid_sentences': self.meta.get('num_valid', 0),
            'test_sentences': self.meta.get('num_test', 0),
            'vocab_size': len(self.vocab),
            'train_files': len(self.get_data_files('train')),
            'valid_files': len(self.get_data_files('valid')),
            'test_files': len(self.get_data_files('test'))
        }


def validate_preprocessed_data(data_dir: str) -> bool:
    """전처리된 데이터의 무결성 검증"""
    print(f"🔍 데이터 검증 시작: {data_dir}")
    
    try:
        # 데이터 로더 생성
        loader = ELMoDataLoader(data_dir)
        
        # 기본 통계 확인
        stats = loader.get_stats()
        print(f"📊 데이터셋 통계:")
        for key, value in stats.items():
            print(f"   - {key}: {value:,}")
        
        # 각 분할별 샘플 확인
        for split in ['train', 'valid', 'test']:
            files = loader.get_data_files(split)
            if not files:
                print(f"⚠️ {split} 분할에 파일이 없습니다.")
                continue
            
            # 첫 번째 파일에서 몇 개 샘플 확인
            sample_count = 0
            for sentence in loader.load_sentences(split, max_files=1):
                sample_count += 1
                if sample_count == 1:
                    print(f"✅ {split} 샘플: {sentence[:100]}...")
                if sample_count >= 3:  # 처음 3개만 확인
                    break
            
            print(f"   - {split}: {len(files)}개 파일, 첫 파일에서 {sample_count}개 샘플 확인")
        
        # 어휘 사전 검증
        special_tokens = ['<S>', '</S>', '<UNK>']
        missing_special = [token for token in special_tokens if token not in loader.vocab]
        if missing_special:
            print(f"⚠️ 누락된 특수 토큰: {missing_special}")
        else:
            print(f"✅ 특수 토큰 확인 완료: {special_tokens}")
        
        # 인덱싱 테스트
        test_sentence = "This is a test sentence ."
        tokens = test_sentence.split()
        indices = [loader.word_to_id.get(token, loader.word_to_id.get('<UNK>', 0)) for token in tokens]
        reconstructed = [loader.id_to_word.get(idx, '<UNK>') for idx in indices]
        
        print(f"🧪 인덱싱 테스트:")
        print(f"   - 원본: {tokens}")
        print(f"   - 인덱스: {indices}")
        print(f"   - 복원: {reconstructed}")
        
        print(f"✅ 데이터 검증 완료!")
        return True
        
    except Exception as e:
        print(f"❌ 데이터 검증 실패: {e}")
        return False


def sample_data(data_dir: str, split: str = 'train', num_samples: int = 10):
    """데이터 샘플 출력"""
    print(f"📋 {split} 데이터 샘플 ({num_samples}개):")
    
    try:
        loader = ELMoDataLoader(data_dir)
        
        count = 0
        for sentence in loader.load_sentences(split, max_files=1):
            print(f"   {count+1:2d}: {sentence}")
            count += 1
            if count >= num_samples:
                break
                
    except Exception as e:
        print(f"❌ 샘플 출력 실패: {e}")


def main():
    """메인 함수 - 데이터 검증 및 샘플 출력"""
    import sys
    
    if len(sys.argv) < 2:
        print("사용법: python data_loader.py <data_dir> [action]")
        print("  action: validate (기본값), sample")
        return
    
    data_dir = sys.argv[1]
    action = sys.argv[2] if len(sys.argv) > 2 else 'validate'
    
    if action == 'validate':
        validate_preprocessed_data(data_dir)
    elif action == 'sample':
        sample_data(data_dir)
    else:
        print(f"알 수 없는 액션: {action}")


if __name__ == "__main__":
    main()