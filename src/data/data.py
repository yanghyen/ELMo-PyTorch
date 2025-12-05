# src/data.py
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
import numpy as np
import os
import pickle
from typing import Generator, List

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR    = os.path.dirname(SCRIPT_DIR)
ROOT_DIR   = os.path.dirname(SRC_DIR)
TOKEN_INDICES_PATH = os.path.join(ROOT_DIR, "data/pretrain/token_indices.npy")


class ELMoIterableDataset(IterableDataset):
    """
    ELMo 학습을 위한 Iterable Dataset
    Forward와 Backward 언어 모델 학습을 위한 시퀀스 생성 
    """
    def __init__(self, file_path, word2idx, seq_len=20):
        super().__init__()
        self.file_path = file_path
        self.word2idx = word2idx
        self.seq_len = seq_len
        self.vocab_size = len(word2idx)
        self.idx2word = {i: w for w, i in word2idx.items()}
        
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Binary index file not found: {self.file_path}. Run preprocessing first!")
        
        try:
            self.token_indices = np.load(self.file_path, mmap_mode='r')
            self.total_tokens = len(self.token_indices)
            print(f"Loaded token indices (mmap): Total tokens = {self.total_tokens:,}")
        except Exception as e:
            raise RuntimeError(f"Error loading token indices file via mmap: {e}")
    
    def __iter__(self):
        """Forward와 Backward 시퀀스 쌍을 yield"""
        
        worker_info = torch.utils.data.get_worker_info()
        overlap = self.seq_len  # 시퀀스 길이만큼 오버랩
        
        if worker_info is None:
            start_idx = 0
            end_idx = self.total_tokens
        else:
            per_worker = self.total_tokens // worker_info.num_workers
            start_idx = worker_info.id * per_worker
            end_idx = start_idx + per_worker
            
            if worker_info.id > 0:
                start_idx = max(0, start_idx - overlap)
            if worker_info.id < worker_info.num_workers - 1:
                end_idx = min(self.total_tokens, end_idx + overlap)
            else:
                end_idx = self.total_tokens
        
        current_idx = start_idx
        
        while current_idx < end_idx - self.seq_len:
            # 시퀀스 추출
            seq_end = min(current_idx + self.seq_len + 1, end_idx)
            sequence = self.token_indices[current_idx:seq_end]
            
            if len(sequence) < 2:
                current_idx += 1
                continue
            
            # Forward: [w1, w2, ..., wn] -> [w2, w3, ..., wn+1]
            forward_input = torch.tensor(sequence[:-1], dtype=torch.long)
            forward_target = torch.tensor(sequence[1:], dtype=torch.long)
            
            # Backward: [wn, wn-1, ..., w1] -> [wn-1, wn-2, ..., w0]
            backward_sequence = sequence[::-1]
            backward_input = torch.tensor(backward_sequence[:-1], dtype=torch.long)
            backward_target = torch.tensor(backward_sequence[1:], dtype=torch.long)
            
            yield forward_input, backward_input, forward_target, backward_target
            
            # 다음 시퀀스로 이동 (오버랩을 위해 seq_len만큼 이동)
            current_idx += self.seq_len


def collate_fn_elmo(batch):
    """
    배치 내 시퀀스들을 패딩하여 동일한 길이로 맞춤
    """
    forward_inputs, backward_inputs, forward_targets, backward_targets = zip(*batch)
    
    # 최대 길이 계산
    max_len = max(len(seq) for seq in forward_inputs)
    
    batch_size = len(batch)
    
    # Forward 패딩
    padded_forward_input = torch.zeros(batch_size, max_len, dtype=torch.long)
    padded_forward_target = torch.zeros(batch_size, max_len, dtype=torch.long)
    forward_mask = torch.zeros(batch_size, max_len, dtype=torch.float)
    
    # Backward 패딩
    padded_backward_input = torch.zeros(batch_size, max_len, dtype=torch.long)
    padded_backward_target = torch.zeros(batch_size, max_len, dtype=torch.long)
    backward_mask = torch.zeros(batch_size, max_len, dtype=torch.float)
    
    for i, (f_in, b_in, f_tgt, b_tgt) in enumerate(batch):
        f_len = len(f_in)
        b_len = len(b_in)
        
        padded_forward_input[i, :f_len] = f_in
        padded_forward_target[i, :f_len] = f_tgt
        forward_mask[i, :f_len] = 1.0
        
        padded_backward_input[i, :b_len] = b_in
        padded_backward_target[i, :b_len] = b_tgt
        backward_mask[i, :b_len] = 1.0
    
    return (
        padded_forward_input,
        padded_backward_input,
        padded_forward_target,
        padded_backward_target,
        forward_mask,
        backward_mask
    )


def get_dataloader(file_path, config, word2idx):
    """
    ELMo 학습을 위한 DataLoader 생성
    
    Args:
        file_path: 토큰 인덱스 파일 경로
        config: 설정 딕셔너리
        word2idx: 단어-인덱스 매핑
    
    Returns:
        DataLoader
    """
    dataset = ELMoIterableDataset(
        file_path=file_path,
        word2idx=word2idx,
        seq_len=config.get("seq_len", 20)
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config.get("num_workers", 16),
        pin_memory=True,
        collate_fn=collate_fn_elmo
    )
    
    return dataloader


