#!/usr/bin/env python3

# parser
# --config, --data_path, --vocab_path, --checkpoint_dir, --resume 

"""
ELMo BiLM 사전학습 스크립트
elmo_depth-1_seed-42.yaml 설정으로 모델 학습
"""

import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pickle
import time
import argparse
from tqdm import tqdm
import logging
import json
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import psutil
import GPUtil
from datetime import datetime
import pandas as pd

# 프로젝트 루트 디렉토리를 Python path에 추가
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(SCRIPT_DIR)
ROOT_DIR = os.path.dirname(SRC_DIR)
sys.path.append(SRC_DIR)

# 경로 상수 정의
CHECKPOINT_DIR = "runs/checkpoints"
DATA_DIR = "data/pretrain"
RESULTS_DIR = "results"
LOGS_DIR = "logs"

# 기본 파일 경로
DEFAULT_DATA_PATH = os.path.join(DATA_DIR, "token_indices.npy")
DEFAULT_VOCAB_PATH = os.path.join(DATA_DIR, "vocab.pkl")

from model.bilm import create_elmo_model
from data.data import get_dataloader

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('train_bilm.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def set_seed(seed):
    """재현 가능한 결과를 위한 시드 설정"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path):
    """YAML 설정 파일 로드"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def load_vocabulary(vocab_path):
    """어휘 사전 로드"""
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Vocabulary file not found: {vocab_path}")
    
    with open(vocab_path, 'rb') as f:
        vocab_data = pickle.load(f)
    
    # vocab.pkl이 딕셔너리 형태로 저장된 경우 word2idx 추출
    if isinstance(vocab_data, dict) and 'word2idx' in vocab_data:
        word2idx = vocab_data['word2idx']
    else:
        # 기존 형식 (직접 word2idx가 저장된 경우)
        word2idx = vocab_data
    
    logger.info(f"Loaded vocabulary: {len(word2idx):,} words")
    return word2idx


def save_checkpoint(model, optimizer, epoch, loss, checkpoint_path, metrics=None, config=None):
    """체크포인트 저장 (메트릭 포함)"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'metrics': metrics or {},
        'config': config,
        'timestamp': datetime.now().isoformat(),
    }
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Checkpoint saved: {checkpoint_path}")


def load_checkpoint(model, optimizer, checkpoint_path):
    """체크포인트 로드"""
    if not os.path.exists(checkpoint_path):
        logger.info("No checkpoint found, starting from scratch")
        return 0, float('inf')
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    logger.info(f"Checkpoint loaded: epoch {epoch}, loss {loss:.4f}")
    return epoch, loss


class MetricsTracker:
    """학습 메트릭 추적 및 분석"""
    
    def __init__(self, log_dir="logs", results_dir="runs/metrics"):
        self.log_dir = log_dir
        self.results_dir = results_dir
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)
        
        self.metrics = defaultdict(list)
        self.efficiency_metrics = defaultdict(list)
        self.error_analysis = defaultdict(list)
        
        # CSV 저장을 위한 데이터프레임
        self.metrics_df = pd.DataFrame()
        self.efficiency_df = pd.DataFrame()
        self.error_df = pd.DataFrame()
        
        # GPU 정보 초기화
        self.gpu_available = torch.cuda.is_available()
        if self.gpu_available:
            self.gpu_info = {
                'name': torch.cuda.get_device_name(),
                'memory_total': torch.cuda.get_device_properties(0).total_memory,
                'compute_capability': torch.cuda.get_device_properties(0).major
            }
    
    def compute_perplexity(self, loss):
        """Perplexity 계산"""
        return torch.exp(loss).item()
    
    def compute_accuracy(self, logits, targets, mask=None):
        """Top-1 accuracy 계산"""
        predictions = torch.argmax(logits, dim=-1)
        correct = (predictions == targets).float()
        
        if mask is not None:
            correct = correct * mask
            accuracy = correct.sum() / mask.sum()
        else:
            accuracy = correct.mean()
        
        return accuracy.item()
    
    def compute_top_k_accuracy(self, logits, targets, k=5, mask=None):
        """Top-k accuracy 계산"""
        _, top_k_preds = torch.topk(logits, k, dim=-1)
        targets_expanded = targets.unsqueeze(-1).expand_as(top_k_preds)
        correct = (top_k_preds == targets_expanded).any(dim=-1).float()
        
        if mask is not None:
            correct = correct * mask
            accuracy = correct.sum() / mask.sum()
        else:
            accuracy = correct.mean()
        
        return accuracy.item()
    
    def track_efficiency(self, batch_size, seq_len, forward_time, memory_used, batch_idx=None):
        """효율성 메트릭 추적"""
        throughput = batch_size / forward_time  # samples per second
        memory_per_sample = memory_used / batch_size if batch_size > 0 else 0
        
        # 기존 딕셔너리에 추가
        self.efficiency_metrics['throughput'].append(throughput)
        self.efficiency_metrics['memory_per_sample'].append(memory_per_sample)
        self.efficiency_metrics['forward_time'].append(forward_time)
        self.efficiency_metrics['batch_size'].append(batch_size)
        self.efficiency_metrics['seq_len'].append(seq_len)
        
        # CSV용 데이터 준비
        efficiency_row = {
            'batch_idx': batch_idx or len(self.efficiency_metrics['throughput']) - 1,
            'timestamp': datetime.now().isoformat(),
            'throughput': throughput,
            'memory_per_sample': memory_per_sample,
            'forward_time': forward_time,
            'batch_size': batch_size,
            'seq_len': seq_len,
            'memory_used': memory_used
        }
        
        # DataFrame에 추가
        new_row = pd.DataFrame([efficiency_row])
        self.efficiency_df = pd.concat([self.efficiency_df, new_row], ignore_index=True)
        
        # Efficiency CSV는 최종에만 저장 (배치마다 저장하지 않음)
    
    def analyze_errors(self, logits, targets, vocab, mask=None, sample_size=20):
        """에러 분석 - 언어학적 현상별 분류"""
        predictions = torch.argmax(logits, dim=-1)
        errors = (predictions != targets)
        
        if mask is not None:
            errors = errors * mask.bool()
        
        # 에러 샘플링
        error_positions = torch.nonzero(errors, as_tuple=False)
        if len(error_positions) > sample_size:
            indices = torch.randperm(len(error_positions))[:sample_size]
            error_positions = error_positions[indices]
        
        error_samples = []
        for pos in error_positions:
            batch_idx, seq_idx = pos[0].item(), pos[1].item()
            
            target_id = targets[batch_idx, seq_idx].item()
            pred_id = predictions[batch_idx, seq_idx].item()
            
            # 어휘 복원 (가능한 경우)
            target_word = vocab.get(target_id, f"<UNK_{target_id}>")
            pred_word = vocab.get(pred_id, f"<UNK_{pred_id}>")
            
            error_info = {
                'target': target_word,
                'prediction': pred_word,
                'target_id': target_id,
                'pred_id': pred_id,
                'position': seq_idx,
                'category': self._categorize_error(target_word, pred_word)
            }
            error_samples.append(error_info)
        
        self.error_analysis['samples'].extend(error_samples)
        
        # CSV용 에러 데이터 준비
        for error in error_samples:
            error_row = {
                'timestamp': datetime.now().isoformat(),
                'target': error['target'],
                'prediction': error['prediction'],
                'target_id': error['target_id'],
                'pred_id': error['pred_id'],
                'position': error['position'],
                'category': error['category']
            }
            new_row = pd.DataFrame([error_row])
            self.error_df = pd.concat([self.error_df, new_row], ignore_index=True)
        
        return error_samples
    
    def _categorize_error(self, target, prediction):
        """에러를 언어학적 현상으로 분류"""
        # 간단한 휴리스틱 기반 분류
        if target.startswith('<') or prediction.startswith('<'):
            return 'OOV'
        elif target.isupper() or prediction.isupper():
            return 'named_entities'
        elif target.lower() in ['not', 'no', 'never', 'nothing']:
            return 'negation'
        elif target.lower() in ['he', 'she', 'it', 'they', 'his', 'her', 'their']:
            return 'coreference'
        elif len(target) <= 2 or len(prediction) <= 2:
            return 'short_words'
        elif target.isdigit() or prediction.isdigit():
            return 'numbers'
        else:
            return 'other'
    
    def log_metrics(self, epoch, metrics_dict):
        """메트릭 로깅 (JSON + CSV)"""
        # 기존 딕셔너리에 추가
        for key, value in metrics_dict.items():
            self.metrics[key].append(value)
        
        # JSON으로 저장 (상세 로그용)
        metrics_file = os.path.join(self.log_dir, f'metrics_epoch_{epoch}.json')
        with open(metrics_file, 'w') as f:
            json.dump(metrics_dict, f, indent=2)
        
        # CSV용 데이터 준비
        csv_row = {'epoch': epoch, 'timestamp': datetime.now().isoformat()}
        csv_row.update(metrics_dict)
        
        # DataFrame에 추가
        new_row = pd.DataFrame([csv_row])
        self.metrics_df = pd.concat([self.metrics_df, new_row], ignore_index=True)
        
        # CSV는 최종에만 저장 (매 에포크마다 저장하지 않음)
    
    def save_efficiency_report(self):
        """효율성 리포트 저장 (JSON + CSV)"""
        if not self.efficiency_metrics['throughput']:
            return
        
        # JSON 리포트 (요약)
        report = {
            'gpu_info': self.gpu_info if self.gpu_available else None,
            'avg_throughput': np.mean(self.efficiency_metrics['throughput']),
            'std_throughput': np.std(self.efficiency_metrics['throughput']),
            'avg_memory_per_sample': np.mean(self.efficiency_metrics['memory_per_sample']),
            'std_memory_per_sample': np.std(self.efficiency_metrics['memory_per_sample']),
            'avg_forward_time': np.mean(self.efficiency_metrics['forward_time']),
            'std_forward_time': np.std(self.efficiency_metrics['forward_time']),
            'total_samples': sum(self.efficiency_metrics['batch_size']),
            'total_batches': len(self.efficiency_metrics['batch_size'])
        }
        
        report_file = os.path.join(self.log_dir, 'efficiency_report.json')
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # CSV 최종 저장
        csv_file = os.path.join(self.results_dir, 'efficiency_metrics.csv')
        self.efficiency_df.to_csv(csv_file, index=False)
        
        # 요약 통계 CSV
        summary_stats = {
            'metric': ['throughput', 'memory_per_sample', 'forward_time'],
            'mean': [report['avg_throughput'], report['avg_memory_per_sample'], report['avg_forward_time']],
            'std': [report['std_throughput'], report['std_memory_per_sample'], report['std_forward_time']],
            'total_samples': [report['total_samples']] * 3,
            'total_batches': [report['total_batches']] * 3
        }
        summary_df = pd.DataFrame(summary_stats)
        summary_csv = os.path.join(self.results_dir, 'efficiency_summary.csv')
        summary_df.to_csv(summary_csv, index=False)
        
        logger.info(f"Efficiency report saved: {report_file}")
        logger.info(f"Efficiency CSV saved: {csv_file}")
        logger.info(f"Efficiency summary saved: {summary_csv}")
    
    def save_error_analysis(self):
        """에러 분석 결과 저장 (JSON + CSV)"""
        if not self.error_analysis['samples']:
            return
        
        # 카테고리별 에러 통계
        categories = Counter([sample['category'] for sample in self.error_analysis['samples']])
        
        # JSON 분석 (상세)
        analysis = {
            'total_errors_analyzed': len(self.error_analysis['samples']),
            'error_categories': dict(categories),
            'error_samples': self.error_analysis['samples'][:50],  # 상위 50개만 저장
        }
        
        analysis_file = os.path.join(self.log_dir, 'error_analysis.json')
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2, ensure_ascii=False)
        
        # CSV 저장 (전체 에러 데이터)
        if not self.error_df.empty:
            csv_file = os.path.join(self.results_dir, 'error_analysis.csv')
            self.error_df.to_csv(csv_file, index=False)
            
            # 카테고리별 통계 CSV
            category_stats = pd.DataFrame([
                {'category': cat, 'count': count, 'percentage': count/len(self.error_analysis['samples'])*100}
                for cat, count in categories.items()
            ])
            category_csv = os.path.join(self.results_dir, 'error_categories.csv')
            category_stats.to_csv(category_csv, index=False)
            
            logger.info(f"Error analysis CSV saved: {csv_file}")
            logger.info(f"Error categories CSV saved: {category_csv}")
        
        logger.info(f"Error analysis JSON saved: {analysis_file}")
        logger.info(f"Error categories: {dict(categories)}")
    
    def plot_metrics(self):
        """메트릭 시각화"""
        if not self.metrics:
            return
        
        # Loss 및 Perplexity 플롯
        plt.figure(figsize=(15, 10))
        
        # Loss plot
        plt.subplot(2, 3, 1)
        if 'total_loss' in self.metrics:
            plt.plot(self.metrics['total_loss'], label='Total Loss')
        if 'forward_loss' in self.metrics:
            plt.plot(self.metrics['forward_loss'], label='Forward Loss')
        if 'backward_loss' in self.metrics:
            plt.plot(self.metrics['backward_loss'], label='Backward Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.legend()
        plt.grid(True)
        
        # Perplexity plot
        plt.subplot(2, 3, 2)
        if 'perplexity' in self.metrics:
            plt.plot(self.metrics['perplexity'])
        plt.xlabel('Epoch')
        plt.ylabel('Perplexity')
        plt.title('Perplexity')
        plt.grid(True)
        
        # Accuracy plot
        plt.subplot(2, 3, 3)
        if 'accuracy' in self.metrics:
            plt.plot(self.metrics['accuracy'], label='Top-1')
        if 'top5_accuracy' in self.metrics:
            plt.plot(self.metrics['top5_accuracy'], label='Top-5')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Accuracy')
        plt.legend()
        plt.grid(True)
        
        # Throughput plot
        plt.subplot(2, 3, 4)
        if self.efficiency_metrics['throughput']:
            plt.plot(self.efficiency_metrics['throughput'])
        plt.xlabel('Batch')
        plt.ylabel('Samples/sec')
        plt.title('Throughput')
        plt.grid(True)
        
        # Memory usage plot
        plt.subplot(2, 3, 5)
        if self.efficiency_metrics['memory_per_sample']:
            plt.plot(self.efficiency_metrics['memory_per_sample'])
        plt.xlabel('Batch')
        plt.ylabel('Memory/Sample (MB)')
        plt.title('Memory Usage')
        plt.grid(True)
        
        # Error categories plot
        plt.subplot(2, 3, 6)
        if self.error_analysis['samples']:
            categories = Counter([sample['category'] for sample in self.error_analysis['samples']])
            plt.pie(categories.values(), labels=categories.keys(), autopct='%1.1f%%')
            plt.title('Error Categories')
        
        plt.tight_layout()
        plot_file = os.path.join(self.log_dir, 'training_metrics.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Metrics plot saved: {plot_file}")
    
    def save_final_results(self):
        """최종 결과를 CSV로 저장"""
        logger.info("Saving all CSV files...")
        
        # 1. 학습 메트릭 CSV 저장
        if not self.metrics_df.empty:
            csv_file = os.path.join(self.results_dir, 'training_metrics.csv')
            self.metrics_df.to_csv(csv_file, index=False)
            logger.info(f"Training metrics CSV saved: {csv_file}")
            
            # 최종 메트릭 요약
            final_metrics = self.metrics_df.iloc[-1].to_dict() if len(self.metrics_df) > 0 else {}
            
            # 전체 학습 통계
            training_summary = {
                'total_epochs': len(self.metrics_df),
                'best_loss': self.metrics_df['total_loss'].min() if 'total_loss' in self.metrics_df.columns else 0,
                'final_loss': final_metrics.get('total_loss', 0),
                'best_accuracy': self.metrics_df['accuracy'].max() if 'accuracy' in self.metrics_df.columns else 0,
                'final_accuracy': final_metrics.get('accuracy', 0),
                'best_perplexity': self.metrics_df['perplexity'].min() if 'perplexity' in self.metrics_df.columns else 0,
                'final_perplexity': final_metrics.get('perplexity', 0),
                'avg_epoch_time': self.metrics_df['epoch_time'].mean() if 'epoch_time' in self.metrics_df.columns else 0,
                'total_training_time': self.metrics_df['epoch_time'].sum() if 'epoch_time' in self.metrics_df.columns else 0
            }
            
            # 요약 통계를 CSV로 저장
            summary_df = pd.DataFrame([training_summary])
            summary_csv = os.path.join(self.results_dir, 'training_summary.csv')
            summary_df.to_csv(summary_csv, index=False)
            logger.info(f"Training summary CSV saved: {summary_csv}")
        
        # 2. 효율성 메트릭 CSV 저장
        if not self.efficiency_df.empty:
            csv_file = os.path.join(self.results_dir, 'efficiency_metrics.csv')
            self.efficiency_df.to_csv(csv_file, index=False)
            logger.info(f"Efficiency metrics CSV saved: {csv_file}")
        
        # 3. 에러 분석 CSV 저장
        if not self.error_df.empty:
            csv_file = os.path.join(self.results_dir, 'error_analysis.csv')
            self.error_df.to_csv(csv_file, index=False)
            logger.info(f"Error analysis CSV saved: {csv_file}")
        
        # 4. 최종 효율성 리포트 저장 (JSON + 요약 CSV)
        self.save_efficiency_report()
        
        # 5. 최종 에러 분석 저장 (JSON + 카테고리 CSV)
        self.save_error_analysis()


def get_gpu_memory_usage():
    """GPU 메모리 사용량 조회"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**3  # GB
    return 0


def evaluate_model(model, dataloader, device, metrics_tracker=None, vocab=None, max_batches=None):
    """모델 평가"""
    model.eval()
    total_loss = 0.0
    total_forward_loss = 0.0
    total_backward_loss = 0.0
    total_accuracy = 0.0
    total_top5_accuracy = 0.0
    num_batches = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Evaluating')
        
        for batch_idx, batch in enumerate(pbar):
            if max_batches and batch_idx >= max_batches:
                break
                
            try:
                # 배치 데이터 언패킹
                (forward_input, backward_input, forward_target, backward_target, 
                 forward_mask, backward_mask) = batch
                
                # GPU로 이동
                forward_input = forward_input.to(device)
                backward_input = backward_input.to(device)
                forward_target = forward_target.to(device)
                backward_target = backward_target.to(device)
                forward_mask = forward_mask.to(device)
                backward_mask = backward_mask.to(device)
                
                # Forward pass
                forward_logits, backward_logits = model(
                    forward_input, backward_input, 
                    forward_mask, backward_mask
                )
                
                # Loss 계산
                total_batch_loss, forward_loss, backward_loss = model.compute_loss(
                    forward_logits, backward_logits,
                    forward_target, backward_target,
                    forward_mask, backward_mask
                )
                
                # 메트릭 계산
                if metrics_tracker:
                    # Accuracy 계산
                    forward_acc = metrics_tracker.compute_accuracy(
                        forward_logits, forward_target, forward_mask
                    )
                    backward_acc = metrics_tracker.compute_accuracy(
                        backward_logits, backward_target, backward_mask
                    )
                    avg_accuracy = (forward_acc + backward_acc) / 2
                    
                    # Top-5 accuracy 계산
                    forward_top5 = metrics_tracker.compute_top_k_accuracy(
                        forward_logits, forward_target, k=5, mask=forward_mask
                    )
                    backward_top5 = metrics_tracker.compute_top_k_accuracy(
                        backward_logits, backward_target, k=5, mask=backward_mask
                    )
                    avg_top5_accuracy = (forward_top5 + backward_top5) / 2
                    
                    total_accuracy += avg_accuracy
                    total_top5_accuracy += avg_top5_accuracy
                
                # 통계 업데이트
                total_loss += total_batch_loss.item()
                total_forward_loss += forward_loss.item()
                total_backward_loss += backward_loss.item()
                num_batches += 1
                
                # 진행률 업데이트
                avg_loss = total_loss / num_batches
                avg_acc = total_accuracy / num_batches if metrics_tracker else 0
                
                pbar.set_postfix({
                    'Loss': f'{avg_loss:.4f}',
                    'Acc': f'{avg_acc:.3f}' if metrics_tracker else 'N/A'
                })
                
            except Exception as e:
                logger.error(f"Error in evaluation batch {batch_idx}: {str(e)}")
                continue
    
    # 평균 계산
    avg_loss = total_loss / max(num_batches, 1)
    avg_forward_loss = total_forward_loss / max(num_batches, 1)
    avg_backward_loss = total_backward_loss / max(num_batches, 1)
    avg_accuracy = total_accuracy / max(num_batches, 1) if metrics_tracker else 0
    avg_top5_accuracy = total_top5_accuracy / max(num_batches, 1) if metrics_tracker else 0
    
    eval_metrics = {
        'eval_loss': avg_loss,
        'eval_forward_loss': avg_forward_loss,
        'eval_backward_loss': avg_backward_loss,
        'eval_accuracy': avg_accuracy,
        'eval_top5_accuracy': avg_top5_accuracy,
        'eval_perplexity': metrics_tracker.compute_perplexity(torch.tensor(avg_loss)) if metrics_tracker else 0
    }
    
    return eval_metrics


def train_epoch(model, dataloader, optimizer, device, epoch, total_epochs, config=None, metrics_tracker=None, vocab=None):
    """한 에포크 학습 (메트릭 추적 포함)"""
    model.train()
    total_loss = 0.0
    total_forward_loss = 0.0
    total_backward_loss = 0.0
    total_accuracy = 0.0
    total_top5_accuracy = 0.0
    num_batches = 0
    
    # 진행률 표시 (total 설정으로 진행률 표시 개선)
    try:
        total_batches = len(dataloader)
    except TypeError:
        # IterableDataset의 경우 예상 배치 수 사용
        # seq_len=64, batch_size=16 기준으로 대략적인 추정
        estimated_tokens = config.get('estimated_tokens', 100000000)  # 기본값 100M 토큰
        seq_len = config.get('seq_len', 64)
        batch_size = config.get('batch_size', 16)
        total_batches = estimated_tokens // (seq_len * batch_size)
        logger.info(f"⚠️ IterableDataset has no definite length. Using estimated batches: {total_batches:,}")
    
    pbar = tqdm(
        dataloader,
        total=total_batches,
        desc=f'Epoch {epoch+1}/{total_epochs}',
        dynamic_ncols=True,
        mininterval=0.5  # 최소 0.5초 간격으로 업데이트
    )
    
    for batch_idx, batch in enumerate(pbar):
        try:
            # 배치 데이터 언패킹
            (forward_input, backward_input, forward_target, backward_target, 
             forward_mask, backward_mask) = batch
            
            # GPU로 이동
            forward_input = forward_input.to(device)
            backward_input = backward_input.to(device)
            forward_target = forward_target.to(device)
            backward_target = backward_target.to(device)
            forward_mask = forward_mask.to(device)
            backward_mask = backward_mask.to(device)
            
            # 효율성 측정 시작
            batch_start_time = time.time()
            memory_before = get_gpu_memory_usage()
            
            # 큰 vocab_size의 경우 배치를 더 작게 나누어 처리 (메모리 부족 방지)
            original_batch_size = forward_input.size(0)
            effective_batch_size = original_batch_size
            vocab_size = config.get('vocab_size', 0)
            
            # vocab_size가 매우 큰 경우 (1M 이상) 배치 크기를 자동으로 줄임
            if vocab_size > 1000000 and original_batch_size > 2:
                # 메모리 사용량 추정: batch_size * seq_len * vocab_size * 4 bytes * 2 (forward+backward)
                seq_len = forward_input.size(1)
                estimated_memory_gb = (original_batch_size * seq_len * vocab_size * 4 * 2) / (1024**3)
                
                # 더 적극적인 분할: 목표 메모리 사용량을 8GB로 제한
                target_memory_gb = 8.0
                if estimated_memory_gb > target_memory_gb:
                    # 필요한 분할 비율 계산
                    split_ratio = estimated_memory_gb / target_memory_gb
                    effective_batch_size = max(1, int(original_batch_size / split_ratio))
                    
                    # 최소 배치 크기 보장
                    effective_batch_size = max(1, min(effective_batch_size, 4))
                    
                    # 첫 번째 배치에서만 경고 출력
                    if batch_idx == 0:
                        logger.warning(f"Large vocab_size ({vocab_size:,}) detected. "
                                     f"Splitting batch {original_batch_size} -> {effective_batch_size} "
                                     f"(estimated: {estimated_memory_gb:.1f}GB -> target: {target_memory_gb}GB)")
            
            # 배치 분할 처리
            if effective_batch_size < original_batch_size:
                # 배치를 더 작게 나누어 처리
                num_splits = original_batch_size // effective_batch_size
                forward_losses = []
                backward_losses = []
                total_batch_loss_val = 0.0
                
                for split_idx in range(num_splits):
                    start_idx = split_idx * effective_batch_size
                    end_idx = min((split_idx + 1) * effective_batch_size, original_batch_size)
                    
                    # 부분 배치 추출
                    f_input = forward_input[start_idx:end_idx]
                    b_input = backward_input[start_idx:end_idx]
                    f_target = forward_target[start_idx:end_idx]
                    b_target = backward_target[start_idx:end_idx]
                    f_mask = forward_mask[start_idx:end_idx] if forward_mask is not None else None
                    b_mask = backward_mask[start_idx:end_idx] if backward_mask is not None else None
                    
                    # Forward pass
                    f_logits, b_logits = model(f_input, b_input, f_mask, b_mask)
                    
                    # Loss 계산
                    split_loss, f_loss, b_loss = model.compute_loss(
                        f_logits, b_logits, f_target, b_target, f_mask, b_mask
                    )
                    
                    forward_losses.append(f_loss.item())
                    backward_losses.append(b_loss.item())
                    total_batch_loss_val += split_loss.item()
                    
                    # 적극적인 메모리 정리
                    del f_logits, b_logits, split_loss, f_loss, b_loss
                    del f_input, b_input, f_target, b_target, f_mask, b_mask
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        # synchronize() 제거: 불필요한 동기화 오버헤드 제거로 GPU 활용률 향상
                
                # 평균 loss 계산
                total_batch_loss = torch.tensor(total_batch_loss_val / num_splits, device=device, requires_grad=True)
                forward_loss = torch.tensor(sum(forward_losses) / len(forward_losses), device=device)
                backward_loss = torch.tensor(sum(backward_losses) / len(backward_losses), device=device)
                
                # 메트릭 계산은 첫 번째 split만 사용 (메모리 절약)
                if metrics_tracker:
                    # 첫 번째 split만 사용하여 메트릭 계산
                    f_input = forward_input[:effective_batch_size]
                    b_input = backward_input[:effective_batch_size]
                    f_target = forward_target[:effective_batch_size]
                    b_target = backward_target[:effective_batch_size]
                    f_mask = forward_mask[:effective_batch_size] if forward_mask is not None else None
                    b_mask = backward_mask[:effective_batch_size] if backward_mask is not None else None
                    
                    f_logits, b_logits = model(f_input, b_input, f_mask, b_mask)
                    
                    forward_acc = metrics_tracker.compute_accuracy(f_logits, f_target, f_mask)
                    backward_acc = metrics_tracker.compute_accuracy(b_logits, b_target, b_mask)
                    avg_accuracy = (forward_acc + backward_acc) / 2
                    
                    forward_top5 = metrics_tracker.compute_top_k_accuracy(f_logits, f_target, k=5, mask=f_mask)
                    backward_top5 = metrics_tracker.compute_top_k_accuracy(b_logits, b_target, k=5, mask=b_mask)
                    avg_top5_accuracy = (forward_top5 + backward_top5) / 2
                    
                    total_accuracy += avg_accuracy
                    total_top5_accuracy += avg_top5_accuracy
                    
                    del f_logits, b_logits, f_input, b_input, f_target, b_target, f_mask, b_mask
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        # synchronize() 제거: 불필요한 동기화 오버헤드 제거로 GPU 활용률 향상
                
                # Backward pass
                optimizer.zero_grad()
                total_batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                
                # 통계 업데이트
                total_loss += total_batch_loss.item()
                total_forward_loss += forward_loss.item()
                total_backward_loss += backward_loss.item()
                num_batches += 1
                
                # 진행률 업데이트
                avg_loss = total_loss / num_batches
                avg_acc = total_accuracy / num_batches if metrics_tracker else 0
                avg_top5 = total_top5_accuracy / num_batches if metrics_tracker else 0
                
                # 간결한 출력 형식
                postfix = {
                    'loss': f'{avg_loss:.4f}',
                    'F': f'{total_forward_loss/num_batches:.4f}',
                    'B': f'{total_backward_loss/num_batches:.4f}'
                }
                if metrics_tracker:
                    postfix.update({
                        'Acc': f'{avg_acc:.3f}',
                        'Top5': f'{avg_top5:.3f}'
                    })
                
                pbar.set_postfix(postfix)
                continue
            
            # Forward pass (정상 크기 배치)
            forward_logits, backward_logits = model(
                forward_input, backward_input, 
                forward_mask, backward_mask
            )
            
            # Loss 계산
            total_batch_loss, forward_loss, backward_loss = model.compute_loss(
                forward_logits, backward_logits,
                forward_target, backward_target,
                forward_mask, backward_mask
            )
            
            # 메트릭 계산
            if metrics_tracker:
                # Accuracy 계산
                forward_acc = metrics_tracker.compute_accuracy(
                    forward_logits, forward_target, forward_mask
                )
                backward_acc = metrics_tracker.compute_accuracy(
                    backward_logits, backward_target, backward_mask
                )
                avg_accuracy = (forward_acc + backward_acc) / 2
                
                # Top-5 accuracy 계산
                forward_top5 = metrics_tracker.compute_top_k_accuracy(
                    forward_logits, forward_target, k=5, mask=forward_mask
                )
                backward_top5 = metrics_tracker.compute_top_k_accuracy(
                    backward_logits, backward_target, k=5, mask=backward_mask
                )
                avg_top5_accuracy = (forward_top5 + backward_top5) / 2
                
                total_accuracy += avg_accuracy
                total_top5_accuracy += avg_top5_accuracy
                
                # 에러 분석 (일부 배치에서만)
                if batch_idx % 100 == 0 and vocab:
                    idx2word = {v: k for k, v in vocab.items()}
                    metrics_tracker.analyze_errors(
                        forward_logits, forward_target, idx2word, forward_mask, sample_size=10
                    )
            
            # Backward pass
            optimizer.zero_grad()
            total_batch_loss.backward()
            
            # Gradient clipping (안정적인 학습을 위해)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            optimizer.step()
            
            # 효율성 측정 완료
            batch_time = time.time() - batch_start_time
            memory_after = get_gpu_memory_usage()
            memory_used = max(0, memory_after - memory_before)
            
            # 효율성 메트릭 추적
            if metrics_tracker:
                batch_size = forward_input.size(0)
                seq_len = forward_input.size(1)
                metrics_tracker.track_efficiency(batch_size, seq_len, batch_time, memory_used, batch_idx)
            
            # 통계 업데이트
            total_loss += total_batch_loss.item()
            total_forward_loss += forward_loss.item()
            total_backward_loss += backward_loss.item()
            num_batches += 1
            
            # 진행률 업데이트 (매 배치마다 업데이트, tqdm이 자동으로 출력 빈도 조절)
            avg_loss = total_loss / num_batches
            avg_acc = total_accuracy / num_batches if metrics_tracker else 0
            avg_top5 = total_top5_accuracy / num_batches if metrics_tracker else 0
            
            # 간결한 출력 형식 (너무 많은 정보로 인한 잘림 방지)
            postfix = {
                'loss': f'{total_batch_loss.item():.4f}',
                'avg': f'{avg_loss:.4f}',
                'F': f'{total_forward_loss/num_batches:.4f}',
                'B': f'{total_backward_loss/num_batches:.4f}'
            }
            if metrics_tracker:
                postfix.update({
                    'Acc': f'{avg_acc:.3f}',
                    'Top5': f'{avg_top5:.3f}'
                })
            
            pbar.set_postfix(postfix)
            
            # 메모리 정리 (더 적극적으로)
            del forward_logits, backward_logits, total_batch_loss, forward_loss, backward_loss
            if metrics_tracker:
                del forward_acc, backward_acc, avg_accuracy, forward_top5, backward_top5, avg_top5_accuracy
            # 입력 텐서도 정리
            del forward_input, backward_input, forward_target, backward_target, forward_mask, backward_mask
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                # synchronize() 제거: 불필요한 동기화 오버헤드 제거로 GPU 활용률 향상
                
        except Exception as e:
            logger.error(f"Error in batch {batch_idx}: {str(e)}")
            continue
    
    avg_loss = total_loss / max(num_batches, 1)
    avg_forward_loss = total_forward_loss / max(num_batches, 1)
    avg_backward_loss = total_backward_loss / max(num_batches, 1)
    avg_accuracy = total_accuracy / max(num_batches, 1) if metrics_tracker else 0
    avg_top5_accuracy = total_top5_accuracy / max(num_batches, 1) if metrics_tracker else 0
    
    return {
        'total_loss': avg_loss,
        'forward_loss': avg_forward_loss,
        'backward_loss': avg_backward_loss,
        'accuracy': avg_accuracy,
        'top5_accuracy': avg_top5_accuracy,
        'perplexity': metrics_tracker.compute_perplexity(torch.tensor(avg_loss)) if metrics_tracker else 0
    }


def main():
    parser = argparse.ArgumentParser(description='Train ELMo BiLM')
    parser.add_argument('--config', type=str, 
                       default='configs/elmo_depth-1_seed-42.yaml',
                       help='Path to config file')
    parser.add_argument('--data_path', type=str,
                       default=DEFAULT_DATA_PATH,
                       help='Path to training data')
    parser.add_argument('--vocab_path', type=str,
                       default=DEFAULT_VOCAB_PATH,
                       help='Path to vocabulary file')
    parser.add_argument('--checkpoint_dir', type=str,
                       default=CHECKPOINT_DIR,
                       help='Directory to save checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # 설정 로드
    config_path = os.path.join(ROOT_DIR, args.config)
    config = load_config(config_path)
    logger.info(f"Loaded config: {config}")
    
    # 시드 설정
    set_seed(config['seed'])
    
    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    if torch.cuda.is_available():
        # GPU 최적화 설정 (CUDA_LAUNCH_BLOCKING 제거로 비동기 실행 활성화)
        # 학습 시에는 비동기 실행이 더 효율적
        if 'CUDA_LAUNCH_BLOCKING' in os.environ:
            del os.environ['CUDA_LAUNCH_BLOCKING']
        
        os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'
        torch.cuda.empty_cache()
        
        logger.info(f"GPU: {torch.cuda.get_device_name()}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        logger.info("GPU 비동기 실행 모드 활성화 (성능 최적화)")
        
        # 배치 크기 권장 사항
        batch_size = config.get('batch_size', 16)
        if batch_size < 32:
            logger.warning(f"⚠️  배치 크기({batch_size})가 작습니다. GPU 활용률을 높이려면 batch_size를 32 이상으로 늘리는 것을 권장합니다.")
    
    # 어휘 사전 로드
    vocab_path = os.path.join(ROOT_DIR, args.vocab_path)
    try:
        word2idx = load_vocabulary(vocab_path)
    except FileNotFoundError:
        logger.error(f"Vocabulary file not found: {vocab_path}")
        logger.info("Please run preprocessing first to generate vocabulary")
        return
    
    # vocab_size 제한 (메모리 절약)
    original_vocab_size = len(word2idx)
    max_vocab_size = 50000  # 최대 50K 단어로 제한
    
    if original_vocab_size > max_vocab_size:
        logger.warning(f"Limiting vocab from {original_vocab_size:,} to {max_vocab_size:,} for memory efficiency")
        
        # 상위 빈도 단어만 사용하도록 word2idx 제한
        limited_word2idx = {}
        for word, idx in word2idx.items():
            if idx < max_vocab_size:
                limited_word2idx[word] = idx
        
        # UNK 토큰 확인 (인덱스 1)
        if '<UNK>' not in limited_word2idx and 1 < max_vocab_size:
            limited_word2idx['<UNK>'] = 1
        
        word2idx = limited_word2idx
        config['vocab_size'] = max_vocab_size
        
        logger.info(f"Limited vocab contains {len(word2idx):,} words")
        logger.info(f"UNK token available: {'<UNK>' in word2idx}")
    else:
        config['vocab_size'] = original_vocab_size
    
    logger.info(f"Using vocab_size: {config['vocab_size']:,}")
    
    # 모델 생성
    logger.info("Creating ELMo BiLM model...")
    model = create_elmo_model(config)
    
    # 옵티마이저 설정
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    
    # 체크포인트 디렉토리 생성
    checkpoint_dir = os.path.join(ROOT_DIR, args.checkpoint_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)
    logger.info(f"Checkpoints will be saved to: {checkpoint_dir}")
    
    # 메트릭 추적기 생성
    run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_dir = os.path.join(ROOT_DIR, LOGS_DIR, run_name)
    results_dir = os.path.join(ROOT_DIR, RESULTS_DIR, 'metrics')
    metrics_tracker = MetricsTracker(log_dir, results_dir)
    
    # 체크포인트 로드 (resume인 경우)
    start_epoch = 0
    best_loss = float('inf')
    
    if args.resume:
        resume_path = os.path.join(ROOT_DIR, args.resume)
        start_epoch, best_loss = load_checkpoint(model, optimizer, resume_path)
    
    # 데이터 로더 생성
    data_path = os.path.join(ROOT_DIR, args.data_path)
    logger.info("Creating data loader...")
    
    try:
        dataloader = get_dataloader(data_path, config, word2idx)
        logger.info(f"Data loader created successfully")
    except Exception as e:
        logger.error(f"Failed to create data loader: {str(e)}")
        logger.info("Please check if the data file exists and run preprocessing if needed")
        return
    
    # 학습 시작
    logger.info("Starting training...")
    logger.info(f"Config: {config}")
    logger.info(f"Metrics will be saved to: {log_dir}")
    
    # 설정 저장
    config_file = os.path.join(log_dir, 'config.json')
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    for epoch in range(start_epoch, config['epochs']):
        epoch_start_time = time.time()
        
        # 한 에포크 학습
        epoch_metrics = train_epoch(
            model, dataloader, optimizer, device, epoch, config['epochs'],
            config, metrics_tracker, word2idx
        )
        
        epoch_time = time.time() - epoch_start_time
        epoch_metrics['epoch_time'] = epoch_time
        epoch_metrics['gpu_memory_peak'] = get_gpu_memory_usage()
        
        # 메트릭 로깅
        metrics_tracker.log_metrics(epoch, epoch_metrics)
        
        # 로그 출력
        logger.info(f"Epoch {epoch+1}/{config['epochs']} completed in {epoch_time:.2f}s")
        logger.info(f"Average Loss: {epoch_metrics['total_loss']:.4f}")
        logger.info(f"Forward Loss: {epoch_metrics['forward_loss']:.4f}")
        logger.info(f"Backward Loss: {epoch_metrics['backward_loss']:.4f}")
        logger.info(f"Accuracy: {epoch_metrics['accuracy']:.4f}")
        logger.info(f"Top-5 Accuracy: {epoch_metrics['top5_accuracy']:.4f}")
        logger.info(f"Perplexity: {epoch_metrics['perplexity']:.2f}")
        logger.info(f"GPU Memory: {epoch_metrics['gpu_memory_peak']:.2f}GB")
        
        # 체크포인트 저장
        checkpoint_path = os.path.join(checkpoint_dir, f'elmo_bilm_epoch_{epoch+1}.pt')
        save_checkpoint(model, optimizer, epoch+1, epoch_metrics['total_loss'], 
                       checkpoint_path, epoch_metrics, config)
        
        # GPU 메모리 정리
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # 최종 분석 및 리포트 생성
    logger.info("Generating final reports...")
    metrics_tracker.save_final_results()
    metrics_tracker.plot_metrics()
    
    # 최종 요약 저장
    final_summary = {
        'best_loss': best_loss,
        'total_epochs': config['epochs'],
        'total_parameters': sum(p.numel() for p in model.parameters()),
        'config': config,
        'gpu_info': metrics_tracker.gpu_info if metrics_tracker.gpu_available else None
    }
    
    summary_file = os.path.join(log_dir, 'training_summary.json')
    with open(summary_file, 'w') as f:
        json.dump(final_summary, f, indent=2)
    
    logger.info("Training completed!")
    logger.info(f"Best loss: {best_loss:.4f}")


if __name__ == '__main__':
    main()
