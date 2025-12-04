# ELMo 구현 (PyTorch)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

양방향 LSTM 언어 모델을 사용하는 ELMo (Embeddings from Language Models) PyTorch 구현입니다.

## 특징

- **Depth 1, 2 지원**: 1층 또는 2층 양방향 LSTM 지원
- **Forward & Backward LSTM**: 양방향 언어 모델 학습
- **레이어별 임베딩 추출**: 각 레이어의 hidden state를 임베딩으로 사용 가능

## 설치

```bash
# 저장소 클론
cd ELMo_repo

# 의존성 설치
pip install -r requirements.txt
```

## 사용법

### 학습

```bash
# Depth 1 모델 학습
python src/train.py --config configs/elmo_depth-1_seed-42.yaml

# Depth 2 모델 학습
python src/train.py --config configs/elmo_depth-2_seed-42.yaml
```

### 평가

```bash
python src/eval.py \
    configs/elmo_depth-2_seed-42.yaml \
    runs/checkpoints_elmo_depth-2/elmo_depth-2_seed-42.pth \
    data/word_similarity/wordsim353_sim.csv \
    data/word_similarity/SimLex-999.txt \
    data/word_similarity/questions-words.txt \
    --save_csv results/elmo_depth-2_seed-42.csv
```

**출력 예시:**
```
📊 WordSim-353 Spearman: 0.6285
📘 SimLex-999 Spearman: 0.2639
👑 Google Analogy Accuracy: 0.3831
```

## 설정 파일

YAML 파일로 하이퍼파라미터를 설정합니다:

```yaml
# configs/elmo_depth-2_seed-42.yaml
vocab_size: 30000
embedding_dim: 512
hidden_dim: 512
num_layers: 2  # depth: 1 or 2
dropout: 0.1
seq_len: 20
batch_size: 32
lr: 0.001
epochs: 1
seed: 42
enable_subsampling: true
subsample_t: 1e-3
num_workers: 16
```

## 모델 구조

ELMo 모델은 다음과 같은 구조를 가집니다:

1. **단어 임베딩 레이어**: 단어를 벡터로 변환
2. **Forward LSTM**: 왼쪽에서 오른쪽으로 읽는 언어 모델
3. **Backward LSTM**: 오른쪽에서 왼쪽으로 읽는 언어 모델
4. **다층 구조**: Depth 1 또는 2의 LSTM 레이어

각 레이어의 hidden state를 결합하여 최종 임베딩을 생성합니다.

## 평가 항목

ELMo 모델은 다음 항목으로 평가됩니다:

### Intrinsic Evaluation (단어 수준)
- **Depth 1 bi-LSTM**: 1층 양방향 LSTM
- **Depth 2 bi-LSTM**: 2층 양방향 LSTM

각 모델은 다음 데이터셋으로 평가됩니다:
- WordSim-353 (단어 유사도)
- SimLex-999 (단어 유사도)
- Google Analogy (단어 유추)

### Extrinsic Evaluation (다운스트림 태스크)
ELMo 임베딩을 사용한 sequence classification tasks:
- **SST-2**: 감정 분석 (Sentiment Analysis)
- **MRPC**: 문장 쌍 분류 (Paraphrase Detection)
- **CoNLL-03 NER**: 개체명 인식 (Named Entity Recognition)

```bash
# Sequence tasks 평가
python src/eval_sequence.py \
    configs/elmo_depth-2_seed-42.yaml \
    runs/checkpoints_elmo_depth-2/elmo_depth-2_seed-42.pth \
    --sst2_dir data/sequence_tasks/sst2 \
    --mrpc_dir data/sequence_tasks/mrpc \
    --ner_dir data/sequence_tasks/conll03
```

## 프로젝트 구조

```
ELMo_repo/
├── configs/              # 실험 설정 파일 (YAML)
├── data/
│   ├── pretrain/         # 학습 데이터
│   └── word_similarity/  # 평가 데이터셋
├── src/
│   ├── model.py          # ELMo 모델
│   ├── data.py           # 데이터 로더
│   ├── train.py          # 학습 스크립트
│   └── eval.py           # 평가 스크립트
├── runs/
│   ├── checkpoints_elmo_depth-1/  # Depth 1 체크포인트
│   ├── checkpoints_elmo_depth-2/  # Depth 2 체크포인트
│   └── metrics/          # 학습 메트릭
└── results/              # 평가 결과
```

## 참고

이 구현은 Word2Vec_repo의 구조를 참고하여 작성되었습니다.

