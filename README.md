# ELMo / BiLM PyTorch 실험 레포

이 레포는 AllenAI의 ELMo 논문인
["Deep contextualized word representations"](http://arxiv.org/abs/1802.05365)의
biLM(Bidirectional Language Model)을 PyTorch로 구현하고, 사전학습한 ELMo
표현을 여러 downstream NLP 태스크에 붙여 성능을 비교하기 위한 실험 코드입니다.

원본 `bilm-tf` TensorFlow 구현의 구조를 참고하되, 현재 레포는 PyTorch 학습,
임베딩 추출, 데이터 전처리, downstream 재현 실험을 한곳에서 실행할 수 있도록
구성되어 있습니다.

## 구현 내용

- ELMo 스타일 biLM 사전학습
  - character CNN 입력
  - highway network
  - 2-layer bidirectional LSTM
  - layer mixing을 통한 ELMo representation 생성
- Wikipedia 기반 사전학습 데이터 준비 파이프라인
  - Hugging Face dataset 다운로드
  - 문장 전처리 및 토큰화
  - train/valid/test shard 생성
  - vocabulary 생성
- ELMo 임베딩 사용 예시
  - character id 기반 즉시 추론
  - 전체 문장 임베딩을 HDF5로 캐싱
- Downstream 실험
  - SST-2: baseline classifier, ELMo classifier
  - SST-5: baseline classifier, ELMo classifier
  - NER(CoNLL-2003): GloVe+CharCNN+BiLSTM-CRF baseline, ELMo 추가 모델
  - SQuAD v1.1: GloVe baseline QA, GloVe+ELMo QA, layer/placement ablation
  - SNLI: ESIM baseline에 ELMo 입력 결합
  - SRL: GloVe baseline과 ELMo 결합 모델

## 디렉토리 구조

```text
.
├── bilm/src/                  # BiLM/ELMo 모델, 학습, 데이터 유틸리티
├── bilm/src/dataset/          # 사전학습 데이터 다운로드/전처리 도구
├── bin/train_elmo.py          # BiLM 사전학습 실행 스크립트
├── checkpoints/bilm/          # 학습 옵션 및 체크포인트 저장 위치
├── downstream/                # SST, NER, SQuAD, SNLI, SRL 실험 코드
├── tests/fixtures/            # 테스트용 작은 vocab/model/data fixture
├── usage_character.py         # character input 기반 ELMo 사용 예시
├── usage_cached.py            # HDF5 임베딩 캐시 사용 예시
├── config.yaml                # BiLM 사전학습 기본 설정
└── requirements.txt
```

## 설치

Python 가상환경을 만든 뒤 의존성을 설치합니다.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

최소 핵심 의존성은 `torch`, `numpy`, `h5py`, `tqdm`, `PyYAML`입니다.
Downstream 스크립트는 태스크에 따라 `datasets`, `scikit-learn`, `pandas`,
`matplotlib` 등을 사용합니다.

설치 확인:

```bash
python test_pytorch_conversion.py
```

## 사전학습 데이터 준비

ELMo 학습용 말뭉치를 준비하려면 dataset 파이프라인을 실행합니다.

```bash
cd bilm/src/dataset
python prepare_data.py
cd ../../..
```

기본 파이프라인은 다음 작업을 수행합니다.

1. Wikipedia 데이터셋 다운로드
2. ELMo 학습에 맞는 텍스트 전처리
3. train/valid/test shard 저장
4. `vocab.txt` 생성
5. 전처리 결과 검증

기본 출력 위치는 다음과 같습니다.

```text
bilm/data/pretrain/elmo/
├── train/train_*.txt
├── valid/valid_*.txt
├── test/test_*.txt
├── vocab.txt
└── preprocess_meta.json
```

전처리만 다시 실행하거나 검증만 하고 싶다면:

```bash
python prepare_data.py --force-preprocess
python prepare_data.py --validate-only
python prepare_data.py --validate-only --sample
```

## BiLM / ELMo 사전학습

레포 루트에서 `config.yaml`을 사용해 학습합니다.

```bash
python bin/train_elmo.py --config config.yaml
```

`config.yaml`에는 데이터 경로, 모델 구조, batch size, max steps, validation
주기, checkpoint 저장 주기 등이 정의되어 있습니다. 기본 저장 위치는
`checkpoints/bilm`입니다.

CLI 인자로 주요 경로를 덮어쓸 수도 있습니다.

```bash
python bin/train_elmo.py \
  --config config.yaml \
  --save_dir checkpoints/bilm \
  --vocab_file bilm/data/pretrain/elmo/vocab.txt \
  --train_prefix "bilm/data/pretrain/elmo/train/train_*.txt" \
  --valid_prefix "bilm/data/pretrain/elmo/valid/valid_*.txt"
```

YAML 없이 실행하려면 필수 경로를 직접 넘깁니다.

```bash
python bin/train_elmo.py \
  --save_dir checkpoints/bilm \
  --vocab_file bilm/data/pretrain/elmo/vocab.txt \
  --train_prefix "bilm/data/pretrain/elmo/train/train_*.txt" \
  --use_character_inputs \
  --batch_size 32 \
  --n_epochs 10 \
  --learning_rate 0.001
```

학습 결과로 `options.json`과 PyTorch checkpoint가 생성됩니다. Downstream ELMo
스크립트들은 기본적으로 `checkpoints/bilm/final_model.pt`를 찾습니다.

## ELMo 임베딩 사용

작은 fixture 모델로 character input 기반 추론 예시를 실행할 수 있습니다.

```bash
python usage_character.py
```

문장별 biLM/ELMo 임베딩을 HDF5 파일로 미리 저장하려면:

```bash
python usage_cached.py
```

`usage_cached.py`는 예시 문장을 `dataset_file.txt`에 쓰고,
`elmo_embeddings.hdf5`를 생성한 뒤 저장된 임베딩을 읽습니다.

## Downstream 실험 실행

각 downstream 스크립트는 레포 루트에서 실행하는 것을 기준으로 작성되어 있습니다.
ELMo 모델을 쓰는 실험은 먼저 `checkpoints/bilm/final_model.pt`와
`bilm/data/pretrain/elmo/vocab.txt`가 준비되어 있어야 합니다.

### SST-2

```bash
python downstream/SST_2/sst2_baseline_classifier.py
python downstream/SST_2/sst2_elmo_classifier.py
```

### SST-5

```bash
python downstream/SST_5/sst5_baseline_classifier.py
python downstream/SST_5/sst5_elmo_classifier.py
```

SST-5 스크립트는 기본적으로 `downstream/SST_5/SST-5/train.tsv`,
`dev.tsv`, `test.tsv` 형식의 데이터를 기대합니다.

### NER

```bash
python downstream/NER/conll2003_ner_bilstm_crf.py --model baseline
python downstream/NER/conll2003_ner_bilstm_crf.py \
  --model elmo \
  --bilm-checkpoint checkpoints/bilm/final_model.pt
```

### SQuAD

```bash
python downstream/SQuAD/squad_baseline_glove_qa.py
python downstream/SQuAD/squad_glove_elmo_qa.py
python downstream/SQuAD/squad_layer_ablation.py
python downstream/SQuAD/squad_elmo_placement_ablation.py
```

### SNLI / SRL

```bash
python downstream/SNLI/snli_esim_glove_elmo.py
python downstream/SRL/deep_srl_glove_elmo.py
```

실험 결과는 각 downstream 디렉토리의 `*_metrics.csv` 파일로 저장됩니다.

## 테스트

```bash
python test_pytorch_conversion.py
pytest tests
```

`pytest`가 설치되어 있지 않다면:

```bash
pip install pytest
```

## README에 추가하면 좋은 내용

현재 README에는 실행 흐름을 중심으로 정리했습니다. 프로젝트 제출용 또는 재현성을
높이려면 아래 내용도 추가하는 것이 좋습니다.

- 실험 환경: GPU 모델, CUDA 버전, Python/PyTorch 버전
- 데이터셋 버전과 split 크기
- 학습 완료 checkpoint 파일명과 다운로드/공유 위치
- 각 downstream 태스크의 최종 성능 표
- baseline 대비 ELMo 성능 차이 분석
- ablation 결과 해석
- 알려진 제한사항
  - 사전학습 비용이 큼
  - 일부 downstream 데이터는 실행 시 다운로드가 필요함
  - GloVe 캐시 파일이 없으면 첫 실행 시간이 길 수 있음

## Citation

```bibtex
@inproceedings{Peters:2018,
  author={Peters, Matthew E. and Neumann, Mark and Iyyer, Mohit and Gardner, Matt and Clark, Christopher and Lee, Kenton and Zettlemoyer, Luke},
  title={Deep contextualized word representations},
  booktitle={Proc. of NAACL},
  year={2018}
}
```

## License

원본 `bilm-tf` 구현과 동일하게 Apache License 2.0을 따릅니다.
