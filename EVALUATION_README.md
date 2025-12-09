# ELMo 모델 평가 가이드

## 평가 스크립트 실행

학습된 ELMo 모델을 평가하려면 다음 명령을 실행하세요:

```bash
python src/eval/evaluate_elmo.py --checkpoint runs/checkpoints/depth-1/elmo_bilm_epoch_1.pt
```

## 평가 항목

### 1. 동일 단어의 문맥별 벡터 차이 확인

동일한 단어가 다른 문맥에서 다른 벡터를 생성하는지 확인합니다.

**테스트 케이스:**
- "bank" in "river bank" vs "bank" in "I deposited money in the bank"
- "bass" in "bass fish" vs "bass guitar"
- "crane" in "crane bird" vs "construction crane"
- "mouse" in "mouse animal" vs "computer mouse"

**평가 기준:**
- 평균 코사인 유사도 < 0.7: ✅ 우수 (문맥 구분 잘 함)
- 평균 코사인 유사도 0.7-0.9: ⚠️ 보통 (어느 정도 구분)
- 평균 코사인 유사도 > 0.9: ❌ 개선 필요 (문맥 구분 약함)

### 2. Coherence 테스트

PCA/t-SNE를 사용하여 hidden state를 시각화하고, 유사한 단어/문맥이 군집을 이루는지 확인합니다.

**시각화 결과:**
- `runs/evaluations/coherence_visualization.png`에 저장됩니다.

**평가 기준:**
- 의미적으로 유사한 단어들(예: 동물 관련 단어들)이 군집을 이루는지 확인
- 평균 코사인 유사도가 적절한 범위(0.3-0.7)에 있는지 확인

## 출력 예시

```
============================================================
ELMo 모델 평가
============================================================
체크포인트: runs/checkpoints/depth-1/elmo_bilm_epoch_1.pt
설정: configs/elmo_depth-1_seed-42.yaml
어휘: data/pretrain/vocab.pkl

✅ Vocabulary loaded: 50,000 words
✅ 체크포인트 로드 완료 (Epoch 1)

============================================================
테스트 1: 동일 단어의 문맥별 벡터 차이 확인
============================================================

단어: 'bank'
  문맥 1: 'I walked along the river bank'
    위치: 6, 벡터 차원: (256,)
  문맥 2: 'I deposited money in the bank'
    위치: 5, 벡터 차원: (256,)
    문맥 1 vs 문맥 2 유사도: 0.6234
  평균 유사도: 0.6234
  ✅ 좋음: 문맥에 따라 다른 벡터 생성 (유사도 0.6234)

...

============================================================
최종 평가 요약
============================================================
문맥 구분 능력 (평균 유사도): 0.6543
  ✅ 우수: 문맥에 따라 다른 벡터를 잘 생성합니다

Coherence (평균 유사도): 0.4521
PCA 설명 분산: 78.45%
  ✅ 우수: 의미적으로 유사한 단어들을 구분합니다

✅ 평가 완료!
```

## 필요한 라이브러리

```bash
pip install torch numpy scikit-learn matplotlib seaborn pyyaml
```

## 문제 해결

### 체크포인트를 찾을 수 없는 경우
- 체크포인트 경로를 절대 경로로 지정하거나
- `--checkpoint` 옵션으로 정확한 경로를 지정하세요

### 메모리 부족 오류
- GPU 메모리가 부족한 경우, CPU 모드로 실행됩니다
- 더 작은 배치 크기로 평가하도록 스크립트를 수정할 수 있습니다

### 시각화 파일이 생성되지 않는 경우
- `runs/evaluations/` 디렉토리가 자동으로 생성됩니다
- matplotlib 백엔드 문제인 경우, 스크립트는 자동으로 'Agg' 백엔드를 사용합니다
