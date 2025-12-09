#!/bin/bash
# ELMo 모델 평가 스크립트 실행

cd "$(dirname "$0")"
python3 src/eval/evaluate_elmo.py --checkpoint runs/checkpoints/depth-1/elmo_bilm_epoch_1.pt
