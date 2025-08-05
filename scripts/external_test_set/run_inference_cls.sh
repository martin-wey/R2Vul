#!/bin/bash

export HF_HOME="/data/.cache/huggingface"
export PYTHONPATH="$PYTHONPATH:$PWD"

CUDA_VISIBLE_DEVICES=0 python src/inference/inference_cls.py \
  --model_name_or_path "runs/cls/Qwen2.5-Coder-1.5B-Instruct/java" \
  --output_dir "results/external_test_set/cls" \
  --data_dir data \
  --dataset_name external_java_test \
  --languages java \
  --batch_size 8
