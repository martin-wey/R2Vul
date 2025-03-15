#!/bin/bash

export HF_HOME="/data/.cache/huggingface"
export PYTHONPATH="$PYTHONPATH:$PWD"

languages=("c_sharp" "javascript" "java" "python" "c")

models=(
  "Qwen2.5-Coder-7B-Instruct"
  "Qwen2.5-Coder-1.5B-Instruct"
  "Qwen2.5-Coder-0.5B-Instruct"
)

for model in "${models[@]}"; do
  for lang in "${languages[@]}"; do
    CUDA_VISIBLE_DEVICES=0 python src/inference/inference_cls.py \
      --model_name_or_path "runs/cls/${model}/${lang}" \
      --output_dir "runs/cls/${model}" \
      --languages "$lang" \
      --batch_size 8
  done
done