#!/bin/bash

export HF_HOME="/data/.cache/huggingface"
export PYTHONPATH="$PYTHONPATH:$PWD"

strategies=("zero-shot" "cot" "cot-reflection" "cot-contrastive")

languages=("c_sharp" "javascript" "java" "python" "c")

models=(
  "Qwen2.5-Coder-32B-Instruct"
  "Qwen2.5-Coder-7B-Instruct"
  "Qwen2.5-Coder-1.5B-Instruct"
  "Qwen2.5-Coder-0.5B-Instruct"
)

for model in "${models[@]}"; do
  for strategy in "${strategies[@]}"; do
    for lang in "${languages[@]}"; do
      CUDA_VISIBLE_DEVICES=0 python src/inference/inference_cot.py \
        --model_name_or_path "Qwen/${model}" \
        --output_dir "runs/cot/${model}" \
        --languages ${lang} \
        --strategy ${strategy} \
        --do_sample
    done
  done
done
