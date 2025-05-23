#!/bin/bash

export HF_HOME="/data/.cache/huggingface"
export PYTHONPATH="$PYTHONPATH:$PWD"

languages=("c_sharp" "javascript" "java" "python" "c")

for lang in "${languages[@]}"; do
  CUDA_VISIBLE_DEVICES=0 python src/inference/inference_vulllm.py \
    --model_name_or_path "runs/VulLLM" \
    --output_dir "runs/VulLLM" \
    --languages ${lang} \
    --do_sample
done
