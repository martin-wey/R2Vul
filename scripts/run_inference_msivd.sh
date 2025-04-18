#!/bin/bash

export HF_HOME="/data/.cache/huggingface"
export PYTHONPATH="$PYTHONPATH:$PWD"

languages=("c_sharp" "javascript" "java" "python" "c")

for lang in "${languages[@]}"; do
  CUDA_VISIBLE_DEVICES=0 python src/inference/inference_msivd.py \
    --model_name_or_path "runs/MSIVD" \
    --output_dir "runs/MSIVD" \
    --languages ${lang} \
    --do_sample
done
