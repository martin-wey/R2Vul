#!/bin/bash

export HF_HOME="/data/.cache/huggingface"
export PYTHONPATH="$PYTHONPATH:$PWD"

ratios=(
  "1:1" "1:2" "1:3" "1:4" "1:5" "1:10" "1:20"
)

for ratio in "${ratios[@]}"; do
    python src/inference/inference_tgi.py \
      --model_name_or_path "r2vul-1.5B" \
      --output_dir "results/class_imbalance_r2vul" \
      --languages "python" \
      --test_ratio "$ratio" \
      --do_sample
done
