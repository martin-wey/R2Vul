#!/bin/bash

export HF_HOME="/data/.cache/huggingface"
export PYTHONPATH="$PYTHONPATH:$PWD"

run_ids=(1 2 3)

for run_id in "${run_ids[@]}"; do
  python src/inference/inference_tgi.py \
    --model_name_or_path "sft-1.5B" \
    --output_dir "results/external_test_set/sft_${run_id}" \
    --dataset_name "external_java_test" \
    --languages "java" \
    --do_sample \
    --seed 42
done
