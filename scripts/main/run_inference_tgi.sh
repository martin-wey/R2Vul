#!/bin/bash

export HF_HOME="/data/.cache/huggingface"
export PYTHONPATH="$PYTHONPATH:$PWD"

run_ids=(1 2 3)
model="sft-7B"  # depends on what model you serve using `tgi_serve.sh`

for run_id in "${run_ids[@]}"; do
  python src/inference/inference_tgi.py \
    --model_name_or_path "${model}" \
    --output_dir "aaai_results/main/${model}_${run_id}" \
    --languages "c_sharp" "javascript" "java" "python" "c" \
    --do_sample \
    --seed 42
done
