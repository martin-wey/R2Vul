#!/bin/bash

export HF_HOME="/data/.cache/huggingface"
export PYTHONPATH="$PYTHONPATH:$PWD"

# set OPENAI_API_KEY and ANTHROPIC_API_KEY before running this script
models=("gpt-4o" "gpt-4.1" "claude-opus-4-20250514" "claude-sonnet-4-20250514")

for model in "${models[@]}"; do
  python src/inference/inference_oai.py \
    --model_name_or_path "${model}" \
    --output_dir "results/main/${model}" \
    --languages "c_sharp" "javascript" "java" "python" "c" \
    --do_sample \
    --seed 42
done
