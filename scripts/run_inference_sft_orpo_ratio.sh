#!/bin/bash

export HF_HOME="/data/.cache/huggingface"
export PYTHONPATH="$PYTHONPATH:$PWD"

ratios=("1:2" "1:3" "1:4" "1:5" "1:10")

languages=("c_sharp" "java")

model_dir="runs/orpo/Qwen2.5-Coder-1.5B-Instruct_lora_r16a32_b0.3/checkpoint-29356"

for ratio in "${ratios[@]}"; do
  for lang in "${languages[@]}"; do
    CUDA_VISIBLE_DEVICES=0 python src/inference/inference_cot.py \
      --model_name_or_path ${model_dir} \
      --output_dir ${model_dir} \
      --languages ${lang} \
      --test_ratio ${ratio} \
      --strategy cot \
      --do_sample
  done
done
