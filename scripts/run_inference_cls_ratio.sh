#!/bin/bash

export HF_HOME="/data/.cache/huggingface"
export PYTHONPATH="$PYTHONPATH:$PWD"

ratios=("1:2" "1:3" "1:4" "1:5" "1:10")

languages=("c_sharp" "java")

model_dir="runs/cls/Qwen2.5-Coder-1.5B-Instruct_lora_r16a32"
# model_dir="runs/cls/CodeBERT"

for ratio in "${ratios[@]}"; do
  for lang in "${languages[@]}"; do
    CUDA_VISIBLE_DEVICES=0 python src/inference/inference_cls.py \
      --model_name_or_path ${model_dir}/${lang}/best_model_checkpoint \
      --output_dir ${model_dir}/${lang}/best_model_checkpoint \
      --languages ${lang} \
      --test_ratio ${ratio} \
      --batch_size 8
  done
done
