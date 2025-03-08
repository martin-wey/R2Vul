#!/bin/bash

export HF_HOME="/data/.cache/huggingface"
export PYTHONPATH="$PYTHONPATH:$PWD"

model_dirs=(
  "runs/orpo/Qwen2.5-Coder-1.5B-Instruct_lora_r16a32_b0.3/checkpoint-29356"
  "runs/sft/Qwen2.5-Coder-1.5B-Instruct_lora_r16a32/multi/best_model_checkpoint"

  "runs/orpo/Qwen2.5-Coder-7B-Instruct_lora_r16a32_b0.3/checkpoint-29356"
  "runs/sft/Qwen2.5-Coder-7B-Instruct_lora_r16a32/multi/best_model_checkpoint"

  "runs/orpo/Qwen2.5-Coder-0.5B-Instruct_lora_r16a32_b0.3/checkpoint-29356"
  "runs/sft/Qwen2.5-Coder-0.5B-Instruct_lora_r16a32/multi/best_model_checkpoint"
)

for model_dir in "${model_dirs[@]}"; do
  CUDA_VISIBLE_DEVICES=0 python src/inference/inference_cot.py \
    --model_name_or_path ${model_dir} \
    --output_dir ${model_dir}/realworld_dedup \
    --data_dir data \
    --dataset_name realworld_dedup \
    --languages java \
    --strategy cot \
    --do_sample
done
