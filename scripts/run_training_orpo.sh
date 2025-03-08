#!/bin/bash

export HF_HOME="/data/.cache/huggingface"
export WANDB_PROJECT="R2Vul"
export PYTHONPATH="$PYTHONPATH:$PWD"

models=(
  "Qwen2.5-Coder-7B-Instruct"
  "Qwen2.5-Coder-1.5B-Instruct"
  "Qwen2.5-Coder-0.5B-Instruct"
)
# change to desired beta
b=0.3

for model in "${models[@]}"; do
  CUDA_VISIBLE_DEVICES=0 python src/training/train_orpo.py \
    --output_dir runs/orpo/${model}_lora_r16a32_b${b} \
    --run_name orpo/${model}_lora_r16a32_b${b} \
    --model_name_or_path Qwen/${model} \
    --learning_rate 3.0e-4 \
    --lora_r 16 \
    --lora_alpha 32 \
    --beta ${b} \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --gradient_checkpointing \
    --optim adafactor \
    --logging_steps 1 \
    --num_train_epochs 5 \
    --eval_strategy 'epoch' \
    --save_strategy 'epoch' \
    --load_best_model_at_end \
    --bf16 \
    --report_to wandb
done