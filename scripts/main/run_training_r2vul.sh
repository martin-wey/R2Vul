#!/bin/bash

export HF_HOME="/data/.cache/huggingface"
export WANDB_PROJECT="R2Vul"
export PYTHONPATH="$PYTHONPATH:$PWD"

model="Qwen2.5-Coder-1.5B-Instruct"   # change to target LLM variant
b="0.3"

CUDA_VISIBLE_DEVICES=0 python src/training/train_orpo.py \
  --output_dir runs/r2vul/${model}_b${b} \
  --run_name r2vul/${model}_b${b} \
  --model_name_or_path Qwen/${model} \
  --learning_rate 3.0e-4 \
  --lora_r 16 \
  --lora_alpha 32 \
  --beta 0.3 \
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
  --report_to wandb \
  --seed 42
