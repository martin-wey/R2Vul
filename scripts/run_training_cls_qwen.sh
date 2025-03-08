#!/bin/bash

export HF_HOME="/data/.cache/huggingface"
export WANDB_PROJECT="R2Vul"
export PYTHONPATH="$PYTHONPATH:$PWD"

languages=("c_sharp" "javascript" "java" "python" "c")

models=(
  "Qwen2.5-Coder-7B-Instruct"
  "Qwen2.5-Coder-1.5B-Instruct"
  "Qwen2.5-Coder-0.5B-Instruct"
)

for model in "${models[@]}"; do
  for lang in "${languages[@]}"; do
    CUDA_VISIBLE_DEVICES=0 python src/training/train_cls.py \
      --output_dir "runs/cls/${model}_lora_r16a32/${lang}" \
      --run_name "cls/${model}_lora_r16a32/${lang}" \
      --model_name_or_path "Qwen/${model}" \
      --languages ${lang} \
      --lora_r 16 \
      --lora_alpha 32 \
      --learning_rate 5.0e-5 \
      --per_device_train_batch_size 1 \
      --per_device_eval_batch_size 8 \
      --gradient_accumulation_steps 16 \
      --logging_steps 10 \
      --num_train_epochs 10 \
      --eval_strategy 'epoch' \
      --save_strategy 'epoch' \
      --load_best_model_at_end True \
      --save_total_limit 1 \
      --bf16 \
      --report_to wandb
  done
done