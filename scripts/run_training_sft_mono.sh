#!/bin/bash

export HF_HOME="/data/.cache/huggingface"
export WANDB_PROJECT="R2Vul"
export PYTHONPATH="$PYTHONPATH:$PWD"

# train a separate model for each language
languages=("c_sharp" "javascript" "java" "python" "c")

models=(
  "Qwen2.5-Coder-7B-Instruct"
  "Qwen2.5-Coder-1.5B-Instruct"
  "Qwen2.5-Coder-0.5B-Instruct"
)

for lang in "${languages[@]}"; do
  for model in "${models[@]}"; do
    CUDA_VISIBLE_DEVICES=0 python src/training/train_sft.py \
      --output_dir runs/sft/${model}/${lang} \
      --run_name sft/${lang}/${model} \
      --model_name_or_path Qwen/${model} \
      --completion_only \
      --languages ${lang} \
      --learning_rate 3.0e-4 \
      --lora_r 16 \
      --lora_alpha 32 \
      --per_device_train_batch_size 1 \
      --per_device_eval_batch_size 1 \
      --gradient_accumulation_steps 2 \
      --gradient_checkpointing \
      --optim adafactor \
      --logging_steps 1 \
      --num_train_epochs 10 \
      --eval_strategy 'epoch' \
      --save_strategy 'epoch' \
      --load_best_model_at_end \
      --bf16 \
      --report_to wandb
  done
done