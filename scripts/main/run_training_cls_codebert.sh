#!/bin/bash

export HF_HOME="/data/.cache/huggingface"
export WANDB_PROJECT="R2Vul"
export PYTHONPATH="$PYTHONPATH:$PWD"

languages=("c_sharp" "javascript" "java" "python" "c")

for lang in "${languages[@]}"; do
  CUDA_VISIBLE_DEVICES=0 python src/training/train_cls.py \
    --output_dir "runs/cls/CodeBERT/${lang}" \
    --run_name "cls/CodeBERT/${lang}" \
    --model_name_or_path "microsoft/codebert-base" \
    --languages ${lang} \
    --learning_rate 5.0e-5 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --logging_steps 10 \
    --num_train_epochs 10 \
    --eval_strategy "epoch" \
    --save_strategy "epoch" \
    --load_best_model_at_end True \
    --save_total_limit 1 \
    --report_to wandb
done