#!/bin/bash

export HF_HOME="/data/.cache/huggingface"
export PYTHONPATH="$PYTHONPATH:$PWD"

# multi-lingual -> run inference on each language in the list
languages=("c_sharp" "javascript" "java" "python" "c")
# For mono-lingual:
# languages=("c_sharp")

# model checkpoint -> change to your model checkpoint dir
model_dir="runs/sft/Qwen2.5-Coder1.5B-Instruct/multi"

for lang in "${languages[@]}"; do
  CUDA_VISIBLE_DEVICES=0 python src/inference/inference_cot.py \
    --model_name_or_path ${model_dir}/best_model_checkpoint \
    --output_dir ${model_dir} \
    --languages ${lang} \
    --strategy cot \
    --do_sample
done