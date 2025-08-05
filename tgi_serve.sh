#!/bin/bash

# This script is used to serve any HF model.
# In this case, it serves `Qwen/Qwen2.5-Coder-32B-Instruct`.

export CUDA_VISIBLE_DEVICES=0   # change with your device ID(s)
export HF_HOME="YOUR HF CACHE DIRECTORY"

volume=data   # change with your actual volume to mount
network=r2vul-inference-net

qwen32b=Qwen/Qwen2.5-Coder-32B-Instruct

# you can change the ports to your liking
docker run -d -it \
  --name qwen32b-model \
  --network $network \
  --gpus "device=${CUDA_VISIBLE_DEVICES}" \
  --shm-size 64g \
  -p 9100:80 \
  -v $volume:/data \
  -e HF_HOME=$HF_HOME \
  ghcr.io/huggingface/text-generation-inference:3.3.4 \
  --model-id $qwen32b \
  --dtype bfloat16
