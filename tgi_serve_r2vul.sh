#!/bin/bash

# This script is used to serve a SFT/R2Vul model.
# Note that the LoRA adapter has to be merged with the base model prior to running this script.

export CUDA_VISIBLE_DEVICES=0   # change with your device ID(s)
export HF_HOME="YOUR HF CACHE DIRECTORY"

volume=data   # change with your actual volume to mount
network=r2vul-inference-net

r2vul_model=/models/r2vul-1.5B    # path to the merged R2Vul model

# you can change the ports to your liking
docker run -d -it \
  --name r2vul-model \
  --network $network \
  --gpus "device=${CUDA_VISIBLE_DEVICES}" \
  --shm-size 64g \
  -p 9100:80 \
  -v $volume:/data \
  -e HF_HOME=$HF_HOME \
  ghcr.io/huggingface/text-generation-inference:3.3.4 \
  --model-id $r2vul_model \
  --dtype bfloat16 \
  --enable-prefill-logprobs   # required for model calibration experiments
