# R2Vul: Learning to Reason about Software Vulnerabilities with Reinforcement Learning and Structured Reasoning Distillation

This is the replication package accompanying our paper *"R2Vul: Learning to Reason about Software Vulnerabilities with Reinforcement Learning and Structured Reasoning Distillation"*.

The datasets and models checkpoints of this paper are available on [Zenodo](https://zenodo.org/records/15029989).

Project structure
---
The project is structured as follows.

    .
    ├── scripts/            # bash scripts to run specific training and inference
    ├── src/                # source code of the project
        ├── inference/      # inference scripts
        ├── training/       # training scripts for CLS, SFT, and ORPO
    ├── Dockerfile          # Dockerfile to setup the docker container
    ├── requirements.txt    # required Python libraries

## Environment setup

We provide a `Dockerfile` to setup a docker image to run our code.
The image is based on `nvidia/cuda:12.4.0` for Ubuntu. Depending on your machine, you can look for an appropriate base image that runs cuda 12.4.0 on [dockerhub](https://hub.docker.com/r/nvidia/cuda/tags?name=12.4.0).

1. **Download the repository**

First, download this repository (top-right button). 
Next, open a terminal and change current directory to `R2Vul`.

2. **Build the docker image**  

```bash
   docker build -t r2vul-image .
```
This builds the docker image and ensures Python 3 is properly installed.

3. **Create the docker container**  

Next, you can instantiate a new docker container based on the image we just created.
```bash
docker run -it --name r2vul -d -v R2Vul:/r2vul --gpus all r2vul-image
```
You can then start the container and attach to it:
```bash
docker start r2vul
docker exec r2vul -it bash
cd r2vul # go to the source code directory
```

4. **Setup the virtual environment**

Create a new virtual environment and install the required Python libraries.
```bash
python -m venv venv
pip install -r requirements.txt
source venv/bin/activate # activate the venv
```
Note that if you do not wish to use Docker, you can simply rely on the Python venv, but we cannot guarantee that everything will run smoothly.

## Datasets and Models

The datasets and models checkpoints are available on [Zenodo](https://zenodo.org/records/15029989).

Included data:
1. `r2vul_dataset.zip` - the main dataset used for training and testing.
2. `realworld_dedup.zip` - the manually annotated Java dataset used in RQ2.
3. `raw_dataset.json`, the raw data mined from NVD.

*Depending on the experiments you want to conduct, download the according dataset zip file, and extract it under a `/data` folder.*

The models include:
1. `cls.zip` - Models fine-tuned using CLS.
2. `sft.zip` - Models fine-tuned using SFT () 
3. `orpo.zip` - Models fine-tuned using ORPO (R2Vul)
4. `MSIVD.zip` - MSIVD model checkpoint 
5. `VulLLM.zip` - VulLLM model checkpoint

*If you want to replicate experiments with existing checkpoints, download them, and place them in a `/runs` folder.*

Replicating Main Experiments
---
Below, we detail how to replicate the main experiments of the paper (RQ1 and RQ2).
For each experiment, a serie of scripts is available under the `/scripts` folder.
For each script, you can easily edit the variables to run the script for a specific model, language, or method.

### 1 - Performance Comparison
- `run_inference_cot.sh` - to run inference with chain-of-thought baselines (zero-shot, CoT, CoT-Reflection, and CoT-Contrastive).


- `run_inference.cls.sh` - to run inference with sequence classification baselines.


- `run_inference_sft_orpo.sh` - to run inference with either SFT or ORPO-tuned LLMs.


- `run_inference_msivd.sh` - to run inference using MSIVD (requires downloading `MSIVD.zip`).


- `run_inference_vulllm.sh` - to run inference using VulLLM (requires downloading `VulLLM.zip`).

### 2 - OOD Generalization and Class Imbalance

#### 2.1. Class Imbalance Experiment.
- `run_inference_cls_ratio.sh` - to run inference for the different class imbalance ratio with CLS checkpoints..


- `run_inference_sft_orpo_ratio.sh` - to run inference for the different class imbalance ratios with SFT or ORPO checkpoints.

#### 2.2. Transferability to External Test Set Experiment.
- `run_inference_cls_real.sh` - to run inference on the external Java test set using CLS checkpoints.


- `run_inference_sft_orpo_ratio.sh` - to run inference on the external Java test set using SFT or ORPO checkpoints.

Model Fine-Tuning
---
We also provide scripts to fine-tune your own model using CLS, SFT (monolingual or multilingual), and ORPO.
Again, you can change any variable and hyperparameter to your convenience. 

- `run_training_cls_codebert.sh` - to fine-tune CodeBERT using CLS.


- `run_training_cls_qwen.sh` - to fine-tune Qwen using CLS (or any other LLM available on Hugging Face).


- `run_training_sft_mono.sh` - to fine-tune an LLM using monolingual SFT.


- `run_training_sft_multi.sh` - to fine-tune an LLM using multilingual SFT.


- `run_training_orpo.sh` - to fine-tune an LLM using ORPO.

