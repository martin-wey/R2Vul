# R2Vul: Learning to Reason about Software Vulnerabilities with Reinforcement Learning and Structured Reasoning Distillation

This is the replication package accompanying our paper *"R2Vul: Learning to Reason about Software Vulnerabilities with Reinforcement Learning and Structured Reasoning Distillation"*.

You can find the paper appendix in the supplementary material (`appendix.pdf`).

Codebase structure
---
The project is structured as follows.

    .
    ├── scripts/                # bash scripts to run specific training and inference
    ├── src/                    # source code of the project
        ├── data_annotation/    # code to generate data annotations, positive/negative reasoning, and LLM-as-a-judge rankings.
        ├── inference/          # inference scripts
        ├── training/           # training scripts for CLS, SFT, and ORPO
    ├── Dockerfile              # Dockerfile to setup the docker container
    ├── requirements.txt        # required Python libraries
    ├── tgi_serve.sh            # script to serve a HuggingFace model using TGI.
    ├── tgi_serve_r2vul.sh      # script to serve a local model using TGI.

## Environment setup

We provide a `Dockerfile` to setup a docker image to run our code.
The image is based on `nvidia/cuda:12.4.0` for Ubuntu. Depending on your machine, you can look for an appropriate base image that runs cuda 12.4.0 on [dockerhub](https://hub.docker.com/r/nvidia/cuda/tags?name=12.4.0).

1. **Build the docker image**  

```bash
docker build -t r2vul-image .
```
This builds the docker image and ensures Python 3 is properly installed.

3. **Create the docker container**  

Next, you can instantiate a new docker container based on the image we just created.
```bash
docker run -it --name r2vul -d -v R2Vul:/r2vul --gpus all r2vul-image
```
Note that if you plan to run inference with a local TGI model, add `--network=r2vul-inference-net` to have both containers on the same network.

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

## Results

We share our raw results for all experiments discussed in the paper under the `/results` folder:

1. `/main`- runs related to the main experiments (**Section 5.1**). Includes a `compute_metrics.py` script to compute paired bootstrap tests.
2. `/external_test_set` - runs related to experiments on the external test set (**Section 5.2**). Includes a `compute_metrics.py` script to compute paired bootstrap tests.
3. `/data_ablation` - runs related to the data ablation (**Section 5.3**). Includes a `render_plot.py` script to reproduce Figure 2.
4. `/class_imbalance` - runs related to class imbalance experiments (**Section 5.4**). Includes a `render_plot.py` script to reproduce Figure 3.
5. `/calibration` - runs related to model calibration experiments (**Section 5.5**). Includes a `render_plot.py` script to reproduce Figure 4.

## Datasets and Models

We make our datasets and models checkpoints available on Zenodo: https://zenodo.org/records/16741648.

### Data
1. `raw_dataset.json` - our raw data mined from NVD.
2. `paired_dataset.json` - pre/post-commit function pairs.
3. `/r2vul_dataset` - our dataset for training and inference.
4. `/external_java_test` - external Java test set in Huggingface format.

*Create a folder `/data` containing each dataset.*

### Models
1. `orpo.zip` - Qwen2.5-Coder-Instruct models fine-tuned using R2Vul (with ORPO).
2. `sft.zip` - Qwen2.5-Coder-Instruct models fine-tuned using SFT.
3. `cls.zip` - Models fine-tuned using CLS
4. `MSIVD.zip` - MSIVD model checkpoint 
5. `VulLLM.zip` - VulLLM model checkpoint

*If you want to replicate experiments with existing checkpoints, you need to download them and place them in a `/runs` folder.*

*For SFT / R2Vul models, the adapter needs to be merged with the base model before serving them using TGI (see `merge.py` script).*

Replicating Experiments
---

We provide bash scripts to run specific experiments. You simply have to run them and change some variables depending on what you want to run.
### 1. Main 

#### Inference 
- `MSIVD`: `/scripts/main/run_inference_msivd.sh`
- `VulLLM`: `/scripts/main/run_inference_vulllm.sh`
- `CLS`: `/scripts/main/run_inference_cls.sh`
- `Commercial LLMs`: `/scripts/main/run_inference_oai.sh`
- `CoT`: `/scripts/main/run_inference_tgi.sh` (run `tgi_serve.sh` first)
- `SFT` and  `R2Vul`: `/scripts/main/run_inference_tgi.sh` (run `tgi_serve_r2vul.sh` first)

#### Fine-Tuning 
- `CodeBERT (CLS)`: `/scripts/main/run_training_cls_codebert.sh`
- `Qwen2.5-Coder-Instruct (CLS)`: `/scripts/main/run_training_cls_qwen.sh`
- `SFT`: `/scripts/main/run_training_sft.sh`
- `R2Vul`: `/scripts/main/run_training_r2vul.sh`

### 2. External Test Set
- `CLS`: `/scripts/external_test_set/run_inference_cls.sh`
- `SFT`: `/scripts/external_test_set/run_inference_sft.sh` (run `tgi_serve_r2vul.sh` first)
- `R2Vul`: `/scripts/external_test_set/run_inference_r2vul.sh` (run `tgi_serve_r2vul.sh` first)

### 3. Data Ablation

#### Fine-Tuning 
- `SFT`: `/scripts/data_ablation/run_training_sft.sh`
- `R2Vul`: `/scripts/data_ablation/run_training_r2vul.sh`

For inference, run `tgi_serve_r2vul.sh` with a specific checkpoint, then run inference using `run/main/run_inference_tgi.sh`.

### 4. Class Imbalance
- `SFT`: `/scripts/class_imbalance/run_inference_sft.sh`
- `R2Vul`: `/scripts/class_imbalance/run_inference_r2vul.sh`

### 5. Model Calibration
- `R2Vul`: `/scripts/calibration/run_inference_r2vul.sh`
