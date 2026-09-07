# `jiawen` Environment: Core `vllm` Packages

This file summarizes the packages in the current `jiawen` environment that are most relevant for running `vllm`, along with the versions observed on this machine.

## Runtime Baseline

- `python==3.10.18`
- `pip==25.1`
- `setuptools==78.1.1`

## Core Packages Required for `vllm`

- `vllm==0.19.1`
- `torch==2.10.0`
- `triton==3.6.0`
- `xformers==0.0.31`
- `numpy==2.2.6`

## Model Loading / Tokenization

- `transformers==5.12.1`
- `tokenizers==0.22.2`
- `safetensors==0.7.0`
- `huggingface-hub==1.20.1`
- `sentencepiece==0.2.0`

## CUDA Runtime Packages Present in `jiawen`

These are important because the current environment uses the CUDA 12.8 build of PyTorch.

- `nvidia-cublas-cu12==12.8.4.1`
- `nvidia-cuda-cupti-cu12==12.8.90`
- `nvidia-cuda-nvrtc-cu12==12.8.93`
- `nvidia-cuda-runtime-cu12==12.8.90`
- `nvidia-cudnn-cu12==9.10.2.21`
- `nvidia-cufft-cu12==11.3.3.83`
- `nvidia-cufile-cu12==1.13.1.3`
- `nvidia-curand-cu12==10.3.9.90`
- `nvidia-cusolver-cu12==11.7.3.90`
- `nvidia-cusparse-cu12==12.5.8.93`
- `nvidia-cusparselt-cu12==0.7.1`
- `nvidia-nccl-cu12==2.27.5`
- `nvidia-nvjitlink-cu12==12.8.93`
- `nvidia-nvtx-cu12==12.8.90`

## Useful but Scenario-Dependent

These are not always strictly required for the simplest `vllm` usage, but they often matter depending on how you serve or load models:

- `fastapi==0.137.2`
- `uvicorn==0.22.0`
- `ray==2.47.1`
- `flashinfer-python==0.6.6`
- `flashinfer-cubin==0.6.6`
- `gguf==0.17.1`
- `compressed-tensors==0.15.0.1`
- `bitsandbytes==0.46.0`
- `autoawq==0.2.9`

## Probably Not Core for a Minimal `vllm` Environment

These exist in `jiawen`, but they are usually not necessary if the target is only "run `vllm`":

- `deepspeed==0.15.0`
- `accelerate==1.12.0`
- `datasets==5.0.0`
- `trl==1.6.0`
- `wandb==0.24.2`
- `anthropic==0.111.0`
- `openai==2.43.0`
- `matplotlib==3.10.3`
- `pandas==2.3.3`
- `seaborn==0.13.2`

## Notes

- Observed runtime probe:
  - `torch.__version__ == 2.10.0+cu128`
  - `torch.version.cuda == 12.8`
  - `vllm.__version__ == 0.19.1`
- `torch.cuda.is_available()` was `False` in the probe context on this machine, so GPU availability on the target server still needs to be verified separately with `nvidia-smi`.
- If you rebuild the environment instead of copying it, start from the "Core Packages Required for `vllm`" and "Model Loading / Tokenization" sections first, then only add the scenario-dependent packages if needed.
