# InfoTok: Adaptive Discrete Video Tokenizer via Information-Theoretic Compression

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv%20paper-2512.16975-b31b1b.svg)](https://arxiv.org/abs/2512.16975)&nbsp;
[![huggingface weights](https://img.shields.io/badge/%F0%9F%A4%97%20Weights-infotok--flex-yellow)](https://huggingface.co/qyoo/infotok-flex)&nbsp;
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

</div>

<p align="center" style="font-size: larger;">
  <a href="https://arxiv.org/abs/2512.16975">InfoTok: Adaptive Discrete Video Tokenizer via Information-Theoretic Compression</a>
</p>

<p align="center">
<img src="assets/infotok-header.png" width=95%>
</p>

**InfoTok** is an adaptive discrete video tokenizer designed for efficient and accurate compression of long video sequences. Unlike traditional tokenizers that use a fixed compression rate, InfoTok allocates tokens dynamically based on the informational content of video segments in a principled way, significantly improving the efficiency.

## Installation

### Prerequisites

- Linux (tested on Ubuntu 20.04, 22.04, 24.04)
- Python 3.10.x
- NVIDIA GPU (H100-80GB or A100-80GB recommended)

### Setup

```bash
# Create conda environment
conda env create --file infotok.yaml
conda activate infotok

# Set PYTHONPATH
export PYTHONPATH=$(pwd)

# Install dependencies
pip install -r requirements.txt
```

### Download Checkpoints

```bash
# Download from HuggingFace
huggingface-cli download qyoo/infotok-flex --local-dir checkpoints/
```

## Inference

### Quick Start

```bash
bash exp_scripts/infotok_inference.sh
```

### Detailed Usage

```bash
python3 -m cosmos_predict1.tokenizer.inference.video_cli \
    --video_pattern "/path/to/videos/*.mp4" \
    --checkpoint /path/to/infotok_mse.pt \
    --output_dir /path/to/output \
    --tokenizer_type OURS4x8x8-mse-256p-88 \
    --temporal_window 33 \
    --overlap_window 5 \
    --strategy global_elbo \
    --avg_rate 0.5 \
    --mode torch
```

### Parameters

| Parameter | Description | Default |
|:----------|:------------|:-------:|
| `--video_pattern` | Glob pattern for input videos | Required |
| `--checkpoint` | Path to model checkpoint | Required |
| `--output_dir` | Output directory | Required |
| `--tokenizer_type` | Model architecture | `OURS4x8x8-mse-256p-88` |
| `--temporal_window` | Frames per window | `33` |
| `--overlap_window` | Overlap frames for blending | `5` |
| `--strategy` | Rate allocation (`global_elbo`, `static`) | `global_elbo` |
| `--avg_rate` | Target average token usage ratio (0.0625~1.0) | `0.5` |

## Post-Training

### 0. Additional Dependencies (Training Only)

Post-training requires additional dependencies:

```bash
# Patch Transformer Engine linking
ln -sf $CONDA_PREFIX/lib/python3.10/site-packages/nvidia/*/include/* $CONDA_PREFIX/include/
ln -sf $CONDA_PREFIX/lib/python3.10/site-packages/nvidia/*/include/* $CONDA_PREFIX/include/python3.10

# Install Transformer Engine
pip install transformer-engine[pytorch]==1.12.0

# Install Apex
git clone https://github.com/NVIDIA/apex && cd apex
CUDA_HOME=$CONDA_PREFIX pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation \
    --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" .
cd ..
```

### 1. Prepare Dataset

Register your dataset in `cosmos_predict1/tokenizer/training/datasets/dataset_provider.py`:

```python
_VIDEO_PATTERN_DICT = {
    "custom_video": "/path/to/videos/*.mp4",
}
```

### 2. Run Training

**Single GPU (Debug):**

```bash
bash exp_scripts/infotok_posttrain.sh
```

**Multi-GPU:**

```bash
export PYTHONPATH=$(pwd)
export OUTPUT_ROOT="/path/to/checkpoints"

torchrun --nproc_per_node=8 --rdzv_endpoint=localhost:29501 \
    -m cosmos_predict1.tokenizer.training.train \
    --config=cosmos_predict1/tokenizer/training/configs/config.py -- \
    experiment=ADV4x8x8_256p_CUSTOM_Posttrain \
    checkpoint.load_path=/path/to/infotok_mse.pt \
    checkpoint.strict_resume=False \
    checkpoint.load_training_state=False \
    dataloader_train.batch_size=1 \
    dataloader_train.dataset.num_video_frames=33
```

### Output Structure

```
checkpoints/
└── infotok_posttraining/tokenizer/{NAME}/checkpoints/
    ├── iter_{N}.pt           # Full checkpoint
    ├── iter_{N}_enc.jit      # Encoder (JIT)
    ├── iter_{N}_dec.jit      # Decoder (JIT)
    └── iter_{N}_ema.jit      # EMA model
```

## Citation

If you find InfoTok useful, please cite:

```bibtex
@misc{ye2025infotok,
      title={InfoTok: Adaptive Discrete Video Tokenizer via Information-Theoretic Compression}, 
      author={Haotian Ye and Qiyuan He and Jiaqi Han and Puheng Li and Jiaojiao Fan and Zekun Hao and Fitsum Reda and Yogesh Balaji and Huayu Chen and Sheng Liu and Angela Yao and James Zou and Stefano Ermon and Haoxiang Wang and Ming-Yu Liu},
      year={2025},
      eprint={2512.16975},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2512.16975}, 
}
```

## Acknowledgments

InfoTok is built on [NVIDIA Cosmos-Predict1](https://github.com/nvidia-cosmos/cosmos-predict1). We thank the Cosmos team for their infrastructure and pre-trained models.
