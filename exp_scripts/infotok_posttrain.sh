#!/bin/bash
# Posttrain script for OURS4x8x8-mse-256p-88 tokenizer
# Experiment config: cosmos_predict1/tokenizer/training/configs/experiments/cosmos_tokenize1_vit.py

# # Basic settings
# cd ~/cosmos-predict1
# source ~/.bashrc
# eval "$(conda shell.bash hook)"
# conda activate cosmos

export PYTHONPATH=$(pwd)
export OUTPUT_ROOT="./checkpoints"
export CUDA_HOME=$CONDA_PREFIX
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-$HOME/.cache/torch}"
mkdir -p $TORCH_HOME/hub/checkpoints

# Training parameters
timestamp=$(date +%Y%m%d%H%M%S)
EXPERIMENT_NAME="ADV4x8x8_256p_CUSTOM_Posttrain"  # Registered name in cosmos_tokenize1_vit.py
CHECKPOINT_PATH="./checkpoints/infotok_mse.pt"  # Pretrained checkpoint
BATCH_SIZE=1
GRAD_ACCUM=1
NUM_GPUS_PER_NODE=8
PORT=29502

# Dataloader settings (should match cosmos_tokenize1_vit.py for consistency)
CROP_HEIGHT=256
NUM_VIDEO_FRAMES=33  # Must be temporal_compression * n + 1 (4*8+1=33)

# ============================================================
# DEBUG: Single GPU test run
# ============================================================
WANDB_MODE=offline CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run --nproc_per_node=1 --rdzv_endpoint=localhost:${PORT} \
    -m cosmos_predict1.tokenizer.training.train \
    --config=cosmos_predict1/tokenizer/training/configs/config.py -- \
    experiment=${EXPERIMENT_NAME} \
    job.name=DEBUG_OURS_${timestamp} \
    checkpoint.load_path=${CHECKPOINT_PATH} \
    checkpoint.strict_resume=False \
    checkpoint.load_training_state=False \
    checkpoint.save_iter=10 \
    trainer.validation_iter=5 \
    trainer.logging_iter=1 \
    dataloader_train.dataset.crop_height=${CROP_HEIGHT} \
    dataloader_train.dataset.num_video_frames=${NUM_VIDEO_FRAMES} \
    dataloader_val.dataset.crop_height=${CROP_HEIGHT} \
    dataloader_val.dataset.num_video_frames=${NUM_VIDEO_FRAMES}

# ============================================================
# Full training: Multi-GPU
# ============================================================
# python -m torch.distributed.run --nproc_per_node=${NUM_GPUS_PER_NODE} --rdzv_endpoint=localhost:${PORT} \
#     -m cosmos_predict1.tokenizer.training.train \
#     --config=cosmos_predict1/tokenizer/training/configs/config.py -- \
#     experiment=${EXPERIMENT_NAME} \
#     job.name=OURS4x8x8_mse_Posttrain_${timestamp} \
#     checkpoint.load_path=${CHECKPOINT_PATH} \
#     checkpoint.strict_resume=False \
#     checkpoint.load_training_state=False \
#     dataloader_train.batch_size=${BATCH_SIZE} \
#     dataloader_val.batch_size=${BATCH_SIZE} \
#     dataloader_train.dataset.crop_height=${CROP_HEIGHT} \
#     dataloader_train.dataset.num_video_frames=${NUM_VIDEO_FRAMES} \
#     dataloader_val.dataset.crop_height=${CROP_HEIGHT} \
#     dataloader_val.dataset.num_video_frames=${NUM_VIDEO_FRAMES} \
#     trainer.grad_accum_iter=${GRAD_ACCUM}