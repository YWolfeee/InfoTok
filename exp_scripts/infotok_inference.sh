#!/usr/bin/env bash
set -e

CHECKPOINT_DIR="./infotok/checkpoints"
GT_VIDEO_DIR="./default_videos"
OUT_VIDEO_DIR="./outputs"
MODEL_NAME="infotok_mse.pt"
strategy="global_elbo"
tokenizer_type="OURS4x8x8-mse-256p-88"
avg_rate=0.5

python3 -m cosmos_predict1.tokenizer.inference.video_cli \
    --video_pattern "${GT_VIDEO_DIR}/*.mp4" \
    --checkpoint ${CHECKPOINT_DIR}/${MODEL_NAME} \
    --output_dir ${OUT_VIDEO_DIR} \
    --temporal_window 33 \
    --mode torch \
    --strategy ${strategy} \
    --tokenizer_type ${tokenizer_type} \
    --overlap_window 5 \
    --avg_rate ${avg_rate}
