#!/usr/bin/env bash
set -e

# Optional first argument: visualize_mask → output [Original | Recon | Mask | Token bar] to *_compare.mp4
VISUALIZE_MASK=0
if [ "${1:-}" = "visualize_mask" ]; then
    VISUALIZE_MASK=1
    shift
fi

CHECKPOINT_DIR="./infotok-flex"
GT_VIDEO_DIR="./default_videos"
OUT_VIDEO_DIR="./outputs"
MODEL_NAME="infotok_mse.pt"
strategy="elbo"
tokenizer_type="OURS4x8x8-mse-256p-88"
avg_rate=0.5

if [ "$VISUALIZE_MASK" = "1" ]; then
    python3 -m cosmos_predict1.tokenizer.inference.video_cli_compare \
        --video_pattern "${GT_VIDEO_DIR}/*.mp4" \
        --checkpoint "${CHECKPOINT_DIR}/${MODEL_NAME}" \
        --output_dir "${OUT_VIDEO_DIR}" \
        --temporal_window 81 \
        --mode torch \
        --strategy "${strategy}" \
        --tokenizer_type "${tokenizer_type}" \
        --overlap_window 3 \
        --avg_rate "${avg_rate}"
else
    python3 -m cosmos_predict1.tokenizer.inference.video_cli \
        --video_pattern "${GT_VIDEO_DIR}/*.mp4" \
        --checkpoint ${CHECKPOINT_DIR}/${MODEL_NAME} \
        --output_dir ${OUT_VIDEO_DIR} \
        --temporal_window 81 \
        --mode torch \
        --strategy ${strategy} \
        --tokenizer_type ${tokenizer_type} \
        --overlap_window 3 \
        --avg_rate ${avg_rate}
fi
