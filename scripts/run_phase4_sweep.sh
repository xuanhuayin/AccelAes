#!/bin/bash
# Phase 4 Quick Sweep: 10 prompts × 3 seeds = 30 images per config
set -e
cd "$(dirname "$0")/.."

CONFIG=configs/base.yaml
PROMPTS=prompts/prompts_dev.txt
SEEDS="0,1,2"
MAX_P=10

echo "=== Phase 4 Quick Sweep (${MAX_P} prompts × 3 seeds = 30 imgs/group) ==="
echo ""

# P3-baseline-ref (v1 path for regression check)
echo "[1/7] P3-baseline-ref"
python scripts/run_generate.py --config $CONFIG --prompts $PROMPTS \
    --seeds $SEEDS --max_prompts $MAX_P \
    --skip_update --skip_mask_type complexity --extrapolation linear \
    --skip_ratio 0.25 --grid_size 8 \
    --exp_name p3-baseline-ref --save_masks

# P4.1-region-only
echo "[2/7] P4.1-region-only"
python scripts/run_generate.py --config $CONFIG --prompts $PROMPTS \
    --seeds $SEEDS --max_prompts $MAX_P \
    --skip_update --skip_mask_type complexity --extrapolation linear \
    --skip_ratio 0.25 --region_mask --n_segments 64 \
    --exp_name p4.1-region-only --save_masks

# P4.2-region-soft
echo "[3/7] P4.2-region-soft"
python scripts/run_generate.py --config $CONFIG --prompts $PROMPTS \
    --seeds $SEEDS --max_prompts $MAX_P \
    --skip_update --skip_mask_type complexity --extrapolation linear \
    --skip_ratio 0.25 --region_mask --n_segments 64 \
    --dilation_radius 2 --mask_blur_sigma 1.5 \
    --exp_name p4.2-region-soft --save_masks

# P4.2-grid-soft
echo "[4/7] P4.2-grid-soft"
python scripts/run_generate.py --config $CONFIG --prompts $PROMPTS \
    --seeds $SEEDS --max_prompts $MAX_P \
    --skip_update --skip_mask_type complexity --extrapolation linear \
    --skip_ratio 0.25 --grid_size 8 \
    --dilation_radius 2 --mask_blur_sigma 1.5 \
    --exp_name p4.2-grid-soft --save_masks

# P4.3-velocity
echo "[5/7] P4.3-velocity"
python scripts/run_generate.py --config $CONFIG --prompts $PROMPTS \
    --seeds $SEEDS --max_prompts $MAX_P \
    --skip_update --skip_mask_type complexity --extrapolation velocity \
    --skip_ratio 0.25 --region_mask --n_segments 64 \
    --dilation_radius 2 --mask_blur_sigma 1.5 \
    --exp_name p4.3-velocity --save_masks

# P4.3-x0
echo "[6/7] P4.3-x0"
python scripts/run_generate.py --config $CONFIG --prompts $PROMPTS \
    --seeds $SEEDS --max_prompts $MAX_P \
    --skip_update --skip_mask_type complexity --extrapolation x0 \
    --skip_ratio 0.25 --region_mask --n_segments 64 \
    --dilation_radius 2 --mask_blur_sigma 1.5 \
    --exp_name p4.3-x0 --save_masks

# P4.3-vel-grid
echo "[7/7] P4.3-vel-grid"
python scripts/run_generate.py --config $CONFIG --prompts $PROMPTS \
    --seeds $SEEDS --max_prompts $MAX_P \
    --skip_update --skip_mask_type complexity --extrapolation velocity \
    --skip_ratio 0.25 --grid_size 8 \
    --exp_name p4.3-vel-grid --save_masks

echo ""
echo "=== Quick Sweep done! Run eval: ==="
echo "python scripts/run_eval.py --exp_dirs outputs/p3-baseline-ref outputs/p4.1-region-only outputs/p4.2-region-soft outputs/p4.2-grid-soft outputs/p4.3-velocity outputs/p4.3-x0 outputs/p4.3-vel-grid"
