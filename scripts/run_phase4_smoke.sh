#!/bin/bash
# Phase 4 Smoke Test: 1 prompt × 1 seed per config, verify no errors
set -e
cd "$(dirname "$0")/.."

CONFIG=configs/base.yaml
PROMPTS=prompts/prompts_dev.txt

echo "=== Phase 4 Smoke Test ==="
echo ""

# P3-baseline-ref (v1 path, sanity check)
echo "[1/7] P3-baseline-ref"
python scripts/run_generate.py --config $CONFIG --prompts $PROMPTS \
    --seeds 0 --max_prompts 1 \
    --skip_update --skip_mask_type complexity --extrapolation linear \
    --skip_ratio 0.25 --grid_size 8 \
    --exp_name smoke_p3-baseline-ref

# P4.1-region-only
echo "[2/7] P4.1-region-only"
python scripts/run_generate.py --config $CONFIG --prompts $PROMPTS \
    --seeds 0 --max_prompts 1 \
    --skip_update --skip_mask_type complexity --extrapolation linear \
    --skip_ratio 0.25 --region_mask --n_segments 64 \
    --exp_name smoke_p4.1-region-only

# P4.2-region-soft
echo "[3/7] P4.2-region-soft"
python scripts/run_generate.py --config $CONFIG --prompts $PROMPTS \
    --seeds 0 --max_prompts 1 \
    --skip_update --skip_mask_type complexity --extrapolation linear \
    --skip_ratio 0.25 --region_mask --n_segments 64 \
    --dilation_radius 2 --mask_blur_sigma 1.5 \
    --exp_name smoke_p4.2-region-soft

# P4.2-grid-soft
echo "[4/7] P4.2-grid-soft"
python scripts/run_generate.py --config $CONFIG --prompts $PROMPTS \
    --seeds 0 --max_prompts 1 \
    --skip_update --skip_mask_type complexity --extrapolation linear \
    --skip_ratio 0.25 --grid_size 8 \
    --dilation_radius 2 --mask_blur_sigma 1.5 \
    --exp_name smoke_p4.2-grid-soft

# P4.3-velocity
echo "[5/7] P4.3-velocity"
python scripts/run_generate.py --config $CONFIG --prompts $PROMPTS \
    --seeds 0 --max_prompts 1 \
    --skip_update --skip_mask_type complexity --extrapolation velocity \
    --skip_ratio 0.25 --region_mask --n_segments 64 \
    --dilation_radius 2 --mask_blur_sigma 1.5 \
    --exp_name smoke_p4.3-velocity

# P4.3-x0
echo "[6/7] P4.3-x0"
python scripts/run_generate.py --config $CONFIG --prompts $PROMPTS \
    --seeds 0 --max_prompts 1 \
    --skip_update --skip_mask_type complexity --extrapolation x0 \
    --skip_ratio 0.25 --region_mask --n_segments 64 \
    --dilation_radius 2 --mask_blur_sigma 1.5 \
    --exp_name smoke_p4.3-x0

# P4.3-vel-grid
echo "[7/7] P4.3-vel-grid"
python scripts/run_generate.py --config $CONFIG --prompts $PROMPTS \
    --seeds 0 --max_prompts 1 \
    --skip_update --skip_mask_type complexity --extrapolation velocity \
    --skip_ratio 0.25 --grid_size 8 \
    --exp_name smoke_p4.3-vel-grid

echo ""
echo "=== All smoke tests passed! ==="
