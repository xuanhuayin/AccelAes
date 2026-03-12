#!/bin/bash
# Run all P0 eval configs sequentially, each in a fresh process to avoid OOM.
# Includes: RAS (new), bug-fixed re-runs (attn_only, attn_ffn, sasd_full).

set -e
cd /home/runkai/xuanhua/SASD

COMMON="--num_prompts 20 --seeds 0 1 2"
export PYTORCH_ALLOC_CONF=expandable_segments:True

run_config() {
    local mode=$1
    local name=$2
    echo ""
    echo "=============================="
    echo " Running: $name (mode=$mode)"
    echo "=============================="
    python scripts/run_p0_eval.py --mode "$mode" --only_config "$name" $COMMON \
        2>&1 | tee "logs/p0_${mode}_${name}.log"
}

# ── COMPARE configs ──
# baseline: already done (skip)
# delta_dit: already done (skip)
# fora: already done (skip)
run_config baseline_compare ras        # NEW: RAS baseline
run_config baseline_compare sasd_full  # RE-RUN: bug-fixed sparse attn+FFN

# ── ABLATION configs ──
# baseline: already done (skip)
# fskip_only: already done, unaffected by bug (skip)
run_config ablation attn_only   # RE-RUN: bug fixed (was always dense)
run_config ablation attn_ffn    # RE-RUN: bug fixed (sparse FFN now works)
# sasd_full ablation: symlink -> compare/sasd_full, already re-run above (skip)

echo ""
echo "=============================="
echo " All generation done!"
echo " Now running metrics..."
echo "=============================="

python scripts/run_p0_eval.py --mode all --skip_gen $COMMON \
    2>&1 | tee logs/p0_metrics.log

echo "=== DONE ==="
