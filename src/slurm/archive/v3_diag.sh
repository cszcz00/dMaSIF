#!/bin/bash
# v3 diagnostic ladder: submit three short runs that add ONE change at a time
# on top of runs/b32's known-good setup, to find what flattened runs/v3
# (val MRR at chance for 4,125 steps, train nce flat at ~6.5 against ~7.3
# chance, pooled cosdiff climbing 0.83 -> 0.99; docs/STATUS.md "v3").
#
#   A  v3a_site_mean      b32 + fpocket site pockets, MEAN pooling
#   B  v3b_site_pairs     A + PAIR pooling
#   C  v3c_site_pairs_bn  B + BatchNorm on the pooled vector (--anchor_norm batch)
#
# All three: batch 32, 1,500 steps, constant lr 3e-4, no chemo, no bank, no
# extra ligands, the same 400 group-balanced val rows, eval every 125 steps.
# ~4 s/step, so ~2 h each. No smoke (v3's passed); a shape error in the one
# new path (C's BatchNorm on 312 pair dims) fails at the first step instead.
#
# Read at step 1,250 (runs/b32 was at MRR 0.087 there, blind, head regions;
# known-site should do better):
#   A up, B at chance           -> pair pooling is the problem; C says whether
#                                  normalisation fixes it
#   A at chance too             -> the problem is upstream (site-pocket training)
#   all three up                -> chemo or bank / extra ligands; split next
# Judge C by mrr and train nce, not cosdiff: cosdiff is measured on the pooled
# vectors BEFORE the BatchNorm, so it can stay high while retrieval works.
#
# Usage (from $WORK, login node):  bash src/slurm/v3_diag.sh
# Logs: v3a_<jobid>.log, v3b_<jobid>.log, v3c_<jobid>.log in $WORK.
set -e
cd $WORK
submit() {   # name, then VAR=value overrides
    local name=$1; shift
    env SKIP_SMOKE=1 STEPS=1500 CHEMO=0 BANK= EXTRA_LIG= LR_SCHEDULE=const WARMUP=0 \
        RUN=$WORK/runs/$name "$@" \
        sbatch --job-name=${name%%_*} --output=${name%%_*}_%j.log \
        $WORK/src/slurm/train_v3.sbatch
}
submit v3a_site_mean     POOL_MODE=mean
submit v3b_site_pairs    POOL_MODE=pairs
submit v3c_site_pairs_bn POOL_MODE=pairs ANCHOR_NORM=batch
