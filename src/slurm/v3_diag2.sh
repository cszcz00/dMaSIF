#!/bin/bash
# v3 diagnostic ladder, round 2. Round 1 (v3_diag.sh) showed fpocket site
# pockets and pair pooling both TRAIN: A site+mean 0.096, B +pairs 0.103,
# C +pairs+BatchNorm 0.114 val MRR (mean of the evaluations at steps
# 1000-1500, chance 0.017), all still rising. So runs/v3's collapse came from
# what it added on top: chemo, the memory bank + extra ligands, or the cosine
# schedule. Each run below differs from C by ONE thing.
#
#   v3d  runs/v3d_pairs_bn_chemo   C + chemo channels (--chemo)
#   v3e  runs/v3e_pairs_bn_bank    C + bank 2048 + 1024 extra ligands
#   v3c+ runs/v3c_site_pairs_bn    C itself, RESUMED 1500 -> 5000 steps
#
# Read: a run at chance, or cosdiff climbing to ~0.99, names v3's culprit.
# v3e's nce is over ~2,000 candidates, so judge it by mrr and cosdiff only.
# Both D and E learning -> the cosine schedule or the combination; one more
# run (C + chemo + bank) settles it. v3c+ keeps the constant lr, so its curve
# simply continues; it is where the best configuration plateaus.
#
# Usage (from $WORK, login node):  bash src/slurm/v3_diag2.sh
# Logs: v3d_<jobid>.log, v3e_<jobid>.log, v3c+_<jobid>.log in $WORK.
set -e
cd $WORK
submit() {   # job name, run name, then VAR=value overrides
    local job=$1 name=$2; shift 2
    env SKIP_SMOKE=1 STEPS=1500 CHEMO=0 BANK= EXTRA_LIG= LR_SCHEDULE=const WARMUP=0 \
        POOL_MODE=pairs ANCHOR_NORM=batch RUN=$WORK/runs/$name "$@" \
        sbatch --job-name=$job --output=${job}_%j.log \
        $WORK/src/slurm/train_v3.sbatch
}
submit v3d  v3d_pairs_bn_chemo CHEMO=1
submit v3e  v3e_pairs_bn_bank  BANK=2048 EXTRA_LIG=1024
submit v3c+ v3c_site_pairs_bn  STEPS=5000
