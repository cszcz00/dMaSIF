#!/bin/bash
# v3 ladder, round 3: the two false-negative levers, each ONE change from
# C (v3c_site_pairs_bn: site pockets + pair pooling + BatchNorm, batch 32,
# const lr, no chemo / bank / extras; val MRR 0.114 at steps 1000-1500).
#
#   v3f  runs/v3f_pairs_bn_decoys   C + site decoys: up to 7 of the receptor's
#        other fpocket pockets per row (top 8 by druggability, clean of the
#        site, its 4 A label and sibling sites), trained against the row's
#        ligand by --lam_site 1. Negatives that cannot be homologues: same
#        protein. Never scored across proteins, so mrr / nce mean what they
#        meant for C.
#   v3g  runs/v3g_pairs_bn_bank_tau C + bank 2048 / 1024 extra ligands (= v3e)
#        + --tau_min 0.06. Whether the temperature floor keeps the bank from
#        sharpening onto its false negatives. 0.06 because every run that
#        learned ended at 0.064-0.076, while the two bank runs ended at 0.036
#        (v2) and 0.043 (v3) - a floor that blocks only what the bank added.
#
# Read (1,500 steps, the same 400 val rows, eval every 125):
#   v3f  mrr against C (0.114) at matched steps; `site` = how often a val
#        ligand scores its own pocket above every decoy on its receptor (C has
#        no decoys, so its log says NaN; its baseline is the eval line below).
#        The point of decoys is the BLIND number: eval both checkpoints with
#        --pocket_select topk and compare. Known-site mrr may not move at all.
#   v3g  against v3e (v3_diag2.sh). If v3e stalls or its cosdiff climbs back
#        toward 0.99 and v3g does not, the temperature was the mechanism. If
#        both stall, it is the bank itself (staleness, or false negatives the
#        floor does not soften) - next lever: debiased / soft-weighted InfoNCE.
#        Read the new `tau` column in both: v3e's shows where an unfloored
#        temperature goes with ~2,000 candidates (if v3e ran before the
#        column existed, its final value is in last.pt: docs/STATUS.md).
#
# Baselines from checkpoints that already exist (container, any node):
#   # C's own decoy `site` rate, never trained on decoys:
#   CKPTS="$WORK/runs/v3c_site_pairs_bn/best.pt" EVAL_ARGS="--use_ckpt_split \
#       --pocket_select site --n_decoys 7 --pocket_rank drug" sbatch src/slurm/eval.sbatch
#   # blind, for C and (when done) v3f:
#   CKPTS="$WORK/runs/v3c_site_pairs_bn/best.pt $WORK/runs/v3f_pairs_bn_decoys/best.pt" \
#       EVAL_ARGS="--use_ckpt_split --pocket_select topk --pocket_rank drug" \
#       sbatch src/slurm/eval.sbatch
#
# Usage (from $WORK, login node):  bash src/slurm/v3_diag3.sh
# Logs: v3f_<jobid>.log, v3g_<jobid>.log in $WORK. Needs `train.py selftest`
# group [11] to pass first.
set -e
cd $WORK
submit() {   # job name, run name, then VAR=value overrides
    local job=$1 name=$2; shift 2
    env SKIP_SMOKE=1 STEPS=1500 CHEMO=0 BANK= EXTRA_LIG= LR_SCHEDULE=const WARMUP=0 \
        POOL_MODE=pairs ANCHOR_NORM=batch RUN=$WORK/runs/$name "$@" \
        sbatch --job-name=$job --output=${job}_%j.log \
        $WORK/src/slurm/train_v3.sbatch
}
# v3f keeps the smoke: the decoy path (pooling + lam_site) has never run on a
# GPU, and the smoke now runs its loss too.
submit v3f  v3f_pairs_bn_decoys   SKIP_SMOKE=0 DECOYS=7 LAM_SITE=1 POCKET_RANK=drug
submit v3g  v3g_pairs_bn_bank_tau BANK=2048 EXTRA_LIG=1024 TAU_MIN=0.06
