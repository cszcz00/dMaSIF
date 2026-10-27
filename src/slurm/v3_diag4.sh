#!/bin/bash
# v3 ladder, round 4. What rounds 1-3 settled (docs/STATUS.md "v3"):
#   A/B/C   site pockets, pair pooling and BatchNorm all TRAIN; C is the best
#           configuration, mrr 0.114 at 1,500 steps.
#   v3c+    C PLATEAUS at ~0.18 by step ~3,600 while train nce keeps falling.
#           The ceiling is MEMORISATION (~69 visits per target group), so more
#           steps buy nothing and more TARGETS is the lever.
#   v3d     chemo neither breaks nor helps. Dropped.
#   v3e/v3g the bank + extra ligands is what killed v3, and it is NOT the
#           temperature: tau never leaves 0.07, so v3g's floor never bound and
#           it reproduced v3e exactly. Bank dropped.
#   v3f     decoys + lam_site 1 raise within-protein `site` 0.605 -> 0.794 but
#           cost cross-protein mrr (0.114 -> 0.093) and 1.5x the time.
#
# RUN THE EVALS FIRST - they are minutes on the test queue and they decide
# whether decoys stay at all. Blind (topk / drug) is the only number
# comparable across configurations and the only one a screen would see:
#   CKPTS="$WORK/runs/v3c_1500/best.pt $WORK/runs/v3f_pairs_bn_decoys/best.pt \
#   $WORK/runs/v3c_site_pairs_bn/best.pt" \
#   EVAL_ARGS="--use_ckpt_split --pocket_select topk --pocket_rank drug" \
#       sbatch src/slurm/eval.sbatch
# C@1,500 blind was 0.088 (= b32 at matched data). Read: v3f above it means
# the decoys' within-protein skill IS what blind screening needs, and lam_site
# stays; v3f at or below it means decoys only traded one metric for another and
# C stands. C@5,000 is in the same job because its plateau has never been
# scored blind - that is the number that says whether the v3 stack beats b32.
#
# Then the two runs here, each ONE change from C:
#   v3h  runs/v3h_decoys_ks0_lam03   C + 7 decoys, --k_neg 0, --lam_site 0.3.
#        Two questions at once, deliberately: is v3f's mrr loss the lam_site
#        WEIGHT (0.3 should recover most of it while keeping the site gain),
#        and does dropping the 8 redundant k_neg patches restore the speed
#        (16 regions -> 8, so expect ~4 s/step against v3f's 6.3). If mrr
#        comes back to ~0.11 with site still ~0.75, this is the decoy
#        configuration to keep. If mrr stays ~0.09, the decoys themselves
#        cost it and the trade is real.
#   v3i  runs/v3i_kneg0              C with --k_neg 0 and nothing else. The
#        throughput baseline: pooling drops 9 regions -> 1, so expect ~2.5
#        s/step against C's 4.2 - a 1.7x on every future run, which matters
#        because the ceiling is data, not steps. The risk is the BatchNorm:
#        it would see 32 vectors per step instead of 288, so if mrr falls
#        below C's 0.114 the statistics were load-bearing and k_neg stays.
#        Compare mrr AND cosdiff, since BatchNorm is what moves cosdiff.
#
# Also worth doing while these run - the retrieval inspection on C's plateau,
# which says WHAT the model retrieves rather than how well (hub collapse, the
# size-only baseline, chemotype rank; src/retrieval_report.py):
#   for SEL in site topk; do
#     python src/retrieval_report.py dump --ckpt $WORK/runs/v3c_site_pairs_bn/best.pt \
#       --items $WORK/runs/v1/items.tsv --lig_feats $WORK/runs/v1/ligand_feats.npz \
#       --surface_cache $WORK/surfaces --pocket_select $SEL --pocket_rank drug \
#       --n_train 400 --n_val 400 --out $WORK/reports/v3c5000_$SEL.npz
#   done
#
# Usage (from $WORK, login node):  bash src/slurm/v3_diag4.sh
# Logs: v3h_<jobid>.log, v3i_<jobid>.log in $WORK.
set -e
cd $WORK
submit() {   # job name, run name, then VAR=value overrides
    local job=$1 name=$2; shift 2
    env SKIP_SMOKE=1 STEPS=1500 CHEMO=0 BANK= EXTRA_LIG= LR_SCHEDULE=const WARMUP=0 \
        POOL_MODE=pairs ANCHOR_NORM=batch RUN=$WORK/runs/$name "$@" \
        sbatch --job-name=$job --output=logs/${job}_%j.log \
        $WORK/src/slurm/train_v3.sbatch
}
# Both touch the k_neg path for the first time, so both keep the smoke: with
# --k_neg 0 a batch can yield one pooled vector per protein and the BatchNorm
# guard for fewer than two samples has never been exercised on a GPU.
# K_NEG, not TRAIN_ARGS: TRAIN_ARGS reaches only the main run, so the smoke
# would have tested the k_neg 8 path instead of the one being changed.
submit v3h v3h_decoys_ks0_lam03 SKIP_SMOKE=0 K_NEG=0 DECOYS=7 LAM_SITE=0.3 \
    POCKET_RANK=drug
submit v3i v3i_kneg0            SKIP_SMOKE=0 K_NEG=0
