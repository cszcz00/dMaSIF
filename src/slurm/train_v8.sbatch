#!/bin/bash
#SBATCH --partition=alpha
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/train_v8_%j.log
# v8: the pocket logit places the anchors again, trained on contact + PLIP;
# learned atom typing; pair pooling; a light Tanimoto pull on the ligands.
# A preset over train.sbatch, like train_v3.sbatch: every value below is a
# default the submitting shell can override, the smoke + resume check runs
# first, and the job resumes itself (resubmit, SKIP_SMOKE=1 to save time).
#
# The stack, per training row:
#   whole receptor surface -> curvatures + AtomNet + DirAtomNet over the 13
#      atom features and polar directions (--atom_feats, 32 input channels)
#   -> dMaSIF conv -> pocket head logit per point
#   -> ball_R8 regions from the logit's top 5% (--region_source predicted,
#      k_pos 8 pocket anchors; the original b32-era selector)
#   -> pair pooling of each region (--pool_mode pairs: 8 types x 8 types x 8
#      distance bins -> 36 x 8 = 288 pair dims + mean + mass), weighted by
#      the pocket probability -> BatchNorm -> projection
#   -> max over the 8 anchors -> InfoNCE against the ligand GNN.
# Training signals beyond InfoNCE:
#   pocket BCE, target graded (--pocket_target contact_plip): a surface
#     point within 4 A of the row's ligand is a positive at POCKET_PLIP_BASE
#     (0.5), and a full positive where PLIP says the ligand interacts there;
#     PLIP labels outside the 4 A pocket are dropped. So the logit separates
#     pocket from surface as before and peaks at interaction points inside
#     it, which is where the ball selector's seeds and centres go.
#   ligand Tanimoto pull (--lam_lig_sim, lig_sim.py): pairs of TRAIN ligands
#     at Tanimoto >= LIG_SIM_THR asked for cosine >= their Tanimoto in the
#     scoring space; one-sided, weight 0.05.
#
# This is a BLIND evaluation: each val protein's anchors come from its own
# pocket head, with no knowledge of the ligand. It is not comparable to C /
# v4-v7 (known site, fpocket at the ligand); the comparable rows are a blind
# eval of those (eval.sbatch with --pocket_select topk --pocket_rank drug)
# and the ablations below. `fb` is the share of val proteins none of whose 8
# anchors matched the pocket (scored on the fallback ball): the honest
# measure of how well the logit places anchors.
#
# Needs: dataset_v2 built (items.tsv, ligand_feats.npz), PLIP arrays in
# $WORK/data/plip, atom_feats in the surface cache (ATOM_FEATS=1
# RUN=$WORK/out/runs/dataset_v2 sbatch src/slurm/cache_surfaces.sbatch, or it
# is added lazily), and the pocket head $WORK/out/encoder_test/
# head_pretrained_300.pt (inflated to 32 channels at load, so step 0 is its
# function).
#
# First, on the login node (container, CPU):
#   EX="apptainer exec --cleanenv --env OMP_NUM_THREADS=8 $WORK/dmasif_sandbox python"
#   $EX src/lig_sim.py selftest && $EX src/plip_loss.py selftest
#
# Run:
#   sbatch --job-name=v8 src/slurm/train_v8.sbatch
# Ablations (same seed, data, batches - partners come from their own stream):
#   POCKET_TARGET=contact RUN=$WORK/out/runs/v8_contact      ... the PLIP grading
#   LAM_LIG_SIM=0         RUN=$WORK/out/runs/v8_nolsim       ... the Tanimoto pull
#   POCKET_TARGET=contact LAM_LIG_SIM=0 RUN=$WORK/out/runs/v8_base   ... both off
#
# Read, per evaluation line: mrr / top10 against chance; fb (lower = the
# logit finds the pocket); AUROC / AP (the head against the CONTACT label,
# comparable to 0.937 / 0.47 even with the graded target). Under it:
#   "pocket head inside the pocket: AUROC x" - PLIP-interacting vs
#      contact-only points inside the pocket; 0.5 = no grading learned. Should
#      rise above v8_contact's.
#   "ligsim  loss / pulled pairs cos vs Tanimoto / random in-batch cos" - the
#      pulled pairs' cos should climb toward their Tanimoto; random cos is
#      the collapse alarm: if it climbs toward the pulled cos, lower
#      LAM_LIG_SIM or raise LIG_SIM_THR.
#
# Overrides here: POCKET_TARGET (contact_plip | contact), POCKET_PLIP_BASE,
#   PLIP (dir), PLIP_SOFT (1.5; empty = hard labels), LAM_LIG_SIM, LIG_SIM_THR,
#   LIG_SIM_K, ATOM_FEATS (1; 0 = off), PAIR_ARGS, plus everything
#   train.sbatch takes (K_NEG, HEAD, ...). The v6/v7 PLIP type / salience
#   losses (LAM_PLIP) are not wired here: they need --pocket_select site.
export RUN=${RUN:-$WORK/out/runs/v8_logit_plip_af_pairs_lsim}
export DATA=${DATA:-$WORK/out/runs/dataset_v2}
export BATCH=${BATCH:-32}
export STEPS=${STEPS:-10000}
export VAL_BALANCE=${VAL_BALANCE-1}
export POOL_MODE=${POOL_MODE:-pairs}
export ANCHOR_NORM=${ANCHOR_NORM:-batch}
export CHEMO=${CHEMO-0}
export BANK=${BANK-}
export EXTRA_LIG=${EXTRA_LIG-}
export LR_SCHEDULE=${LR_SCHEDULE:-const}
export WARMUP=${WARMUP:-0}
# Negative patches are never scored and no loss reads them under the
# predicted source; with pair pooling they would double the pooling cost and
# only feed the BatchNorm. 8 pocket anchors x 32 proteins is plenty for it.
export K_NEG=${K_NEG-0}
export SURFACE_RAM=${SURFACE_RAM:-32000}
export LABEL_RAM=${LABEL_RAM:-40000}
PLIP=${PLIP:-$WORK/data/plip}
POCKET_TARGET=${POCKET_TARGET:-contact_plip}
PLIP_SOFT=${PLIP_SOFT-1.5}
LAM_LIG_SIM=${LAM_LIG_SIM-0.05}
if [ "$POCKET_TARGET" = contact_plip ] && [ ! -d "$PLIP" ]; then
    echo "no $PLIP; sbatch src/slurm/plip.sbatch first (or POCKET_TARGET=contact)"; exit 1
fi
# In REGION_ARGS so the smoke runs the whole v8 path, losses included.
export REGION_ARGS="--region_source predicted --pocket_target $POCKET_TARGET \
--pocket_plip_base ${POCKET_PLIP_BASE:-0.5} --plip $PLIP ${PLIP_SOFT:+--plip_soft $PLIP_SOFT} \
${LAM_LIG_SIM:+--lam_lig_sim $LAM_LIG_SIM} --lig_sim_thr ${LIG_SIM_THR:-0.5} \
--lig_sim_k ${LIG_SIM_K:-32} ${PAIR_ARGS:-}"
if [ "${ATOM_FEATS-1}" != 0 ] && [ -n "${ATOM_FEATS-1}" ]; then
    export REGION_ARGS="$REGION_ARGS --atom_feats"
fi
export TRAIN_ARGS=${TRAIN_ARGS:-"--eval_every 125 --save_every 125 \
--val_set $WORK/out/valset_v1v2 --keep_steps 2500,5000"}
exec bash $WORK/src/slurm/train.sbatch
