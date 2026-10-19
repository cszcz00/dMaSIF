#!/bin/bash
#SBATCH --partition=alpha
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=train_%j.log
# Contrastive protein-ligand training (src/train.py). The real loop: both
# encoders train, the loss is InfoNCE over max-over-anchor scores plus the
# pocket BCE, and validation reports retrieval metrics every --eval_every.
#
# Needs, from the CPU venv, BEFORE submitting:
#   python src/train.py prepare --index datasets/dataset_v1/index.parquet \
#       --out runs/v1 --split train --exclude encoder_test/items.tsv
#   python src/ligand_features.py --index datasets/dataset_v1/index.parquet \
#       --out runs/v1
# and, strongly recommended, a trained pocket head so the regions are usable
# from step 0 (src/slurm/head_ckpt.sbatch writes it in 30 min):
#   $WORK/encoder_test/head_pretrained_300.pt
#
# This job RESUMES ITSELF. --max_hours stops cleanly inside the walltime and
# checkpoints; resubmitting the same job picks up $RUN/last.pt and carries on,
# so a long run is a chain of these rather than one impossible allocation.
# The surface cache is shared, so the second job in a chain never resamples.
#
# What to read in the log, per evaluation line:
#   mrr / top1   ligand -> protein retrieval on the val groups, against chance
#   site         of proteins with a pocket-matching anchor, how often the
#                ligand's best anchor is that one   <- the ranking weakness
#   AUROC / AP   the pocket head, comparable to encoder_test's 0.937 / 0.47
#   fb           fraction of val proteins needing the fallback ball; high and
#                flat means the loop is leaning on labels it will not have
#   cosdiff      median cosine between DIFFERENT proteins' top anchors. 0.993
#                before any contrastive training. This is the number to move.
#
# Overrides: RUN, STEPS, BATCH, ARM, LR, LAM_POCKET, LAM_SITE, POOL_MODE,
#   ANCHOR_NORM, MAX_HOURS, LIMIT, SURFACES, CHEMO=1, SKIP_SMOKE=1, SMOKE_MIN,
#   FREEZE=1 (--freeze_encoder: the frozen-protein control),
#   VAL_BALANCE=1 (--val_balance: group-balanced validation draw),
#   SURFACE_RAM (training surfaces held in host RAM, ~1.1 MB each plus their
#     neighbour lists; 512), LABEL_RAM (pocket masks, small; 1024),
#   v2 levers, all off by default: BANK (protein memory bank size),
#     EXTRA_LIG (extra ligand negatives per step), DESC_DIMS (descriptor
#     branch width), LR_SCHEDULE (const | cosine), WARMUP (steps),
#     GPU_NEIGHBORS=1 (region neighbour lists on the GPU),
#   TRAIN_ARGS (anything else passed through verbatim).
# src/slurm/train_v2.sbatch is this file with the v2 levers switched on.
# A resumed run keeps its checkpoint's split, so VAL_BALANCE only matters for
# a NEW run directory; FREEZE must match the checkpoint or train.py refuses.
#
# The second round of runs (after runs/v1), one GPU job each. DATA points the
# new run directories at v1's items.tsv / ligand_feats.npz (DATA defaults to
# RUN), and VAL_BALANCE=1 gives all three the same balanced validation set:
#   SKIP_SMOKE=1 sbatch src/slurm/train.sbatch                       # finish v1
#   D=$WORK/runs/v1
#   DATA=$D RUN=$WORK/runs/frozen FREEZE=1 VAL_BALANCE=1 sbatch src/slurm/train.sbatch
#   DATA=$D RUN=$WORK/runs/shells POOL_MODE=shells VAL_BALANCE=1 sbatch src/slurm/train.sbatch
#   DATA=$D RUN=$WORK/runs/b32 BATCH=32 STEPS=5000 VAL_BALANCE=1 \
#       TRAIN_ARGS="--eval_every 125 --save_every 125" sbatch src/slurm/train.sbatch
set -e
module load apptainer
cd $WORK
RUN=${RUN:-$WORK/runs/v1}
SURFACES=${SURFACES:-$WORK/surfaces}
STEPS=${STEPS:-20000}
BATCH=${BATCH:-8}
ARM=${ARM:-gnn}
LR=${LR:-3e-4}
LAM_POCKET=${LAM_POCKET:-1.0}
LAM_SITE=${LAM_SITE:-0.0}
POOL_MODE=${POOL_MODE:-mean}
ANCHOR_NORM=${ANCHOR_NORM:-none}
SURFACE_RAM=${SURFACE_RAM:-512}
LABEL_RAM=${LABEL_RAM:-1024}
LR_SCHEDULE=${LR_SCHEDULE:-const}
WARMUP=${WARMUP:-0}
DATA=${DATA:-$RUN}            # where items.tsv / ligand_feats.npz live
MAX_HOURS=${MAX_HOURS:-11.0}
HEAD=${HEAD:-$WORK/encoder_test/head_pretrained_300.pt}
EXEC="apptainer exec --nv --cleanenv --env KEOPS_CACHE_FOLDER=$KEOPS_CACHE_FOLDER,OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK $WORK/dmasif_sandbox"
RUNPY="$EXEC python $WORK/src/train.py"

echo "node $(hostname): $(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader)"
# Fail fast if this container's PyTorch was not built for this GPU; KeOps dies
# with CUDA_ERROR_INVALID_IMAGE otherwise, hours in.
apptainer exec --nv --cleanenv $WORK/dmasif_sandbox python -c "
import sys, torch
cap = 'sm_%d%d' % torch.cuda.get_device_capability()
print('GPU', cap, '| torch built for', torch.cuda.get_arch_list())
sys.exit(0 if cap in torch.cuda.get_arch_list() else 'container cannot run on ' + cap)
"

for f in $DATA/items.tsv $DATA/ligand_feats.npz; do
    [ -f "$f" ] || { echo "missing $f; run the two CPU steps in the header first"; exit 1; }
done
HEAD_ARG=""
if [ -f "$HEAD" ]; then
    HEAD_ARG="--head_ckpt $HEAD"
else
    echo "[warn] no pocket head at $HEAD: the regions start from a random map, so"
    echo "       the fallback ball will carry the first few thousand steps."
    echo "       sbatch src/slurm/head_ckpt.sbatch writes it in 30 min."
fi
N_ITEMS=$(($(wc -l < $DATA/items.tsv) - 1))
echo "run $RUN (data $DATA); $N_ITEMS interactions; arm $ARM; pool $POOL_MODE; anchor_norm $ANCHOR_NORM"
echo "batch $BATCH; bank ${BANK:-0}; extra ligands ${EXTRA_LIG:-0}; descriptor ${DESC_DIMS:-0}; lr $LR $LR_SCHEDULE warmup $WARMUP; gpu neighbours ${GPU_NEIGHBORS:-0}"
[ -f "$RUN/last.pt" ] && echo "found $RUN/last.pt: this job will RESUME it"

COMMON="--items $DATA/items.tsv --lig_feats $DATA/ligand_feats.npz \
    --surface_cache $SURFACES --arm $ARM --pool_mode $POOL_MODE \
    --anchor_norm $ANCHOR_NORM --region_method ball --ball_radius 8 --ball_shift 2 \
    $HEAD_ARG ${CHEMO:+--chemo} ${LIMIT:+--limit $LIMIT} \
    ${BANK:+--bank $BANK} ${EXTRA_LIG:+--extra_ligands $EXTRA_LIG} \
    ${DESC_DIMS:+--desc_dims $DESC_DIMS}"
# Switches: set and not 0, so FREEZE=0 means off rather than "set, so on".
on() { [ -n "$1" ] && [ "$1" != 0 ]; }
if on "${FREEZE:-}"; then COMMON="$COMMON --freeze_encoder"; fi
if on "${VAL_BALANCE:-}"; then COMMON="$COMMON --val_balance"; fi
if on "${GPU_NEIGHBORS:-}"; then COMMON="$COMMON --gpu_neighbors"; fi

# A 30-step pass on 64 interactions first, in the SAME job, written to a
# throwaway run directory. The loop has never run on a GPU, so a shape error
# would otherwise cost a whole queue wait to find - and a surface sampled here
# lands in the shared cache, so the time is not even wasted.
#
# The smoke then RESUMES itself for 10 more steps and checks last.pt advanced.
# train_check.sh's resume stage never finished on the login node, and the
# chain of resubmissions below depends on resume entirely, so it is proven here
# in minutes rather than discovered 11 hours in. Both passes run under
# `timeout`, so a hang fails the job instead of eating the walltime.
# SMOKE_MIN (60) bounds the first pass, which also compiles KeOps's CUDA
# kernels and samples 64 surfaces; the resume pass gets a third of that.
if [ "${SKIP_SMOKE:-0}" != "1" ]; then
    SMOKE_MIN=${SMOKE_MIN:-60}
    SMOKE="$RUNPY train $COMMON --out $RUN/smoke --limit 64 \
        --eval_every 15 --save_every 30 --val_frac 0.3 --val_max 24 --batch 4"
    echo "=== smoke: 64 interactions, 30 steps (limit ${SMOKE_MIN} min) ==="
    timeout ${SMOKE_MIN}m $SMOKE --steps 30 --no_auto_resume \
        || { echo "smoke FAILED or timed out (exit $?)"; exit 1; }
    echo "=== smoke resume: 30 -> 40 steps from $RUN/smoke/last.pt ==="
    timeout $((SMOKE_MIN / 3))m $SMOKE --steps 40 \
        || { echo "smoke resume FAILED or timed out (exit $?)"; exit 1; }
    AT=$($EXEC python -c "
import torch
print('CKPT_STEP=%d' % torch.load('$RUN/smoke/last.pt', map_location='cpu',
                                  weights_only=False)['step'])
" | sed -n 's/^CKPT_STEP=//p' | tail -1)
    [ "$AT" = "40" ] || { echo "resume did not advance: last.pt at step '$AT', expected 40"; exit 1; }
    echo "smoke ok: trained, checkpointed, resumed and advanced to step $AT"
fi

# --max_hours is measured from train.py's own start, so take off what the
# container check and smoke already used, or a slow smoke would push the clean
# stop past the walltime and the job would die without its last checkpoint.
MAX_HOURS=$(awk -v m=$MAX_HOURS -v s=$SECONDS 'BEGIN { printf "%.2f", m - s / 3600 }')
echo "=== train: $STEPS steps, batch $BATCH, max ${MAX_HOURS}h (after ${SECONDS}s of checks) ==="
$RUNPY train $COMMON --out $RUN \
    --steps $STEPS --batch $BATCH --lr $LR \
    --lam_pocket $LAM_POCKET --lam_site $LAM_SITE \
    --max_hours $MAX_HOURS --surface_ram $SURFACE_RAM --label_ram $LABEL_RAM \
    --lr_schedule $LR_SCHEDULE --warmup $WARMUP ${TRAIN_ARGS:-}

echo
echo "checkpoints: $RUN/best.pt (best val MRR) and $RUN/last.pt (resume point)"
echo "history:     $RUN/history.json"
echo "resubmit this same script to continue from last.pt."
