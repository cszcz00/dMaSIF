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
#   ANCHOR_NORM, MAX_HOURS, LIMIT, SURFACES, CHEMO=1, SKIP_SMOKE=1,
#   TRAIN_ARGS (anything else passed through verbatim).
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

for f in $RUN/items.tsv $RUN/ligand_feats.npz; do
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
N_ITEMS=$(($(wc -l < $RUN/items.tsv) - 1))
echo "run $RUN; $N_ITEMS interactions; arm $ARM; pool $POOL_MODE; anchor_norm $ANCHOR_NORM"
[ -f "$RUN/last.pt" ] && echo "found $RUN/last.pt: this job will RESUME it"

COMMON="--items $RUN/items.tsv --lig_feats $RUN/ligand_feats.npz \
    --surface_cache $SURFACES --arm $ARM --pool_mode $POOL_MODE \
    --anchor_norm $ANCHOR_NORM --region_method ball --ball_radius 8 --ball_shift 2 \
    $HEAD_ARG ${CHEMO:+--chemo} ${LIMIT:+--limit $LIMIT}"

# A 30-step pass on 64 interactions first, in the SAME job, written to a
# throwaway run directory. The loop has never run on a GPU, so a shape error
# would otherwise cost a whole queue wait to find - and a surface sampled here
# lands in the shared cache, so the time is not even wasted.
if [ "${SKIP_SMOKE:-0}" != "1" ]; then
    echo "=== smoke: 64 interactions, 30 steps ==="
    $RUNPY train $COMMON --out $RUN/smoke --limit 64 --steps 30 \
        --eval_every 15 --save_every 30 --val_frac 0.3 --val_max 24 \
        --batch 4 --no_auto_resume
    echo "smoke ok"
fi

echo "=== train: $STEPS steps, batch $BATCH, max ${MAX_HOURS}h ==="
$RUNPY train $COMMON --out $RUN \
    --steps $STEPS --batch $BATCH --lr $LR \
    --lam_pocket $LAM_POCKET --lam_site $LAM_SITE \
    --max_hours $MAX_HOURS ${TRAIN_ARGS:-}

echo
echo "checkpoints: $RUN/best.pt (best val MRR) and $RUN/last.pt (resume point)"
echo "history:     $RUN/history.json"
echo "resubmit this same script to continue from last.pt."
