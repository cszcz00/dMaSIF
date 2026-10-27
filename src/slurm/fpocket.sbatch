#!/bin/bash
#SBATCH --partition=alpha
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --array=0-15
#SBATCH --output=logs/fpocket_%A_%a.log
# fpocket on every receptor surface of items.tsv -> $POCKETS/<input_id>.npz
# (src/external_pockets.py run). CPU only: no GPU, no KeOps. The container is
# only for Biopython, which reads the receptor exactly as the surface does.
#
# Before: bash src/slurm/build_fpocket.sh   (login node, once)
# After:  the score step below, then src/slurm/train_fpocket.sbatch.
#
# Shards are disjoint (every n-th receptor of a fixed sorted order, the same
# order as `train.py cache`), a receptor already done is skipped, and files are
# written to a temporary name and renamed, so rerunning the array is cheap and
# safe. fpocket takes ~1-3 s on a typical receptor; ~19.3k receptors over 16
# tasks is well under an hour. Failures go to $POCKETS/failed_<i>of<n>.tsv.
#
# Then, CPU venv (numpy only), how good the pockets are as regions:
#   python src/external_pockets.py score --items $RUN/items.tsv \
#       --surface_cache $WORK/data/surfaces --pockets $POCKETS --n 1000 \
#       --out $POCKETS/score.tsv --per_item $POCKETS/score_items.tsv
#
# Overrides: RUN, POCKETS, FPOCKET (binary), FPOCKET_ARGS (extra flags), LIMIT.
set -e
module load apptainer
cd $WORK
RUN=${RUN:-$WORK/out/runs/v1}
POCKETS=${POCKETS:-$WORK/data/pockets/fpocket}
FPOCKET=${FPOCKET:-$WORK/tools/fpocket-src/bin/fpocket}
N=${SLURM_ARRAY_TASK_COUNT:-1}
I=${SLURM_ARRAY_TASK_ID:-0}
[ -f "$RUN/items.tsv" ] || { echo "missing $RUN/items.tsv (train.py prepare)"; exit 1; }
[ -x "$FPOCKET" ] || { echo "no fpocket at $FPOCKET; bash src/slurm/build_fpocket.sh"; exit 1; }
mkdir -p $POCKETS
SCRATCH=${TMPDIR:-/tmp}/fpocket_${SLURM_JOB_ID:-0}_$I
mkdir -p $SCRATCH

apptainer exec --cleanenv --env OMP_NUM_THREADS=1 $WORK/dmasif_sandbox \
    python $WORK/src/external_pockets.py run --items $RUN/items.tsv \
    --out $POCKETS --fpocket $FPOCKET --workdir $SCRATCH --shard $I/$N \
    ${FPOCKET_ARGS:+--fpocket_args "$FPOCKET_ARGS"} ${LIMIT:+--limit $LIMIT}

rm -rf $SCRATCH
echo "shard $I/$N done; $(ls $POCKETS | grep -c '\.npz$') pocket files in $POCKETS"
