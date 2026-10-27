#!/bin/bash
#SBATCH --partition=alpha
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --array=0-15
#SBATCH --output=logs/plip_%A_%a.log
# PLIP interaction labels for every row of items.tsv -> $PLIP/<system_id>.npz
# (src/plip_labels.py run). CPU only.
#
# Before: bash src/slurm/setup_plip.sh   (login node, once)
# After, CPU venv (numpy only), the sanity check on cached surfaces:
#   python src/plip_labels.py check --items $RUN/items.tsv --plip $PLIP \
#       --surface_cache $WORK/surfaces --n 500
#
# Per row: the receptor's protein atoms (as the surface sees them) + the row's
# ligand from its SDF, merged, PLIP run on it. ~1-2 s per row (0.8 s for a
# 2,900-atom receptor, plus the Biopython parse), so ~20.7k rows over 16 tasks
# is under an hour. Shards are disjoint (every n-th row by system_id), a row
# already done is skipped, files are written to a temporary name and renamed,
# so rerunning the array is cheap and safe. Failures (including the per-row
# --timeout) go to $PLIP/failed_<i>of<n>.tsv.
#
# Overrides: RUN, PLIP, PLIP_PKGS, LIMIT, TIMEOUT.
set -e
module load apptainer
cd $WORK
RUN=${RUN:-$WORK/runs/v1}
PLIP=${PLIP:-$WORK/plip}
DIR=${PLIP_PKGS:-$WORK/envs/plip_pkgs}
N=${SLURM_ARRAY_TASK_COUNT:-1}
I=${SLURM_ARRAY_TASK_ID:-0}
[ -f "$RUN/items.tsv" ] || { echo "missing $RUN/items.tsv (train.py prepare)"; exit 1; }
[ -d "$DIR/plip" ] || { echo "no PLIP in $DIR; bash src/slurm/setup_plip.sh"; exit 1; }
mkdir -p $PLIP
SCRATCH=${TMPDIR:-/tmp}/plip_${SLURM_JOB_ID:-0}_$I
mkdir -p $SCRATCH

apptainer exec --cleanenv --env OMP_NUM_THREADS=1,PYTHONPATH=$DIR $WORK/dmasif_sandbox \
    python $WORK/src/plip_labels.py run --items $RUN/items.tsv --out $PLIP \
    --workdir $SCRATCH --shard $I/$N --timeout ${TIMEOUT:-300} ${LIMIT:+--limit $LIMIT}

rm -rf $SCRATCH
echo "shard $I/$N done; $(ls $PLIP | grep -c '\.npz$') PLIP files in $PLIP"
