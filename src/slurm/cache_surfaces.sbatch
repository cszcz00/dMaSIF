#!/bin/bash
#SBATCH --partition=alpha
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-15
#SBATCH --output=logs/cache_surfaces_%A_%a.log
# Prefill the surface cache src/train.py reads, as an array job.
#
# Optional: training fills the cache lazily, so this only moves work earlier.
# Worth it because the first epoch is otherwise dominated by surface sampling
# (a Biopython parse plus a dMaSIF sampling pass, ~1 s per receptor), and 16
# tasks turn ~6 h of that into ~25 min.
#
# Shards are disjoint by construction: `cache --shard i/n` takes every n-th
# receptor of a fixed sorted order, so two tasks never sample the same surface
# and a rerun of one task covers exactly the receptors it covered before.
# A surface already on disk is skipped, so rerunning the array is cheap and
# safe - the write is to a temporary file and renamed, so a task killed at the
# walltime leaves no half-written .npz for training to pick up.
#
# ATOM_FEATS=1 also gives every surface its atom_features.py per-atom
# chemistry (the learned-typing runs need it): a surface already on disk is
# kept and only its atoms re-parsed (~1-2 s a receptor, CPU), the file
# rewritten through the same temporary-file rename, so running training jobs
# that read the cache are unaffected.
#
# Needs $RUN/items.tsv (src/train.py prepare).
# Overrides: RUN, SURFACES, LIMIT, ATOM_FEATS.
set -e
module load apptainer
cd $WORK
RUN=${RUN:-$WORK/out/runs/v1}
SURFACES=${SURFACES:-$WORK/data/surfaces}
N=${SLURM_ARRAY_TASK_COUNT:-1}
I=${SLURM_ARRAY_TASK_ID:-0}
EXEC="apptainer exec --nv --cleanenv --env KEOPS_CACHE_FOLDER=$KEOPS_CACHE_FOLDER,OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK $WORK/dmasif_sandbox"

echo "node $(hostname) shard $I/$N: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
[ -f "$RUN/items.tsv" ] || { echo "missing $RUN/items.tsv (train.py prepare)"; exit 1; }

$EXEC python $WORK/src/train.py cache \
    --items $RUN/items.tsv --surface_cache $SURFACES \
    --shard $I/$N ${LIMIT:+--limit $LIMIT} \
    $( [ -n "${ATOM_FEATS:-}" ] && [ "$ATOM_FEATS" != 0 ] && echo --atom_feats )

echo "shard $I/$N done; $(ls $SURFACES | wc -l) surfaces now cached in total"
