#!/bin/bash
#SBATCH --partition=alpha
#SBATCH --qos=test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/eval_%j.log
# Validation metrics from saved checkpoints (`train.py eval`); nothing trains.
#
# Written to put out/runs/v1 on the SAME validation set as the second round of
# runs. v1 validated on a row-uniform 400 that covered 112 of its ~283 val
# groups; the new runs use VAL_BALANCE=1, a round-robin over groups. --recarve
# rebuilds v1's full group-disjoint val pool from its own val_frac /
# split_seed, and --val_max 400 --val_balance then draws exactly the rows a
# new run with VAL_BALANCE=1 validates on (same table, same split_seed). So
# the numbers here are v1's row in the comparison table, not the 0.098 its
# own log printed.
#
# Overrides: CKPTS (space-separated; default v1's best.pt and last.pt), DATA
#   (items.tsv / ligand_feats.npz; out/runs/v1), SURFACES, POCKETS
#   (data/pockets/fpocket), EVAL_ARGS (verbatim; default "--recarve --val_max
#   400 --val_balance").
# --pockets is always passed: checkpoints from before the data/ refactor
# stored $WORK/pockets/fpocket, which no longer exists. A checkpoint with
# predicted regions ignores it (train.pocket_source warns).
# `train.py eval` had never run before this job; if it fails, that is why.
set -e
module load apptainer
cd $WORK
DATA=${DATA:-$WORK/out/runs/v1}
SURFACES=${SURFACES:-$WORK/data/surfaces}
POCKETS=${POCKETS:-$WORK/data/pockets/fpocket}
CKPTS=${CKPTS:-"$WORK/out/runs/v1/best.pt $WORK/out/runs/v1/last.pt"}
EVAL_ARGS=${EVAL_ARGS:-"--recarve --val_max 400 --val_balance"}
EXEC="apptainer exec --nv --cleanenv --env KEOPS_CACHE_FOLDER=$KEOPS_CACHE_FOLDER,OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK $WORK/dmasif_sandbox"

echo "node $(hostname): $(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader)"
for CKPT in $CKPTS; do
    echo "=== $CKPT ==="
    $EXEC python $WORK/src/train.py eval --ckpt $CKPT \
        --items $DATA/items.tsv --lig_feats $DATA/ligand_feats.npz \
        --surface_cache $SURFACES --pockets $POCKETS $EVAL_ARGS
done
