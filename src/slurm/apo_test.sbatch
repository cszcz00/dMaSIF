#!/bin/bash
#SBATCH --partition=alpha
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/apo_test_%j.log
# Apo vs holo test of the protein encoder (src/apo_eval.py):
#   1. train the pocket head once (pretrained start, as in region_test.sbatch) and
#      save it - skipped when the checkpoint already exists
#   2. evaluate holo / resampled holo / apo / AlphaFold versions of each pair
#
# Needs, from the CPU venv first:
#   python src/apo_pairs.py --links $PLINDER_DIR/links --index datasets/dataset_v1/index.parquet \
#       --exclude encoder_test/items.tsv --out apo_test/
# and $PLINDER_DIR / $KEOPS_CACHE_FOLDER set in the submitting shell.
#
# Overrides: N_TRAIN (32), STEPS (300), LIMIT (0 = every pair; e.g. LIMIT=10 for a
# quick check). Runs ~1-3 h for ~500 pairs; drop --time / add a --qos as your
# cluster requires.
set -e
module load apptainer
cd $WORK
N_TRAIN=${N_TRAIN:-32}
STEPS=${STEPS:-300}
LIMIT=${LIMIT:-0}
CKPT=$WORK/encoder_test/head_pretrained_${STEPS}.pt
EXEC="apptainer exec --nv --cleanenv --env KEOPS_CACHE_FOLDER=$KEOPS_CACHE_FOLDER,OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK $WORK/dmasif_sandbox"

echo "node $(hostname): $(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader)"
echo "pairs: $(($(wc -l < $WORK/apo_test/pairs.tsv) - 1)) in apo_test/pairs.tsv; links from $PLINDER_DIR"

if [ ! -f "$CKPT" ]; then
    echo "=== train pocket head (pretrained start, $STEPS steps) -> $CKPT ==="
    $EXEC python $WORK/src/test_encoder.py overfit --items $WORK/encoder_test/items.tsv \
        --n $N_TRAIN --holdout 0 --steps $STEPS --eval_every 100 --pretrained \
        --save_ckpt $CKPT
fi

echo "=== apo vs holo evaluation ==="
$EXEC python $WORK/src/apo_eval.py --ckpt $CKPT --pairs $WORK/apo_test/pairs.tsv \
    --links_root $PLINDER_DIR/linked_structures --out $WORK/apo_test --limit $LIMIT
