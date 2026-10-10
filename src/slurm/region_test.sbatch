#!/bin/bash
#SBATCH --partition=alpha
#SBATCH --qos=test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=region_test_%j.log
# Region-selector comparison (anchors.py options 1-3) on ONE trained pocket head:
#   1. train the head (pretrained start, pocket BCE, as in encoder_test.sbatch)
#      and dump its probability maps for train + held-out items
#   2. score every selector setting on those maps (src/region_sweep.py, CPU)
#
# Needs $WORK/encoder_test/items.tsv with N_TRAIN + N_HOLD rows. More held-out
# items than the first test's 16 (CPU venv; same seed, so the first 48 rows are
# the items used so far):
#   python src/test_encoder.py prepare --index datasets/dataset_v1/index.parquet --n 96 --out encoder_test/
#
# Overrides: N_TRAIN (32), N_HOLD (64), STEPS (300).
# SWEEP_ONLY=1 reuses an existing maps.pt and only reruns step 2.
set -e
module load apptainer
cd $WORK
N_TRAIN=${N_TRAIN:-32}
N_HOLD=${N_HOLD:-64}
STEPS=${STEPS:-300}
OUT=$WORK/encoder_test
MAPS=$OUT/maps_pretrained_${STEPS}.pt
EXEC="apptainer exec --nv --cleanenv --env KEOPS_CACHE_FOLDER=$KEOPS_CACHE_FOLDER,OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK $WORK/dmasif_sandbox"

echo "node $(hostname): $(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader)"
echo "items: $(($(wc -l < $OUT/items.tsv) - 1)) in items.tsv; using $N_TRAIN train + $N_HOLD held-out"

if [ "${SWEEP_ONLY:-0}" != "1" ]; then
    echo "=== train pocket head (pretrained start, $STEPS steps), dump maps ==="
    $EXEC python $WORK/src/test_encoder.py overfit --items $OUT/items.tsv \
        --n $N_TRAIN --holdout $N_HOLD --steps $STEPS --eval_every 100 --pretrained \
        --dump $MAPS
fi

echo "=== region sweep: watershed+persistence / components / NMS balls ==="
$EXEC python $WORK/src/region_sweep.py --maps $MAPS \
    --out $OUT/region_sweep_${STEPS}.tsv --per_item $OUT/region_items_${STEPS}.tsv
