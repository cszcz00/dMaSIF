#!/bin/bash
#SBATCH --partition=alpha
#SBATCH --qos=test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/report_%j.log
# The retrieval inspection's GPU half (src/retrieval_report.py dump): a
# checkpoint -> anchors for a train and a val slice + every ligand's vector,
# one .npz per --pocket_select. Nothing is scored here, so one dump serves
# every question the CPU `report` stage can ask.
#
# Why both selects by default: `site` is what training optimised (the pocket
# is handed to the model), `topk` is what a screen sees (the receptor's best
# fpocket pockets, no ligand). The two reports side by side say how much of
# the retrieval survives losing the site.
#
# Overrides: CKPT (out/runs/v3c_site_pairs_bn/best.pt), DATA (items.tsv /
#   ligand_feats.npz; out/runs/v1), SURFACES, OUT (out/reports/), TAG (names the
#   files; defaults to the checkpoint's run directory), SELECTS ("site topk"),
#   N_TRAIN / N_VAL (400 each, group-balanced), DUMP_ARGS (verbatim).
set -e
module load apptainer
cd $WORK
CKPT=${CKPT:-$WORK/out/runs/v3c_site_pairs_bn/best.pt}
DATA=${DATA:-$WORK/out/runs/v1}
SURFACES=${SURFACES:-$WORK/data/surfaces}
# Always passed: pre-refactor checkpoints stored $WORK/pockets/fpocket.
POCKETS=${POCKETS:-$WORK/data/pockets/fpocket}
OUT=${OUT:-$WORK/out/reports}
SELECTS=${SELECTS:-"site topk"}
N_TRAIN=${N_TRAIN:-400}
N_VAL=${N_VAL:-400}
TAG=${TAG:-$(basename $(dirname $CKPT))}
[ -f "$CKPT" ] || { echo "no checkpoint at $CKPT"; exit 1; }
for f in $DATA/items.tsv $DATA/ligand_feats.npz; do
    [ -f "$f" ] || { echo "missing $f"; exit 1; }
done
mkdir -p $OUT
EXEC="apptainer exec --nv --cleanenv --env KEOPS_CACHE_FOLDER=$KEOPS_CACHE_FOLDER,OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK $WORK/dmasif_sandbox"

echo "node $(hostname): $(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader)"
echo "checkpoint $CKPT -> $OUT/${TAG}_<select>.npz"
for SEL in $SELECTS; do
    echo "=== dump --pocket_select $SEL ==="
    $EXEC python $WORK/src/retrieval_report.py dump --ckpt $CKPT \
        --items $DATA/items.tsv --lig_feats $DATA/ligand_feats.npz \
        --surface_cache $SURFACES --pockets $POCKETS --pocket_select $SEL --pocket_rank drug \
        --n_train $N_TRAIN --n_val $N_VAL \
        --out $OUT/${TAG}_${SEL}.npz ${DUMP_ARGS:-}
done

echo
echo "now the CPU half, in the CPU venv (numpy only, no torch, no GPU):"
for SEL in $SELECTS; do
    echo "  python src/retrieval_report.py report --dump $OUT/${TAG}_${SEL}.npz \\"
    echo "      --items $DATA/items.tsv --lig_feats $DATA/ligand_feats.npz \\"
    echo "      --out $OUT/${TAG}_${SEL}"
done
