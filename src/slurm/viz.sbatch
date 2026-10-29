#!/bin/bash
#SBATCH --partition=alpha
#SBATCH --qos=test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/viz_%j.log
# Channel visualisation and the PLIP diagnostic (src/channel_viz.py): per
# checkpoint a dump of every per-point channel on N val proteins, then the
# numpy report (collapse test for the site types, channel naming, pocket and
# PLIP AUROCs) + HTML viewers + PyMOL files. One job: the report is numpy and
# runs in the same container after the dumps.
#
# Default checkpoints: C after the resume (the plateaued model), v3d (C +
# chemo: its type head also reads the 12 chemotype channels) and v3f (C +
# site decoys). The first dump picks the proteins; the others reuse them
# (--rows_from) so every table compares the same surfaces. Each dump also
# carries the PUBLISHED dMaSIF AtomNet channels on the same points, the
# baseline for what fine-tuning changed.
#
# Overrides: CKPTS (space-separated), DATA (items.tsv / ligand_feats.npz;
#   out/runs/v1), SURFACES, POCKETS, PLIP, OUT (out/viz), N (64), TARGETS
#   (not GROUPS: bash's own GROUPS holds your numeric group ids)
#   (comma-separated UniProt ids to add, e.g. P56817,P0DTD1), N_HTML (12),
#   DUMP_ARGS / REPORT_ARGS (verbatim).
# Then open $OUT/report/<run>_html/*.html in a browser (scp them down), or
#   cd $OUT/report/<run>_pymol/<system_id> && pymol load.pml
set -e
module load apptainer
cd $WORK
CKPTS=${CKPTS:-"$WORK/out/runs/v3c_site_pairs_bn/best.pt $WORK/out/runs/v3d_pairs_bn_chemo/best.pt $WORK/out/runs/v3f_pairs_bn_decoys/best.pt"}
DATA=${DATA:-$WORK/out/runs/v1}
SURFACES=${SURFACES:-$WORK/data/surfaces}
POCKETS=${POCKETS:-$WORK/data/pockets/fpocket}
PLIP=${PLIP:-$WORK/data/plip}
OUT=${OUT:-$WORK/out/viz}
N=${N:-64}
N_HTML=${N_HTML:-12}
for f in $DATA/items.tsv $DATA/ligand_feats.npz; do
    [ -f "$f" ] || { echo "missing $f"; exit 1; }
done
mkdir -p $OUT
EXEC="apptainer exec --nv --cleanenv --env KEOPS_CACHE_FOLDER=$KEOPS_CACHE_FOLDER,OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK $WORK/dmasif_sandbox"
echo "node $(hostname): $(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader)"

FIRST=""
DUMPS=""
for CKPT in $CKPTS; do
    [ -f "$CKPT" ] || { echo "[skip] no checkpoint at $CKPT"; continue; }
    TAG=$(basename $(dirname $CKPT))
    echo "=== dump $TAG ==="
    $EXEC python $WORK/src/channel_viz.py dump --ckpt $CKPT --tag $TAG \
        --items $DATA/items.tsv --lig_feats $DATA/ligand_feats.npz \
        --surface_cache $SURFACES --pockets $POCKETS --pocket_select site \
        --n $N ${TARGETS:+--groups $TARGETS} ${FIRST:+--rows_from $FIRST} \
        --out $OUT/$TAG.npz ${DUMP_ARGS:-}
    FIRST=${FIRST:-$OUT/$TAG.npz}
    DUMPS="$DUMPS $OUT/$TAG.npz"
done
[ -n "$DUMPS" ] || { echo "no checkpoint found"; exit 1; }

echo "=== report ==="
$EXEC python $WORK/src/channel_viz.py report --dumps $DUMPS --plip $PLIP \
    --out $OUT/report --n_html $N_HTML ${REPORT_ARGS:-}
