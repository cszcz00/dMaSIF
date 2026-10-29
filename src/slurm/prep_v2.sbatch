#!/bin/bash
#SBATCH --partition=alpha
#SBATCH --qos=standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/prep_v2_%j.log
# dataset_v2's CPU half, in order (CPU venv; pandas / rdkit / numpy):
#   1. build_dataset.py: data/datasets/<name>/ from the spec (dry run first)
#   2. train.py prepare: out/runs/<name>/items.tsv, train split, the same
#      --exclude as dataset_v1 (the head checkpoint's own receptors)
#   3. ligand_features.py: out/runs/<name>/ligand_feats.npz (fp_row is
#      re-indexed per dataset, so this is always rebuilt)
#   4. valset.py make: the fixed validation set shared with dataset_v1 runs
#      -> out/valset_v1v2/
# Normally submitted by src/slurm/build_v2.sh, which chains surfaces, fpocket
# and PLIP after it.
#
# Overrides: SPEC (configs/datasets/dataset_v2.yaml), RUN (out/runs/<name>),
#   V1 (out/runs/v1), VALSET (out/valset_v1v2), OVERWRITE=1 to rebuild an
#   existing data/datasets/<name>/.
set -e
export PYTHONUNBUFFERED=1
source $HOME/venvs/plinder/bin/activate
cd $WORK
SPEC=${SPEC:-configs/datasets/dataset_v2.yaml}
NAME=$(basename $SPEC .yaml)
RUN=${RUN:-$WORK/out/runs/$NAME}
V1=${V1:-$WORK/out/runs/v1}
VALSET=${VALSET:-$WORK/out/valset_v1v2}
INDEX=$WORK/data/datasets/$NAME/index.parquet
EXCLUDE=$WORK/out/encoder_test/items.tsv
[ -f "$V1/items.tsv" ] && [ -f "$V1/ligand_feats.npz" ] || { echo "missing $V1/items.tsv or ligand_feats.npz"; exit 1; }
[ -f data/meta/structure_scan.parquet ] || { echo "missing data/meta/structure_scan.parquet (structure_scan.py merge)"; exit 1; }

echo "===== 1. build_dataset $SPEC ====="
python src/build_dataset.py $SPEC --root $WORK --dry_run
if [ -f "$INDEX" ] && [ -z "$OVERWRITE" ]; then
    echo "$INDEX exists; keeping it (OVERWRITE=1 to rebuild)"
else
    python src/build_dataset.py $SPEC --root $WORK --check_files ${OVERWRITE:+--overwrite}
fi

echo "===== 2. train.py prepare -> $RUN/items.tsv ====="
python src/train.py prepare --index $INDEX --out $RUN --split train \
    $( [ -f "$EXCLUDE" ] && echo --exclude $EXCLUDE )

echo "===== 3. ligand features -> $RUN/ligand_feats.npz ====="
python src/ligand_features.py --index $INDEX --out $RUN --overwrite

echo "===== 4. fixed validation set -> $VALSET ====="
python src/valset.py make --items_a $V1/items.tsv --lig_feats_a $V1/ligand_feats.npz \
    --items_b $RUN/items.tsv --lig_feats_b $RUN/ligand_feats.npz --out $VALSET

echo "done: $RUN/items.tsv, $RUN/ligand_feats.npz, $VALSET/"
