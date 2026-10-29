#!/bin/bash
#SBATCH --partition=alpha
#SBATCH --qos=standard
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/funnel_%j.log
# The dataset-expansion numbers, off the login node. CPU venv, all pandas:
#   1. build_manifest.py --inspect: which per-pocket / per-ligand validation
#      columns PLINDER's annotation table has (needs $PLINDER_DIR)
#   2. structure_scan.py merge, if the scan parts exist and are not merged yet
#   3. dataset_funnel.py report --per_entry (+ --scan when merged)
#   4. companion_funnel.py funnel and distance
# Everything goes to this job's log; paste it back for RESULTS.md.
#
#   sbatch src/slurm/funnel.sbatch
#   sbatch --dependency=afterok:<scan array job id> src/slurm/funnel.sbatch
#
# Overrides: LADDER (dataset_funnel --ladder), SKIP_INSPECT=1, SKIP_COMPANION=1.
set -e
# Unbuffered: a batch log otherwise stays empty until Python exits.
export PYTHONUNBUFFERED=1
source $HOME/venvs/plinder/bin/activate
cd $WORK
MANIFEST=$WORK/data/meta/manifest.parquet
PARTS=$WORK/data/meta/scan
SCAN=$WORK/data/meta/structure_scan.parquet

if [ -z "$SKIP_INSPECT" ]; then
    echo "===== validation columns in the annotation table ====="
    [ -n "$PLINDER_DIR" ] || { echo "PLINDER_DIR is not set in the submitting shell"; exit 1; }
    python src/build_manifest.py --plinder $PLINDER_DIR --out data/meta/ --inspect_columns
fi

if [ ! -f "$SCAN" ] && ls $PARTS/part_*of*.parquet >/dev/null 2>&1; then
    echo "===== merging the structure scan ====="
    python src/structure_scan.py merge --parts $PARTS --manifest $MANIFEST --out $SCAN
fi

echo "===== dataset funnel ====="
python src/dataset_funnel.py report --manifest $MANIFEST --per_entry \
    ${LADDER:+--ladder "$LADDER"} $( [ -f "$SCAN" ] && echo --scan $SCAN )

if [ -z "$SKIP_COMPANION" ]; then
    echo "===== companion funnel ====="
    python src/companion_funnel.py funnel --manifest $MANIFEST
    python src/companion_funnel.py distance --manifest $MANIFEST --systems systems --n 400
fi
