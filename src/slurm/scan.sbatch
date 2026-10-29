#!/bin/bash
#SBATCH --partition=alpha
#SBATCH --qos=standard
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --array=0-63
#SBATCH --output=logs/scan_%A_%a.log
# Structure scan over the WHOLE manifest -> $OUT/part_<i>of<n>.parquet
# (src/structure_scan.py run): per row, ligand burial, pocket completeness /
# mobility / altlocs, and companion size x distance for strip-and-gate. CPU
# only, CPU venv (gemmi; `pip install gemmi` into ~/venvs/plinder once).
#
# ~224k receptor files + ~307k system files, each parsed once. Shards are
# disjoint (every n-th receptor of a fixed sorted order) and a shard whose
# part file exists is skipped, so resubmitting the array only redoes failed
# or timed-out tasks. Time per receptor is printed every 500; at 64 tasks the
# shard is ~3.5k receptors.
# Measured on the 50-receptor test: ~0.12 s per receptor, so a shard is
# ~7-10 min and 1 h leaves a wide margin. QOS: `sacctmgr show assoc
# user=$USER format=account,qos%40` lists the ones you may use; override
# at submit with `sbatch --qos=<name> ...`.
#
# Quick test first (one shard, 50 receptors, written to a scratch dir):
#   OUT=$WORK/tmp/scan_test LIMIT=50 sbatch --array=0 src/slurm/scan.sbatch
#
# After every shard is done:
#   sbatch src/slurm/funnel.sbatch     (merges the parts, then the funnel with --scan)
#
# Overrides: OUT, MANIFEST, LIMIT.
set -e
source $HOME/venvs/plinder/bin/activate
cd $WORK
OUT=${OUT:-$WORK/data/meta/scan}
MANIFEST=${MANIFEST:-$WORK/data/meta/manifest.parquet}
N=${SLURM_ARRAY_TASK_COUNT:-1}
I=${SLURM_ARRAY_TASK_ID:-0}
python -c "import gemmi" 2>/dev/null || { echo "gemmi missing: source ~/venvs/plinder/bin/activate && pip install gemmi"; exit 1; }
mkdir -p $OUT

python src/structure_scan.py run --manifest $MANIFEST --structures structures \
    --systems systems --out $OUT --shard $I/$N ${LIMIT:+--limit $LIMIT}
