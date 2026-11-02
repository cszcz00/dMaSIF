#!/bin/bash
#SBATCH --partition=alpha
#SBATCH --qos=test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/pocket_figure_%j.log
# The pocket-selection figure (src/pocket_figure.py): pocket logit -> top 5%
# -> ball anchors -> 128-d anchor vectors vs the ligand's, on real val proteins.
#
#   1. dump    GPU, a few minutes: N val proteins of the checkpoint's own
#              validation rows (group-balanced), everything the figure draws
#   2. figure  numpy, in the same container: one protein -> an SVG. Prints a
#              table of the dumped proteins first; the default protein is the
#              clean success with the largest margin
#
# Safe while the run trains: it only READS the checkpoint (best.pt is written
# by a temp-file rename) and the shared caches.
#
# Overrides: CKPT (default: v8's best.pt), DATA (dataset_v2), SURFACES, N (12),
#   TARGETS (comma-separated UniProt ids to add), OUT (out/figures),
#   SYSTEM_ID (pick the protein), FIG_ARGS (verbatim, e.g. "--px 1400").
# Another protein from the same dump, no GPU (CPU venv is fine):
#   python src/pocket_figure.py figure --dump out/figures/pocket_<run>.npz \
#       --out out/figures/pocket_<run>_<id>.svg --system_id <system_id>
set -e
module load apptainer
cd $WORK
CKPT=${CKPT:-$WORK/out/runs/v8_logit_plip_af_pairs_lsim/best.pt}
DATA=${DATA:-$WORK/out/runs/dataset_v2}
SURFACES=${SURFACES:-$WORK/data/surfaces}
OUT=${OUT:-$WORK/out/figures}
N=${N:-12}
[ -f "$CKPT" ] || { echo "no checkpoint at $CKPT"; exit 1; }
mkdir -p $OUT
TAG=$(basename $(dirname $CKPT))
EXEC="apptainer exec --nv --cleanenv --env KEOPS_CACHE_FOLDER=$KEOPS_CACHE_FOLDER,OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK $WORK/dmasif_sandbox"
echo "node $(hostname): $(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader)"

$EXEC python $WORK/src/pocket_figure.py dump --ckpt $CKPT \
    --items $DATA/items.tsv --lig_feats $DATA/ligand_feats.npz \
    --surface_cache $SURFACES --n $N ${TARGETS:+--groups $TARGETS} \
    --out $OUT/pocket_$TAG.npz
$EXEC python $WORK/src/pocket_figure.py figure --dump $OUT/pocket_$TAG.npz \
    --out $OUT/pocket_$TAG.svg ${SYSTEM_ID:+--system_id $SYSTEM_ID} ${FIG_ARGS:-}
echo "figure: $OUT/pocket_$TAG.svg (scp it down; opens in any browser)"
