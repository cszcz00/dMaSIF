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
#   SYSTEM_ID (the protein to draw; it is also added to the dump, whatever its
#   split, and the files are named pocket_<run>_<SYSTEM_ID>), FIG_ARGS
#   (verbatim, e.g. "--px 1400").
# One named row only, e.g. a TRAIN row's per-point pocket logits (N=0: no val
# proteins); the .npz keys are p0_xyz, p0_pocket_logit, p0_contact, ...:
#   N=0 SYSTEM_ID=1ett__1__1.B__1.C sbatch src/slurm/pocket_figure.sbatch
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
NAME=pocket_$TAG${SYSTEM_ID:+_$SYSTEM_ID}
EXEC="apptainer exec --nv --cleanenv --env KEOPS_CACHE_FOLDER=$KEOPS_CACHE_FOLDER,OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK $WORK/dmasif_sandbox"
echo "node $(hostname): $(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader)"

$EXEC python $WORK/src/pocket_figure.py dump --ckpt $CKPT \
    --items $DATA/items.tsv --lig_feats $DATA/ligand_feats.npz \
    --surface_cache $SURFACES --n $N ${TARGETS:+--groups $TARGETS} \
    ${SYSTEM_ID:+--system_ids $SYSTEM_ID} --out $OUT/$NAME.npz
$EXEC python $WORK/src/pocket_figure.py figure --dump $OUT/$NAME.npz \
    --out $OUT/$NAME.svg ${SYSTEM_ID:+--system_id $SYSTEM_ID} ${FIG_ARGS:-}
echo "dump: $OUT/$NAME.npz"
echo "figure: $OUT/$NAME.svg (scp it down; opens in any browser)"
