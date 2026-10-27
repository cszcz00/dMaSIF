#!/bin/bash
#SBATCH --partition=alpha
#SBATCH --qos=test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=logs/probe_retrieval_%j.log
# Which ligand encoder pairs with our surface anchors? Two stages:
#   1. dump   frozen anchors from a trained pocket head (GPU, one pass)
#   2. probe  train each ligand arm on those anchors (CPU is enough)
#
# Needs, from the CPU venv, BEFORE submitting:
#   python src/probe_retrieval.py prepare --index datasets/dataset_v1/index.parquet \
#       --exclude encoder_test/items.tsv --n 1500 --out probe/
#   python src/ligand_features.py --index datasets/dataset_v1/index.parquet --out probe/
# and a checkpoint from a run with --save_ckpt:
#   $WORK/encoder_test/head_pretrained_300.pt
#
# Overrides: CKPT, LIMIT (items to dump; unset = all), STEPS, ARMS,
#   REGION_ARGS (selector, default the region sweep's leader: balls R 8 shift 2),
#   OTHER_LIGANDS (a labels.py directory, so a receptor's other ligand sites are
#   ignored rather than trained as negatives),
#   PROBE_ONLY=1 reuses an existing anchors.pt and reruns stage 2 only.
set -e
module load apptainer
cd $WORK
OUT=$WORK/probe
CKPT=${CKPT:-$WORK/encoder_test/head_pretrained_300.pt}
STEPS=${STEPS:-2000}
ARMS=${ARMS:-fp,fp_linear,desc,fp_desc,gnn}
REGION_ARGS=${REGION_ARGS:-"--region_method ball --ball_radius 8 --ball_shift 2"}
ANCHORS=$OUT/anchors.pt
EXEC="apptainer exec --nv --cleanenv --env KEOPS_CACHE_FOLDER=$KEOPS_CACHE_FOLDER,OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK $WORK/dmasif_sandbox"

echo "node $(hostname): $(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader)"
# Fail fast if this container's PyTorch was not built for this GPU; KeOps would
# otherwise die with CUDA_ERROR_INVALID_IMAGE (as in encoder_test.sbatch).
apptainer exec --nv --cleanenv $WORK/dmasif_sandbox python -c "
import sys, torch
cap = 'sm_%d%d' % torch.cuda.get_device_capability()
archs = torch.cuda.get_arch_list()
print('GPU', cap, '| torch built for', archs)
sys.exit(0 if cap in archs else 'this container cannot run on ' + cap)
"

for f in $OUT/items.tsv $OUT/ligand_feats.npz; do
    [ -f "$f" ] || { echo "missing $f; run the two CPU steps in the header first"; exit 1; }
done
echo "items: $(($(wc -l < $OUT/items.tsv) - 1)); arms: $ARMS"

if [ "${PROBE_ONLY:-0}" != "1" ]; then
    [ -f "$CKPT" ] || { echo "missing checkpoint $CKPT (produce it with test_encoder.py --save_ckpt)"; exit 1; }
    # A 10-item pass first, in the SAME job. The full dump is ~40 min for 1500
    # items, and `dump` has never run on a GPU, so a shape error would otherwise
    # cost a whole queue wait to discover. set -e aborts before the full pass.
    if [ -z "${LIMIT:-}" ] && [ "${SKIP_SMOKE:-0}" != "1" ]; then
        echo "=== 1a. dump smoke: first ${SMOKE_LIMIT:-10} items ==="
        $EXEC python $WORK/src/probe_retrieval.py dump \
            --items $OUT/items.tsv --ckpt $CKPT --out $OUT/anchors_smoke.pt \
            --limit ${SMOKE_LIMIT:-10} \
            ${OTHER_LIGANDS:+--other_ligands $OTHER_LIGANDS} $REGION_ARGS
        echo "smoke dump ok"
    fi
    echo "=== 1. dump frozen anchors from $CKPT ==="
    $EXEC python $WORK/src/probe_retrieval.py dump \
        --items $OUT/items.tsv --ckpt $CKPT --out $ANCHORS \
        ${LIMIT:+--limit $LIMIT} ${OTHER_LIGANDS:+--other_ligands $OTHER_LIGANDS} \
        $REGION_ARGS
fi

echo "=== 2. probe: train each ligand arm on the frozen anchors ==="
$EXEC python $WORK/src/probe_retrieval.py probe \
    --anchors $ANCHORS --ligands $OUT/ligand_feats.npz \
    --arms $ARMS --steps $STEPS --oracle --device cuda:0 \
    --out $OUT/arms.tsv --save_dir $OUT/models
