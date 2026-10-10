#!/bin/bash
#SBATCH --partition=alpha
#SBATCH --qos=test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=encoder_test_%j.log
# Needs $WORK/encoder_test/items.tsv from (CPU venv):
#   python src/test_encoder.py prepare --index datasets/dataset_v1/index.parquet --n 48 --out encoder_test/
set -e
module load apptainer
cd $WORK
RUN="apptainer exec --nv --cleanenv --env KEOPS_CACHE_FOLDER=$KEOPS_CACHE_FOLDER $WORK/dmasif_sandbox python $WORK/src/test_encoder.py"
ITEMS=$WORK/encoder_test/items.tsv

echo "node $(hostname): $(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader)"
# Fail fast if this container's PyTorch was not built for this GPU (e.g. Blackwell
# sm_120 vs the CUDA 12.6 image); KeOps would otherwise die with CUDA_ERROR_INVALID_IMAGE.
apptainer exec --nv --cleanenv $WORK/dmasif_sandbox python -c "
import sys, torch
cap = 'sm_%d%d' % torch.cuda.get_device_capability()
archs = torch.cuda.get_arch_list()
print('GPU', cap, '| torch built for', archs)
sys.exit(0 if cap in archs else 'this container cannot run on ' + cap + '; pin the job to a supported GPU')
"

echo "=== smoke ==="
$RUN smoke --items $ITEMS --n 8

echo "=== overfit, fresh weights, regions from each protein's top 5% ==="
$RUN overfit --items $ITEMS --n 32 --holdout 16 --steps 300

echo "=== overfit, pretrained start, regions from each protein's top 5% ==="
$RUN overfit --items $ITEMS --n 32 --holdout 16 --steps 300 --pretrained

echo "=== control: overfit, pretrained start, OLD fixed threshold p >= 0.5 ==="
$RUN overfit --items $ITEMS --n 32 --holdout 16 --steps 300 --pretrained --top_fraction 0
