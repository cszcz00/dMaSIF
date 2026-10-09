#!/bin/bash
#SBATCH --partition=alpha
#SBATCH --qos=test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:15:00
#SBATCH --output=extract_test_%j.log
set -e
module load apptainer
apptainer exec --nv --cleanenv --env KEOPS_CACHE_FOLDER=$KEOPS_CACHE_FOLDER $WORK/dmasif_sandbox \
    python $WORK/dMaSIF/affinity/extract.py --inputs $WORK/test_pdbs --out $WORK/feats/test_moved --repo $WORK/dMaSIF
