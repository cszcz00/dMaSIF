#!/bin/bash
#SBATCH --partition=alpha
#SBATCH --qos=test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:15:00
#SBATCH --output=probe_%j.log
set -e
module load apptainer
RUN="apptainer exec --nv --cleanenv --env KEOPS_CACHE_FOLDER=$KEOPS_CACHE_FOLDER $WORK/dmasif_sandbox"
$RUN python $WORK/src/extract.py --merge_models \
    --inputs $WORK/test_pdbs/assembly --out $WORK/feats/test --repo $WORK/dMaSIF
$RUN python $WORK/src/probe_pocket.py \
    --ligand $WORK/test_pdbs/1STP.pdb --ligand_resname BTN \
    --feats $WORK/feats/test/1STP.npz $WORK/feats/test/1STP_assembly.npz \
    --labels ASU assembly
