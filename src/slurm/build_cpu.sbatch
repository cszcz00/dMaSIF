#!/bin/bash
#SBATCH --partition=alpha
#SBATCH --qos=test
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=build_%j.log
set -e
module load apptainer
cd $WORK
apptainer build --force --fakeroot --ignore-fakeroot-command $WORK/dmasif.sif $WORK/src/slurm/dmasif.def
rm -rf dmasif_sandbox
apptainer build --sandbox dmasif_sandbox dmasif.sif
echo BUILD_DONE
