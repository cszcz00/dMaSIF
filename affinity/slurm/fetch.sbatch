#!/bin/bash
#SBATCH --account=cu_vvenkatasubrama_exploring
#SBATCH --partition=alpha
#SBATCH --qos=standard
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=fetch_%j.log
set -e
source $HOME/venvs/plinder/bin/activate
cd $WORK
python dMaSIF/affinity/fetch_structures.py --receptors meta/receptors.txt --out structures/ \
    --manifest meta/manifest.parquet --systems_out systems/ --workers 8