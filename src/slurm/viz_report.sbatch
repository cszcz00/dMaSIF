#!/bin/bash
#SBATCH --partition=alpha
#SBATCH --qos=standard
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/viz_report_%j.log
# The numpy half of src/channel_viz.py alone, on dumps viz.sbatch already
# wrote: report + viewers again without the GPU. Use it to change the viewer
# (mesh vs points, smoothing, crop, how many proteins) or the PLIP radius.
# CPU venv; numpy only.
#
# Overrides: DUMPS (default: every out/viz/*.npz), PLIP, OUT (out/viz/report),
#   N_HTML (12), REPORT_ARGS (verbatim, e.g. "--smooth 1.5 --grid 0.5" or
#   "--viewer points" or "--crop 0" for the whole surface).
set -e
source $HOME/venvs/plinder/bin/activate
cd $WORK
DUMPS=${DUMPS:-$(ls $WORK/out/viz/*.npz)}
PLIP=${PLIP:-$WORK/data/plip}
OUT=${OUT:-$WORK/out/viz/report}
N_HTML=${N_HTML:-12}
python src/channel_viz.py report --dumps $DUMPS --plip $PLIP --out $OUT \
    --n_html $N_HTML ${REPORT_ARGS:-}
