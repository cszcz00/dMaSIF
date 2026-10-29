#!/bin/bash
# Build dataset_v2 end to end: one CPU prep job, then the three per-receptor
# arrays after it (Slurm dependencies; nothing runs on the login node):
#
#   prep_v2.sbatch        build_dataset -> items.tsv -> ligand features -> fixed val set
#   cache_surfaces.sbatch surfaces for the new receptors          (GPU array)
#   fpocket.sbatch        fpocket pockets for the new receptors    (CPU array)
#   plip.sbatch           PLIP labels for the new rows             (CPU array)
#
# All three arrays skip what is already on disk (dataset_v1's 19,293 surfaces
# and pockets, its PLIP files), so only ~23k new receptors are processed.
# Resubmitting is safe: finished shards and files are skipped.
#
#   bash src/slurm/build_v2.sh
#   ARRAY=0-7 bash src/slurm/build_v2.sh     # more shards, if the submit cap allows
#
# ARRAY sets the shard count for all three arrays (default 0-3, i.e. 4 each:
# 13 jobs in all, under the per-user submit cap that refused 64). At 4 shards
# a shard is ~5.8k new receptors: surfaces ~1.3 s each (~2 h), fpocket 1-3 s
# (~2-5 h), PLIP ~1 s per row; time limits below leave a margin.
# Overrides: ARRAY, QOS (standard), SPEC, RUN (out/runs/<spec name>).
set -e
cd $WORK
ARRAY=${ARRAY:-0-3}
QOS=${QOS:-standard}
SPEC=${SPEC:-configs/datasets/dataset_v2.yaml}
export RUN=${RUN:-$WORK/out/runs/$(basename $SPEC .yaml)}
export SPEC
[ -n "$KEOPS_CACHE_FOLDER" ] || { echo "KEOPS_CACHE_FOLDER is not set (the surface jobs need it)"; exit 1; }

PREP=$(sbatch --parsable --qos=$QOS src/slurm/prep_v2.sbatch)
AFTER="--dependency=afterok:$PREP --kill-on-invalid-dep=yes --qos=$QOS --array=$ARRAY"
CACHE=$(sbatch --parsable $AFTER --time=06:00:00 src/slurm/cache_surfaces.sbatch)
FPOCKET=$(sbatch --parsable $AFTER --time=10:00:00 src/slurm/fpocket.sbatch)
PLIP=$(sbatch --parsable $AFTER --time=10:00:00 src/slurm/plip.sbatch)

echo "prep      $PREP   logs/prep_v2_$PREP.log"
echo "surfaces  $CACHE  logs/cache_surfaces_${CACHE}_<i>.log   (after prep)"
echo "fpocket   $FPOCKET  logs/fpocket_${FPOCKET}_<i>.log     (after prep)"
echo "plip      $PLIP  logs/plip_${PLIP}_<i>.log           (after prep)"
echo "RUN=$RUN"
echo "watch: squeue -u \$USER"
