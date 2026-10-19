#!/bin/bash
#SBATCH --partition=alpha
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=train_v2_%j.log
# v2: train.sbatch with the four v2 levers switched on. Everything below is a
# DEFAULT, so any of it can be overridden from the submitting shell exactly as
# with train.sbatch, and this file stays a preset rather than a second copy of
# the job. Like train.sbatch it resumes itself: resubmit the same command to
# continue from $RUN/last.pt (with SKIP_SMOKE=1 to save the two minutes).
#
# The levers, and the evidence for each (docs/STATUS.md, "Second round"):
#   BANK=2048, EXTRA_LIG=1024  negatives. batch 8 -> 32 was the best single
#       change and still climbing; this takes each ligand from 31 wrong
#       proteins to ~2,000 and each protein from 31 wrong ligands to ~1,000,
#       for the price of re-projecting cached vectors and encoding small
#       molecules.
#   GPU_NEIGHBORS=1, SURFACE_RAM=4000, LABEL_RAM=20000  throughput. Every run
#       stopped on the clock while still improving, at ~0.4 s per protein.
#       The new "s/step" line under each evaluation says where that time goes;
#       RSS peak on the same line says whether SURFACE_RAM can go higher.
#   DESC_DIMS=32  capacity: a second, fresh 32-wide conv for region
#       descriptors next to the pretrained 16-wide one, which still places the
#       regions. The frozen control showed the 16 pretrained dims alone give
#       ~0.08 of the ~0.13.
#   LR_SCHEDULE=cosine, WARMUP=200  over STEPS=6000.
#
# Compare against runs/b32 (0.127 MRR over 76k-92k proteins seen, same
# balanced val set) at MATCHED PROTEINS SEEN = step x 32, mean of the last
# few evaluations, not the best one.
#
# Leave-one-out ablations, one GPU each, when the full run looks worth it:
#   RUN=$WORK/runs/v2_nobank BANK=0 EXTRA_LIG=0 sbatch src/slurm/train_v2.sbatch
#   RUN=$WORK/runs/v2_nodesc DESC_DIMS=0 sbatch src/slurm/train_v2.sbatch
export RUN=${RUN:-$WORK/runs/v2}
export DATA=${DATA:-$WORK/runs/v1}
export BATCH=${BATCH:-32}
export STEPS=${STEPS:-6000}
export VAL_BALANCE=${VAL_BALANCE-1}
export BANK=${BANK-2048}
export EXTRA_LIG=${EXTRA_LIG-1024}
export DESC_DIMS=${DESC_DIMS-32}
export GPU_NEIGHBORS=${GPU_NEIGHBORS-1}
export SURFACE_RAM=${SURFACE_RAM:-4000}
export LABEL_RAM=${LABEL_RAM:-20000}
export LR_SCHEDULE=${LR_SCHEDULE:-cosine}
export WARMUP=${WARMUP:-200}
export TRAIN_ARGS=${TRAIN_ARGS:-"--eval_every 125 --save_every 125"}
exec bash $WORK/src/slurm/train.sbatch
