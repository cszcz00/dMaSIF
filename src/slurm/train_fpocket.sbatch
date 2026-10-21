#!/bin/bash
#SBATCH --partition=alpha
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=train_fpocket_%j.log
# train.sbatch with the pocket regions taken from fpocket instead of the
# pocket head (--region_source external). A preset: every train.sbatch
# override still works, and this resumes itself the same way.
#
# Why: the head's map is the known limit on the regions (its top 5% covers
# only 0.68 of the labelled pocket), region selection has no gradient, and
# every run so far has scored regions the head placed. Here fpocket decides
# WHERE (its top k_pos pockets, in its own rank order), and the encoder only
# learns what each pocket looks like to a ligand.
#
# Needs $POCKETS from src/slurm/fpocket.sbatch, and should follow its `score`
# step: if fpocket's hit / dccA there are not at least ball_R8's (0.84 / 0.77),
# this run is not worth its GPU, and the score table also says which
# POCKET_SHAPE / POCKET_MARGIN to use.
#
# Batch, val set and data match runs/b32 (the best head-region run: 0.127 MRR
# at 76k-92k proteins seen on the balanced val set), so the two are directly
# comparable at MATCHED PROTEINS SEEN.
#
# Read `fb` first: it is now the rate at which fpocket's top k_pos pockets miss
# the ligand. A val protein with no fpocket pocket at all has no anchors and
# drops out of the retrieval metrics; the store line under each evaluation
# counts them ("without a file" / "with no usable pocket").
#
# Variants, one GPU each:
#   POCKET_SHAPE=ball RUN=$WORK/runs/fp_ball sbatch src/slurm/train_fpocket.sbatch
#   POCKET_RANK=drug  RUN=$WORK/runs/fp_drug sbatch src/slurm/train_fpocket.sbatch
#   REGION_SOURCE=external_map RUN=$WORK/runs/fp_map sbatch src/slurm/train_fpocket.sbatch
#   LAM_POCKET=0 RUN=$WORK/runs/fp_nohead sbatch src/slurm/train_fpocket.sbatch
#       (the head no longer places regions; this asks whether its BCE still
#       helps as an auxiliary loss on the shared conv)
# Overrides here: POCKETS, REGION_SOURCE, POCKET_SHAPE, POCKET_RANK,
#   POCKET_MARGIN, plus everything train.sbatch takes.
POCKETS=${POCKETS:-$WORK/pockets/fpocket}
[ -d "$POCKETS" ] || { echo "no $POCKETS; sbatch src/slurm/fpocket.sbatch first"; exit 1; }
export RUN=${RUN:-$WORK/runs/fpocket}
export DATA=${DATA:-$WORK/runs/v1}
export BATCH=${BATCH:-32}
export STEPS=${STEPS:-5000}
export VAL_BALANCE=${VAL_BALANCE-1}
# REGION_ARGS, not TRAIN_ARGS: train.sbatch puts these in the smoke too, so
# the smoke exercises the external path before the real run starts.
export REGION_ARGS="--region_source ${REGION_SOURCE:-external} --pockets $POCKETS \
--pocket_shape ${POCKET_SHAPE:-pocket} --pocket_rank ${POCKET_RANK:-score} \
--pocket_margin ${POCKET_MARGIN:-0}"
export TRAIN_ARGS=${TRAIN_ARGS:-"--eval_every 125 --save_every 125"}
exec bash $WORK/src/slurm/train.sbatch
