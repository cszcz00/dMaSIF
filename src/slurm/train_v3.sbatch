#!/bin/bash
#SBATCH --partition=alpha
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=train_v3_%j.log
# v3: known-site pockets from fpocket, pooled pairwise. A preset over
# train.sbatch, like train_v2.sbatch: every value below is a default that the
# submitting shell can override, the smoke + resume check runs first, and the
# job resumes itself (resubmit the same command, SKIP_SMOKE=1 to save time).
#
# The stack, per training row:
#   whole receptor surface -> dMaSIF conv (+ --chemo pharmacophore channels)
#   -> fpocket's pocket AT this row's ligand (--pocket_select site: the union
#      of fpocket pockets the ligand occupies; external_pockets.site_pockets)
#   -> pair pooling of that pocket's point embeddings (--pool_mode pairs:
#      learned soft site types x types x distance bins, pair_pool.py)
#   -> projection -> InfoNCE against the ligand GNN.
# One anchor per protein, so the score is a plain cosine (the DrugCLIP
# setting). The pocket head no longer places anything; LAM_POCKET keeps its
# BCE as an auxiliary loss on the shared conv (0 to drop it).
#
# Needs, in order (docs/STATUS.md "v3"):
#   bash src/slurm/build_fpocket.sh                 # once
#   sbatch src/slurm/fpocket.sbatch                 # -> $POCKETS
#   python src/external_pockets.py score ...        # the site table: `found`
#                                                   # is the rows v3 can train on
#   $EX src/pair_pool.py ; $EX src/train.py selftest  # groups [9], [10] are new
#
# Read, per evaluation line: mrr / top1 / top10 against chance; `fb` is now
# the share of val rows where fpocket found NO pocket at the ligand (they are
# scored on the fallback ball); AUROC / AP are the auxiliary head. cosdiff is
# measured on the pooled vectors, here the pair fingerprints.
#
# This validates KNOWN-SITE retrieval (each val protein's pocket is chosen by
# its own ligand's position), so it is not comparable to runs/b32's 0.127,
# which scored the head's top-8 regions blind. The blind number, top k_pos
# fpocket pockets with a max:
#   CKPTS="$WORK/runs/v3/best.pt" \
#   EVAL_ARGS="--use_ckpt_split --pocket_select topk" sbatch src/slurm/eval.sbatch
# and the same with "--pocket_select site" reproduces the log's own number.
#
# Ablations, one GPU each (compare at matched steps):
#   POOL_MODE=mean   RUN=$WORK/runs/v3_mean   ... is it the pooling?
#   POOL_MODE=shells RUN=$WORK/runs/v3_shells
#   CHEMO=0          RUN=$WORK/runs/v3_nochemo   (CHEMO= empty also works)
#   BANK= EXTRA_LIG= RUN=$WORK/runs/v3_nobank
#   POCKET_SHAPE=ball RUN=$WORK/runs/v3_ball
#
# Overrides here: POCKETS, POCKET_SHAPE, POCKET_MARGIN, PAIR_ARGS (e.g.
#   "--pair_types 12 --pair_bins 10"), plus everything train.sbatch takes.
POCKETS=${POCKETS:-$WORK/pockets/fpocket}
[ -d "$POCKETS" ] || { echo "no $POCKETS; sbatch src/slurm/fpocket.sbatch first"; exit 1; }
export RUN=${RUN:-$WORK/runs/v3}
export DATA=${DATA:-$WORK/runs/v1}
export BATCH=${BATCH:-32}
export STEPS=${STEPS:-6000}
export VAL_BALANCE=${VAL_BALANCE-1}
export POOL_MODE=${POOL_MODE:-pairs}
export CHEMO=${CHEMO-1}
export BANK=${BANK-2048}
export EXTRA_LIG=${EXTRA_LIG-1024}
export SURFACE_RAM=${SURFACE_RAM:-4000}
export LABEL_RAM=${LABEL_RAM:-20000}
export LR_SCHEDULE=${LR_SCHEDULE:-cosine}
export WARMUP=${WARMUP:-200}
# In REGION_ARGS so the smoke runs the v3 path too, not just the main run.
export REGION_ARGS="--region_source external --pockets $POCKETS --pocket_select site \
--pocket_shape ${POCKET_SHAPE:-pocket} --pocket_margin ${POCKET_MARGIN:-0} ${PAIR_ARGS:-}"
export TRAIN_ARGS=${TRAIN_ARGS:-"--eval_every 125 --save_every 125"}
exec bash $WORK/src/slurm/train.sbatch
