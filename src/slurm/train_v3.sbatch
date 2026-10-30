#!/bin/bash
#SBATCH --partition=alpha
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=logs/train_v3_%j.log
# v3: known-site pockets from fpocket, pooled pairwise. A preset over
# train.sbatch, like archive/train_v2.sbatch: every value below is a default that the
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
# its own ligand's position), so it is not comparable to out/runs/b32's 0.127,
# which scored the head's top-8 regions blind. The blind number, top k_pos
# fpocket pockets with a max:
#   CKPTS="$WORK/out/runs/v3/best.pt" \
#   EVAL_ARGS="--use_ckpt_split --pocket_select topk --pocket_rank drug" \
#       sbatch src/slurm/eval.sbatch
# Rank by druggability, not fpocket's score: on 1,000 training rows the top 8
# by drug hit the ligand's pocket for 0.78 of proteins against 0.65 by score
# (external_pockets.py score, 2026-10-22). Adding --k_pos 16 widens the net.
# and the same with "--pocket_select site" reproduces the log's own number.
#
# Ablations, one GPU each (compare at matched steps):
#   POOL_MODE=mean   RUN=$WORK/out/runs/v3_mean   ... is it the pooling?
#   POOL_MODE=shells RUN=$WORK/out/runs/v3_shells
#   CHEMO=0          RUN=$WORK/out/runs/v3_nochemo   (CHEMO= empty also works)
#   BANK= EXTRA_LIG= RUN=$WORK/out/runs/v3_nobank
#   POCKET_SHAPE=ball RUN=$WORK/out/runs/v3_ball
#
# Site decoys (off unless DECOYS is set): DECOYS=7 pools up to 7 of the
# receptor's other fpocket pockets per row as within-protein negatives,
# trained by --lam_site (LAM_SITE, 1 when DECOYS is set). POCKET_RANK=drug
# draws them from the top k_pos by druggability, the pockets the blind eval
# scores. The `site` column then reads "own pocket above every decoy".
#
# Overrides here: LAM_PLIP / LAM_PLIP_SAL / LAM_PLIP_TYPE / PLIP_SOFT / PLIP
#   (PLIP supervision, below), ATOM_FEATS (learned atom typing, below),
#   POCKETS, POCKET_SHAPE, POCKET_MARGIN, PAIR_ARGS (e.g.
#   "--pair_types 12 --pair_bins 10"), DECOYS, POCKET_RANK, plus everything
#   train.sbatch takes (TAU_MIN among them).
POCKETS=${POCKETS:-$WORK/data/pockets/fpocket}
[ -d "$POCKETS" ] || { echo "no $POCKETS; sbatch src/slurm/fpocket.sbatch first"; exit 1; }
export RUN=${RUN:-$WORK/out/runs/v3}
export DATA=${DATA:-$WORK/out/runs/v1}
export BATCH=${BATCH:-32}
export STEPS=${STEPS:-6000}
export VAL_BALANCE=${VAL_BALANCE-1}
export POOL_MODE=${POOL_MODE:-pairs}
export CHEMO=${CHEMO-1}
export BANK=${BANK-2048}
export EXTRA_LIG=${EXTRA_LIG-1024}
# RAM caches. 4,000 surfaces left ~65% of dataset_v2's training reads on disk
# (v4: 44,087 of 68,000 by step 2,125, ~0.8 s/step of load) at RSS 5.1 GB of
# the 96 GB asked for, CUDA context included, so <= ~1.2 MB per surface.
# 32,000 holds every dataset_v2 training receptor (31,861 rows) after its
# first visit, ~40 GB at most; LABEL_RAM likewise every row. Check the log's
# "RSS peak" after the first evaluations. Results do not depend on these:
# sampling is seeded per receptor, not by cache state.
export SURFACE_RAM=${SURFACE_RAM:-32000}
export LABEL_RAM=${LABEL_RAM:-40000}
export LR_SCHEDULE=${LR_SCHEDULE:-cosine}
export WARMUP=${WARMUP:-200}
# In REGION_ARGS so the smoke runs the v3 path too, not just the main run.
export REGION_ARGS="--region_source external --pockets $POCKETS --pocket_select site \
--pocket_shape ${POCKET_SHAPE:-pocket} --pocket_margin ${POCKET_MARGIN:-0} ${PAIR_ARGS:-} \
${POCKET_RANK:+--pocket_rank $POCKET_RANK}"
if [ -n "${DECOYS:-}" ] && [ "$DECOYS" != 0 ]; then
    # --lam_site here too, so the smoke runs the decoy LOSS and not only the
    # pooling; train.sbatch passes the same LAM_SITE again to the main run.
    export LAM_SITE=${LAM_SITE:-1.0}
    export REGION_ARGS="$REGION_ARGS --n_decoys $DECOYS --lam_site $LAM_SITE"
fi
# PLIP supervision of the site types and salience (off unless LAM_PLIP is set):
# LAM_PLIP=0.1 weights both losses (LAM_PLIP_SAL / LAM_PLIP_TYPE override one
# each); PLIP_SOFT=1.5 ramps the labels across 1.5 A instead of stepping at
# the radius (unset = hard, as v6); labels from PLIP (data/plip). In
# REGION_ARGS, so the smoke runs the losses.
if [ -n "${LAM_PLIP:-}" ] && [ "$LAM_PLIP" != 0 ]; then
    export REGION_ARGS="$REGION_ARGS --plip ${PLIP:-$WORK/data/plip} \
--lam_plip_sal ${LAM_PLIP_SAL:-$LAM_PLIP} --lam_plip_type ${LAM_PLIP_TYPE:-$LAM_PLIP} \
${PLIP_SOFT:+--plip_soft $PLIP_SOFT}"
fi
# Learned atom typing (atom_features.py; off unless ATOM_FEATS is set): a
# second AtomNet over 16 per-atom chemistry values, 16 more input channels.
# The surface cache needs atom_feats first - ATOM_FEATS=1 sbatch
# src/slurm/cache_surfaces.sbatch adds them in place - or the run adds them
# lazily, re-parsing each receptor on its first read. Typed attention is
# POOL_MODE=typed_attn; its sizes go in PAIR_ARGS ("--attn_heads 16
# --attn_vdim 8 --attn_bins 12" are the defaults).
if [ -n "${ATOM_FEATS:-}" ] && [ "$ATOM_FEATS" != 0 ]; then
    export REGION_ARGS="$REGION_ARGS --atom_feats"
fi
export TRAIN_ARGS=${TRAIN_ARGS:-"--eval_every 125 --save_every 125"}
exec bash $WORK/src/slurm/train.sbatch
