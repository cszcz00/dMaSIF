#!/bin/bash
#SBATCH --partition=alpha
#SBATCH --qos=test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/head_ckpt_%j.log
# Train the pocket head and save ONLY the checkpoint, nothing else.
#
# Step 1 of apo_test.sbatch, split out so it can be queued on its own. That job
# asks for 4 h, a GPU and 64 G to train the head and then run the whole apo
# evaluation; the probe (src/slurm/probe_retrieval.sbatch) and apo_eval both
# only need the checkpoint, which is ~15 min of that. A smaller, shorter job
# backfills into gaps a 4 h one cannot fit, so it typically starts much sooner
# on a busy queue.
#
# Nothing is duplicated and nothing needs cancelling: apo_test.sbatch writes
# this exact path and skips its own training step when the file already exists
# (`if [ ! -f "$CKPT" ]`), so submit this alongside a pending apo_test and
# whichever runs first, the work happens once.
#
# Overrides: N_TRAIN (32), STEPS (300), ITEMS.
#
# Size this from evidence, not from this file's guesses: after a previous
# encoder_test or region_test job,
#   sacct -j <jobid> -o jobid,jobname,elapsed,maxrss,reqmem
# gives the real peak RSS and runtime. Raise --mem if it OOMs - an OOM costs
# another trip through the queue, which is worse than having asked for more.
set -e
module load apptainer
cd $WORK
N_TRAIN=${N_TRAIN:-32}
STEPS=${STEPS:-300}
ITEMS=${ITEMS:-$WORK/encoder_test/items.tsv}
CKPT=$WORK/encoder_test/head_pretrained_${STEPS}.pt
EXEC="apptainer exec --nv --cleanenv --env KEOPS_CACHE_FOLDER=$KEOPS_CACHE_FOLDER,OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK $WORK/dmasif_sandbox"

echo "node $(hostname): $(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader)"
if [ -f "$CKPT" ]; then
    echo "$CKPT already exists; nothing to do"
    exit 0
fi
[ -f "$ITEMS" ] || { echo "missing $ITEMS (test_encoder.py prepare writes it)"; exit 1; }

echo "=== train pocket head (pretrained start, $STEPS steps, $N_TRAIN items) -> $CKPT ==="
$EXEC python $WORK/src/test_encoder.py overfit --items $ITEMS \
    --n $N_TRAIN --holdout 0 --steps $STEPS --eval_every $STEPS --pretrained \
    --save_ckpt $CKPT

echo
echo "$CKPT written. This unblocks:"
echo "  sbatch src/slurm/probe_retrieval.sbatch   (needs probe/items.tsv + probe/ligand_feats.npz)"
echo "  sbatch src/slurm/apo_test.sbatch          (now skips its own training step)"
