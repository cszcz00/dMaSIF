#!/bin/bash
#SBATCH --partition=alpha
#SBATCH --qos=test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/chemo_test_%j.log
# Do the pharmacophore channels raise the pocket head's ceiling?
#
# Two arms on the SAME items, seed and step count, differing only in the conv's
# input width:
#     base    16 channels (10 curvature + 6 AtomNet)         - today's encoder
#     chemo   28 channels (+ 12 chemotype.py)                - pretrained weights inflated
#
# The number to compare is the `candidates` oracle row that region_sweep.py
# prints: the union of each protein's top 5% of points after smoothing, whose
# coverage of the labelled pocket is the CEILING for every selector. It is 0.68
# today, and STATUS calls the head the main limit. Also compare the ball_R8 row
# (the chosen selector) and the per-point AUROC / AP the overfit run prints.
#
# Needs $WORK/out/encoder_test/items.tsv (test_encoder.py prepare).
#
# Overrides: N_TRAIN (32), N_HOLD (64), STEPS (300),
#   ARM (both | base | chemo)   - run one arm per job if the QoS walltime is
#                                 too short for two training runs
#   SWEEP_ONLY=1                - reuse existing maps, rerun the sweeps only
#   SKIP_SMOKE=1                - skip the 4-item shape check
set -e
module load apptainer
cd $WORK
N_TRAIN=${N_TRAIN:-32}
N_HOLD=${N_HOLD:-64}
STEPS=${STEPS:-300}
ARM=${ARM:-both}
OUT=$WORK/out/encoder_test
EXEC="apptainer exec --nv --cleanenv --env KEOPS_CACHE_FOLDER=$KEOPS_CACHE_FOLDER,OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK $WORK/dmasif_sandbox"
RUN="$EXEC python $WORK/src/test_encoder.py"

echo "node $(hostname): $(nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader)"
apptainer exec --nv --cleanenv $WORK/dmasif_sandbox python -c "
import sys, torch
cap = 'sm_%d%d' % torch.cuda.get_device_capability()
print('GPU', cap, '| torch built for', torch.cuda.get_arch_list())
sys.exit(0 if cap in torch.cuda.get_arch_list() else 'container cannot run on ' + cap)
"
[ -f "$OUT/items.tsv" ] || { echo "missing $OUT/items.tsv (test_encoder.py prepare)"; exit 1; }
echo "items: $(($(wc -l < $OUT/items.tsv) - 1)); arm(s): $ARM; $N_TRAIN train + $N_HOLD held-out"

# The typing tables are pure python and cost nothing; a wrong table would make
# every chemo channel meaningless, so check it before spending the GPU.
echo "=== chemotype self-test ==="
$EXEC python $WORK/src/chemotype.py

if [ "${SKIP_SMOKE:-0}" != "1" ] && [ "$ARM" != "base" ]; then
    echo "=== smoke, chemo arm (shapes, inflate, backward) ==="
    $RUN smoke --items $OUT/items.tsv --n 4 --pretrained --chemo
fi

for arm in base chemo; do
    [ "$ARM" = both ] || [ "$ARM" = "$arm" ] || continue
    MAPS=$OUT/maps_${arm}_${STEPS}.pt
    EXTRA=""
    [ "$arm" = chemo ] && EXTRA="--chemo"
    if [ "${SWEEP_ONLY:-0}" != "1" ]; then
        echo "=== train: $arm arm, $STEPS steps, pretrained start ==="
        $RUN overfit --items $OUT/items.tsv --n $N_TRAIN --holdout $N_HOLD \
            --steps $STEPS --eval_every 100 --pretrained $EXTRA --dump $MAPS
    fi
    echo "=== region sweep: $arm arm ==="
    $EXEC python $WORK/src/region_sweep.py --maps $MAPS \
        --out $OUT/sweep_${arm}_${STEPS}.tsv \
        --per_item $OUT/sweep_items_${arm}_${STEPS}.tsv ${SWEEP_ARGS:-}
done

echo
echo "compare, between $OUT/sweep_base_${STEPS}.tsv and $OUT/sweep_chemo_${STEPS}.tsv:"
echo "  candidates  oracle cover/prec  <- the ceiling; 0.68 cover today"
echo "  ball_R8     iou / cover / prec / hit / dccA"
echo "and the held-out AUROC / AP lines printed by each overfit run."
