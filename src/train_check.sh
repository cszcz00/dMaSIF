#!/bin/bash
# End-to-end check of the training pipeline, on the LOGIN NODE, no GPU, no queue.
#
# Seven stages, each printing what it proves and what a failure means. Nothing
# here touches runs/ or surfaces/: everything lands in $OUT (default
# $WORK/train_check), so `rm -rf` on that directory undoes the whole run.
#
#   1  imports            every module loads and every name resolves
#   2  selftest           the loop minus dMaSIF, on a planted signal
#   3  prepare            the real dataset index -> items.tsv
#   4  ligand features    SMILES -> ECFP4 + descriptors + graphs
#   5  train on CPU       the REAL encoder: Biopython parse, dMaSIF surface
#                         sampling, AtomNet, the convolution, region selection,
#                         pooling, InfoNCE, pocket BCE, a checkpoint
#   6  resume             a second invocation picks up last.pt and advances
#   7  eval               metrics from that checkpoint, nothing trained
#
# Stages 1, 2, 5, 6, 7 run in the container; 3 and 4 in the CPU venv. Both are
# invoked by path, so this script does not care which one you are in.
#
# Stage 5 is the slow one and the one worth waiting for. It runs the sampler at
# --sup_sampling 6 instead of 20, so the surfaces are ~3x sparser and the pass
# takes minutes rather than an hour of login-node CPU. That is a SHAPE check,
# not a training run: the point is that every stage of the real pipeline
# executes and the numbers are finite, not that they are any good. KeOps also
# compiles its CPU kernels the first time, which is a few minutes of g++ on top.
#
# Overrides: OUT, N (interactions, 12), STEPS (2), SUP (6), LIG (an existing
#   ligand_feats.npz to reuse - probe/ligand_feats.npz is built from the same
#   index, so pointing at it skips stage 4 entirely), VENV, SANDBOX, INDEX,
#   SKIP (a space-separated list of stage numbers), FULL=1 (sup_sampling 20).
set -u
cd ${WORK:?set WORK to the repo root}

OUT=${OUT:-$WORK/train_check}
INDEX=${INDEX:-$WORK/datasets/dataset_v1/index.parquet}
VENV=${VENV:-$HOME/venvs/plinder/bin/python}
SANDBOX=${SANDBOX:-$WORK/dmasif_sandbox}
N=${N:-12}
STEPS=${STEPS:-2}
SUP=${SUP:-6}
[ "${FULL:-0}" = "1" ] && SUP=20
SKIP=${SKIP:-}

# No --nv: there is no GPU here, and asking for one makes apptainer complain.
EXEC="apptainer exec --cleanenv --env KEOPS_CACHE_FOLDER=${KEOPS_CACHE_FOLDER:-$WORK/keops_cache},OMP_NUM_THREADS=${OMP:-8} $SANDBOX python"
RUN=$OUT/run
declare -a RESULTS=()
FAILED=0

stage () {         # stage <number> <name>
    echo
    echo "=============================================================="
    echo "[$1] $2"
    echo "=============================================================="
}

record () {        # record <number> <name> <exit code> <seconds>
    if [ "$3" -eq 0 ]; then
        RESULTS+=("  PASS  [$1] $2  (${4}s)")
    else
        RESULTS+=("  FAIL  [$1] $2  (exit $3 after ${4}s)")
        FAILED=1
    fi
}

skipped () { case " $SKIP " in *" $1 "*) return 0;; *) return 1;; esac; }

run_stage () {     # run_stage <number> <name> <command...>
    local num=$1 name=$2; shift 2
    if skipped "$num"; then
        RESULTS+=("  SKIP  [$num] $name")
        return 0
    fi
    stage "$num" "$name"
    local t0=$SECONDS
    "$@"
    local rc=$?
    record "$num" "$name" "$rc" "$((SECONDS - t0))"
    [ $rc -eq 0 ] || echo ">>> [$num] $name FAILED (exit $rc); later stages may fail for this reason alone"
    return 0
}

mkdir -p $OUT
echo "repo      $WORK"
echo "out       $OUT   (delete this directory to undo everything below)"
echo "container $SANDBOX"
echo "venv      $VENV"
echo "settings  N=$N interactions, STEPS=$STEPS, sup_sampling=$SUP"
[ -f "$SANDBOX" ] || [ -d "$SANDBOX" ] || { echo "no container at $SANDBOX"; exit 1; }
[ -x "$VENV" ] || echo "[warn] no CPU venv python at $VENV; stages 3 and 4 will fail"
[ -f "$INDEX" ] || echo "[warn] no dataset index at $INDEX; stages 3-7 will fail"

# ---------------------------------------------------------------- 1. imports
# Seconds, and it catches the whole class of typo that would otherwise surface
# twenty minutes into stage 5. Importing surface_encoder does NOT pull in
# dMaSIF - that happens when SurfaceEncoder is constructed - so this needs no
# KeOps and no GPU.
import_check () {
    $EXEC - <<'PY'
import sys
sys.path.insert(0, "src")
import anchors, chemotype, ids, ligand_encoder, ligand_features
import probe_retrieval, surface_encoder, test_encoder, train, train_data
from surface_encoder import EncoderConfig

for mode, want in (("mean", 16), ("mean_std", 32), ("shells", 48)):
    cfg = EncoderConfig(pool_mode=mode, n_shells=3)
    got = cfg.pooled_dims
    assert got == want, f"{mode}: pooled_dims {got}, expected {want}"
    print(f"  pooled_dims {mode:9s} = {got:3d}   in_channels = {cfg.in_channels}")
print(f"  in_channels with chemo  = {EncoderConfig(chemo=True).in_channels} (expect 28)")
assert EncoderConfig(chemo=True).in_channels == 28
for bad in ("meen", "shell"):
    try:
        EncoderConfig(pool_mode=bad).pooled_dims
    except ValueError:
        print(f"  pool_mode {bad!r} rejected at config time, as it should be")
    else:
        raise AssertionError(f"pool_mode {bad!r} was accepted")
print("imports and config arithmetic OK")
PY
}
run_stage 1 "imports and config arithmetic" import_check

# --------------------------------------------------------------- 2. selftest
# The loop minus dMaSIF, on synthetic proteins with a planted signal. This is
# the stage that tells you whether the LOGIC is right: region labels, the
# fallback, the sampler, the InfoNCE, the metrics, the checkpoint round-trip.
# A failure here is a bug in the code; a failure only in stage 5 is a bug in
# how the code meets the real encoder.
run_stage 2 "train.py selftest (no dMaSIF, no KeOps)" \
    $EXEC src/train.py selftest --tmp $OUT/selftest

# ---------------------------------------------------------------- 3. prepare
# Real index, real structure paths, real ligand SDFs. Proves the item table and
# its target groups come out of the actual dataset, not just the fixture.
run_stage 3 "prepare: the real index -> items.tsv" \
    $VENV src/train.py prepare --index $INDEX --out $OUT --n $N --split train

# -------------------------------------------------------- 4. ligand features
# fp_row indexes the WHOLE ligand table of the index, not just these rows, so
# any ligand_feats.npz built from the same index is interchangeable -
# probe/ligand_feats.npz is, and reusing it skips an rdkit pass over 8,378
# molecules.
LIG=${LIG:-$OUT/ligand_feats.npz}
if [ -f "$LIG" ]; then
    echo
    echo "[4] reusing $LIG"
    RESULTS+=("  SKIP  [4] ligand features (reusing $LIG)")
else
    run_stage 4 "ligand features (rdkit over every ligand in the index)" \
        $VENV src/ligand_features.py --index $INDEX --out "$(dirname "$LIG")"
fi

# ----------------------------------------------------------- 5. the real loop
# The whole protein side on CPU: Biopython parse -> dMaSIF surface sampling ->
# curvatures + AtomNet -> dMaSIFConv_seg -> pocket head -> region selection ->
# pooling -> projection -> InfoNCE + pocket BCE -> a checkpoint.
#
# shells + batch anchor_norm on purpose: those two paths have never gone
# through the real convolution, and this is the cheapest place to find out
# that they do. A real run should start from the defaults.
TRAIN_ARGS="--items $OUT/items.tsv --lig_feats $LIG \
    --surface_cache $OUT/surfaces --device cpu \
    --sup_sampling $SUP --pool_mode shells --anchor_norm batch \
    --region_method ball --ball_radius 8 --ball_shift 2 \
    --batch 2 --val_frac 0.4 --val_max 4 --eval_batch 2 \
    --surface_ram 32 --val_ram 8 --lam_site 0.5"

# Exercises load_head against a REAL checkpoint, including its shape filter:
# that head was saved with pool_mode "mean", so its projection does not fit a
# shells run and has to be reported and left at init rather than crashing.
HEAD=${HEAD:-$WORK/encoder_test/head_pretrained_300.pt}
if [ -f "$HEAD" ]; then
    TRAIN_ARGS="$TRAIN_ARGS --head_ckpt $HEAD"
    echo "stage 5 will also load the pocket head from $HEAD"
fi
run_stage 5 "train on CPU: the real encoder, $STEPS steps" \
    $EXEC src/train.py train $TRAIN_ARGS --out $RUN --steps $STEPS \
        --eval_every $STEPS --save_every $STEPS

# -------------------------------------------------------------- 6. resume
# A second invocation must find $RUN/last.pt, report the step it resumed at,
# and carry on from there. Nothing else in the project exercises this path, and
# train.sbatch depends on it entirely - a long run is a chain of resubmissions.
# A TAGGED line, pulled out with sed, rather than the last line of output:
# apptainer and the container entrypoint both print chatter of their own, and
# `tail -1` would silently hand an error message to an integer comparison.
ckpt_step () {
    $EXEC - <<PY | sed -n 's/^CKPT_STEP=//p' | tail -1
import torch
print("CKPT_STEP=%d" % torch.load("$1", map_location="cpu",
                                  weights_only=False)["step"])
PY
}

resume_check () {
    local before after
    before=$(ckpt_step $RUN/last.pt)
    case "$before" in
        ''|*[!0-9]*) echo "could not read a step number out of $RUN/last.pt"
                     echo "(got: '$before')"; return 1;;
    esac
    echo "last.pt is at step $before; running $STEPS more"
    $EXEC src/train.py train $TRAIN_ARGS --out $RUN \
        --steps $((STEPS * 2)) --eval_every $STEPS --save_every $STEPS || return 1
    after=$(ckpt_step $RUN/last.pt)
    case "$after" in
        ''|*[!0-9]*) echo "could not read a step number back"; echo "(got: '$after')"
                     return 1;;
    esac
    echo "last.pt is now at step $after"
    [ "$after" -gt "$before" ] || {
        echo "the step did not advance ($before -> $after): the resume path"
        echo "picked up the checkpoint but made no progress"; return 1; }
    # The surfaces must all have come from the disk cache this time: if the
    # second run resamples, the cache key or the write is broken, and a chain
    # of sbatch jobs would pay the sampling cost every single time.
    echo "surfaces now cached: $(ls $OUT/surfaces 2>/dev/null | wc -l)"
}
run_stage 6 "resume from last.pt" resume_check

# ---------------------------------------------------------------- 7. eval
# The eval subcommand rebuilds the encoder from the checkpoint's own config,
# which is the path apo_eval-style follow-ups will take.
eval_check () {
    local ck=$RUN/best.pt
    [ -f "$ck" ] || ck=$RUN/last.pt
    [ -f "$ck" ] || { echo "no checkpoint in $RUN; stage 5 did not get that far"; return 1; }
    echo "evaluating $ck"
    $EXEC src/train.py eval --ckpt $ck \
        --items $OUT/items.tsv --lig_feats $LIG \
        --surface_cache $OUT/surfaces --device cpu \
        --eval_batch 2 --surface_ram 32 --use_ckpt_split
}
run_stage 7 "eval from the checkpoint" eval_check

# ---------------------------------------------------------------- summary
echo
echo "=============================================================="
echo "SUMMARY"
echo "=============================================================="
if [ ${#RESULTS[@]} -gt 0 ]; then
    for line in "${RESULTS[@]}"; do echo "$line"; done
fi
echo
if [ $FAILED -eq 0 ]; then
    echo "All stages passed. The pipeline runs end to end on real data."
    echo
    echo "What this does NOT tell you:"
    echo "  - anything about GPU execution. KeOps compiles different kernels for"
    echo "    CUDA, and the sm_120 check in train.sbatch is what catches that."
    echo "  - anything about model quality. $STEPS steps on $N interactions at"
    echo "    sup_sampling $SUP is a shape check; the metrics are noise."
    echo "  - how long a real step takes. CPU KeOps is orders of magnitude off."
    echo
    echo "Next: sbatch src/slurm/train.sbatch   (it smokes 64 items on the GPU first)"
else
    echo "Something failed. Read the first FAIL above, not the last - later"
    echo "stages consume earlier ones, so one break cascades."
fi
echo
echo "rm -rf $OUT   undoes everything this script wrote."
exit $FAILED
