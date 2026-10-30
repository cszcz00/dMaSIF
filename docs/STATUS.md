# Project status (handoff)

**Branch `claude/model-training`. Commit and push there; do not open a PR
unless asked.** Read this file first; it is the current state only. Anything
historical is in `RESULTS.md`, anything about how the method works is in
`DESIGN.md`, anything about the data on disk is in `DATA.md`.

Companion documents, split out of this file on 2026-10-27:
- [`RESULTS.md`](RESULTS.md) - every run and ablation, chronologically. The lab notebook.
- [`DESIGN.md`](DESIGN.md) - how the method works and the decisions behind it.
- [`DATA.md`](DATA.md) - the layout under `$WORK`, PLINDER, and the dataset build.
- [`REFACTOR.md`](REFACTOR.md) - the staged plan this split is stage 1 of.

## Where the work stands (2026-10-27)

**`C` is the configuration: fpocket's pocket at the row's ligand
(`--pocket_select site`), pair pooling (`--pool_mode pairs`), BatchNorm on the
pooled vector (`--anchor_norm batch`), batch 32, constant lr, no chemo, no
bank.** `out/runs/v3c_site_pairs_bn`. Five things are settled:

1. **C plateaus at ~0.18 known-site MRR** by step ~3,600 of 5,000, while train
   nce keeps falling. The ceiling is MEMORISATION (~69 visits per target
   group), so more steps buy nothing and **more distinct TARGETS is the
   lever**. (`v3c+`, RESULTS.)
2. **Blind, C at 1,125 steps scores 0.088 - the same as b32's 0.087 at matched
   data.** Its known-site lead was the known SITE, not a better
   representation. The blind number for the PLATEAUED checkpoint has never
   been measured and is the most important open number.
3. **The memory bank + extra ligands is what killed v3, and this is now
   proven twice over.** `v3e` reproduces it exactly (loss pinned at its random
   value, cosdiff 0.87 -> 0.986 in 125 steps), and `out/runs/v2` - recovered
   2026-10-27, having been run and never recorded - dies the same way from a
   completely different encoder: head regions, mean pooling, no BatchNorm, a
   descriptor branch, a cosine schedule. Every bank-free run learns. Dropped.
   NOT a temperature effect (`v3g`), and NOT the BatchNorm train/eval mismatch
   either, since v2 ran `anchor_norm none`. The surviving suspect is the
   detached bank's one-sided repulsion on the ligand encoder; see RESULTS.
4. **Chemo is neutral at scale** (`v3d`), negative twice now. `cfg.chemo`
   stays off.
5. **Site decoys work at their own job and cost the main one** (`v3f`):
   within-protein `site` 0.605 -> 0.794, cross-protein MRR 0.114 -> 0.093.
   Whether that trade is worth it depends entirely on the blind eval.

Standing constraints, from the user, which have held all through this work:

- **The user runs every job on the HPC themselves, via sbatch or the login
  node. Claude writes code, commits and pushes; Claude does not run jobs.**
  Changes reach the HPC by `git pull` on this branch.
- **Claude's own container has no torch and must not install it.** Anything
  needing torch is for the user to run. Numpy, pandas and rdkit paths can be
  checked locally, and geometry arithmetic can and should be verified in numpy
  before asserting anything about it.
- Every experiment gets a script with a docstring saying what it measures and
  how to read it, an sbatch file, and its results recorded in `RESULTS.md`.

## Goal

Contrastive protein-ligand retrieval. Protein side: dMaSIF surface features +
convolution (retrained), a per-point pocket head, and K pooled **region**
anchors per protein compared against ligand vectors.

Built, tested and RUN on a GPU: all of it - the protein encoder and its
region selectors, the apo/holo/AlphaFold evaluation, the ligand encoder, the
frozen-anchor probe, and the contrastive training loop over twelve runs (v1
through the v3 A-I ladder; see `RESULTS.md`). What is built and NOT yet wired
in: the PLIP interaction labels. What is written and not yet run: the
retrieval inspection, the data inventory and the companion funnel, all below.

## How we work

- The user runs everything on the HPC (sbatch / login node). Claude writes
  code, commits and pushes; it does not run jobs. Changes reach the HPC by
  `git pull` on the working branch.
- Every experiment gets: a script with a docstring saying what it measures and
  how to read it, an sbatch file, and results recorded in `RESULTS.md`.
- Prefer comparing methods on the SAME trained head / maps (dump once, sweep
  offline on CPU) over retraining per variant.

## Pending work

**Run the evals first.** They are minutes on the test queue and they settle the
two biggest open questions - whether decoys help blind, and whether C's
plateau beats b32:
```
CKPTS="$WORK/runs/v3c_1500/best.pt $WORK/runs/v3f_pairs_bn_decoys/best.pt \
$WORK/runs/v3c_site_pairs_bn/best.pt" \
EVAL_ARGS="--use_ckpt_split --pocket_select topk --pocket_rank drug" \
    sbatch src/slurm/eval.sbatch
```
Then, in any order:

1. **Round 4**, `bash src/slurm/v3_diag4.sh`: `v3h` = decoys with `K_NEG=0`
   and `lam_site 0.3` (is v3f's MRR loss the WEIGHT, and does dropping the
   redundant patches restore the speed), `v3i` = C with `K_NEG=0` alone (a
   ~1.7x throughput win if BatchNorm survives on 32 vectors per step). Both
   keep the smoke: `k_neg 0` has never run on a GPU.
2. **The retrieval inspection**, `sbatch src/slurm/report.sbatch` then the CPU
   half it prints. WHAT the model retrieves rather than how well - hub
   collapse, the size-only baseline, chemotype rank. See below.
3. **Dataset expansion - NOW FIRST** (user, 2026-10-29: before the channel
   analysis). `src/dataset_funnel.py report` (CPU venv, seconds) says which
   filter costs how many NEW TARGET GROUPS, then `companion_funnel.py
   funnel` / `distance` for the `n_ligand_chains` question; then build the
   DRAFT `configs/datasets/dataset_v2.yaml`. See "Dataset expansion" below.
4. **`dataset_v2`**: one rebuild carrying the relaxed filters, the per-entry
   dedup for the symmetry copies, and (if `distance` allows) the relaxed
   `n_ligand_chains`. Larger val/test come with it for free, since PLINDER's
   val/test rows are filtered by the same rules.
5. **PLIP labels into training**: built and checked, never wired. The
   intended use is supervising pair pooling's site types; see DESIGN.
6. Older and still unrun: a **probe rerun** for the anchor-spread diagnostic
   and `--anchor_norm center / whiten` (CPU-only, `PROBE_ONLY=1`), and a
   **paired base-vs-chemo bootstrap** from the per-item sweep TSVs (a small
   offline CPU script, not written). The follow-up selector sweep
   (`SWEEP_ONLY=1 sbatch src/slurm/region_test.sbatch`) is superseded by the
   apo test and the chemo ablation, both of which scored every selector.

## Next steps (model training, on `claude/model-training`)

1. **Second round of runs (code in place, not yet run).** Four GPU jobs plus
   one short eval, commands in `src/slurm/train.sbatch`'s header:
   - finish v1: `SKIP_SMOKE=1 sbatch src/slurm/train.sbatch` (15,642 -> 20,000)
   - `out/runs/frozen`, `FREEZE=1`: `--freeze_encoder` freezes the surface encoder
     below the projection (conv, inputs, pocket head, so the regions are fixed
     too); projection, ligand and temperature train. The probe's setting inside
     this loop, on this split: **the control that says whether training the
     protein side buys the MRR.**
   - `out/runs/shells`, `POOL_MODE=shells`: 48-dim pooled vector that keeps
     arrangement inside a region. Head loads 117/118 (projection width).
   - `out/runs/b32`, `BATCH=32 STEPS=5000`, eval/save every 125: 4x the
     negatives; ~3,900 steps is v1's 125k proteins seen. Memory at batch 32
     is unmeasured; an OOM would fail at the first step.
   - all three new runs use `VAL_BALANCE=1` (`--val_balance`: the 400 val rows
     drawn round-robin over target groups, so all ~283 val groups appear
     instead of 112) and `DATA=$WORK/runs/v1` for items/features. Same table +
     same split_seed = the same 400 rows in every run.
   - `sbatch src/slurm/eval.sbatch` re-scores v1's `best.pt` / `last.pt` on
     those same 400 rows (`train.py eval --recarve --val_max 400
     --val_balance`): v1's row in the comparison table. **`train.py eval` has
     never run before**, and the eval path now shares `load_rows()` with
     train so its row indices match a checkpoint's even when ligands were
     dropped.
   Read: frozen vs v1 at matched steps on MRR / top10 (cosdiff will not move
   much under frozen, by construction - only the projection changes);
   shells and b32 vs v1 on all of them. Val noise is ~+-0.015 MRR per
   evaluation, so compare the mean of the last few evaluations, not maxima.
   Deferred: a wider conv (`emb_dims` > 16). The head checkpoint and the
   published dMaSIF weights are 16-wide, so widening discards the pretrained
   head and the regions with it; it needs a function-preserving widening
   first.
2. **The 0.993 question, at the vector level.** This is the whole point of the
   loop and `cosdiff` in its log is the readout. The two levers built for it
   are `--anchor_norm batch` and `--pool_mode shells`; both are OFF by default,
   so the first run is a clean baseline and the next two are the ablation. If
   `cosdiff` does not move, nothing downstream of it will.
   Once it does move, **re-ask the apo question**: `apo_eval.py`'s
   vector-level numbers are uninterpretable while everything sits at 0.993,
   and a contrastively trained encoder is the first thing worth re-measuring
   there.
3. **Scale.** `--val_max` caps validation at 400 proteins and `--batch` IS the
   number of in-batch negatives, so batch size is a loss-quality knob, not
   just a speed one. Expect to measure seconds per step from the first run and
   size the chain from that.
4. **dataset_v2**: per-entry dedup, larger val/test (PLINDER's own val/test are
   891 / 1,120 against our 94 / 305). The loop's `--val_split` already accepts
   the dataset's own splits, so this is a data job, not a code one.
5. **Re-test the chemo channels at scale** - the ablation's negative result was
   measured on 32 training proteins with 28 input channels, which is where
   extra capacity shows up as memorisation rather than generalisation.
6. **A size-only baseline arm** for the probe, to settle whether the ~7x chance
   signal is size matching rather than chemical complementarity.
7. Later: apo / AlphaFold receptors as training augmentation. The apo test
   argues for it - AlphaFold inputs cost only 6% of AP against holo, and
   `D_closed` holds up far better than expected, so closure is augmentation
   rather than exclusion.
8. Small known wart: `test_encoder.py` has no selector CLI, so the region
   columns `encoder_test` / `chemo_test` print inline are the dead `watershed`
   default. Harmless (overfit's loss is pocket BCE alone, so the maps are
   selector-independent and `region_sweep.py` scores every selector offline on
   them), but misleading to read. `train.py` and `probe_retrieval.py` both
   take the flags.

## v6: PLIP supervision of the site types and salience (`src/plip_loss.py`) - WRITTEN, numpy part of selftest PASSES, torch part + train path untested

Why: channel_viz on C (RESULTS) - types coarse (polar / apolar / shape,
never donor vs acceptor), salience = concavity (PLIP AUROC 0.55), and v3d
showed more input chemistry alone does not sharpen them. Two auxiliary
losses on the row's SITE region (`--pocket_select site`, slot 0):
- salience: BCE(sal logit, contacted), contacted = any PLIP class at the
  point for ANY ligand of the receptor (union over sibling rows), positive-
  weighted by the batch's neg / pos ratio (capped at 20);
- types: CE of the (T + 1)-way type softmax against the PLIP classes on
  contacted points only, type t <- class t for the 6 classes (donor,
  acceptor, cation, anion, aromatic, hydrophobic); types 6-7 and "not a
  site" get no target; uncontacted points get no type loss (PLIP labels are
  positive-unlabelled).
Plumbing: `PairPool.heads()` (raw logits; `site_types` now wraps it),
`LabelCache(plip=...)` attaches `plip_union` per row, train flags `--plip
--lam_plip_sal --lam_plip_type` (refused without pairs / site select / >= 6
types), a `plip` line under every evaluation (losses + contacted share; type
CE at chance = log 9 = 2.197). `train_v3.sbatch`: LAM_PLIP=0.1 puts it in
REGION_ARGS, so the smoke runs the losses too.

Run order (login node first, container, CPU):
```
apptainer exec --cleanenv --env OMP_NUM_THREADS=8 $WORK/dmasif_sandbox python src/plip_loss.py selftest
apptainer exec --cleanenv --env OMP_NUM_THREADS=8 $WORK/dmasif_sandbox python src/pair_pool.py
apptainer exec --cleanenv --env OMP_NUM_THREADS=8 $WORK/dmasif_sandbox python src/train.py selftest
# v6 = v5 (C + chemo on dataset_v2) + PLIP supervision
LAM_PLIP=0.1 SKIP_SMOKE=0 STEPS=10000 CHEMO=1 BANK= EXTRA_LIG= LR_SCHEDULE=const WARMUP=0 \
POOL_MODE=pairs ANCHOR_NORM=batch RUN=$WORK/out/runs/v6_C_chemo_plip_dataset_v2 \
DATA=$WORK/out/runs/dataset_v2 \
TRAIN_ARGS="--eval_every 125 --save_every 125 --val_set $WORK/out/valset_v1v2" \
sbatch --job-name=v6_plip src/slurm/train_v3.sbatch
```
Read: v6 against v5 (same everything but the two losses) on val MRR, and
channel_viz on both for the per-type PLIP AUROCs. The `plip` line should show
type CE falling well below 2.197 and salience BCE falling; if MRR drops while
they fall, the weight is too high (try 0.03).

## Ladder: pooling P0 / P2 / P3 x features chemo / learned atom typing - WRITTEN; numpy self-tests PASS (atom_features, paired_eval), torch parts untested

The question: does pooling that is more chemistry-aware extract more, and
more so from richer chemistry? Same seed, data and batches for every run
(as v4 / v5), 5,000 steps, the winners extended.

|                       | P0 `mean`     | P2 `pairs`    | P3 `typed_attn` |
|---|---|---|---|
| chemo (28 ch)         | `lad_p0_chemo` | **v5** (running) | `lad_p3_chemo` |
| learned atoms (32 ch) | `lad_p0_af`    | `lad_p2_af`   | `lad_p3_af`     |

What is new (all off by default; v4-v7 and their resubmits are unchanged):
- `src/atom_features.py`: 13 values per heavy atom - element, sp3 / sp2 /
  aromatic, H count, delocalised formal charge, donor, acceptor, backbone,
  burial at 10 A - plus each polar atom's direction (`atom_dir`, the N-H /
  lone-pair bisector or C->O axis, a unit vector). Dropped as trivially
  learned: carbon bonded to N/O, a histidine flag, burial at 6 A. Residue
  and atom names are only lookup keys: 175 (residue, atom) entries -> 23
  descriptors, 8 single-residue, all distinct chemical groups (self-test
  prints them). `--atom_feats` / `ATOM_FEATS=1`: `dir_atomnet.DirAtomNet`
  (`SurfaceEncoder.chemnet`, fresh) - AtomNet's 3 atom->atom and 3
  atom->point rounds at width 16, input Linear(13 -> 16), distances as
  d / 4 A, and every atom->point message carrying cos(polar direction,
  atom -> point), which is rotation / translation invariant and bounded.
  16 more input channels after the pretrained ones, which the inflation
  zero-pads, so step 0 is the pretrained function. Without chemo: 10
  curvature + 6 AtomNet + 16 = 32 channels.
- Cache: `atom_feats` / `atom_dir` are optional cached arrays (an older
  width is recomputed). `ATOM_FEATS=1 sbatch
  src/slurm/cache_surfaces.sbatch` adds it to existing surfaces in place
  (atoms re-parsed, surface kept, temp-file rename, so running jobs are
  unaffected); otherwise a run adds it lazily on first read.
- `--pool_mode typed_attn` (`pair_pool.TypedAttnPool`): PairPool's type /
  salience heads, then 16 learned heads, each (from-type mix, to-type mix,
  distance profile over 12 Gaussians 3-16 A) with an 8-dim partner
  description; output mean + mass + per head (log count, partners) = 168
  dims. Sizes via `PAIR_ARGS="--attn_heads .. --attn_vdim .. --attn_bins .."`.
- `--keep_steps 5000` saves `step5000.pt`, never overwritten.
- `train.py eval --per_row FILE` writes each scored row's rank;
  `src/paired_eval.py A.tsv B.tsv` gives B - A with a 95% interval from a
  bootstrap over TARGET GROUPS (numpy only; `python src/paired_eval.py
  selftest`).

Run order:
```
# login node, container, CPU: the parts that need torch
EX="apptainer exec --cleanenv --env OMP_NUM_THREADS=8 $WORK/dmasif_sandbox python"
$EX src/atom_features.py && $EX src/dir_atomnet.py && $EX src/pair_pool.py && $EX src/train.py selftest
# atom_feats into the shared cache (16 shards, ~1-2 s a receptor)
ATOM_FEATS=1 RUN=$WORK/out/runs/dataset_v2 sbatch src/slurm/cache_surfaces.sbatch
# the five runs
COMMON="STEPS=5000 BANK= EXTRA_LIG= LR_SCHEDULE=const WARMUP=0 ANCHOR_NORM=batch \
DATA=$WORK/out/runs/dataset_v2"
TA="--eval_every 125 --save_every 125 --val_set $WORK/out/valset_v1v2 --keep_steps 2500,5000"
for spec in "p0_chemo CHEMO=1 POOL_MODE=mean" "p3_chemo CHEMO=1 POOL_MODE=typed_attn" \
            "p0_af CHEMO=0 ATOM_FEATS=1 POOL_MODE=mean" "p2_af CHEMO=0 ATOM_FEATS=1 POOL_MODE=pairs" \
            "p3_af CHEMO=0 ATOM_FEATS=1 POOL_MODE=typed_attn"; do
  set -- $spec; name=$1; shift
  env $COMMON "$@" RUN=$WORK/out/runs/lad_$name TRAIN_ARGS="$TA" \
      sbatch --job-name=lad_$name src/slurm/train_v3.sbatch
done
```
v5 is P2 + chemo but started without --keep_steps: copy its last.pt when its
history.json reaches step 5,000 (`last.pt` is rewritten every 125 steps), as
`out/runs/v5_C_chemo_dataset_v2/step5000.pt`.

Reading: window means at matched steps first (as for v4 / v5), then the
paired test on step5000.pt of each pair:
```
for r in lad_p0_chemo v5_C_chemo_dataset_v2 lad_p3_chemo lad_p0_af lad_p2_af lad_p3_af; do
  CKPTS=$WORK/out/runs/$r/step5000.pt DATA=$WORK/out/runs/dataset_v2 \
  EVAL_ARGS="--val_set $WORK/out/valset_v1v2 --pocket_select site \
             --per_row $WORK/out/per_row/$r.tsv" sbatch src/slurm/eval.sbatch
done
python src/paired_eval.py out/per_row/lad_p0_chemo.tsv out/per_row/v5_C_chemo_dataset_v2.tsv
```
The comparisons that answer the question: P0 -> P2 -> P3 within each row
(does the pooling help?), and chemo vs learned atoms within each column
(do the features?); the interaction is whether the column gaps grow
left to right. Watch: s/step of typed_attn against pairs (estimated close),
the learned branch's first ~1,000 steps (fresh weights, like chemo's), and
cosdiff only as a collapse alarm.

## v7: soft PLIP labels + a heavier salience weight - WRITTEN, numpy part of `plip_loss.py selftest` PASSES, torch part untested

Why: v6's salience BCE sits at its constant-prediction value (0.948 against
0.930 at a 0.33 contacted share, step 1,750; RESULTS) while its type CE
learns (1.316 against 2.197). Salience is the one PLIP loss with negatives,
so the hard 3 A step - between surface points ~1 A apart that the conv
cannot tell apart - hits it hardest; and at 0.1 it is ~0.09 of a ~3.4 loss.
- `--plip_soft S` (`PLIP_SOFT`): labels ramp linearly from 1 at r - S/2 to 0
  at r + S/2 (r = 3 A atoms, 4 A ring / charge centres), max over a class's
  interactions; same label mass as the step (selftest: 1.06x). Salience
  target = the point's max class weight, type target = the class weights
  normalised, each point's type CE weighted by its salience target. 0 = the
  v6 losses exactly (selftest checks bool vs 0/1 float).
- `LAM_PLIP_SAL` overrides `LAM_PLIP` for the salience weight alone.
- The `plip` line now prints the salience BCE's constant-prediction value
  next to it (`train.plip_sal_floor`): below it = salience is learning.
- `train_v3.sbatch` RAM defaults raised (SURFACE_RAM 4,000 -> 32,000,
  LABEL_RAM 20,000 -> 40,000): every dataset_v2 training receptor stays
  resident after its first read. Speed only; resubmits of v4-v6 get it too.

All new behaviour is off by default, so resubmitting v4 / v5 / v6 after a
pull continues them unchanged. Run order (login node, container, CPU):
```
apptainer exec --cleanenv --env OMP_NUM_THREADS=8 $WORK/dmasif_sandbox python src/plip_loss.py selftest
# v7 = v6 + soft labels (1.5 A) + salience weight 0.5 (types stay at 0.1)
LAM_PLIP=0.1 LAM_PLIP_SAL=0.5 PLIP_SOFT=1.5 SKIP_SMOKE=0 STEPS=10000 CHEMO=1 BANK= EXTRA_LIG= \
LR_SCHEDULE=const WARMUP=0 POOL_MODE=pairs ANCHOR_NORM=batch \
RUN=$WORK/out/runs/v7_C_chemo_plip_soft_dataset_v2 DATA=$WORK/out/runs/dataset_v2 \
TRAIN_ARGS="--eval_every 125 --save_every 125 --val_set $WORK/out/valset_v1v2" \
sbatch --job-name=v7_soft src/slurm/train_v3.sbatch
```
Read: salience BCE below its constant-prediction value (the v6 symptom
gone); type CE as v6's or lower; val MRR window means against v6 and v5 at
matched steps. v7 changes two things against v6; if it moves MRR, a
`PLIP_SOFT=1.5` run with `LAM_PLIP_SAL` unset (0.1) separates them.

## Channel visualisation + PLIP diagnostic (`src/channel_viz.py`) - FIRST RUN DONE (RESULTS 2026-10-29)

Result in one line: the site types did NOT collapse but encode a polar /
apolar axis + shape, never donor vs acceptor; v3d shows more input
chemistry alone does not sharpen them; salience is learned concavity (PLIP
AUROC 0.55). The case for PLIP supervision (types + salience, with
`--chemo` on) is made; see RESULTS.


What the per-point channels look like and mean, never inspected so far: the
16 inputs (10 curvatures, fixed; 6 AtomNet channels, LEARNED and fine-tuned,
so checkpoint-specific) and pair pooling's 8 site types + salience (trained
only through the contrastive loss; whether they differentiated or collapsed
is unknown). Also the gate for PLIP supervision of those heads (DESIGN, PLIP
labels): how well they already match what ligands contact.

`sbatch src/slurm/viz.sbatch` (GPU, test QOS, ~minutes): dumps C (resumed),
v3d and v3f on the same 64 group-balanced val proteins (`--rows_from`),
each with the PUBLISHED AtomNet channels on the same points and chemotype's
12 channels as an interpretable reference; then the numpy report:
- types: usage share, effective number of types, mean |r| between type
  masses, per-point entropy -> "COLLAPSED / partly redundant /
  differentiated"
- naming: each learned channel's top-3 correlates among chemotype, local
  element shares (6 A) and curvatures
- pocket: AUROC of every channel for the 4 A label (also curvature sign)
- plip: AUROC per channel x PLIP class inside the site region; chemotype's
  own rows are the reference supervision could reach
- `out/viz/report/<run>_html/*.html` (plotly viewer, every channel in a
  dropdown, ligand drawn, points within 18 A) and `<run>_pymol/<system>/`
  (`pymol load.pml`, B-factor = channel)
`TARGETS=P56817,P0DTD1 sbatch ...` adds named targets. Viewers are a SURFACE MESH by default
(2026-10-29): the zero set of an oriented-point signed distance through the
sampled points, triangulated by surface nets (numpy), every channel smoothed
onto the vertices (Gaussian 1.2 A); `--viewer points` for the raw samples.
`sbatch src/slurm/viz_report.sbatch` redoes report + viewers on existing
dumps, CPU only. Checked: a sampled sphere meshes closed (Euler 2), on the
samples, faces outward; a rendered page draws in headless Chromium. Since
2026-10-29 the mesh covers the WHOLE protein by default (an 18 A crop
sphere left ragged edges and see-through holes), with geometry and 8-bit
channel levels base64-encoded: ~3-4 MB and ~45 s per protein. The dump stage is
untested (no torch here); it reuses train.py's load_model_from_ckpt /
build_batch / anchor_block exactly as retrieval_report.py does, and every
attribute anchor_block reads is supplied by the parser. The report, viewer
and PyMOL output are checked on a planted dump (11 checks).

## Dataset expansion - dataset_v2 BUILT (2026-10-30): 7,895 train groups, 100% surfaces / pockets / PLIP, fixed val set in out/valset_v1v2. C re-scored on the fixed set: **MRR 0.151** (top10 0.268; RESULTS). NEXT: C on dataset_v2 (`v4_C_dataset_v2`, command below), compared against 0.151

**Build + train order (user, 2026-10-29: cap 50, fixed validation).**
```
git pull
bash src/slurm/build_v2.sh          # prep (CPU) -> surfaces (GPU) / fpocket / PLIP arrays
# when all four are done: re-score the v1 checkpoints on the FIXED set
CKPTS="$WORK/out/runs/v3c_site_pairs_bn/best.pt" \
EVAL_ARGS="--val_set $WORK/out/valset_v1v2 --pocket_select site" sbatch src/slurm/eval.sbatch
# then C's configuration on dataset_v2, validated on the same rows
SKIP_SMOKE=0 STEPS=10000 CHEMO=0 BANK= EXTRA_LIG= LR_SCHEDULE=const WARMUP=0 \
POOL_MODE=pairs ANCHOR_NORM=batch RUN=$WORK/out/runs/v4_C_dataset_v2 \
DATA=$WORK/out/runs/dataset_v2 \
TRAIN_ARGS="--eval_every 125 --save_every 125 --val_set $WORK/out/valset_v1v2" \
sbatch src/slurm/train_v3.sbatch
```
`src/valset.py` (selftest PASSES, numpy only): the fixed validation set is
dataset_v1's carved val TARGETS (re-derived exactly as every v1 run carved
them: same rows dropped for failed ligands, split_by_group at val_frac 0.1,
split_seed 0), 400 group-balanced rows that exist in BOTH datasets with the
same ligand; a v2 run trains on everything outside those targets.
`train.py train / eval --val_set DIR` use it (eval refuses a set that
overlaps the checkpoint's train rows). The v1 numbers so far were on a
different (per-run carved) set, so C is re-scored on the fixed one before
comparing. `train.py` changes are static-checked only: the first eval /
train with --val_set is their test.

Third run (RESULTS): 8,398 train groups (from 2,379; visits 67 -> 19) with
the scan gates, but nucleotide analogues (ANP, GNP, ACP), NAG, maltose and
GLU were uncapped hubs and DMU a detergent. Now: a 50-row cap on every
ligand and a detergent list. `dataset_v2.yaml` mirrors LADDER_SCAN row for
row and is ready to build once the fourth run confirms the numbers:
`SKIP_INSPECT=1 SKIP_COMPANION=1 sbatch src/slurm/funnel.sbatch`.

Decisions (user, 2026-10-29): add covalent ligands and SMALL oligomers;
cofactors as targets WITH A CAP per ligand; companions by STRIP-AND-GATE
(keep extraction protein-only, admit a system by companion size x
distance); the structure scan over the WHOLE manifest; everything through
Slurm, nothing heavy on the login node.

Run order, now (CPU venv, Slurm):
```
source ~/venvs/plinder/bin/activate && pip install gemmi      # once
python src/structure_scan.py selftest                           # seconds, login node is fine
OUT=$WORK/tmp/scan_test LIMIT=50 sbatch --array=0 src/slurm/scan.sbatch   # timing check
sbatch src/slurm/funnel.sbatch            # inspect_columns + funnel (no scan yet) + companion funnel
sbatch src/slurm/scan.sbatch              # 64 shards over the whole manifest
sbatch --dependency=afterok:<scan job id> src/slurm/funnel.sbatch   # merge + funnel with --scan
```
New since the first run:
- funnel: `not_oligo<=HA40/HA60` (oligomers up to N heavy atoms),
  `not_cofactor=cap25/50/100` (cofactor targets, at most N train rows per
  CCD code, kept one per target first so the cap keeps as many targets as
  it can), a cofactor CCD census, the most frequent train ligands at the
  end of the ladder, and with `--scan` the gates `ligand_chains=gate_ion8 /
  gate_small8` (companions within 8 A no bigger than 1 / 6 heavy atoms),
  `mw=buried` (120-200 Da only when burial45 >= 0.9) and
  `resolution<=4.5q` (past 3.5 A only with <= 2 missing pocket atoms and
  pocket B ratio <= 1.5). Default ladder: no_altlocs=drop, resolution<=3.5,
  mw150-900, not_covalent=drop, not_oligo<=HA60, not_cofactor=cap50.
- `structure_scan.py` (gemmi): per manifest row, ligand burial and
  contacts, pocket completeness / B ratio / occupancy / altlocs on the
  receptor file the surface is built from, and companion size x distance
  from system.cif matched by position (no reliance on its chain naming).
  `selftest` PASSES (planted receptor, ligand, Mg at 3 A, 6-atom group at
  6 A, 12-atom group at 20 A, water); run -> merge -> join checked end to
  end on planted files.
- `build_dataset.py`: `join` (extra per-row tables on (system_id,
  ligand_id)) and `selection.cap` (`cap_rows`, shared with the funnel).
  `build_manifest.py --inspect_columns`: the annotation table's validation /
  quality / ligand-class columns, for replacing proxies with PLINDER's own
  local measures where they exist.
- `dataset_v2.yaml` DRAFT now matches the new default ladder; checked row
  for row against the funnel on a planted 6,000-row manifest.

Why: `v3c+` put the ceiling at memorisation (~69 visits per target group at
5,000 x 32), so the lever is more distinct TARGET GROUPS (UniProt, the
sampler's unit), not rows. dataset_v1 keeps 21,209 of the manifest's 341,945
rows and nobody has measured which filter is responsible. The manifest is
loose (only HARD filters applied) and `structures/` was fetched for all of
it (223,972 receptors), so **any relaxation needs no new downloads**: only
surface sampling, fpocket and PLIP for the new receptors, all of which skip
work already done.

`dataset_funnel.py report` counts, against a baseline that is dataset_v1's
selection: per single-criterion variant (each filter dropped, plus loosened
thresholds for resolution, altlocs, unresolved pocket atoms, missing
residues, crystal contacts, MW, interactions, pocket size, ligand chains)
the train rows, train groups, **NEW groups**, rows in them, new receptor
inputs, PLINDER val/test rows and visits per group; then a cumulative
`--ladder` and the largest groups at its end. Dedup prefers `strict_rep`
rows, so every variant is a SUPERSET of dataset_v1: existing surfaces,
pocket files and PLIP labels stay valid. `--per_entry` adds the (PDB entry,
ligand) dedup for symmetry copies, which removes rows.

Supporting changes:
- `build_dataset.py`: `dedup` may be a list of keys applied in turn, new
  `dedup_sort` (leading `-` = descending; default `system_id`, i.e. the old
  behaviour), derived column `entry`. dataset_v1's spec reproduces exactly.
- `companion_funnel.py` could not have run: `n_ligand_chains` is derived by
  `build_dataset.add_derived`, not stored in the manifest, so `funnel` exited
  at once; `distance` imported `extract` (torch at module level) in the CPU
  venv, and looked for a `ligand_instance_chain` column that does not exist
  (it now takes the chain from `ligand_id`). Its groups default to the
  sampler's target groups.
- `configs/datasets/dataset_v2.yaml`, DRAFT: the default ladder (altlocs
  dropped, resolution <= 3.0 A, <= 5 unresolved pocket heavy atoms) + per-
  entry dedup, `n_ligand_chains == 1` kept. Checked on a planted 3,000-row
  manifest: `build_dataset.select` on this spec picks exactly the rows the
  funnel's ladder does, and contains every dataset_v1 row.

Caveat for `ligand_chains`: PLINDER builds a system from ligands that sit
close together, so a second ligand chain in the SAME system is probably near
the site by construction, and the "glycerol 30 A away" case may be rare.
`distance` measures it; expect most of `ligand_chains=drop`'s gain to need
the gate.
Also: `funnel`'s companion-kind table reads companion ROWS from the
manifest, and ions and artifacts are HARD filters dropped when the manifest
was built, so it can only see cofactor / oligo / covalent companions. The
ion and additive share has to come from `distance`'s SDFs.

Build order once the spec is settled (the funnel / scan steps are in the
Slurm run order at the top of this section; the build steps are seconds to
minutes of pandas and rdkit, but run them in an interactive or batch job too):
```
python src/build_dataset.py configs/datasets/dataset_v2.yaml --root $WORK --dry_run
python src/build_dataset.py configs/datasets/dataset_v2.yaml --root $WORK --check_files
python src/train.py prepare --index data/datasets/dataset_v2/index.parquet \
    --out out/runs/dataset_v2 --split train
python src/ligand_features.py --index data/datasets/dataset_v2/index.parquet --out out/runs/dataset_v2
RUN=$WORK/out/runs/dataset_v2 sbatch src/slurm/cache_surfaces.sbatch   # GPU, new receptors only
RUN=$WORK/out/runs/dataset_v2 sbatch src/slurm/fpocket.sbatch          # CPU, new receptors only
RUN=$WORK/out/runs/dataset_v2 sbatch src/slurm/plip.sbatch             # CPU, new rows only
```
Ligand features MUST be rebuilt (`fp_row` is re-indexed per dataset). Not
`out/runs/v2`: that name is the dead bank run.

Open before training on it: **validation comparability.** `train.py` carves
val by target group from whatever items it is given, so a dataset_v2 run
validates on different rows from every run so far, and "did more data help"
needs a FIXED val set (e.g. v1's carved val groups held out of dataset_v2's
train). Not built yet.

## Retrieval inspection (`src/retrieval_report.py`) - WRITTEN, selftest PASSES, never run on real data

Every number so far is a scalar (mrr, top1, site, cosdiff): it says how well
the ranking does, never WHAT it ranks. This dumps the top-k ligands per
receptor for a TRAIN and a VAL slice and reports the diagnostics that tell
apart three different models at the same mrr - one finding the right chemotype
and missing the molecule, one matching ligand SIZE to pocket size, and one
returning the same hub ligands for every receptor. `dump` (GPU, ~2 min) writes
anchors + every ligand vector; `report` (numpy, CPU) scores and analyses, so
one dump answers every later question. It imports from `train.py` and never
writes to it, so it is safe to run while a job holds the same files.

What it answers, and the decision each line drives:
- **hubs**: share of top-k slots held by the 20 most-retrieved ligands,
  normalised entropy, ligands covering half the slots, and the TRAIN/VAL
  Jaccard of the hub sets. A low entropy is a ligand prior, not retrieval; a
  high Jaccard means ONE global prior rather than per-receptor matching.
- **chemotype rank**: best rank among candidates at Tanimoto >= 0.4 / 0.5 /
  0.7 to the truth. If molecule mrr is 0.18 and chemotype mrr is ~0.45, the
  model finds the class and misses the molecule - a success, and an argument
  for reporting (and maybe training) at chemotype level.
- **size-only baseline**: rank candidates by |heavy atoms - heavy atoms of
  truth| and take its mrr. If the model is near it, the signal is size
  matching. This is "Next steps" item 6, answered for free.
- **coherence**: mean pairwise Tanimoto inside each top-k against a random-k
  baseline, plus the top-k's Tanimoto to truth. Coherent but wrong = something
  systematic to find; both low = noise.
- **properties**: Spearman between the true ligand's descriptors and the
  top-1's, per column, so "it matches polarity but not shape" is readable.
- per-row TSV carries fallback, site_size and group, for slicing success by
  pocket size / merged sites / target-group size (the memorisation axis).

`selftest` (numpy only) PASSES and already earned itself: the top-k list was
built over the masked candidates, and a row's own ligand is masked because it
shares its own target group, so every statistic would have been computed on
lists that systematically EXCLUDED the right answer. Groups [1]-[4] cover the
Tanimoto / rank / entropy helpers, anchor masking, a planted perfect ranking,
and a planted hub collapse.

Run it on C at 5,000 steps, both selects, before tweaking anything.
`src/slurm/report.sbatch` is the GPU half (both selects, ~30 min test queue)
and prints the CPU commands to follow it:
```
sbatch src/slurm/report.sbatch                 # -> out/reports/<run>_{site,topk}.npz
# then, CPU venv (numpy only; load_ligand_arrays avoids ligand_encoder's torch)
python src/retrieval_report.py report --dump out/reports/v3c_site_pairs_bn_topk.npz \
    --items out/runs/v1/items.tsv --lig_feats out/runs/v1/ligand_feats.npz \
    --out out/reports/v3c_topk
```

## Data inventory (`src/inventory.py`) - WRITTEN, selftest PASSES, never run on real data

What is on disk, is it complete, is any of it stale. Read-only (parquet, .npz
headers, history.json), so it is safe while jobs train, and numpy/pandas only -
no torch, so it runs in the CPU venv. `python src/inventory.py report --items
out/runs/v1/items.tsv`. Three things `du` and `ls` cannot answer:
- **coverage**: a surface / pocket / PLIP file per row of an items.tsv, keyed
  correctly per artefact (surfaces and pockets by `input_id`, PLIP by
  `system_id`), with the missing ones named. A missing surface costs sampling
  time; a missing pocket file silently drops that receptor's regions and
  raises `fb`.
- **staleness**, as an exact test rather than a date: `ligand_feats.npz` built
  before the RemoveAllHs fix kept isotopic H / H2 / hydrides as GRAPH atoms,
  so `n_heavy` disagrees with `diff(atom_ptr)`. Any mismatch means rebuild.
  START HERE records `probe/` and `train_check/` as affected; this says which
  files on disk actually are.
- **runs**: every `out/runs/<name>/` summarised from history.json alone -
  evaluations, last step, best val MRR and where, last cosdiff / tau / fb,
  which checkpoints exist. The index of what each directory holds, which
  otherwise lives only in this file's prose.
`selftest` (planted tree, numpy only) PASSES: 13 checks over tree counts, the
pocket subdirectory and its failure log, per-artefact coverage keys, the
staleness test on a planted bad file, run summaries, and the degenerate cases
(a NaN-only history, an unreadable history, absent directories).

## Refactor plan - `docs/REFACTOR.md`, NOTHING DONE YET

13,382 lines over 24 modules, 26 slurm files, and this file at 1,591 lines.
Five stages ordered by risk: (1) split this file into STATUS / RESULTS / DATA /
DESIGN; (2) extract `train.py`'s selftest, which is 990 of its 2,436 lines;
(3) move the shared mask / sampling / metric functions out of
`probe_retrieval.py`, which `train.py` imports from today, into `masks.py` /
`sampling.py` / `metrics.py`; (4) split the rest of `train.py` into model /
batching / losses / evaluate; (5) archive the spent `v3_diag*.sh` and
`train_v2.sbatch`. The rule for every stage: move code VERBATIM, change
imports only, run the affected selftests before and after. Stage 3 is the one
most likely to break something, because those are the functions that keep the
probe and the loop from drifting apart.

**Stages 1, 2 and 5 are DONE** (2026-10-27): this file split four ways,
`train.py` 2,437 -> 1,612 lines with its selftest in `src/train_selftest.py`,
and the spent ladders in `src/slurm/archive/`. **One check outstanding: run
`python src/train.py selftest` in the container once** - it is verified only
statically, and it is also the first run of group [11]. Stages 3 and 4 are
deliberately held until v3h / v3i are in.

## Companion ligands and the `n_ligand_chains == 1` restriction (`src/companion_funnel.py`) - WRITTEN, NOT RUN

`configs/datasets/dataset_v1.yaml` keeps a system only if it has exactly ONE
ligand chain of ANY kind, ions and crystallisation additives included. So a
drug-like ligand with a magnesium, or with a glycerol 30 A away, is discarded
whole. (The row-level `not_cofactor` / `not_ion` filters are a different and
wanted thing: they stop a cofactor being the TARGET.) That costs data, and
`v3c+` showed the ceiling is MEMORISATION - ~69 visits per target group with
val mrr flat while train nce falls - so more distinct targets is the lever
that matters.

The restriction has one real justification: `extract.load_protein_atoms` keeps
amino acids only, so a companion molecule leaves a VOID, fpocket fills that
void with alpha spheres, and the void can merge into the site region the model
pools over. Whether that matters is a DISTANCE question - a companion far from
the proper ligand cannot reach its site region.

`companion_funnel.py funnel` (manifest, pandas) gives the rows / receptors /
groups a relaxed query would add and what the companions are by kind;
`distance` (manifest + PLINDER system SDFs) gives the minimum companion-to-
ligand heavy-atom distance for a sample of them, against 4 A (the pocket
label), 8 A (roughly a site region's span) and a proposed `--gate`. Then:
- few rows added -> leave the query alone.
- many rows, companions far -> relax to one PROPER ligand and keep the
  current extraction. **Cheap**: existing surfaces and pocket files stay
  valid, only new receptors need sampling and fpocket, and both jobs skip
  work already done.
- many rows, companions close -> relax with a distance gate, or pay the
  expensive path (keep metals / cofactors as receptor atoms), which
  invalidates all 19,293 surfaces, every pocket file, the PLIP labels and the
  comparability of every run here - and costs AlphaFold compatibility, since
  predicted structures have no cofactors and the apo test's 6% AP finding was
  measured with everything stripped on both sides.
Fold into **dataset_v2** with the two issues already queued there (per-`pdb_id`
dedup for the symmetry copies, and a properly sized val/test): one rebuild
rather than three.
