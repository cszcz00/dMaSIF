# Results

Every run and ablation, oldest first. Appended to after each one; nothing here
is rewritten once recorded. Current state and what to run next are in
[`STATUS.md`](STATUS.md); the method itself is in [`DESIGN.md`](DESIGN.md).

Split out of `STATUS.md` on 2026-10-27 and reordered chronologically - the
original interleaved the earliest work (the region sweep) with the latest (the
v3 ladder). Section text is unchanged.

## Results so far

**Pocket head** (pretrained start, 300 steps, 32 training receptors, pocket BCE
pos_weight ~42): held-out AUROC 0.937, AP 0.470 on 64 receptors (AP ~20x the
0.023 base rate); reproduces the 16-receptor test (0.950 / 0.470). Pretrained
start beats fresh weights.

**Region selectors** (`src/region_sweep.py`, job 129929; 64 held-out receptors):
- Labelled pocket ~260 points (2.8% of surface), extent ~18 A, Rg ~6 A; one
  piece at 3 A linkage.
- Candidate ceiling: top 5% covers 0.68 of the pocket (prec 0.38); top 8% 0.80.

| held-out | #reg | size | iou | cover | prec | hit | top1 | iou1 | dcc1 | dccA |
|---|---|---|---|---|---|---|---|---|---|---|
| old watershed | 7.2 | 53 | 0.18 | 0.20 | 0.65 | 0.00 | 0.00 | 0.16 | 0.23 | 0.58 |
| watershed link 3, pers 0.02 | 1.7 | 373 | 0.39 | 0.66 | 0.48 | 0.73 | 0.69 | 0.35 | 0.38 | 0.47 |
| components top 5%, link 3 | 1.5 | 406 | 0.38 | 0.68 | 0.45 | 0.70 | 0.66 | 0.34 | 0.33 | 0.41 |
| ball R 8, shift 2 | 7.1 | 293 | 0.53 | 0.70 | 0.63 | 0.84 | 0.64 | 0.34 | 0.44 | 0.77 |

(iou/cover/prec of the best-matching region; hit = some region covers >= 0.5 at
prec >= 0.3; top1/iou1/dcc1 = the top-ranked region; dcc = centre within 4 A of
the ligand centroid.) Fragmentation was the whole problem for watershed /
components; merged, they hit the candidate ceiling but overshoot the pocket.
Balls give the best region shape and precision but rank weakly; on the
top-ranked region all methods tie. Leader so far: balls R 8 + mean shift. The
head (candidates miss 1/3 of the pocket) is now the main limit.

## Chemo ablation (DONE, job 139099: the pharmacophore channels do NOT help)

`src/slurm/chemo_test.sbatch`, two arms on the same 32 train / 64 held-out
items, same seed, same 300 steps, differing only in the conv's input width:
16 channels (10 curvature + 6 AtomNet) against 28 (+ 12 `chemotype.py`).
The GPU paths it was written to exercise all worked: `inflate=True` widened the
three input tensors 16 -> 28, 110 pretrained tensors loaded, no parameter was
left without a gradient.

The headline is the **candidate ceiling**, the union of each protein's top 5%
of smoothed points, which caps every selector:

| held-out, smooth 2 | top 3% | top 5% | top 8% | top 10% |
|---|---|---|---|---|
| base (16 ch) | 0.48 / 0.45 | **0.67** / 0.38 | 0.80 / 0.28 | 0.83 / 0.23 |
| chemo (28 ch) | 0.48 / 0.45 | 0.66 / 0.38 | 0.78 / 0.28 | 0.82 / 0.23 |

(cover / precision.) Unmoved, if anything marginally worse. So is everything
else:

| held-out at step 300 | base | chemo |
|---|---|---|
| per-point AUROC | 0.937 | 0.937 |
| per-point AP | 0.467 | 0.474 |
| `ball_R8` iou / cover / prec / hit / dccA | 0.54 / 0.72 / 0.63 / 0.86 / 0.77 | 0.51 / 0.69 / 0.61 / 0.80 / 0.72 |

**But the training numbers say why, and it is not "the features are useless".**
The chemo arm fits the TRAINING set better and generalises identically: train
AP 0.540 against 0.523, train hit 0.12 against 0.03. That is extra input
capacity being spent on memorisation - 28 channels against **32 training
proteins**. The ablation is conclusive that the channels do not help at this
scale and genuinely inconclusive about whether they help with real training
data. `cfg.chemo` stays off by default; re-test once `train.py` has run at
scale.

It also rules something out. The 0.68 ceiling is NOT structural - pocket share
is 2.8%, so at top 5% a perfect ranker would reach cover 1.0 - and better input
features do not move it. The limit is the per-point ranking (AP 0.47), and most
plausibly the training set size.

The selector result **reproduced** on this independently trained head:
`ball_R8` held-out iou 0.54 (first sweep: 0.53), cover 0.72 (0.70), prec 0.63
(0.63), hit 0.86 (0.84), dccA 0.77 (0.77). The paired bootstrap sharpens the
known trade-off: `watershed pers=0.02` beats `ball_R8` on **top1 by +0.09
[+0.02, +0.19]**, interval excluding zero, while losing dccA by 0.38. Balls
recall, watershed ranks. Re-ranking balls (`rank=sum`) is a wash on every
metric, every interval including zero.

One number to carry into training: `ball_R8` has `on` = 4.0, i.e. four regions
per protein that are at least 30% pocket. Those must be IGNORED, not trained as
negatives - which is what `train_data.region_labels` does.

Not yet done: a PAIRED base-vs-chemo test. `region_sweep.py` only pairs
selectors WITHIN one map set, so the table above is unpaired and the 0.01-0.02
differences are well inside one standard error on 64 proteins.
`sweep_items_base_300.tsv` and `sweep_items_chemo_300.tsv` hold per-protein
rows for the same proteins, so a paired bootstrap is a small offline CPU
script; it would turn "looks like noise" into an interval.

## Ligand-arm probe (DONE, job 139130: all arms indistinguishable)

`src/slurm/probe_retrieval.sbatch` over frozen anchors from
`head_pretrained_300.pt`. Read `probe/arms.tsv` for the table. The finding:
**the five arms are statistically indistinguishable** - the spread between arms
(~0.007 MRR) is smaller than one arm's own evaluation-to-evaluation variation
(~0.017), against a signal of roughly 7x chance. `desc` holding its own on
top1 is the suggestive detail, since `desc` knows only size, polarity and
shape: the signal may be SIZE MATCHING rather than chemical complementarity,
which a size-only baseline arm would settle.

This does not rank the ligand encoders, because the protein side is the
bottleneck. That job predates the spread diagnostic, so rerun it on the current
code to get the numbers:

```
PROBE_ONLY=1 sbatch src/slurm/probe_retrieval.sbatch                  # spread
PROBE_ONLY=1 PROBE_ARGS="--anchor_norm center" sbatch ...             # and again
PROBE_ONLY=1 PROBE_ARGS="--anchor_norm whiten" sbatch ...
```
(`PROBE_ARGS` passthrough is not in the sbatch yet - add it or edit the file.)

Together with the chemo ablation this is two independent results pointing the
same way: **neither side's representation is the lever.** Better protein input
features change nothing, and better ligand encoders change nothing. What has
never been tried is training the protein encoder with the contrastive
objective, which is what `train.py` is for.

## Apo / holo / AlphaFold (DONE: `src/apo_eval.py`, 613 pairs, 0 failures)

**Pocket finding transfers.** `eval` set, 225 systems, the unbiased stratum:

| version | AUROC | AP | pocket pts |
|---|---|---|---|
| holo | 0.925 | 0.475 | 227 |
| holo_rs (noise floor) | 0.926 | 0.477 | 226 |
| apo | 0.912 | 0.360 | 201 |
| pred (AlphaFold) | 0.915 | 0.447 | 201 |

AUROC barely moves; AP drops 24% on apo but only 6% on AlphaFold. **Predicted
structures are a better input than real unbound crystal forms** - good news,
since AF models are what a prospective screen would use. Degradation scales
with conformational change (`C_large` apo AP 0.364 vs holo 0.534), but
`D_closed` holds up far better than expected (0.453 vs 0.497): clash-defined
closure is not the same as large-scale rearrangement, which argues for keeping
`D_closed` as augmentation rather than excluding it.

**Placement works.** fit_rmsd 0.44 A (apo) / 0.48 A (pred), `kept` 1.00 / 0.89,
zero failures over 613 pairs x 4 versions. `holo_rs` tracks `holo` almost
exactly and its region shift is ~1.0 A against apo's ~2.5 A, so the test has
real resolution.

**The vector-level numbers are NOT interpretable yet.** The median cosine
between top-region vectors of DIFFERENT proteins is 0.993 (0.987 for the old
watershed). For random 16-dim unit vectors it would be ~0. So `site_cos` 0.999
and `cos1` 0.999 on apo are not evidence of invariance - everything is 0.993
from everything. Decomposing, ~88% of each pooled vector is one shared
direction and the protein-distinguishing residual is ~12% of the magnitude
(||a-b|| = sqrt(2(1-c)) = 0.118).

That is expected rather than alarming: pocket BCE is a 1-D objective, so only
the direction `pocket_head` reads was ever shaped, and the other 15 dims still
hold dMaSIF's interface descriptor. Nothing has yet asked them to distinguish
proteins. But it means **the apo question is answered for pocket FINDING and
not at all for retrieval vectors**, and has to be re-asked after contrastive
training. Practical consequences: centre and whiten pooled vectors before the
projection, and report the cosine spread / effective rank alongside any
retrieval metric.

## Getting the loop onto a GPU (2026-10-18, history)

The debugging record of the first end-to-end runs, from the original START
HERE. Kept because every bug here is a class the self-tests now cover, and
because the "never run on a GPU" list at its end is what the later rounds
worked through. Status lines in it are as of 2026-10-18.

**The one open question, answered by output Claude has not seen.** The user ran
`bash src/train_check.sh` on the login node. **The result is unknown** - ask
them to paste the log before doing anything else, and do not assume it passed.
Two bugs were already found by running it:

1. `SurfaceStore.write` called `np.savez` with a path not ending in `.npz`;
   numpy appends the extension, so the write-then-rename always failed. Fixed
   in `0824b70`. This would have broken `cache_surfaces.sbatch` and every
   `--surface_cache` run too, not just the self-test.
2. The self-test fixture was too marginal: protein 0's ball precision was
   0.298 against `pos_prec` 0.3, so a guard assertion would have failed, and
   6% of proteins could never form a positive at all - they would have trained
   on the fallback forever while every `site` assertion passed and measured
   nothing. Retuned to real geometry (pocket 2.7% of the surface against the
   real 2.8%, ball precision 0.65 against the sweep's 0.63) in the same commit.

**First log seen (2026-10-18, on `0824b70`): stages 1-4 PASS; stage 5 was
still compiling KeOps CPU kernels when the log was pasted, so 5-7 are
unknown.** What stages 1-4 showed, fixed in the commit after this note:
- Stage 4: 173 of 8,378 ligands had graph atoms != heavy atoms. `RemoveHs`
  keeps isotopic H (`[2H]`, deuterated ligands), H2 and hydrides, so those H
  were graph nodes and ECFP atoms. Now `RemoveAllHs`; an all-H molecule is
  flagged failed. **Any `ligand_feats.npz` built before this (`probe/`,
  `train_check/`) is stale; `runs/v1/ligand_feats.npz` must be built after.**
- Cosmetic: a requires_grad scalar warning in the selftest print, numpy-2
  `np.int64(1)` reprs in `prepare`'s group sizes and `np.str_` in a warning.
- Not a bug: the selftest's val MRR peaks at step 60 (0.983) and slides to
  0.854 by 180 while the loss keeps falling - overfitting 80 toy items. The
  real run keeps `best.pt` by val MRR for exactly this reason.
- Expected: the head checkpoint loads 117/118 tensors; `project.0.weight` is
  left at init because the check pools with `shells` (48 dims) and the head
  was saved with `mean` (16). Under `train.sbatch`'s default `mean` it fits.

**Second log (same commit): stage 5 PASSED** - the real encoder ran end to
end on CPU: step 0 eval at 698 s (KeOps CPU compile + 4 val surfaces), step 2
at 2,912 s (backward kernels compile + 4 train surfaces + 2 steps), loss 1.422
/ nce 0.717 / pock 0.367 finite, pocket AUROC 0.974 / AP 0.50, fb 0.00,
cosdiff 0.946 -> 0.961 (2 steps on 4 val proteins: no signal, a shape check).
best.pt / last.pt written. The log read `22912.301` because the `sec` column
ran into `step` once past 1,000 s - fixed (integer seconds); it would have hit
every line of a real run after 17 minutes. `nan` loss on an eval row with no
training steps since the start or a resume is expected, not a failure.
Stage 6 resumed at step 2 with its own split, val surfaces from the disk
cache (so the cache write fix works), and was running steps 2 -> 4 on CPU when
pasted: slow, not stalled. Stages 6 (advance) and 7 still unreported.

Stage 6 then made no visible progress (the user called it stalled) and was
abandoned; stages 6-7 never reported. The code has no loop that a resume could
spin in (an unusable batch `continue`s but the step still advances), so the
likeliest cause is login-node CPU throttling, but that is unproven. Instead
`train.sbatch`'s smoke now RESUMES itself 30 -> 40 steps and checks last.pt
reached 40, both passes under `timeout` (SMOKE_MIN, 60 / 20 min), and the main
run's --max_hours is reduced by the time the checks used. `eval` (stage 7) is
still untested; train.sbatch does not call it.

**First GPU job (alphagpu52, sm_120 OK): smoke died at the step-0
evaluation** - `evaluate` indexed the CPU label tensor from `collect` with the
device-side `keep` mask (`indices should be either on cpu or on the same
device`). CPU-only checks cannot see this class of bug. Fixed. `collect` (the
full GPU forward, KeOps CUDA kernels, region selection, fallback) had already
run on all 20 val proteins before the crash; the training step had not.

That fix (`ac72d67`) was the last one needed: the next `train.sbatch` ran
clean end to end - see "First training run" below.

Still **never run on a GPU**: `train.py eval`, `src/slurm/cache_surfaces.sbatch`
as a job (the surfaces were all on disk by the main run - "0 sampled"), and the
non-default `pool_mode shells` / `mean_std` and `anchor_norm batch` paths.

## First training run (`runs/v1`, train.sbatch on `ac72d67`, alphagpu52)

Defaults: arm gnn, pool mean, anchor_norm none, batch 8, lr 3e-4 constant (no
schedule), lam_pocket 1, lam_site 0, ball R8 shift 2, head
`head_pretrained_300.pt` (118/118 tensors). 18,639 train / 400 val rows
(2,315 / 112 target groups). Smoke (64 items, 30 steps) and its resume
30 -> 40 both passed, so **resume is now proven on a GPU**. The main run hit
`--max_hours 10.96` at **step 15,642 of 20,000**; resubmitting finishes it
(~3 h). ~2.5 s/step including evaluation; the step-0 eval took 98 s.

| step | nce | pock | mrr | chance | top1 | top10 | site | AUROC | AP | fb | cosdiff |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | - | - | 0.018 | 0.017 | 0.002 | 0.025 | 0.45 | 0.960 | 0.572 | 0.062 | 0.986 |
| 2,000 | 1.84 | 0.57 | 0.044 | 0.017 | 0.013 | 0.083 | 0.65 | 0.962 | 0.578 | 0.058 | 0.951 |
| 5,000 | 1.56 | 0.56 | 0.060 | 0.017 | 0.020 | 0.102 | 0.69 | 0.963 | 0.582 | 0.065 | 0.884 |
| 10,000 | 1.22 | 0.55 | 0.082 | 0.017 | 0.040 | 0.147 | 0.69 | 0.964 | 0.587 | 0.068 | 0.788 |
| 11,500 (best) | 1.13 | 0.55 | 0.098 | 0.017 | 0.055 | 0.180 | 0.68 | 0.964 | 0.592 | 0.062 | 0.785 |
| 15,000 | 1.04 | 0.54 | 0.095 | 0.017 | 0.050 | 0.168 | 0.69 | 0.964 | 0.580 | 0.062 | 0.677 |
| 15,500 | 0.96 | 0.54 | 0.092 | 0.017 | 0.043 | 0.185 | 0.69 | 0.964 | 0.592 | 0.062 | 0.711 |

Full curve: `runs/v1/history.json`.

What it says:
- **cosdiff moved: 0.986 -> ~0.70, still falling.** The collapse the whole
  loop exists to break is breaking with NEITHER lever (`anchor_norm`,
  `shells`) switched on. Different proteins' top anchors are no longer one
  vector.
- **Retrieval is real but small: MRR 0.018 -> ~0.09 (5-6x chance), top10
  0.025 -> ~0.18.** Rising steeply to ~9k steps, then roughly flat at
  0.08-0.10 while train nce keeps falling (1.27 -> 0.96; ln 8 = 2.08 is in-batch
  chance). Uniform-over-targets sampling over 2,315 groups at 8 per step is
  ~54 visits per group by now, so the widening gap is most likely
  memorisation of training targets. Not proven: val is noisy (below).
- **`site` 0.45 -> ~0.69 without `lam_site`.** The ligand's best anchor
  increasingly IS the pocket one; InfoNCE alone is fixing part of the ranking
  weakness.
- **The pocket head survived**: AUROC 0.960 -> 0.964, AP 0.57 -> 0.59, pocket
  BCE falling. Contrastive gradients did not wreck it. (0.960 is on this val
  set, not encoder_test's 64 receptors, so not directly 0.937's comparator.)
- **fb ~0.06, creeping to ~0.07.** The loop is not leaning on labels.
- **Val is noisy and lopsided.** Step-to-step MRR moves +-0.015 and top1 0.05
  is 20 rows, so `best.pt` at 11,500 is partly a noise peak. Worse,
  `--val_max` draws 400 rows UNIFORMLY over rows: the ~2,070-row group-disjoint
  val pool spans ~283 groups, the 400 drawn cover only 112, so a few big
  targets dominate the metric while training samples uniformly over targets.
  A group-balanced val draw would match the training objective and cut the
  noise.
- **Not yet known: how much of the MRR is the protein encoder training.** Step
  0 here is NOT the frozen-anchor probe (its ligand encoder is untrained), and
  the probe's "~7x chance" was on a different val set, where chance differs, so
  the two ratios cannot be compared. The control is this exact loop with the
  surface encoder frozen, on this split.
- RAM cache: 512 train surfaces resident against 19,293, so 88% of training
  loads hit disk. ~21 GB would hold them all under the 96 GB job; whether disk
  is the bottleneck in 2.5 s/step is unmeasured.

## Second round (jobs 139912 frozen / 139913 shells / 139914 b32)

All three on the SAME group-balanced 400 val rows (`--val_balance`, 283
groups), all from `head_pretrained_300.pt`, all ~10.95 h. Compared at MATCHED
DATA - 76k-92k proteins seen, i.e. batch-8 steps 9,500-11,500 against b32
steps 2,375-2,875 - as the mean of the 5 evaluations in that window, because
one evaluation moves +-0.015-0.02:

| run | proteins/s | mrr | top1 | top10 | site | AUROC / AP | fb | cosdiff |
|---|---|---|---|---|---|---|---|---|
| step 0 (any) | - | 0.020 | 0.003 | 0.03 | 0.44-0.52 | 0.919 / 0.464 | 0.182 | 0.97 |
| frozen, mean, b8 | 2.9 | 0.077 | 0.028 | 0.17 | ~0.54 | 0.919 / 0.464 | 0.182 | 0.976 |
| shells, b8 | 2.5 | 0.113 | 0.045 | 0.23 | ~0.64 | 0.927 / 0.52 | ~0.175 | ~0.68 |
| mean, **b32** | 2.3 | **0.127** | **0.067** | 0.23 | ~0.54 | 0.924 / 0.50 | ~0.175 | ~0.85 |

(chance 0.017.) Ends: frozen 14,554 steps, best 0.091; shells 12,547 steps,
best 0.124; b32 2,875 of 5,000 steps, best 0.142 at 2,500.

What it says:
- **Training the protein side helps, but it is not most of the signal.**
  Frozen encoder + trained projection and ligand reaches ~0.08; the trained
  runs reach 0.11-0.13 on the same data. Roughly 60% of the retrieval is
  already in the frozen pocket features. `--freeze_encoder` worked as meant:
  AUROC, fb and cosdiff are constant to 3 decimals for 14.5k steps.
  cosdiff is measured on the POOLED vectors, before the projection
  (`evaluate` -> `anchor_spread(pooled)`), which is why it cannot move here.
- **Batch 32 is the best arm and still climbing fastest** (0.087 at 1,250 ->
  ~0.13 at 2,875 steps). Per protein it costs about the same wall clock as
  batch 8, so more negatives are nearly free. It also has the best top1.
- **cosdiff is not the whole story.** b32 beats shells on MRR with much LESS
  anchor spread (0.85 vs 0.68). Spread is necessary, not sufficient; stop
  treating it as THE number and read mrr/top10 at matched data.
- **shells vs mean pooling is not yet separable**: the only mean/b8 run is v1,
  whose curve is on the old val set. v1's `best.pt` is at step 11,500 = 92k
  proteins, exactly this window, so `eval.sbatch`'s number for it is the
  mean/b8 row (slightly flattered: it was chosen on a val set that overlaps
  this one).
- **The balanced val set is much harder for the pocket head**: step-0 AUROC
  0.919 / fb 0.18 here against 0.960 / 0.06 on v1's row-uniform set. v1's
  val was dominated by a few big, easy targets. Every earlier val number in
  this file was flattered by that.

## v2 (`src/slurm/archive/train_v2.sbatch`) - WRITTEN, NOT YET RUN

Four levers, all flags defaulting OFF so every earlier run reproduces, and all
switched on by `train_v2.sbatch` (a preset over `train.sbatch`; archived
2026-10-27, see `src/slurm/archive/README.md`):

1. **Negatives.** `--bank N`: `ProteinBank` keeps the POOLED (pre-projection)
   anchors of the last N training proteins, detached, and re-projects them
   with the current projection every step (pool_norm in eval mode), as extra
   PROTEIN candidates for each ligand. `--extra_ligands N`: N more training
   ligands per step (own RNG stream, group-balanced) as extra LIGAND
   candidates for each protein. Both masked by `cross_mask` (group / same
   ligand / Tanimoto, no diagonal - `mask_pairs` clears one whenever Q == C).
   `nce_extended` is InfoNCE over the widened rows; with neither extra it is
   info_nce exactly (selftest [8]). Bank not checkpointed; refills in N/batch
   steps. v2: bank 2048, extra 1024.
2. **Throughput.** `--gpu_neighbors` builds region selection's radius
   neighbour lists on the GPU (`anchors.NEIGHBOR_DEVICE`); `select_regions`
   moves the surface to the CPU, so that was an O(N^2) CPU cdist on ~30k points
   every RAM-cache miss (88% of loads in v1). CPU path untouched. Plus a
   per-phase timing line after every evaluation (load / fwd / select+pool /
   nbr lists / lig+loss / bwd, lr, RSS peak, GPU peak) - THE number to read
   first from the v2 log, since every bottleneck claim so far is a guess.
   v2: SURFACE_RAM 4000, LABEL_RAM 20000.
3. **Capacity.** `--desc_dims D` (multiple of 8): a second, fresh dMaSIF conv
   over the same inputs, concatenated to the pretrained 16-dim embedding before
   pooling (`out["pool_emb"]`; the fallback anchor pools it too). The
   pretrained conv + pocket head still place the regions. v2: 32.
4. **Schedule.** `--lr_schedule cosine --warmup W --min_lr_frac`; `lr_at` is a
   function of the step alone, so resumes need no state. v2: cosine over 6000,
   warmup 200.

Bar: runs/b32's 0.127 MRR at 76k-92k proteins seen on the balanced val set;
appreciable = ~0.20. Leave-one-out ablations are in the sbatch header.
`train.py selftest` PASSED on the login node with [8] (2026-10-19).
Untested on a GPU: everything above. `train.py selftest` [8] covers the
loss, masks, bank and schedule on CPU; the smoke inside the job covers the
descriptor conv (new KeOps kernels: expect a slower first smoke).

## v3 (`src/slurm/train_v3.sbatch`) - WRITTEN, NOT YET RUN

The stack, per training row: the WHOLE receptor surface -> dMaSIF conv (with
`--chemo`) -> fpocket selects the pocket at this row's ligand -> the point
embeddings inside it are pooled PAIRWISE -> projection -> InfoNCE against the
ligand GNN. One anchor per protein; the score is a plain cosine.

Why pairwise: specificity is a handful of sites (H-bond donors / acceptors,
charges, aromatic and hydrophobic patches) in a particular ARRANGEMENT,
matched by complementary ligand groups. `mean` keeps composition, `shells` one
radial order; neither says "donor 7 A from acceptor".

Pieces:
- `--pocket_select site` (`external_pockets.site_pockets` / `site_region`,
  `PocketSource.site`, resolved per ROW in `train.build_batch`, since two
  ligands on one receptor are two sites): the union of the fpocket pockets the
  ligand occupies - >= 0.25 of a pocket's alpha-sphere centres within 3 A of a
  ligand heavy atom, or >= 0.5 of the ligand's atoms within 3 A of the
  pocket's centres. Uses ligand COORDINATES and spheres only, never the
  surface label. On 1ATP: the ATP pocket 0.97 of its centres, the next 0.00.
  `anchor_block` labels that region positive by construction (a merged site can
  fail the 4 A label's precision rule); a row where fpocket has no pocket at
  the site trains on the fallback ball, so `fb` = the site-miss rate.
  `--pocket_select topk` (default) is the blind setting; a site-trained
  checkpoint is evaluated blind with `train.py eval --pocket_select topk`.
- `--pool_mode pairs` (`src/pair_pool.py`, `PairPool`, owned by the encoder as
  `pair_pool`): per point a learned soft site type a_i (T = 8 + "not a site")
  and salience s_i from [conv embedding || raw input channels]
  (`--no_pair_feats` drops the latter); m_i = s_i a_i; output =
  [weighted mean embedding, log1p(type masses / 10),
  log1p(sum_ij m_i m_j^T K_b(d_ij) / 100) over the upper triangle of T x T and
  8 Gaussian distance bins 3-16 A, pairs under 2 A excluded (one atom's own
  surface-point cluster)] = 16 + 8 + 36 x 8 = 312 dims. Rigid-motion and
  permutation invariant; gradient through the types, salience and mean; the
  n x n x B bin tensor is recomputed in backward (checkpoint), and regions over
  4096 points are thinned by distance from the centroid. `--pair_types`,
  `--pair_bins`, `--pair_rmin/rmax`, `--pair_exclude`. `surface_encoder.
  pool_region_vectors` is now the one place a region becomes a vector (the
  selector's anchors and the fallback anchor both).
- `--freeze_encoder` keeps `pair_pool.` trainable (fresh weights).
- `load_head` now zero-pads the input-side tensors when the encoder has more
  input channels than the head (`--chemo` against `head_pretrained_300.pt`), as
  `load_pretrained` already did. Before this, `--chemo --head_ckpt` would have
  left those three tensors RANDOM under a pretrained conv.
- `train.sbatch`: `CHEMO=0` used to switch chemo ON (`${CHEMO:+--chemo}`); now
  `on()` like FREEZE.
- `external_pockets.py score` prints a second table, the site pocket per row:
  `found` (= the rows v3 can train on), size, iou / cover / prec against the
  4 A label, dcc, merged; also written to `<out>.site.tsv`.

Tests: `external_pockets.py selftest` [5] (site rule: single pocket, none,
union of two, lig_frac, PocketSource.site) PASSES locally. `pair_pool.py`'s
own self-test (shape, rotation / translation / permutation invariance, same
composition + different arrangement -> same mean / mass but different pairs in
the right bins, gradient through the checkpointed pair block, thinning) and
`train.py selftest` [10] (build_batch picks exactly the true pocket per row
and nothing for a pocketless row; anchor_block gives one positive usable
anchor each and the fallback for the pocketless row; gradient reaches the
pair heads) **PASSED on the login node (2026-10-22)**, as did groups [1]-[9]
and `external_pockets.py selftest`: pair_pool rigid-motion max diff 1.2e-07,
thinned-region pair-block cosine 1.000; the planted loop still peaks at MRR
0.983. The stub encoder routes only external / pairs configs through the real
`select_and_pool`; groups [1]-[8] are unchanged. Still never run: anything on
a GPU, real fpocket output at scale, `--chemo` with the inflating `load_head`.

Run order:
```
bash src/slurm/build_fpocket.sh                      # once
EX="apptainer exec --cleanenv --env OMP_NUM_THREADS=8 $WORK/dmasif_sandbox python"
$EX src/pair_pool.py && $EX src/train.py selftest    # [9], [10] new
sbatch src/slurm/fpocket.sbatch
python src/external_pockets.py score --items runs/v1/items.tsv \
    --surface_cache surfaces --pockets pockets/fpocket --n 1000 \
    --out pockets/fpocket/score.tsv               # read the site table: found
sbatch src/slurm/train_v3.sbatch
CKPTS="$WORK/runs/v3/best.pt" \
    EVAL_ARGS="--use_ckpt_split --pocket_select topk --pocket_rank drug" \
    sbatch src/slurm/eval.sbatch                     # the same model, blind
```
**fpocket run + score (2026-10-22): DONE.** 19,293 pocket files = every
surface, 0 failures, median 20 pockets per receptor, none empty.
`external_pockets.py score` on 1,000 of the 20,711 rows:

| site pocket (v3 training) | found | size | iou | cover | prec | dcc | merged |
|---|---|---|---|---|---|---|---|
| pocket m=-1 | 0.96 | 230 | 0.43 | 0.55 | 0.71 | 0.74 | 0.22 |
| **pocket m=0 (v3 default)** | 0.96 | 435 | **0.56** | **0.90** | 0.62 | 0.74 | 0.22 |
| pocket m=1 | 0.96 | 541 | 0.51 | 0.95 | 0.53 | 0.73 | 0.22 |
| ball m=0 | 0.96 | 298 | 0.52 | 0.75 | 0.62 | 0.75 | 0.22 |

| blind top 8 | hit | top1 | dcc1 | dccA | iou | size |
|---|---|---|---|---|---|---|
| pocket, rank score, m=0 | 0.65 | 0.35 | 0.28 | 0.51 | 0.41 | 172 |
| **pocket, rank drug, m=0** | **0.78** | **0.54** | 0.41 | 0.59 | 0.48 | 199 |
| ball, rank drug, m=-1 | 0.77 | 0.52 | 0.42 | 0.61 | 0.45 | 261 |
| ball_R8 on the head's map (held-out 64, earlier) | 0.84 | 0.64 | 0.44 | 0.77 | 0.53 | ~290 |

Read: (1) known-site training is well supplied - fpocket has a pocket at the
ligand for 96% of rows (so expect fb ~0.04), and the default shape/margin
(pocket, 0) is the best setting: iou 0.56 and cover 0.90 against the 4 A label,
better than ball_R8's 0.53 / 0.70; prec 0.62 means ~40% rim, which the pair
pooling's salience can down-weight. A fifth of sites are merged from several
fpocket pockets. (2) Blind, rank by DRUGGABILITY, not fpocket's score (hit
0.78 vs 0.65); even so it trails the head's ball_R8 (0.84 / dccA 0.77),
though not on the same proteins and dcc uses plain centroids of irregular
pockets here, so dcc is harsher for fpocket. The blind eval therefore passes
`--pocket_rank drug`; --k_pos 16 is the obvious next lever if blind lags.

**runs/v3 (2026-10-23): FAILED TO LEARN, cancelled after step 4,250/6,000
(its last.pt is at 4,250).**
The plumbing worked - site found for 94% of rows (fb 0.058 as predicted), 0
missing files, GPU peak 7.7 GB, RSS 5.2 GB, ~4.1 s per 32-protein step (~3x
b32's throughput; load 0.5 / fwd 1.6 / bwd 2.0). But val MRR stayed at chance
(0.015-0.026 vs 0.017, top10 ~0.025) the whole way, train nce flat at
6.1-7.0 against ~7.3 chance for 32 + 2,048 bank / 32 + 1,024 extra candidates
(so not overfitting: not fitting), and pooled cosdiff ROSE 0.83 -> 0.98-0.999
(v1's fell 0.99 -> 0.70). Leading hypothesis: pair pooling's scale - 288 of
312 pooled dims are all-positive log pair counts that grow with pocket size,
and with a near-uniform type head at init all 36 type pairs carry the same
histogram, so every pocket points the same way, with anchor_norm none.
Unproven: v3 changed four things at once against b32 (site pockets, pairs,
chemo, bank / extra ligands / cosine, the last never run on a GPU).

Diagnostic ladder, `bash src/slurm/archive/v3_diag.sh` (three 1,500-step runs, one
change each on b32's setup - batch 32, const lr, no chemo / bank / extras, the
same val rows): A `v3a_site_mean` (site pockets, mean pool), B
`v3b_site_pairs` (+ pairs), C `v3c_site_pairs_bn` (+ anchor_norm batch). Read
at step 1,250 against b32's 0.087; how to read each outcome is in the script.

**Results (jobs 145246-8, 2026-10-23): ALL THREE TRAIN.** Val MRR (chance
0.017), means over evaluation windows because one evaluation moves +-0.015:

| run | mrr 500-875 | mrr 1000-1500 | top10 1000-1500 | best | nce @1500 | cosdiff |
|---|---|---|---|---|---|---|
| A site + mean | 0.076 | 0.096 | 0.19 | 0.109 | 2.23 | 0.94 -> 0.80 |
| B + pairs | 0.078 | 0.103 | 0.21 | 0.126 | 2.24 | 0.87 -> 0.74 |
| C + pairs + BN | 0.083 | **0.114** | **0.22** | 0.127 | 2.29 | 0.87 -> 0.82 (pre-BN) |

Read: (1) neither fpocket site pockets nor pair pooling broke v3 - the
pair-pooling-scale hypothesis was WRONG (B learns un-normalised). The culprit
is among what v3 added: chemo, bank + extra ligands, cosine schedule. (2) C >=
B >= A in every window and on top10: pairs + BN lead mean pooling by ~0.02
MRR, about the noise, so suggestive, not proven - the first evidence that
spatial arrangement helps. (3) All still rising at 1,500 steps. (4) At matched
steps b32 (blind, head regions) was 0.087 at 1,250: A matches it (0.088), C is
ahead (0.115); known-site + mean pooling does NOT beat the head's top-8 max.
~2.5 s/step (A), ~3.3 s/step (B, C).

Round 2, `bash src/slurm/archive/v3_diag2.sh`, each one change from C: `v3d` + chemo,
`v3e` + bank 2048 / 1024 extra ligands (judge by mrr / cosdiff, its nce is over
~2,000 candidates), `v3c+` C resumed 1,500 -> 5,000 steps (where it
plateaus). Results: ALL THREE DONE (below). **`v3e` REPRODUCES v3's failure:
the bank + extra ligands is the culprit.**

**`v3e` = C + bank 2048 / 1024 extra ligands (2026-10-27, 1,500 steps): DEAD
AT CHANCE, and it rules out the temperature hypothesis that `--tau_min` was
built for.**

| step | mrr | nce | cosdiff | tau | AUROC / AP | pock |
|---|---|---|---|---|---|---|
| 0 | 0.017 | - | 0.867 | 0.070 | 0.919 / 0.464 | - |
| 125 | 0.018 | 6.918 | **0.979** | 0.069 | 0.919 / 0.486 | 0.631 |
| 750 | 0.026 | 7.310 | 0.982 | 0.071 | 0.926 / 0.502 | 0.552 |
| 1,500 | 0.015 | 7.297 | 0.986 | **0.073** | 0.928 / 0.525 | 0.533 |

mrr over 1000-1500 means 0.019 against chance 0.017; best 0.0257. Read:

1. **The contrastive loss learns NOTHING, exactly.** Symmetric InfoNCE over
   32 + 2048 protein candidates and 32 + 1024 ligand candidates has chance
   (ln 2080 + ln 1056) / 2 = **7.3012**, and nce sits at 7.297-7.314 from step
   250 to the end. Not "high": precisely its random value. (Step 125's 6.918
   is the mean over the bank's 64-step fill.)
2. **Collapse inside 125 steps**: cosdiff 0.867 -> 0.979 by the first
   evaluation, then 0.986. Every protein's anchor points the same way, so
   every score is the same and there is no gradient to escape with. v3 showed
   the same 0.83 -> 0.98-0.999.
3. **Not a plumbing bug.** The pocket BCE falls 0.631 -> 0.533 and AUROC / AP
   rise 0.919 / 0.464 -> 0.928 / 0.525, so gradients flow and the encoder
   trains - through the head, not the contrastive term. fb 0.058, site found
   94%, 0 surfaces sampled: the data path is fine.
4. **The temperature hypothesis is WRONG.** tau goes 0.070 -> 0.073 - it does
   not sharpen at all, it drifts UP while the model sits at chance. So the
   "low tau concentrates the gradient on false negatives" story does not
   describe this failure, and v2/v3's eventual 0.036 / 0.043 must come later
   or from their other changes. **`v3g` (C + bank + `--tau_min 0.06`) is
   therefore a NULL experiment as designed**: its floor is BELOW the 0.069-
   0.073 tau actually reaches here, so it never binds and v3g should just
   reproduce v3e.

Leading hypothesis now, UNTESTED: the bank's anchors are detached
(`ProteinBank.anchors` runs under `no_grad`), so on the ligand -> protein side
2,048 of 2,080 candidates push only the LIGAND encoder, away from 2,048 fixed
points sampled from the whole protein manifold, while the protein side still
sees only 31 negatives. A 64x one-sided repulsion with one positive to balance
it. Second suspect: `pool_norm` runs in EVAL mode for the bank (running
statistics) and TRAIN mode for the live batch, and the two disagree most
during the first ~100 steps, which is exactly when the collapse happens.

Cheap tests if the bank is ever wanted (none run): `BANK=2048 EXTRA_LIG=` and
`BANK= EXTRA_LIG=1024` separately, to see which half does it; and
`BANK=2048 ANCHOR_NORM=none` for the BatchNorm interaction.

**But the strategic reading is to drop the bank.** It exists to supply more
negatives, and `v3c+` showed C's ceiling is MEMORISATION (val mrr flat while
train nce falls, ~69 visits per target group), not a shortage of negatives.
More distinct TARGETS is the lever; more negatives is a solution to a problem
we do not have. C without the bank stands as the best configuration.

**`v3d` = C + chemo (2026-10-27, 1,500 steps): CHEMO IS NOT THE CULPRIT, AND
CHEMO DOES NOT HELP AT SCALE.** Window means against C on the same 400 rows:

| run | mrr 500-875 | mrr 1000-1500 | top10 1000-1500 | best | nce @1500 | cosdiff |
|---|---|---|---|---|---|---|
| C (pairs + BN) | 0.083 | **0.114** | 0.22 | 0.127 | 2.29 | 0.87 -> 0.82 |
| v3d (+ chemo) | 0.084 | 0.102 | 0.21 | 0.116 | 2.27 | 0.83 -> 0.81 |

Read: (1) v3d TRAINS, tracking C step for step (identical at 500-875,
0.012 below at 1000-1500, which is inside the +-0.015 noise). So chemo did
not flatten v3 either, and **the only suspects left are the bank + extra
ligands (`v3e`) and the cosine schedule.** (2) This is also the at-scale
chemo re-test that "Next steps" item 5 asked for: 28 channels against 16 on
18,639 rows / 2,315 groups buys NOTHING, and is a touch worse on mrr and
top10. Taken with the 32-protein ablation, chemo is now negative twice, at
both scales. Keep `cfg.chemo` off. Caveat: 1,500 steps is the LEARNING phase
(C plateaus ~3,600), so whether chemo changes the plateau is untested - and
not worth a GPU day on a channel set that shows nothing while the curve is
still steep. (3) `--chemo` with `--head_ckpt` is now proven on a GPU: the log
shows the three input tensors inflated 16 -> 28 ("pretrained function
preserved at init") and 115/118 head tensors loaded, the 3 skipped being the
projection (312 pooled dims against the head's 16). Before `dbb7c7a` those
inflated tensors would have been left RANDOM under a pretrained conv. (4) tau
flat at 0.070-0.071 the whole run, as for every other bank-free run.

**`v3c+` (C resumed 1,500 -> 5,000 steps, 2026-10-27): C PLATEAUS AT ~0.18
KNOWN-SITE MRR, and the limit is now MEMORISATION, not optimisation.**

| step | mrr | top1 | top10 | nce | cosdiff | tau |
|---|---|---|---|---|---|---|
| 1,500 (C) | 0.114 | - | 0.22 | 2.29 | 0.82 | 0.071 |
| 3,625 | 0.173 | 0.100 | 0.322 | 1.470 | 0.760 | 0.065 |
| 4,875 (best) | 0.184 | 0.102 | 0.317 | 1.224 | 0.747 | 0.060 |
| 5,000 | 0.172 | 0.097 | 0.340 | 1.148 | 0.743 | 0.059 |

(Three of the ~12 evaluations in 3,625-5,000; read the window mean off
`history.json`.) Read: (1) 0.114 -> ~0.176 between 1,500 and 3,625, then FLAT
within noise (+-0.015) over the next 1,375 steps, while train nce keeps
falling 1.47 -> 1.15. That is v1's divergence again - at 32 proteins/step over
2,315 groups, step 5,000 is ~69 visits per group. More steps will not help;
more distinct TARGETS will. (2) So do not extend C further, and expect the
1,500-step ladder to rank runs by how fast they start, not by where they end.
(3) cosdiff still creeping down (0.76 -> 0.74): the anchors keep spreading.
(4) tau drifts to 0.059 with NO bank, so v3g's 0.06 floor - chosen above the
bank runs' 0.036 / 0.043 - would start to bind on a run this long. It is still
above C at 1,500 (0.071), which is where v3g is read. (5) STILL NOT comparable
to b32's 0.127 (blind, head regions); the blind eval is. (6) ~4.2 s/step, of
which bwd 2.09 and select+pool 1.18 of the 1.45 forward - pair pooling
dominates both directions. GPU peak 7.6 GB, RSS 5.2 GB, 0 surfaces sampled,
69% RAM hits. `best.pt` is step 4,875.

**Round 3 (2026-10-26), `bash src/slurm/archive/v3_diag3.sh`: WRITTEN, NOT RUN.
`train.py selftest` group [11] is new and has not run yet (no torch in
Claude's container) - run it first.** Motivation: the bank's ~2,000 candidates
per row pass the masks (UniProt group, same ligand, Tanimoto >= 0.8) for
homologues under other UniProt IDs, cofactor binders and scaffold hops, so
false negatives likely grew from ~0.3 per query at batch 32 to tens. Two
levers, both off by default (earlier runs and resumes reproduce):
- `--n_decoys N` (site select only; `train.site_decoys`): up to N of the
  receptor's other mapped fpocket pockets, minus any overlapping the site
  region, its 4 A label or a sibling site (the region_labels 0.1 rule). Label
  0, never usable (not in cross-protein scores, the bank or eval ranking);
  `--lam_site` (`site_loss`, first use) trains the site against them. Same
  protein, so never a homologue. If fpocket misses the site, slot 0 is a
  decoy and the fallback ball is the positive (labs carry n_site / n_decoy).
  `train_v3.sbatch`: `DECOYS=7 POCKET_RANK=drug`; LAM_SITE defaults to 1 and
  reaches the smoke too.
- `--tau_min` (default 0.01, the old cap): logit scale capped at 1 / tau_min.
  `TAU_MIN=0.05` in train.sbatch.
Log changes from this commit: a `tau` column (also history.json, `train.py
eval`); in site mode `site` = share of val rows whose ligand scores its own
pocket above every decoy (NaN without decoys; it read ~1.0 by construction
before). Runs: `v3f` = C + decoys, `v3g` = C + bank + TAU_MIN 0.05 (read
against v3e). Final temperature of runs that predate the column:
```
apptainer exec --cleanenv $WORK/dmasif_sandbox python -c '
import sys, torch
for p in sys.argv[1:]:
    ck = torch.load(p, map_location="cpu", weights_only=False)
    s = min(float(ck["state_dict"]["logit_scale"].exp()), 100.0)
    print(p, "step", ck["step"], "tau %.4f" % (1 / s))
' $WORK/runs/*/last.pt
```
Baselines and the blind comparison (C's decoy `site` rate, then C vs v3f
with topk / drug) are in the script's header.

**C's two baseline evals (2026-10-27, `runs/v3c_1500/best.pt`, which is step
1,125 - C's best val MRR inside its first 1,500 steps, 0.127, a noise peak
over the 0.114 window mean). Same 400 balanced val rows both times:**

| eval | mrr | top1 | top10 | site | fb | cosdiff |
|---|---|---|---|---|---|---|
| known site + 7 decoys | 0.127 | 0.063 | 0.240 | **0.605** | 0.058 | 0.822 |
| **blind (topk, rank drug)** | **0.088** | 0.040 | 0.163 | 0.543 | **0.230** | 0.796 |

Read: (1) **The decoy metric works and C is already well above chance on it
without ever training on decoys**: site 0.605 against ~1/8 = 0.125 for a coin
flip over 7 decoys, site_cov 1.00 (every val row got at least one decoy). That
is the bar v3f has to clear. (2) The known-site eval reproduces C's own
0.127 EXACTLY, which is the check that decoys stay out of the ranking - they
are label 0 and never `usable`, so MRR cannot see them. (3) **Blind costs ~31%
of MRR (0.127 -> 0.088) and quadruples the fallback rate (0.058 -> 0.230)**:
the top 8 fpocket pockets by druggability simply do not contain the site for
~23% of rows, as `score` predicted (hit 0.78). (4) **The headline: blind, C
at ~1,125 steps MATCHES b32 and does not beat it** - 0.088 against b32's
0.087 at 1,250 steps, same batch 32, so matched data. C's known-site lead over
b32 (0.115 vs 0.087 at 1,250) was the known SITE, not a better
representation. (5) So the open question is whether the PLATEAUED C wins:
known-site it went 0.127 -> 0.184 (+45%) by step 4,875, and nothing has
scored that checkpoint blind yet. That eval is the cheap, decisive number:
```
CKPTS="$WORK/runs/v3c_site_pairs_bn/best.pt" \
EVAL_ARGS="--use_ckpt_split --pocket_select topk --pocket_rank drug" \
    sbatch src/slurm/eval.sbatch
```
Note b32's own plateau is also unknown (it stopped at 2,875 of 5,000 steps,
best 0.142 at 2,500), so a plateau-vs-plateau comparison needs b32 finished
too. (6) `site` 0.543 in the blind eval is the ORIGINAL selector-style metric
(argmax over the scored anchors), not the decoy rate - the two are not
comparable, and site_rate picks by `--pocket_select`.

**`v3f` = C + 7 site decoys + lam_site 1 (2026-10-27, 1,500 steps): THE DECOY
LOSS WORKS AT ITS OWN JOB AND COSTS THE MAIN ONE.** Smoke passed including
the resume, so the decoy path is GPU-proven.

| run | mrr 500-875 | mrr 1000-1500 | top10 | site | cosdiff | s/step |
|---|---|---|---|---|---|---|
| C | 0.083 | **0.114** | 0.22 | 0.605 (eval, untrained) | 0.82 | 4.2 |
| v3f | 0.076 | 0.093 | 0.195 | **0.794** | **0.78** | 6.3 |

Read: (1) **Within-protein discrimination rose a lot: site 0.605 -> 0.794**
(chance ~1/8), so `lam_site` on the decoys does exactly what it was built to
do - the ligand now scores its own pocket over the receptor's others. (2) **But
cross-protein retrieval fell, mrr 0.114 -> 0.093**, 0.021 down and just
outside the +-0.015 noise, with top10 0.22 -> 0.195 and train nce 2.29 ->
2.52. One shared encoder is now serving two objectives and the cross-protein
one lost. (3) cosdiff is BETTER (0.782 against 0.820): the decoys spread the
anchors more. (4) **The decisive test is still the blind eval**, because
picking the right pocket among the receptor's others IS what blind screening
needs - and that is the skill that improved. (5) Cost: 16 regions pooled per
protein (1 site + 7 decoys + 8 k_neg patches) against C's 9, so select+pool
1.15 -> 2.65 s and ~6.3 s/step against 4.2.

**`v3g` = v3e + `--tau_min 0.06` (1,500 steps): NULL, exactly as predicted.**
tau goes 0.069 -> 0.073 and never reaches the floor, so the floor never binds,
and every number matches v3e to the third decimal (nce 7.297 both, cosdiff
0.985 / 0.986, mrr 0.017 / 0.015, AP 0.526 / 0.525, best 0.0259 / 0.0257).
**So the bank's collapse has nothing to do with the temperature**, the
hypothesis `--tau_min` was built for is dead, and the flag survives only as a
cheap guard worth leaving at its 0.01 default. The detached-bank /
BatchNorm-mode suspects in the v3e entry are untouched by this.

**`--k_neg` is now a real throughput lever.** `pool_region_vectors` runs on
EVERY region, and in site mode only the site is ever scored across proteins:
the 8 k_neg patches are label -1 (so `site_loss` ignores them), never
`usable`, and exist solely to feed the BatchNorm's statistics. With decoys
present they are redundant even for that - the 7 decoys already give
BatchNorm 8 anchors per protein. `--k_neg 0` is safe (`negative_regions`
returns [] for k <= 0) and would cut v3f from 16 regions to 8, recovering
most of its 1.5x slowdown; for C it cuts 9 -> 1, though that leaves BatchNorm
only 32 vectors per step, which is why `v3i` below measures it rather than
assuming. Throughput matters now precisely because `v3c+` says the ceiling is
target diversity, so more data per GPU-hour is the point.

Round 4, `bash src/slurm/v3_diag4.sh` (two runs) plus three cheap evals and
the retrieval dumps - the EVALS FIRST, they decide whether decoys stay:
```
# the decoy verdict, matched at 1,500 steps, and C's plateau scored blind
CKPTS="$WORK/runs/v3c_1500/best.pt $WORK/runs/v3f_pairs_bn_decoys/best.pt \
$WORK/runs/v3c_site_pairs_bn/best.pt" \
EVAL_ARGS="--use_ckpt_split --pocket_select topk --pocket_rank drug" \
    sbatch src/slurm/eval.sbatch
bash src/slurm/v3_diag4.sh
```

Final temperatures (2026-10-27, that readout; all started at 0.07, cap 0.01):

| run | step | tau |
|---|---|---|
| v1 | 20,000 | 0.068 |
| frozen | 14,554 | 0.076 |
| shells | 12,547 | 0.073 |
| b32 | 2,875 | 0.064 |
| v3a / v3b / v3c | 1,500 | 0.068 / 0.070 / 0.071 |
| **v2** (bank + extras + cosine + descriptor) | 4,884 | **0.036** |
| **v3** (bank + extras + cosine + chemo + site + pairs) | 4,250 | **0.043** |

No run reached the cap. Every run without a bank stayed within 0.064-0.076;
only the two with a bank sharpened, to about half the start. So the bank does
drive the temperature down, but not to an extreme, and v3g's floor is 0.06
(above both bank runs, below every run that learned) rather than 0.05, which
would barely have bound. `runs/v2` exists at 4,884 steps although the v2
section ABOVE says not run: its results are not recorded here and are the
key datum (bank on the head's regions, no site pockets or pairs) - get its
log / history.json. `runs/v3c_1500` is the copy of C at 1,500 steps; C itself
was still at 1,500, so v3c+ had not run when it was made. Not built: soft-weighted or
debiased InfoNCE, property-matched decoy ligands, a false-negative exposure
count.

v3's log is KNOWN-SITE retrieval and is not comparable to runs/b32 (blind,
head regions); the blind eval is. Ablations in the sbatch header (pool
mean / shells, no chemo, no bank, ball shape). v3 also switches on v2's bank
and extra ligands and the cosine schedule; if v2 has still not run on a GPU,
those are untested there too, and `BANK= EXTRA_LIG=` removes them.
