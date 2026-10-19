# Project status (handoff)

Read this first in a new session. Last updated 2026-10-18 on
`claude/model-training`, after the FIRST REAL TRAINING RUN (`runs/v1`, 15,642
steps, 10.96 h) came back. Its results are the "First training run" section
below; read that before anything else in START HERE, which is the history of
getting the loop onto a GPU.

## START HERE

**Branch `claude/model-training`. Commit and push there; do not open a PR
unless asked.**

Standing constraints, from the user, which have held all through this work:

- **The user runs every job on the HPC themselves, via sbatch or the login
  node. Claude writes code, commits and pushes; Claude does not run jobs.**
  Changes reach the HPC by `git pull` on this branch.
- **Claude's own container has no torch and must not install it.** Anything
  needing torch is for the user to run. Numpy, pandas and rdkit paths can be
  checked locally, and geometry arithmetic can and should be verified in numpy
  before asserting anything about it.
- Every experiment gets a script with a docstring saying what it measures and
  how to read it, an sbatch file, and its results recorded in this file.

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

## Goal

Contrastive protein-ligand retrieval. Protein side: dMaSIF surface features +
convolution (retrained), a per-point pocket head, and K pooled **region**
anchors per protein compared against ligand vectors.

Built and tested: the protein encoder up to the anchors, the region selectors,
the apo/holo/AlphaFold evaluation, the ligand encoder and the frozen-anchor
probe that ranks its arms. Built but **never run on a GPU: the contrastive
training loop** (`src/train.py`, `src/train_data.py`) and its data pipeline.
Its login-node check (`src/train_check.sh`) has been run once, with the outcome
still unseen - see START HERE above.

## How we work

- The user runs everything on the HPC (sbatch / login node). Claude writes
  code, commits and pushes; it does not run jobs. Changes reach the HPC by
  `git pull` on the working branch.
- Every experiment gets: a script with a docstring saying what it measures and
  how to read it, an sbatch file, and results recorded here.
- Prefer comparing methods on the SAME trained head / maps (dump once, sweep
  offline on CPU) over retraining per variant.

## Layout and environments

- Repo root = `$WORK` = `/mnt/home/cjs2301/dmasif` on the HPC. `dMaSIF/` is the
  vendored upstream (+ PyTorch 2 fixes), `src/` is ours. Data and outputs
  (`meta/ structures/ systems/ datasets/ feats/ keops_cache/ encoder_test/
  apo_test/`) are gitignored.
- CPU work: `~/venvs/plinder` (`envs/cpu-requirements.txt`; pandas, pyarrow,
  rdkit; **no torch, no Biopython**). GPU work: Apptainer container
  `$WORK/dmasif_sandbox` from `src/slurm/dmasif.def`: NGC PyTorch 25.06 (CUDA
  12.9, Python 3.12, torch 2.8, PyG 2.8, pykeops 2.2.3, numpy 1.26, Biopython).
  Single-GPU jobs land on the RTX PRO 6000 Blackwell pool (sm_120); the
  container is built for it.
- `$KEOPS_CACHE_FOLDER` and `$PLINDER_DIR` must be set in the submitting shell.
- Log noise that is harmless: `not a valid test operator` and the nvidia-smi
  bind-mount INFO lines (container entrypoint).

## Data

PLINDER 2024-06/v2. Local copy `$PLINDER_DIR = /mnt/home/cjs2301/plinder/2024-06/v2`:

| folder | contents |
|---|---|
| `index/annotation_table.parquet` | PLINDER's main table, 1 GB, ~745 columns, one row per (system, ligand) |
| `splits/split.parquet` | system_id -> train / val / test (+ uniqueness) |
| `fingerprints/ligands_per_system.parquet` | per-system ligand info incl. SMILES |
| `links/kind={apo,pred}/links.parquet` | holo system (`reference_system_id`) -> apo PDB chain / AlphaFold model (`id`), with pocket_lddt, pocket_fident, clash checks. 433,865 apo / 163,816 pred rows; up to 5 apo links per system |
| `linked_structures/{apo,pred}.zip` (+ unpacked `apo/`, `pred/`) | one `<id>.cif` per linked chain, 29,270 apo / 11,687 AlphaFold; each in its OWN frame (not superposed); every linked id has a file |

Ours, under `$WORK`:
- `meta/manifest.parquet` (`src/build_manifest.py`): 341,945 rows (system x
  proper ligand), 37 PLINDER columns + split + `pass_*` filter flags,
  `pass_all`, `strict_rep`, `receptor_key`; plus `receptors.txt`,
  `ligand_fps.npy` (ECFP4).
- `structures/<receptor_key>.cif`: PLINDER receptor.cif (chains with an atom
  within 6 A of a system ligand, one biological assembly, + PLIP waters; no
  ligand). `systems/<system_id>/`: `system.cif` + `ligand_files/<chain>.sdf`
  (crystal coordinates). Fetched by `src/fetch_structures.py`.
- IDs: `system_id = pdb__assembly__receptorchains__ligandchains`, chains
  `<copy>.<label_asym_id>`; `ligand_id = pdb__assembly__ligandchain`;
  `receptor_key` = first three fields.

## Dataset

`configs/datasets/dataset_v1.yaml` -> `datasets/dataset_v1/index.parquet`
(`src/build_dataset.py`, receptor input policy `plinder_receptor` in
`src/receptor_inputs.py`): `strict_rep` and one ligand chain. 21,209
interactions, 19,782 receptors, 8,378 ligands; splits 20,810 / 94 / 305.

Known issues, not yet acted on:
- val/test far too small (PLINDER's own val/test are 891 / 1,120 overall).
- Symmetry copies survive dedup (TRP: 507 rows from 57 entries); proposed
  per-`pdb_id` dedup -> dataset_v2.
- Heavy target concentration (PDE10A 747, SARS-CoV-2 Mpro 538, BACE1 418);
  handle by balanced sampling and same-cluster negative masking, not deletion.
- Hydrogens inconsistent across structures (encoder tests strip them).
- Receptor input lacks assembly chains PLINDER did not assign, so some buried
  interfaces look exposed (`full_assembly` policy reserved, not implemented).

## Protein encoder (`src/surface_encoder.py`, `src/anchors.py`)

Input: receptor atoms only (`extract.load_protein_atoms`: protein residues +
MSE, elements C/H/O/N/S/Se, first model) as coordinates + 6-way element one-hot.
The ligand is used only for labels (surface points within 4 A of a ligand heavy
atom).

atoms -> dMaSIF surface (no grad; can be cached) -> 10 curvatures + 6 AtomNet_MP
chemistry features (+ 12 optional pharmacophore channels, `cfg.chemo`, see
`chemotype.py`) -> dMaSIFConv_seg (16-dim point embeddings; `load_pretrained()`
starts from the published search checkpoint, then trains; `inflate=True`
zero-pads the three input-side tensors so extra channels keep the pretrained
function exactly at init) -> pocket head (logit
per point) -> region anchors -> probability-weighted pooling -> optional
normalisation -> projection MLP -> K = k_pos + k_neg (8 + 8) normalised
128-dim vectors.

Pooling is one weighted mean per region by default (`cfg.pool_mode="mean"`),
and can be `mean_std` or `shells` - see `anchors.pool_regions`. The
normalisation before the projection is identity by default and a BatchNorm1d
over the batch's real anchors with `cfg.anchor_norm="batch"`. Both default to
the original behaviour so every earlier result reproduces, and both exist for
the 0.993 collapse described below. `project_anchors()` is public and
`forward(project=False)` exists so a caller can add an anchor of its own - the
training loop's fallback ball - and put the whole set through one
normalisation.

Everything from the pocket head onward trained only on pocket BCE until
`train.py`; the projection has never seen a contrastive gradient.

Region selection (`anchors.pocket_regions`, `EncoderConfig.region_method`;
default still the OLD `watershed` until a winner is picked):
- `watershed` (+ `persistence`, ToMATo merging in per-protein quantile units)
- `components` (P2Rank-style single linkage, ranked by sum p^2)
- `ball` (greedy fixed-radius balls, mean-shift centring, `ball_rank`)
- `nms` (score-then-suppress: mean-shift proposals from every candidate, score =
  probability inside, NMS by point-set IoU, stop below alpha x best)
- `compball` (components pick and rank pockets, a ball shapes each)
All work on candidates = each protein's top 5% after smoothing. Negative
regions: low-probability patches, farthest-point seeded, sized like the pocket
regions. In training the labelled pocket is currently `forced` in as the first
positive region (see open decisions).

## Ligand encoder (`src/ligand_encoder.py`, `src/ligand_features.py`)

Decision: **a small encoder trained from scratch, with fingerprint baselines as
controls**, not an off-the-shelf pretrained model. The reasons are all about
matching the protein side:

- **Capacity.** The protein branch is dMaSIFConv_seg at 16 dims over 3 layers
  (~10^5 parameters). dataset_v1 has 8,378 distinct ligands and heavy target
  concentration, so a large ligand branch can drive the loss down by
  memorising ligands, and `max` over anchors lets it ignore the protein. Every
  arm here is 10^5-10^6 parameters.
- **Cost.** A step is thousands of surface points through KeOps plus CPU region
  selection; a ligand is ~30 heavy atoms. Freezing the ligand side would save
  nothing and give up the gradient, so every arm trains end-to-end.
- **Evidence.** A 2026 benchmark of 25 pretrained embedding models over 25
  datasets (arXiv 2508.06199) finds nearly all of them no better than plain
  ECFP under frozen-embedding evaluation, with GNN-learned representations
  weaker than fingerprints. DrugCLIP's own ablation (DrugBA, same Uni-Mol
  encoder, regression objective: 69.5 vs 80.9 AUROC on DUD-E) says the
  contrastive formulation carried its result, not the pretrained encoder.
  So a pretrained encoder would cost a container rebuild, a 2048-dim
  projection and unknown PLINDER leakage for little expected gain.

Arms, all mapping to one 128-dim normalised vector and differing by one config
field (`LigandConfig.arm`):

| arm | input | params | role |
|---|---|---|---|
| `fp` | ECFP4 2048 -> MLP | 1.11 M | baseline; fingerprints already exist |
| `fp_linear` | ECFP4 2048 -> Linear | 0.26 M | capacity control for `fp` |
| `desc` | 33 physchem descriptors -> MLP | 0.04 M | size / polarity / shape only |
| `fp_desc` | both, concatenated | 1.13 M | does shape add to substructure |
| `gnn` | GINE over the heavy-atom graph | 0.32 M | the intended model |
| `frozen` | a precomputed embedding -> Linear | varies | unused unless `--lig_emb` is given |

`fp`'s parameters are nearly all in the first layer of a 2048-wide input, so
`fp` vs `gnn` is not capacity-matched on its own; `fp_linear` separates "more
parameters" from "better features". `desc` exists because ECFP identifies
substructures but says nothing about extent or polarity, which is what a
surface patch can be complementary to.

Features are 2D and hydrogen-free, from `ligand_features.py` in the CPU venv
(rdkit), packed into one `ligand_feats.npz` of plain arrays so the container
never needs rdkit. 2D deliberately: the crystal pose would leak the answer,
since a screening candidate has no pose. GINE is implemented directly on
`index_add_` / `scatter_reduce` instead of importing PyG, so the whole ligand
side and every test of it runs on plain CPU torch.

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

## v2 (`src/slurm/train_v2.sbatch`) - WRITTEN, NOT YET RUN

Four levers, all flags defaulting OFF so every earlier run reproduces, and all
switched on by `train_v2.sbatch` (a preset over `train.sbatch`):

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

## Training loop (`src/train.py`, `src/train_data.py`) - RUN, see "First training run"

The loss:

    s(P, L) = max over P's PREDICTED pocket anchors of cos(a, L) / tau
    loss    = symmetric InfoNCE over the B x B batch matrix
            + lam_pocket * per-point pocket BCE
            + lam_site   * within-protein anchor cross-entropy (default 0)

The pocket term is not decoration: anchors come from the predicted pocket map
and region selection has no gradient, so if the head drifts the regions drift
and the contrastive loss loses the places it was scoring. When no predicted
region clears the positive rule, the step falls back to a **selector-shaped
ball at the labelled pocket's centroid** (`train_data.fallback_region`), which
closes open decision 3. A ball rather than the labelled point set itself: the
point set has a size and shape the selector never produces, so training on it
would be learning an anchor that cannot exist at inference. The RATE of that
fallback is printed every evaluation and is the honest measure of how much the
loop leans on labels it will not have.

`lam_site` is the new, optional term: a cross-entropy over one protein's own
pocket anchors with its labelled region as the (multi-)positive. InfoNCE only
ever sees the max over anchors, so it is satisfied by a protein whose best
anchor is in the wrong place; this says WHERE. It targets the known weakness,
top1 0.64 against hit 0.86. Default 0 so the baseline stays clean.

Data pipeline (`src/train_data.py`):
- One ROW is an interaction; one SURFACE belongs to a receptor `input_id` and
  several rows share it, so the two are cached separately.
- `SurfaceStore`: disk cache (one ~1.1 MB npz per receptor, written to a temp
  name and renamed) plus a bounded host-RAM LRU. Sampling is seeded from the
  receptor's own identity, and the global RNG is restored afterwards - without
  that, a cache MISS mid-training would reseed the process and every later draw
  of the run would depend on which surfaces happened to be resident.
- Training and validation get SEPARATE RAM stores over one disk cache, or every
  evaluation would evict the whole training window and the next step would
  evict it back.
- Pocket labels are recomputed from the ligand's COORDINATES against the
  surface in hand, never from a saved index.
- **Other sites come from the item table itself**: every row sharing an
  `input_id` is a sibling, and its ligand's neighbourhood is an ignore mask.
  `probe_retrieval` needed `--other_ligands` pointing at a `labels.py`
  directory for this; the table already knows, so nothing external is needed.
- Sampling, splitting, masking and metrics are `probe_retrieval`'s functions,
  imported rather than re-derived, so the probe and the full loop agree by
  construction on what a batch, a split, a mask and a metric are.

Two shared modules grew to support it, both default-off so every earlier result
reproduces:
- `anchors.pool_regions(mode=)`: `mean` (the original), `mean_std`, or
  `shells` - n weighted means in equal-COUNT radial shells around the region's
  own centroid. A single mean over ~290 points discards all arrangement inside
  the region, which pocket DETECTION does not need and ligand DISCRIMINATION
  does. Equal-count, not fixed-radius, so no shell is ever empty whatever the
  selector's region sizes; distance to the region's own centroid, so it stays
  rigid-motion invariant. `EncoderConfig.pool_mode`, `n_shells`.
- `EncoderConfig.anchor_norm="batch"`: a BatchNorm1d on the pooled vector
  before the projection, which subtracts the per-feature mean the whole dataset
  shares. Learned running statistics rather than statistics fitted once,
  because the encoder trains and anything fitted up front goes stale. This is
  the direct attack on the 0.993 collapse.
  `SurfaceEncoder.project_anchors()` is now public and `forward(project=False)`
  exists, so the training loop can put the fallback anchor through the same
  normalisation as the rest, in ONE call.

Stages: `prepare` (CPU venv) -> `cache` (GPU, array-shardable, optional) ->
`train` -> `eval`, plus `selftest`.

How to run it:
```
# CPU venv
python src/train.py prepare --index datasets/dataset_v1/index.parquet \
    --out runs/v1 --split train --exclude encoder_test/items.tsv
python src/ligand_features.py --index datasets/dataset_v1/index.parquet --out runs/v1
# login node, container, no GPU: ~2-5 min
apptainer exec --cleanenv --env OMP_NUM_THREADS=8 $WORK/dmasif_sandbox \
    python src/train.py selftest
# GPU
sbatch src/slurm/cache_surfaces.sbatch          # optional, 16 tasks, ~25 min
sbatch src/slurm/train.sbatch                   # smoke + the real run
```
`train.sbatch` **resumes itself**: `--max_hours 11` stops cleanly inside the
12 h walltime and checkpoints, and resubmitting the same script picks up
`last.pt`. A long run is a chain of these. The surface cache is shared, so the
second job never resamples. A resumed run is reproducible from its own
checkpoint but is NOT step-for-step identical to an uninterrupted one: the
numpy sampler's state is not in the checkpoint, so it is reseeded from
(seed, step). Said here rather than silently approximated.

Read, per evaluation line: `mrr`/`top1` against `chance`, `site` (the ranking
weakness), `AUROC`/`AP` (comparable to encoder_test's 0.937 / 0.47), `fb` (the
fallback rate - high and flat means the loop is leaning on labels), and
`cosdiff`, the median cosine between different proteins' top anchors. **0.993
is the number to beat**, and nothing else in the log matters as much.

## Pending jobs (ready to submit)

0. **`bash src/train_check.sh`** on the login node, before anything is
   submitted: seven stages, no queue, and stage 5 runs the real encoder on
   CPU. See "End-to-end pipeline check" below. This is the thing standing
   between a typo and a wasted queue wait.
1. **The training loop itself** (`src/slurm/train.sbatch`), the one that
   matters. Self-tested, never on a GPU. See "Training loop" above for the
   three commands.
2. **Surface cache prefill** (`src/slurm/cache_surfaces.sbatch`), optional,
   16 array tasks, ~25 min. Only moves work earlier - training fills the cache
   lazily - but the first epoch is otherwise dominated by surface sampling.
3. **Probe rerun** for the anchor-spread diagnostic and
   `--anchor_norm center / whiten`; see "Ligand-arm probe" above. CPU-only via
   `PROBE_ONLY=1`.
4. **Paired base-vs-chemo bootstrap** from the per-item sweep TSVs; see the
   chemo section. A small offline CPU script, not yet written.
5. **Follow-up selector sweep** (`SWEEP_ONLY=1 sbatch
   src/slurm/region_test.sbatch`): superseded by the apo test and the chemo
   ablation, both of which scored every selector. Note it only runs
   `region_sweep.py`, which is CPU-only, so it does not need the
   `--gres=gpu:1` / `--nv` the sbatch requests.

DONE and recorded above: the apo/holo/AlphaFold test, the chemo ablation, the
ligand-arm probe. `encoder_test/head_pretrained_300.pt` exists;
`src/slurm/head_ckpt.sbatch` rebuilds just that checkpoint in 30 min if it is
ever lost, without the 4 h apo evaluation. To re-run the apo test:
`python src/apo_pairs.py --links $PLINDER_DIR/links --index
datasets/dataset_v1/index.parquet --exclude encoder_test/items.tsv --out
apo_test/` (CPU), then `LIMIT=10 sbatch src/slurm/apo_test.sbatch` as a check,
then the full run.

## Open design decisions (needed for training)

Decisions 1-5 are SETTLED and implemented. `src/train.py` imports
`probe_retrieval`'s versions of 2, 4 and 5 rather than re-deriving them, so the
probe and the full loop cannot drift apart. What remains open is below them.

1. **Which selector**: SETTLED as **`ball_R8`** - greedy fixed-radius balls,
   R 8 A, 2 mean-shift steps, seeded from each protein's top 5% after
   smoothing, ranked by seed strength (`anchors.ball_regions`). It wins dccA
   ("some top-k anchor lands on the site") by a wide margin on `eval` and holds
   it across conformations: 0.70 holo / 0.61 apo / 0.65 pred, against 0.59 /
   0.41 / 0.52 for the next best (`nms_R8`). dccA is the operative metric
   because scoring is max-over-anchors.
   Its known weakness is RANKING, not recall - apo dcc1 0.26, worst of the ball
   family, where `nms_R8` leads at 0.31. If scoring ever changes to top-1,
   that flips to `nms_R8`. The `k_pos = 8` budget is load-bearing: those dccA
   numbers assume the right ball is somewhere in the top 8.
   `watershed_old` is dead - its top region moves 5.0 A from RESAMPLING alone,
   before any conformational change.
   It stays a config field, not code, and `EncoderConfig`'s default is still
   `watershed` so earlier results reproduce. But NOTE: only
   `probe_retrieval.sbatch` passes `--region_method ball --ball_radius 8
   --ball_shift 2`. `test_encoder.py` has NO selector CLI, so anything run
   through it - `encoder_test`, `chemo_test` - silently uses the dead watershed
   default, and the region columns those jobs print inline (`hit`, `top1`,
   `#reg`, `size`, `cover`, `prec`) are watershed numbers.
   That does not invalidate those runs: `overfit`'s loss is pocket BCE alone,
   the anchors are computed and never used, so no gradient flows through region
   selection and both the trained head and the dumped maps are
   selector-independent. `region_sweep.py` then scores all six selectors
   offline on those maps. Read the sweep TSV, not the inline region columns.
   Worth fixing by giving `test_encoder.py` the selector flags, so the inline
   table stops being misleading.
2. **How a ligand is scored against a protein's anchors**: settled as **max
   over the selector's pocket anchors** (multiple-instance,
   `ProbeModel.scores`), because anchor recall is far above top-1 for balls
   (hit 0.84 vs top1 0.64). Negative patches are never scored at inference.
   `--oracle` reports the score restricted to pocket-matching anchors, which
   bounds what better anchor RANKING could buy.
3. **Training labels for regions**: SETTLED. positive = cover >= 0.5 and
   prec >= 0.3 (the `hit` rule); ignore = any other region with cover or
   prec >= 0.1, near-duplicates included (`ball_R8` makes ~4 of them per
   protein); negative = the rest; the selector's negative patches are ignore
   too, since nothing scores them at inference.
   `probe_retrieval.region_labels` for frozen anchors,
   `train_data.region_labels` for the loop. The no-positive case is now
   CLOSED: a selector-shaped ball at the labelled pocket's centroid
   (`train_data.fallback_region`), with the rate reported every evaluation.
   A ball rather than the labelled point set, so the anchor the loss learns on
   is one the selector could actually produce.
4. **Other real sites**: SETTLED, two ways. `probe_retrieval` takes
   `--other_ligands`, a `labels.py` output directory, read for its `lig_xyz`
   COORDINATES. `train.py` needs nothing external: every row sharing an
   `input_id` in the item table is a sibling ligand on the same receptor, so
   the table already says where the other sites are.
   Note `receptor_union.npz` cannot be used for either: it stores point
   INDICES against the surface it was written on, and anything that resamples
   a surface gets unrelated points - the failure `labels.py`'s own docstring
   warns about.
5. **Batch negatives**: implemented. `mask_pairs` masks a pair sharing a target
   group (`system_pocket_UniProt`, falling back to
   `pli_unique_qcov__95__strong__component`), the same ligand, or ECFP4
   Tanimoto >= 0.8; masked pairs leave the denominator entirely rather than
   training as negatives. `sample_weights(alpha=1)` makes sampling uniform over
   targets and `draw_batch` allows at most one row per group per batch.

## Next steps (model training, on `claude/model-training`)

1. **Second round of runs (code in place, not yet run).** Four GPU jobs plus
   one short eval, commands in `src/slurm/train.sbatch`'s header:
   - finish v1: `SKIP_SMOKE=1 sbatch src/slurm/train.sbatch` (15,642 -> 20,000)
   - `runs/frozen`, `FREEZE=1`: `--freeze_encoder` freezes the surface encoder
     below the projection (conv, inputs, pocket head, so the regions are fixed
     too); projection, ligand and temperature train. The probe's setting inside
     this loop, on this split: **the control that says whether training the
     protein side buys the MRR.**
   - `runs/shells`, `POOL_MODE=shells`: 48-dim pooled vector that keeps
     arrangement inside a region. Head loads 117/118 (projection width).
   - `runs/b32`, `BATCH=32 STEPS=5000`, eval/save every 125: 4x the
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

## End-to-end pipeline check (`src/train_check.sh`) - LOGIN NODE, no queue

One command, seven stages, nothing touching `runs/` or `surfaces/`:

```
bash src/train_check.sh                                    # ~15-30 min
LIG=$WORK/probe/ligand_feats.npz bash src/train_check.sh   # skips stage 4
```

1 imports and config arithmetic; 2 `train.py selftest`; 3 `prepare` on the real
index; 4 ligand features; 5 **the real encoder on CPU** - Biopython parse,
dMaSIF surface sampling, AtomNet, `dMaSIFConv_seg`, region selection, pooling,
InfoNCE + pocket BCE, a checkpoint; 6 resume from `last.pt`, checking both that
the step advanced and that no surface was resampled; 7 `eval` from that
checkpoint.

Stage 5 deliberately runs `--pool_mode shells --anchor_norm batch --lam_site
0.5` and `--head_ckpt`, because those four paths have never met the real
convolution and this is the cheapest place to find out that they do
(`load_head`'s shape filter comes along for free: the saved head has a `mean`
projection that cannot fit a shells run). It uses `--sup_sampling 6` instead of
20, so the surfaces are ~3x sparser and the pass takes minutes rather than an
hour of login-node CPU - a SHAPE check, not a training run, and the metrics it
prints are noise. KeOps also compiles its CPU kernels the first time, a few
minutes of g++ on top.

It proves the pipeline executes on real data. It says nothing about GPU
execution (KeOps compiles different kernels for CUDA; `train.sbatch`'s sm_120
check and its 64-item smoke are what cover that), nothing about model quality,
and nothing about step timing - CPU KeOps is orders of magnitude off.
`rm -rf $WORK/train_check` undoes all of it.

The shell plumbing - stage sequencing, the pass/fail/skip summary, the exit
code, the cascade message - was verified against stubbed binaries. The stages
themselves have not been run; there was no torch where this was written.

## Self-tests (no GPU, no data, no KeOps)

They run in the container on the LOGIN node - plain torch, no KeOps, no CUDA.
The first three PASS as of the last session. **`train.py selftest` is new and
has never been run anywhere**, so run it first - or just run
`src/train_check.sh`, whose stage 2 is exactly this:

```
EX="apptainer exec --cleanenv --env OMP_NUM_THREADS=8 $WORK/dmasif_sandbox python"
$EX src/train.py selftest            # ~2-5 min, NEW
$EX src/chemotype.py
$EX src/ligand_encoder.py
$EX src/probe_retrieval.py selftest  # 10-15 min
```

- `chemotype.py`: typing tables (20 cases + full sidechain coverage), element
  order against `extract.ELE2NUM`, batch isolation on DELIBERATELY overlapping
  proteins (0.00e+00), rigid-motion invariance, buriedness ordering.
  `--tables` alone needs only numpy.
- `ligand_encoder.py`: all six arms - shapes, gradients, batch invariance
  (4e-08), permutation consistency, a lone ion and an unparseable molecule.
- `probe_retrieval.py selftest`: plants a latent shared between anchors and
  ligands; every arm must beat max(0.25, 2 x chance) and its untrained floor.
  Reaches MRR 0.98-1.00 for `fp_linear` / `desc` / `fp_desc` / `gnn` against
  chance 0.094, which is the InfoNCE, the false-negative masking, the
  group-disjoint split and the metrics all verified together.
  `fp` peaks at 0.580 and declines - the 1.1 M-parameter MLP overfits 112
  synthetic rows where the 262 k linear map reaches 1.000. Arms are judged on
  `best_mrr` for that reason; all arms overfit this fixture and both figures
  are printed. Nothing here predicts real-molecule behaviour.
- `train.py selftest`: 120 synthetic proteins (a 2400-point sphere with a
  ligand cloud outside one patch) and a stub for the only three pieces that
  need KeOps - the surface sampler, AtomNet and the convolution.
  The fixture's geometry is tuned to the real thing rather than to convenience:
  pocket 2.7% of the surface (real receptors: 2.8%) and ball_R8 precision 0.65
  (the region sweep: 0.63), so every protein can form a genuine positive. A
  tighter ligand cloud leaves the worst proteins under `pos_prec`, which makes
  them train on the fallback forever and silently empties every `site`
  assertion; a looser one clears the label rule trivially and would hide a
  broken one. Two assertions guard that band. Everything else is
  the real code, in seven groups:
  the item table; `SurfaceStore`'s disk round-trip, LRU bound, and the two
  properties a cache lives or dies on (a surface depends only on the
  receptor's identity, and sampling one leaves the global RNG untouched);
  pocket labels and the sibling-ligand other-site mask; all three pooling
  modes including shell pooling's rigid-motion and permutation invariance;
  the region label rules; the training loop on a planted signal, where MRR
  must beat max(0.30, 2 x chance) and the fallback rate must FALL as the head
  learns; and gradients plus a checkpoint round-trip scoring identically,
  which is what makes `--resume` meaningful.
  The planted signal is deliberately two things: a constant pocket marker so
  the head can find the pocket at all, and the ligand's latent so retrieval
  has something to recover. The identity alone would not do - it changes sign
  across proteins, so no linear head would ever find a pocket and the fallback
  would fire forever, which is a fixture failure that looks like a code
  failure.
  Nothing here involves dMaSIF, so it says nothing about surface sampling or
  the convolution.
- `--cleanenv` strips `OMP_NUM_THREADS`, so pass it explicitly or the probe
  selftest runs single-threaded and takes 10x longer.

**Verified on a GPU since**: `chemotype.surface_features` on real atoms
inside the conv and `load_pretrained(inflate=True)` against a real checkpoint,
both by the chemo ablation (job 139099).

**Unverified by anything so far**: the training loop against the real encoder.
`pool_mode="shells"` and `anchor_norm="batch"` have never gone through
`dMaSIFConv_seg`, and the whole of `train.py` has never run on a GPU - which is
why `train.sbatch` smokes 64 interactions for 30 steps inside the same job
before the real run.

## Code map (`src/`)

| file | what |
|---|---|
| `build_manifest.py` | PLINDER annotation table -> `meta/manifest.parquet` |
| `fetch_structures.py` | download receptor / system files the manifest needs |
| `build_dataset.py`, `receptor_inputs.py`, `configs/datasets/*.yaml` | spec-driven dataset builds with provenance |
| `ids.py` | ID parsing, ligand chain / SDF lookup, surface tags |
| `inspect_data.py`, `sanity.py` | read-only data checks |
| `extract.py` | atom loader (`load_protein_atoms`, now with B-factors) and batch dMaSIF feature extraction |
| `labels.py` | pocket labels for cached surfaces + QC table |
| `probe_pocket.py` | earlier probe: do frozen dMaSIF embeddings separate pockets |
| `dmasif_compat.py` | imports the vendored dMaSIF under modern PyTorch |
| `surface_encoder.py` | protein encoder (`EncoderConfig`, `SurfaceEncoder`, `pocket_loss`) |
| `chemotype.py` | pharmacophore typing of protein atoms + 12 per-surface-point channels; `--tables` needs no torch |
| `ligand_features.py` | SMILES -> ECFP4 + descriptors + 2D graph, packed (CPU venv, rdkit) |
| `ligand_encoder.py` | ligand arms (`LigandConfig`, `LigandEncoder`, `LigandData`); `python src/ligand_encoder.py` self-tests |
| `probe_retrieval.py` | frozen-anchor retrieval probe: `prepare` / `dump` / `probe` / `selftest` |
| `anchors.py` | region selectors, negatives, pooling (`mean` / `mean_std` / `shells`) |
| `train_data.py` | items, surface store (disk + RAM LRU), pocket + other-site labels, the fallback region |
| `train.py` | the contrastive loop: `prepare` / `cache` / `train` / `eval` / `selftest` |
| `train_check.sh` | login-node end-to-end check of the whole pipeline, 7 stages, no GPU |
| `test_encoder.py` | prepare / smoke / overfit; `--dump` maps, `--save_ckpt`, `load_checkpoint()` |
| `region_sweep.py` | offline selector comparison on dumped maps |
| `apo_pairs.py`, `apo_eval.py` | apo / AlphaFold vs holo test |
| `slurm/` | `train`, `cache_surfaces`, `encoder_test`, `region_test`, `apo_test`, `probe_retrieval`, `chemo_test`, `head_ckpt`, build / fetch / smoke jobs; `dmasif.def` container |
