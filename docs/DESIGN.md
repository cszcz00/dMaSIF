# Design

How the method works, and the decisions behind it. Settled material, changed
only when a decision changes. Results live in [`RESULTS.md`](RESULTS.md),
current state in [`STATUS.md`](STATUS.md).

Split out of `STATUS.md` on 2026-10-27; section text is unchanged.

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
python src/train.py prepare --index data/datasets/dataset_v1/index.parquet \
    --out out/runs/v1 --split train --exclude out/encoder_test/items.tsv
python src/ligand_features.py --index data/datasets/dataset_v1/index.parquet --out out/runs/v1
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

## External pockets (fpocket) - WRITTEN, NOT YET RUN

Why: the results are still poor, and the one component every run shares is
the pocket head deciding WHERE the regions are (top 5% of its map covers 0.68
of the labelled pocket; region selection has no gradient, so the loop cannot
fix a misplaced region). This takes the head out of region placement: a
geometric pocket finder picks the pockets, the encoder only describes them.
**Nothing was removed**: `--region_source predicted` (the default) is the old
loop exactly, and every earlier command reproduces.

Pieces:
- `src/external_pockets.py`
  - `run` (container, CPU): fpocket on exactly the atoms the surface is built
    from (`load_protein_atoms`, same chain filter, no H), written to a temp
    PDB -> `data/pockets/fpocket/<input_id>.npz`. It stores **alpha spheres**
    (centre, radius, pocket id) plus per-pocket score / druggability / volume,
    NOT surface indices, so one file serves any resampling of the surface.
  - mapping: a surface point is in pocket k if it lies within `r_s + margin`
    of one of k's alpha spheres (`--pocket_margin`, default 0; an alpha sphere's
    boundary runs through the lining atom centres, so the lining surface is
    inside it). Shapes: `pocket` (the covered points) or `ball` (a
    `--ball_radius` ball at their centroid, the fallback's construction).
    Pockets under `min_region_points` or more than 50% inside an earlier one
    are dropped.
  - `score` (numpy, CPU venv): the pockets as regions against the labelled
    pocket on the cached surfaces, region_sweep's metrics, sweeping
    shape x rank x margin. **The gate for the training run.**
  - `selftest` (numpy): parsers (incl. run-together negative coordinates), the
    PDB writer's columns, box-filtered mapping == brute force, ranks, shapes,
    dedup, PocketSource. PASSES locally.
- `EncoderConfig.region_source` / `train.py --region_source`:
  - `external`: the top k_pos external pockets ARE the pocket regions, in
    fpocket's order (`--pocket_rank score|drug|volume`); the selector is not
    run (`anchors.select_regions(fill=False)`).
  - `external_map`: a per-point map (0 outside, 0.5..1 inside, better pocket
    higher) replaces sigmoid(logit) as the selector's input, thresholded at
    `--threshold` (a top-fraction quantile of a map that is 0 on ~97% of the
    surface would be 0).
  - both pool regions as **plain means** (no gradient to the head through the
    weights); the head is still computed, still trained by `--lam_pocket`,
    still reported as AUROC/AP. Negatives are seeded off the external map.
    `fb` now measures how often fpocket's top k_pos miss the ligand. A
    receptor with no pocket file / no usable pocket gets no anchors (training
    falls back, eval drops it); the store line under each evaluation counts
    them.
  - the region loop moved out of `forward` into
    `surface_encoder.select_and_pool` (behaviour unchanged for `predicted`,
    asserted in the selftest) so it can be tested without KeOps.
  - `eval` picks the pocket settings up from the checkpoint unless given.
- `train.py selftest` group [9]: the REAL `select_and_pool` with a
  PocketSource over the synthetic proteins - regions == the external pockets
  in rank order, decoy not positive / true pocket positive, plain-mean
  pooling, no head gradient, rank volume, external_map finds the pocket, a
  missing map refuses, and `predicted` == `select_regions`. **Not run: there
  is no torch here. Run it first.**

Checked locally (no torch): fpocket 4.2.3 built from source; its 1ATP sample
through `write_pdb` -> fpocket -> parser: 28 pockets, the top-ranked one is the
ATP site. On an approximate surface (~3x dMaSIF's point density) the top
pocket at margin 0 has iou 0.76 / cover 0.83 / prec 0.90 against the ATP
label (margin -1: 0.56/0.58/0.96; +2: 0.52/1.00/0.52). One structure: an
existence proof for the code path, not evidence about the dataset.

Run order:
```
bash src/slurm/build_fpocket.sh                     # login node, once; self-checks on 1ATP
EX="apptainer exec --cleanenv --env OMP_NUM_THREADS=8 $WORK/dmasif_sandbox python"
$EX src/train.py selftest                           # group [9] is new
python src/external_pockets.py selftest             # CPU venv
sbatch src/slurm/fpocket.sbatch                     # 16 CPU tasks -> data/pockets/fpocket/
python src/external_pockets.py score --items out/runs/v1/items.tsv \
    --surface_cache data/surfaces --pockets data/pockets/fpocket --n 1000 \
    --out data/pockets/fpocket/score.tsv --per_item data/pockets/fpocket/score_items.tsv
sbatch src/slurm/train_fpocket.sbatch               # b32-matched; set POCKET_SHAPE / POCKET_MARGIN from score
```
Read `score` against ball_R8 (held-out hit 0.84-0.86, dccA 0.77, iou 0.53,
size ~290): if fpocket's hit / dccA are not at least that, the training run is
not worth its GPU. Compare `out/runs/fpocket` with `out/runs/b32` (0.127 MRR at
76k-92k proteins seen, same val set, same batch) at matched proteins seen.
Variants are in `train_fpocket.sbatch`'s header (ball shape, drug rank,
external_map, `LAM_POCKET=0`). `train.sbatch` gained `REGION_ARGS`, passed to
the smoke as well as the main run, so the smoke exercises the external path.

## PLIP interaction labels (`src/plip_labels.py`) - RUN and CHECKED, not yet wired into training

2026-10-26: `plip.sbatch` and `plip_labels.py check --n 500` run; the user
reports the check looks good. Numbers not recorded here yet - paste them in.


For supervising pair pooling's site types (and any later site-based pooling):
which protein atoms each training ligand actually H-bonds, salt-bridges,
pi-stacks or packs against, from PLIP run on every training complex. Labels on
the protein side only, at training time; inference needs none of it. NOT yet
wired into training: that waits for the A/B/C diagnostics.

- Install (`src/slurm/setup_plip.sh`, login node, once): PLIP 3.0.1 +
  OpenBabel 3.2.1 have manylinux wheels for the container's Python 3.12, so the
  container's pip installs them `--no-deps --target $WORK/envs/plip_pkgs`, and
  jobs add that to PYTHONPATH. The container's numpy / torch are untouched. The
  script then checks the imports, runs the self-test and a demo on fpocket's
  1ATP sample.
- Per row (`run`, `src/slurm/plip.sbatch`, 16 CPU tasks): protein atoms exactly
  as the surface sees them (`load_protein_atoms`, chain filter, no H) via
  `external_pockets.write_pdb`, plus the row's ligand converted from its SDF
  by OpenBabel (bond orders kept as repeated CONECT - PLIP protonates the
  complex itself and needs them), merged as TEXT (OpenBabel's residue API, on a
  merged molecule, put the ligand atoms in a protein residue). PLIP finds the
  LIG residue; every interaction is kept as (kind, protein-side xyz,
  ligand-side xyz) -> `plip/<system_id>.npz`. No waters or metals in the
  complex, by design (our receptor input has none).
- Kinds -> classes in the chemotype vocabulary (donor, acceptor, cation, anion,
  aromatic, hydrophobic); `site_labels(xyz, rec)` marks surface points within
  3 A of the protein atom (4 A of a ring / charge centre), optionally only on
  the side facing the ligand partner.
- Checked locally on fpocket's 1ATP: ATP site found in our merged complex, salt
  bridges 3 / 3 against PLIP on the original file, 8 of our 12 H-bonds on the
  original's protein atoms (the original binds the phosphates through Mn ions
  and waters we leave out, so ours finds direct partners instead). 0.8 s per
  complex for a 2,900-atom receptor. `plip_labels.py selftest` PASSES.
- `setup_plip.sh` PASSED on the login node (2026-10-23): wheels installed into
  `$WORK/envs/plip_pkgs`, torch / Biopython / PLIP / OpenBabel import together
  (numpy 1.26.4 untouched), selftest PASS, and the 1ATP demo reproduces the
  local numbers exactly (7 / 5 H-bonds, 3 / 3 salt bridges).
- After the run: `plip_labels.py check` (CPU venv) - interactions per row by
  kind, and on cached surfaces the labelled points per class and the share of
  them inside the 4 A pocket label.

Run order:
```
bash src/slurm/setup_plip.sh
sbatch src/slurm/plip.sbatch
python src/plip_labels.py check --items out/runs/v1/items.tsv --plip data/plip \
    --surface_cache data/surfaces --n 500
```

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

## End-to-end pipeline check (`src/train_check.sh`) - LOGIN NODE, no queue

One command, seven stages, nothing touching `out/runs/` or `surfaces/`:

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
| `build_manifest.py` | PLINDER annotation table -> `data/meta/manifest.parquet` |
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
| `external_pockets.py` | fpocket pockets as regions: `run` / `score` / `selftest`, `PocketSource` for `--region_source external`, the per-row site pocket (`--pocket_select site`) |
| `plip_labels.py` | PLIP interaction labels per training row (`run` / `demo` / `check` / `selftest`) and `site_labels` onto a surface |
| `structure_scan.py` | per manifest row: ligand burial, pocket completeness / mobility, companion size x distance (gemmi); `src/slurm/scan.sbatch`, `selftest` |
| `channel_viz.py` | per-point channel dump (inputs, pretrained AtomNet, chemotype reference, site types, salience) + report: type collapse, channel naming, pocket / PLIP AUROCs, HTML + PyMOL viewers; `src/slurm/viz.sbatch`, `selftest` |
| `dataset_funnel.py` | what each manifest filter costs in NEW target groups, and a cumulative relaxation ladder against dataset_v1; pandas only, `selftest` |
| `pair_pool.py` | `PairPool`: learned site types x distance-bin pair pooling (`--pool_mode pairs`); `python src/pair_pool.py` self-tests |
| `train_data.py` | items, surface store (disk + RAM LRU), pocket + other-site labels, the fallback region |
| `train.py` | the contrastive loop: `prepare` / `cache` / `train` / `eval` / `selftest` |
| `train_check.sh` | login-node end-to-end check of the whole pipeline, 7 stages, no GPU |
| `test_encoder.py` | prepare / smoke / overfit; `--dump` maps, `--save_ckpt`, `load_checkpoint()` |
| `region_sweep.py` | offline selector comparison on dumped maps |
| `apo_pairs.py`, `apo_eval.py` | apo / AlphaFold vs holo test |
| `slurm/` | `train` (base) + `train_v3` (the C preset) + `train_fpocket`; `eval`, `report`, `v3_diag4.sh`; `cache_surfaces`, `fpocket` (+ `build_fpocket.sh`), `plip` (+ `setup_plip.sh`), build / fetch jobs; `encoder_test`, `region_test`, `apo_test`, `probe_retrieval`, `chemo_test`, `head_ckpt`, `smoke`; `dmasif.def` container; `archive/` spent ladders (see its README) |
