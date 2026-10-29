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

## Dataset expansion (`src/dataset_funnel.py`, draft `dataset_v2.yaml`) - WRITTEN, selftest PASSES, never run on real data

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

Run order (CPU venv unless marked):
```
python src/dataset_funnel.py selftest
python src/dataset_funnel.py report --manifest data/meta/manifest.parquet --per_entry
python src/companion_funnel.py funnel --manifest data/meta/manifest.parquet
python src/companion_funnel.py distance --manifest data/meta/manifest.parquet --systems systems --n 400
# paste all three into RESULTS.md, settle the ladder, edit dataset_v2.yaml to match, then:
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
