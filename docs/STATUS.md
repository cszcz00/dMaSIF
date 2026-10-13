# Project status (handoff)

Read this first in a new session. Last updated 2026-10-13, at the hand-off from
the anchor-selection work (branch `claude/sharp-franklin-42hmhj`) to model
training (branch `claude/model-training`, which starts from the same commit).

## Goal

Contrastive protein-ligand retrieval. Protein side: dMaSIF surface features +
convolution (retrained), a per-point pocket head, and K pooled **region**
anchors per protein compared against ligand vectors. Built and tested: the
protein encoder up to the anchors. Built, not yet run: the ligand encoder
(`src/ligand_encoder.py`) and the frozen-anchor retrieval probe that picks
between its arms. **Not built yet: the contrastive training loop over both
sides, and full-scale data loading.**

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
chemistry features -> dMaSIFConv_seg (16-dim point embeddings; `load_pretrained()`
starts from the published search checkpoint, then trains) -> pocket head (logit
per point) -> region anchors -> probability-weighted pooling -> projection MLP
-> K = k_pos + k_neg (8 + 8) normalised 128-dim vectors. The projection is
untrained so far (only pocket BCE has been used).

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

## Pending jobs (submitted or ready, results not recorded yet)

1. **Follow-up sweep** (nms, re-ranked balls, compball, duplicate count `on`,
   paired bootstrap vs the ball leader, results by pocket size):
   `SWEEP_ONLY=1 sbatch src/slurm/region_test.sbatch`.
2. **Ligand-arm probe** (`src/slurm/probe_retrieval.sbatch`), **blocked on the
   apo test's checkpoint** `encoder_test/head_pretrained_300.pt`. Which ligand
   encoder pairs with real surface anchors, on FROZEN protein anchors, in one
   GPU pass plus minutes of CPU. First, in the CPU venv:
   ```
   python src/probe_retrieval.py prepare --index datasets/dataset_v1/index.parquet \
       --exclude encoder_test/items.tsv --n 1500 --out probe/
   python src/ligand_features.py --index datasets/dataset_v1/index.parquet --out probe/
   ```
   then `LIMIT=10 sbatch src/slurm/probe_retrieval.sbatch` as a check, then the
   full run. `PROBE_ONLY=1` reruns only the arm comparison on an existing
   `anchors.pt`, so arms and hyperparameters can be swept without the GPU.
   Read `probe/arms.tsv`: `mrr` against `mrr0` (untrained) and `chance`
   (random), `oracle_mrr` for the headroom in anchor ranking rather than in the
   ligand encoder. These numbers rank the arms; they do NOT predict end-to-end
   performance, because the protein side is frozen and its head saw 32
   receptors for 300 steps.
3. **Apo vs holo test**: `python src/apo_pairs.py --links $PLINDER_DIR/links
   --index datasets/dataset_v1/index.parquet --exclude encoder_test/items.tsv
   --out apo_test/` (CPU), then `LIMIT=10 sbatch src/slurm/apo_test.sbatch`
   as a check, then the full run. First run trains + saves the head checkpoint
   `encoder_test/head_pretrained_300.pt`. `src/apo_eval.py` evaluates holo,
   holo rotated + resampled (noise floor), apo and AlphaFold (pLDDT < 70
   dropped); linked chains are sequence-aligned, superposed on holo pocket
   C-alphas and trimmed to matched residues (placement verified on synthetic
   proteins); labels from the holo ligand. Strata by apo pocket lDDT / clash.

## Open design decisions (needed for training)

Decisions 2-5 now have a WORKING IMPLEMENTATION in `probe_retrieval.py`,
exercised only against frozen anchors so far. The training loop should reuse
those functions rather than re-deriving them.

1. **Which selector** becomes the default (after the pending sweep + apo test).
   Still open. It is a config field everywhere, not code: the probe's sbatch
   passes the sweep's current leader (`--region_method ball --ball_radius 8
   --ball_shift 2`) and `EncoderConfig`'s default is left at the old
   `watershed` so earlier results stay reproducible.
2. **How a ligand is scored against a protein's anchors**: settled as **max
   over the selector's pocket anchors** (multiple-instance,
   `ProbeModel.scores`), because anchor recall is far above top-1 for balls
   (hit 0.84 vs top1 0.64). Negative patches are never scored at inference.
   `--oracle` reports the score restricted to pocket-matching anchors, which
   bounds what better anchor RANKING could buy.
3. **Training labels for regions**: implemented in
   `probe_retrieval.region_labels`. positive = cover >= 0.5 and prec >= 0.3
   (the `hit` rule); ignore = any other region with cover or prec >= 0.1,
   near-duplicates included; negative = the rest. Nothing is forced in.
   Still open: what to do when NO region is positive (the design proposes a
   fallback ball at the labelled pocket's centroid, shaped like the selector's
   own regions, with the rate logged); the probe simply reports the rate.
4. **Other real sites**: handled by `--other_ligands`, a `labels.py` output
   directory, read for its `lig_xyz` COORDINATES. Note `receptor_union.npz`
   cannot be used for this: it stores point INDICES against the surface it was
   written on, and anything that resamples a surface gets unrelated points -
   the failure labels.py's own docstring warns about.
5. **Batch negatives**: implemented. `mask_pairs` masks a pair sharing a target
   group (`system_pocket_UniProt`, falling back to
   `pli_unique_qcov__95__strong__component`), the same ligand, or ECFP4
   Tanimoto >= 0.8; masked pairs leave the denominator entirely rather than
   training as negatives. `sample_weights(alpha=1)` makes sampling uniform over
   targets and `draw_batch` allows at most one row per group per batch.

## Next steps (model training, on `claude/model-training`)

1. Run the ligand-arm probe (pending job 2) once the checkpoint exists, pick
   the arm, and record the table here. Then the contrastive loss over anchor
   vectors in a real training loop: `probe_retrieval.info_nce` and its
   false-negative masking are written and self-tested, but against frozen
   anchors only.
2. Data pipeline for full-scale training: surfaces cached once
   (`src/extract.py` -> `feats/`), pocket labels (`src/labels.py` +
   `pocket_qc.parquet`), a dataset/loader with target-balanced sampling.
3. Training script + sbatch with checkpoints, validation retrieval metrics,
   and the pocket loss kept as an auxiliary term.
4. dataset_v2 (per-entry dedup, larger val/test, e.g. PLINDER's own val/test).
5. Later: train with apo / AlphaFold receptors as augmentation if the apo test
   shows a large holo-apo gap.

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
| `ligand_features.py` | SMILES -> ECFP4 + descriptors + 2D graph, packed (CPU venv, rdkit) |
| `ligand_encoder.py` | ligand arms (`LigandConfig`, `LigandEncoder`, `LigandData`); `python src/ligand_encoder.py` self-tests |
| `probe_retrieval.py` | frozen-anchor retrieval probe: `prepare` / `dump` / `probe` / `selftest` |
| `anchors.py` | region selectors, negatives, pooling |
| `test_encoder.py` | prepare / smoke / overfit; `--dump` maps, `--save_ckpt`, `load_checkpoint()` |
| `region_sweep.py` | offline selector comparison on dumped maps |
| `apo_pairs.py`, `apo_eval.py` | apo / AlphaFold vs holo test |
| `slurm/` | `encoder_test`, `region_test`, `apo_test`, `probe_retrieval`, build / fetch / smoke jobs; `dmasif.def` container |
