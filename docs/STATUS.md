# Project status (handoff)

Last updated after the region-selector sweep (job 129929). Branch:
`claude/sharp-franklin-42hmhj`. Read this first in a new session.

## Goal

Contrastive protein-ligand retrieval. Protein side: dMaSIF surface features +
convolution (retrained), a per-point pocket head, and K pooled **region**
anchors per protein compared against ligand vectors. Ligand side and the
contrastive loss are **not built yet**.

## Layout and environments

- Repo root = `$WORK` = `/mnt/home/cjs2301/dmasif` on the HPC. `dMaSIF/` is the
  vendored upstream (+ PyTorch 2 fixes), `src/` is ours. Data dirs
  (`meta/ structures/ systems/ datasets/ feats/ keops_cache/`) are gitignored.
- CPU work: `~/venvs/plinder` (`envs/cpu-requirements.txt`). GPU work: the
  Apptainer container `dmasif_sandbox` from `src/slurm/dmasif.def`.
- Container base is **NGC PyTorch 25.06** (CUDA 12.9, Python 3.12, torch 2.8,
  PyG 2.8, pykeops 2.2.3, numpy 1.26). Rebuilt because the cluster sends every
  single-GPU job (`--gres=gpu:1`) to the **RTX PRO 6000 Blackwell** pool
  (sm_120); typed GRES requests get rewritten back to that pool. Verified with
  `SMOKE_TEST_PASSED` on alphagpu52.
- `$KEOPS_CACHE_FOLDER` must be set in the submitting shell; jobs pass it in.

## Data (all downloaded, verified by `src/inspect_data.py`)

- `meta/manifest.parquet`: 341,945 rows (one per system x proper ligand),
  307,337 systems, 223,972 receptors, PLINDER 2024-06/v2, loose build of
  `src/build_manifest.py` (`pass_*` flags, `strict_rep`).
- `structures/<receptor_key>.cif`: PLINDER receptor.cif = whole protein chains
  with any atom within 6 A of any system ligand, from one biological assembly,
  plus PLIP-interacting waters. No ligands.
- `systems/<system_id>/`: `system.cif` (receptor + ligand chains, same frame)
  and `ligand_files/<chain>.sdf` (crystal coordinates, bond orders checked
  against SMILES; not generated conformers).
- IDs: `system_id = pdb__assembly__receptorchains__ligandchains`, chains are
  `<copy>.<label_asym_id>`; `ligand_id = pdb__assembly__ligandchain`;
  `receptor_key` (ours) = first three fields.

## Dataset

`configs/datasets/dataset_v1.yaml` -> `datasets/dataset_v1/` (built, provenance
clean at f06c952): `strict_rep` and `n_ligand_chains == 1`. 21,209
interactions, 19,782 receptors, 8,378 ligands; splits 20,810 / 94 / 305.

Known issues, not yet acted on:
- val/test far too small (PLINDER's own val/test are 891 / 1,120 overall).
- Symmetry copies survive dedup (TRP: 507 rows from 57 entries, each its own
  cluster); proposed `pdb_id` dedup -> dataset_v2.
- Heavy target concentration (PDE10A 747, SARS-CoV-2 Mpro 538, BACE1 418;
  pocket cluster c0 = 14%); handle by balanced sampling and same-cluster
  negative masking in training, not by deleting rows.
- Hydrogens inconsistent across structures (tests strip them).

## Protein encoder (`src/surface_encoder.py`, `src/anchors.py`)

atoms -> dMaSIF surface -> curvatures + AtomNet_MP (16 feats) -> dMaSIFConv_seg
(trainable; `load_pretrained()` starts from the search checkpoint) -> pocket head
(per-point logit) -> region anchors -> probability-weighted pooling -> projection.

Regions (`select_regions`): smooth probabilities, take candidates (default each
protein's top 5%, `region_top_fraction`; or absolute `region_threshold`),
watershed flooding, drop < 20 points, keep top `k_pos` by mass; `k_neg`
low-probability negative regions sized like the pocket regions; labelled pocket
can be `forced` in during training.

## Test results (`src/test_encoder.py`, `src/slurm/encoder_test.sbatch`)

32 train / 16 held-out receptors, 300 steps, pocket BCE (pos_weight 42) only.

| held-out, step 300 | AUROC | AP | #reg | size | cover | prec | hit |
|---|---|---|---|---|---|---|---|
| fresh, top 5% | 0.935 | 0.408 | 6.9 | 51 | 0.23 | 0.73 | 0.06 |
| pretrained, top 5% | 0.950 | 0.470 | 6.9 | 48 | 0.20 | 0.71 | 0.00 |
| pretrained, p >= 0.5 | 0.945 | 0.459 | 7.9 | 75 | 0.22 | 0.61 | 0.00 |

Conclusions: point-level pocket scoring works and generalises (AP ~20x the
0.023 base rate); pretrained start is better. Regions **fragment** the pocket
(7-8 small regions, best covers 20-30%) under either threshold; region hit
fails on coverage. The earlier "oversized regions" hypothesis was wrong.

## Region-selector experiment (job 129929, alphagpu53)

`src/slurm/region_test.sbatch`: train the head once (pretrained, 300 steps, 32
train / 64 held-out), dump probability maps (`test_encoder.py overfit --dump`),
then `src/region_sweep.py` scores every selector setting on the same maps.
Methods live in `anchors.pocket_regions` (`method=`, also
`EncoderConfig.region_method`; default still the old watershed).

Head: held AUROC 0.937, AP 0.470 (reproduces the 16-item result).

Oracles (held): labelled pocket ~260 points (2.8% of the surface), extent ~18 A,
Rg ~6 A; one piece at 3 A linkage (2 A splits off slivers). Candidate union
covers 0.68 of the pocket at top 5% (prec 0.38), 0.80 at top 8%; smoothing
radius makes no difference.

| held-out, 64 | #reg | size | iou | cover | prec | hit | top1 | iou1 | dcc1 | dccA |
|---|---|---|---|---|---|---|---|---|---|---|
| old watershed (pers 0, link 2) | 7.2 | 53 | 0.18 | 0.20 | 0.65 | 0.00 | 0.00 | 0.16 | 0.23 | 0.58 |
| watershed link 3, pers 0.02 | 1.7 | 373 | 0.39 | 0.66 | 0.48 | 0.73 | 0.69 | 0.35 | 0.38 | 0.47 |
| components top 5%, link 3 | 1.5 | 406 | 0.38 | 0.68 | 0.45 | 0.70 | 0.66 | 0.34 | 0.33 | 0.41 |
| ball R 8, 2 mean-shift steps | 7.1 | 293 | 0.53 | 0.70 | 0.63 | 0.84 | 0.64 | 0.34 | 0.44 | 0.77 |

Conclusions: fragmentation was the whole problem for watershed/components -
merged, they reach the candidate ceiling, but regions are ~1.5x the pocket
(prec ~0.46). Balls (R 8 ~ pocket extent) give the best best-region (IoU 0.53,
prec 0.63) and find the pocket most often, but always make ~7 balls and rank
them by seed strength: top-ranked region is a tie across methods (top1
0.64-0.69, iou1 ~0.34). Mean-shift centring always helps; smoothing barely
matters; the head (top 5% misses 1/3 of the pocket) is now the limit.

Follow-up, implemented, not yet run: `nms` (score-then-suppress: mean-shift
proposals from every candidate, ball score = probability inside, greedy NMS by
point-set IoU, stop below alpha x best), balls re-ranked by probability inside
(`ball_rank`), and `compball` (components choose and rank, a ball shapes each).
The sweep now also reports `on` (regions >= 30% pocket, i.e. duplicates),
per-protein paired differences vs the ball leader with bootstrap intervals,
and results by pocket-size thirds. Rerun on the saved maps:
`SWEEP_ONLY=1 sbatch src/slurm/region_test.sbatch`.

## Apo vs holo test (set up, not yet run)

PLINDER's links (downloaded to `$PLINDER_DIR/links/kind={apo,pred}/links.parquet`
and `linked_structures/{apo,pred}.zip`) pair holo systems with apo PDB chains
(up to 5 each) and AlphaFold models. On dataset_v1: apo links for 10,974
systems (test 143, val 31), AlphaFold for 12,076. Apo pocket lDDT median 94
(10% below 79); ~5% of links place the holo ligand into clashes (closed pocket).
The zips hold one `<id>.cif` per linked chain in its OWN frame (not superposed).

- `src/apo_pairs.py` (CPU): single-chain receptors, pocket identity >= 95,
  strata A_near (pocket lDDT >= 95) / B_mild / C_large (< 85) / D_closed
  (clash), 100 each, one per target protein, plus every val/test system
  (`eval`); excludes the head's training receptors -> `apo_test/pairs.tsv`.
- `src/apo_eval.py` (container): holo, holo rotated + resampled (noise floor),
  apo, AlphaFold (pLDDT < 70 removed). Linked chains are matched by sequence
  alignment, superposed on holo pocket C-alphas, trimmed to matched residues;
  labels from the holo ligand. Reports AUROC/AP, site-vector cosine, and per
  selector the top region's DCC, IoU, shift and vector cosine vs holo.
- `test_encoder.py overfit --save_ckpt` saves the head; the dump now also
  carries point embeddings. `src/slurm/apo_test.sbatch` trains once if the
  checkpoint is missing, then evaluates.

## Next steps

1. Run the follow-up sweep and the apo test; pick the selector and make it
   the encoder default.
2. Decide how anchors are used in training: label predicted regions by IoU
   with the labelled pocket (positive / ignore / negative) instead of forcing
   the label in, so training and inference see the same kind of region; and
   whether a ligand is scored against the top anchor or the best-matching one.
3. Default to the pretrained start.
4. Then: ligand encoder + contrastive loss; full-scale surface extraction and
   `labels.py` over dataset_v1; dataset_v2 (per-entry dedup, larger eval).
