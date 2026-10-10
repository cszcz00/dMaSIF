# Project status (handoff)

Last updated after the second encoder test (job on alphagpu51). Branch:
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

## Next steps

1. Persistence-based merging in `anchors.watershed_regions` (merge two flooded
   regions when the dip between them is shallow), plus larger smoothing as an
   option; rerun the encoder test with a few merge settings. Target: 1-2
   regions per protein, size ~ pocket size, cover > 0.5 at prec > 0.3.
2. Default to the pretrained start.
3. Then: ligand encoder + contrastive loss; full-scale surface extraction and
   `labels.py` over dataset_v1; dataset_v2 (per-entry dedup, larger eval).
