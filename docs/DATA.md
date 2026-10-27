# Data

What exists under `$WORK`, where it comes from, and what invalidates it.
`python src/inventory.py report --items runs/v1/items.tsv` checks all of it
against the disk: per-artefact coverage, the exact staleness test for ligand
features, and a summary of every run directory. Run that before trusting
anything below.

Split out of `STATUS.md` on 2026-10-27; section text is unchanged.

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
