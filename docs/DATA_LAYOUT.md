# `data/`

Everything the project reads but does not author. Built by jobs, never edited
by hand, all gitignored. `python src/inventory.py report --items
out/runs/v1/items.tsv` checks it against the disk.

**The axis that matters is what each artefact is KEYED by, because that says
what a rebuild costs.** `dataset_v2` makes this concrete: it regenerates the
index and invalidates ligand features, and touches neither surfaces nor
pockets, which are the expensive ones.

| directory | keyed by | built by | survives a dataset rebuild? |
|---|---|---|---|
| `meta/` | the manifest | `build_manifest.py` | no |
| `datasets/<name>/` | the dataset | `build_dataset.py` | no |
| `surfaces/` | `input_id` | `train.py cache`, or lazily by training | **yes** — 19,293 files |
| `pockets/<finder>/` | `input_id` | `external_pockets.py run` | **yes** — 19,293 files |
| `plip/` | `system_id` | `plip_labels.py run` | yes, for rows that survive |

Ligand features (`out/runs/<name>/ligand_feats.npz`) are the trap: `fp_row` is
RE-INDEXED per dataset by `build_dataset.py`, so they do **not** survive a
rebuild even though they look like a cache. `inventory.py` tests them for the
RemoveAllHs staleness bug directly rather than by date.

## What is actually there (`inventory.py report`, 2026-10-27)

| | count | size | note |
|---|---|---|---|
| `structures/` | 223,972 cif | **69.5 GB** | the whole MANIFEST's receptors; only **19,293 (8.6%)** are in dataset_v1 |
| `systems/` | 307,337 dirs | (inodes) | one directory per PLINDER system; the biggest inode consumer here |
| `data/surfaces/` | 19,293 | 5.8 GB | 100% coverage of items.tsv |
| `data/pockets/fpocket/` | 19,293 | 310 MB | 100% coverage, 0 failures |
| `data/plip/` | 20,711 | 19.9 MB | 100% coverage of rows |
| `data/meta/manifest.parquet` | 341,945 rows | 23 MB | **28,678 strict_rep** |
| `data/datasets/dataset_v1/` | 21,209 rows | 2.4 MB | 19,782 receptors, 8,378 ligands |

Three things to read off that:

1. **`structures/` is the disk problem, not `surfaces/`.** 69.5 GB of which
   we use 8.6%. It was fetched for the whole manifest rather than the dataset.
   Do NOT prune it blind: a relaxed `dataset_v2` query will want more of it
   than dataset_v1 does. Prune only against the receptor set that survives the
   new filters, and only under real disk pressure.
2. **`systems/` is 307,337 directories.** HPC filesystems meter inodes as well
   as bytes. Its size is not reported because sizing it means walking 307k
   directories; use `du -sh systems` if it matters.
3. **The `n_ligand_chains == 1` filter costs 26% of strict_rep**: 28,678 ->
   21,209 rows. Relaxing it adds ~7,469 rows, **+35% on the current dataset**.
   That is the headline number `companion_funnel.py funnel` was written to
   get; it still has to say how many new TARGET GROUPS come with them, which
   is what the memorisation ceiling actually cares about.

## Still at the repository root, and why

`structures/` and `systems/` are upstream PLINDER content and belong here too,
but they cannot move yet: **`input_path` and `sdf_paths` are PERSISTED
relative-path columns** inside every `items.tsv` and `index.parquet`
(`ITEM_COLS` in `probe_retrieval.py`, written by `build_dataset.py`), resolved
against the working directory. Moving them silently breaks every existing
table. They move with `dataset_v2`, which regenerates those columns anyway, so
the migration costs nothing then. See `docs/REFACTOR.md` stage 6.

`keops_cache/` stays because `$KEOPS_CACHE_FOLDER` is set in the submitting
shell, not by any script here.
