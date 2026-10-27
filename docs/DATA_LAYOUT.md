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
