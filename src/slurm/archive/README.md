# Archived job scripts

Spent experiments, moved here by refactor stage 5 (`docs/REFACTOR.md`) so
`src/slurm/` holds only what someone would run today. **Nothing here is
deleted or broken**: every script uses absolute `$WORK/src/slurm/...` paths
internally, so each still runs from this directory. Their results are in
`docs/RESULTS.md`, which is the record — these are kept to show exactly what
produced it.

| script | what it ran | why archived |
|---|---|---|
| `v3_diag.sh` | Round 1 of the v3 ladder: A `site+mean`, B `+pairs`, C `+BatchNorm`, 1,500 steps each | Done. All three trained; **C is the configuration** the project now uses. |
| `v3_diag2.sh` | Round 2: `v3d` `+chemo`, `v3e` `+bank/extra ligands`, `v3c+` C to 5,000 steps | Done. chemo neutral, **the bank is what killed v3**, C plateaus at ~0.18. |
| `v3_diag3.sh` | Round 3: `v3f` `+site decoys`, `v3g` `+tau floor` | Done. Decoys raise within-protein `site` 0.605 -> 0.794 but cost cross-protein MRR; the tau floor was a **null experiment** — tau never leaves 0.07, so it never bound. |
| `train_v2.sbatch` | The v2 preset over `train.sbatch`: memory bank 2,048, 1,024 extra ligands, a 32-dim descriptor branch, cosine schedule | The bank and extra ligands are **known to collapse training** (`v3e`/`v3g`), so this preset should not be run as-is. `runs/v2` exists at 4,884 steps and its results were never recorded — if that log turns up, it belongs in `RESULTS.md`. |

Live equivalents in `src/slurm/`: `train.sbatch` (the base), `train_v3.sbatch`
(the C preset), `v3_diag4.sh` (the current round), `eval.sbatch`,
`report.sbatch`.

The descriptor branch (`DESC_DIMS`) is the one v2 lever never tested on its
own — it rode along with the bank that broke everything. `DESC_DIMS=32` on
`train_v3.sbatch` would isolate it, and needs none of this file.
