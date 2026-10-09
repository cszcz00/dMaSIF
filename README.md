# dMaSIF pipeline

This repo is the whole project folder on the HPC: a vendored copy of dMaSIF plus the code we use to run it on our data.

```
dmasif/
├── dMaSIF/            # upstream dMaSIF + PyTorch 2 fixes; keep edits here minimal
├── src/               # our pipeline: fetch, manifest, extract, labels, probes
│   └── slurm/         # sbatch scripts + Apptainer definition (dmasif.def)
├── configs/datasets/  # dataset specs (selection + receptor input policy), tracked
├── datasets/          # built datasets from src/build_dataset.py (ignored, HPC only)
├── job_console/       # shared job-submission console (separate tool, ignored)
├── meta/              # dataset manifest from build_manifest.py (ignored, HPC only)
├── structures/        # raw structures              (ignored, HPC only)
├── systems/           # per-system folders          (ignored, HPC only)
├── feats/             # extracted dMaSIF features   (ignored, HPC only)
└── keops_cache/       # PyKeOps compiled kernels    (ignored, cache)
```

The upstream README is at `dMaSIF/README.md`.

## Environments

| Where | For | Defined by |
|---|---|---|
| `~/venvs/plinder` (CPU) | `build_manifest`, `fetch_structures`, `build_dataset`, `labels`, `sanity` | `envs/cpu-requirements.txt` |
| `dmasif_sandbox` container (GPU) | `extract`, `probe_pocket` | `src/slurm/dmasif.def` |

Not conda `base`: nothing in the pipeline assumes it.

## Building a dataset

A dataset is a spec in `configs/datasets/<name>.yaml`: which manifest rows to
keep, and which receptor input policy (`src/receptor_inputs.py`) decides what
goes into dMaSIF for each interaction.

```bash
python src/build_dataset.py configs/datasets/strict_v1.yaml --dry_run      # attrition only
python src/build_dataset.py configs/datasets/strict_v1.yaml --check_files  # writes datasets/strict_v1/
```

Each build writes `provenance.json` with the git commit, manifest checksum and
per-step row counts. Commit before building so that commit describes the code.
