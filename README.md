# dMaSIF pipeline

This repo is the whole project folder on the HPC: a vendored copy of dMaSIF plus the code we use to run it on our data.

```
dmasif/
├── dMaSIF/            # upstream dMaSIF + PyTorch 2 fixes; keep edits here minimal
├── src/               # our pipeline: fetch, manifest, extract, labels, probes
│   └── slurm/         # sbatch scripts + Apptainer definition (dmasif.def)
├── job_console/       # shared job-submission console (separate tool, ignored)
├── meta/              # dataset manifest from build_manifest.py (ignored, HPC only)
├── structures/        # raw structures              (ignored, HPC only)
├── systems/           # per-system folders          (ignored, HPC only)
├── feats/             # extracted dMaSIF features   (ignored, HPC only)
└── keops_cache/       # PyKeOps compiled kernels    (ignored, cache)
```

The upstream README is at `dMaSIF/README.md`.
