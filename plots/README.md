# plots/

Figures for presentations, made on a laptop from data copied off the HPC.

Nothing here touches torch-on-GPU or KeOps. Scripts read the same relative
paths as the rest of the repo (`out/runs/...`, `data/...`), so run them from
the repository root after copying the inputs into place with
`rsync -avR "$H/./<path>" .` (see the top-level README for the layout).

Environment: the CPU venv (`envs/cpu-requirements.txt`) plus `matplotlib`;
`torch` (CPU build) only for `src/ligand_embed.py`.

Rendered figures go to `plots/output/`, which is gitignored: they are
regenerable from the scripts and the data.
