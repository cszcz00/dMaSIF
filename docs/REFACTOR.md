# Refactor plan

A plan with a lifecycle: delete sections as they land. Nothing here is done
yet. Stages are ordered by risk, lowest first, and each is independently
useful — stopping after any one leaves the tree better than before.

The codebase is 13,382 lines over 24 modules in `src/`, 26 files in
`src/slurm/`, one dataset config, and a 1,591-line `docs/STATUS.md`. It grew
an artefact and a module per experiment and nothing has been consolidated
since the training loop landed. The problems below are real but none of them
is breaking anything today, so **correctness beats tidiness at every step**:
the invariants that keep the probe and the training loop agreeing on what a
mask, a split, a batch and a metric are were expensive to establish, and a
refactor that silently breaks one costs more than the clarity is worth.

Rule for every stage: move code **verbatim**, change imports only, and run the
affected self-tests before and after. If a move requires editing the moved
lines, it belongs in a separate commit from the move.

## Stage 1 — split `docs/STATUS.md` (no code, no risk)

1,591 lines doing three jobs at once: a handoff, a lab notebook, and a design
record. It is appended to after every run and is now hard to navigate.

| file | holds |
|---|---|
| `docs/STATUS.md` | current state, standing constraints, what is running, next steps. Target ~250 lines. |
| `docs/RESULTS.md` | the chronological record: v1, the second round, v2, v3 and its A–I ladder, chemo, apo/holo, the probe, the region sweep. Append-only. |
| `docs/DATA.md` | what each directory under `$WORK` is, which job builds it, what invalidates it, and what `src/inventory.py` checks. |
| `docs/DESIGN.md` | the settled decisions (selector, scoring, region labels, other sites, batch negatives) now buried mid-STATUS. |

`START HERE` stays at the top of `STATUS.md` and links to the other three.

## Stage 2 — extract the self-tests (mechanical, low risk)

`src/train.py` is 2,436 lines, of which **1,101 (45%) is `selftest()` plus
`main()`'s argparse**. The self-test is ~990 lines of pure test code imported
by nothing.

`selftest()` (lines 1336-2325) and its fixtures → `src/train_selftest.py`,
invoked by `train.py selftest` as a one-line delegate so the command does not
change. Nothing else moves. `train.py` drops to ~1,450 lines and the eleven
test groups become findable.

**`train.py` is the only module that needs this.** Measured: probe_retrieval
12% self-test, external_pockets 19%, plip_labels 15%, chemotype 22% — all
healthy, a self-test beside the code it tests. train.py is the outlier at 41%,
and only because it accumulated eleven groups covering the whole pipeline.

Verify: `train.py selftest` prints the same group list and passes.

## Stage 3 — a real shared core (medium risk, highest clarity gain)

Today `train.py` imports `draw_batch`, `group_index`, `info_nce`, `mask_pairs`,
`split_by_group`, `retrieval_metrics`, `tanimoto` and `anchor_spread` from
`probe_retrieval.py`. That is deliberate — the probe and the loop must not
drift — but it means the project's shared library lives inside an experiment
script, and the import direction implies the loop depends on the probe, which
is backwards.

Extract, verbatim, into modules named for what they are:

| new module | from | holds |
|---|---|---|
| `src/masks.py` | probe_retrieval | `mask_pairs`, `tanimoto`, and `train.cross_mask` |
| `src/sampling.py` | probe_retrieval | `group_index`, `draw_batch`, `sample_weights`, `split_by_group` |
| `src/metrics.py` | probe_retrieval | `retrieval_metrics`, `anchor_spread`, `info_nce`, and `train.nce_extended` |

The exact coupling today is nine names over seven call sites in `train.py`:
`mask_pairs`, `tanimoto`, `anchor_spread`, `retrieval_metrics`, `draw_batch`,
`group_index`, `info_nce`, `split_by_group` — and `apply_overrides`, which is
the odd one out: it writes region CLI flags onto an `EncoderConfig` and belongs
with the model/config code, not with masks or metrics. Decide its home when
stage 4 moves `build_model`; until then leave it where it is.

Note every one of these imports is FUNCTION-LOCAL, not module-level. That is
deliberate throughout the repo: it keeps torch-free paths torch-free (the CPU
venv has no torch, and `inventory.py`, `companion_funnel.py` and
`retrieval_report.py report` all depend on that). Preserve the pattern when
moving them.

Both `probe_retrieval.py` and `train.py` then import from these. The anti-drift
property is strengthened, not weakened: there is one definition and neither
module owns it. **This is the stage most likely to break something**, so do it
one module at a time, each with its own commit and a full self-test run of
both `train.py` and `probe_retrieval.py`.

## Stage 4 — split what is left of `train.py` (medium risk, do last)

After stages 2–3 it is ~1,200 lines and still holds five separable concerns:

| new module | holds |
|---|---|
| `src/model.py` | `build_model`, `RetrievalModel`, `load_head`, `pocket_source`, `save_ckpt`, `load_model_from_ckpt` |
| `src/batching.py` | `build_batch`, `site_decoys`, `anchor_block` |
| `src/losses.py` | `ProteinBank`, `site_loss`, `lr_at` |
| `src/evaluate.py` | `collect`, `site_rate`, `evaluate`, `evaluate_only` |
| `src/train.py` | the loop, `cache`, the CLI |

Caveat worth respecting: `train.py`'s module docstring is the best single
explanation of the method anywhere in the repo (what is learned, the three
things that make a batch honest, why the pocket term is not optional). It
should move to `docs/DESIGN.md` or a package `__init__`, not be chopped up
across five files.

Open question for this stage: `src/` is a flat directory on `sys.path` via
`sys.path.insert(0, HERE)` in every module. Five more modules makes a package
(`src/dmasif_retrieval/`) tempting, but that changes every import line and
every sbatch's `python src/<x>.py` entry point. **Not worth it** unless
something else forces it.

## Stage 5 — `src/slurm/` hygiene (low risk)

26 files, including four `v3_diag*.sh` that are now spent history and three
`train_*.sbatch` presets layered over `train.sbatch`.

- Move `v3_diag.sh`, `v3_diag2.sh`, `v3_diag3.sh` to `src/slurm/archive/`
  once their results are in `docs/RESULTS.md`. The ladder they describe is the
  record; the script is not needed to re-read it.
- Keep `train.sbatch` and `train_v3.sbatch` (the live preset). `train_v2.sbatch`
  can go to `archive/` — the bank it switches on is now known to kill training.
- Leave `build*.sbatch`, `fetch.sbatch`, `fpocket.sbatch`, `plip.sbatch`,
  `cache_surfaces.sbatch`, `eval.sbatch`, `report.sbatch` alone: each is the
  only way to rebuild one artefact.

## Not planned, and why

- **Renaming `probe_retrieval.py`.** It is referenced throughout
  `docs/STATUS.md` and in every result table. After stage 3 it is just the
  frozen-anchor probe experiment, which is what the name says.
- **Touching `dMaSIF/`.** Vendored upstream plus PyTorch-2 fixes. Leave it.
- **A `tests/` directory with pytest.** The self-tests run in the container
  with no test runner installed and are invoked as `<module>.py selftest`,
  which works from an sbatch. Adding pytest adds a dependency for no gain.
