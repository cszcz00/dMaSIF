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

## Stage 1 — split `docs/STATUS.md` — DONE 2026-10-27

1,591 lines doing three jobs at once: a handoff, a lab notebook, and a design
record. It is appended to after every run and is now hard to navigate.

| file | holds |
|---|---|
| `docs/STATUS.md` | current state, standing constraints, what is running, next steps. Target ~250 lines. |
| `docs/RESULTS.md` | the chronological record: v1, the second round, v2, v3 and its A–I ladder, chemo, apo/holo, the probe, the region sweep. Append-only. |
| `docs/DATA.md` | what each directory under `$WORK` is, which job builds it, what invalidates it, and what `src/inventory.py` checks. |
| `docs/DESIGN.md` | the settled decisions (selector, scoring, region labels, other sites, batch negatives) now buried mid-STATUS. |

Landed as four files: STATUS 300 lines, RESULTS 790, DESIGN 532, DATA 68.
Verified mechanically: no source line used twice, and of the 29 original `##`
headings none is duplicated and only two are absent - `START HERE` and
`Pending jobs`, both deliberately replaced by STATUS's rewritten header and
`## Pending work`.

Results, design and data text moved VERBATIM. Status text was rewritten,
because a status file is a snapshot and the old one had rotted: it still led
with "the one open question" about a `train_check.sh` log from 2026-10-18,
still listed the training loop as "never run on a GPU", and its `Pending jobs`
section had the finished `train_check` run as item 0. That narrative was not
deleted - it is `RESULTS.md`'s "Getting the loop onto a GPU", where it is
correct as a record.

RESULTS is also reordered chronologically; the original interleaved the region
sweep with the v3 ladder. Six cross-references that the split or the reordering
broke were fixed: three "results recorded in this file" pointers now name
`RESULTS.md`, the Goal section's stale "never run on a GPU / outcome still
unseen" paragraph was rewritten, the Training loop heading in DESIGN now points
at `RESULTS.md`, and one "the v2 section below" became "above".

## Stage 2 — extract the self-tests — DONE 2026-10-27

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

Landed: `train.py` 2,437 -> 1,612 lines, `src/train_selftest.py` 891.
`selftest()` plus its two fixtures (`_fake_sdf`, `_FakeLigands`, used nowhere
else) moved; `train.py selftest` delegates from inside the function so there
is no import cycle.

Verified mechanically, since this container has no torch and cannot run it:
the moved block diffs BYTE-IDENTICAL against `git show HEAD:src/train.py`
lines 1336-2171; all ten names `train_selftest` imports exist as top-level
definitions in `train.py`; no reference to the moved fixtures is left behind;
`main()` still dispatches `selftest`; and the CLI helpers stayed put, so
`retrieval_report.py`'s three import sites still resolve. Both modules compile.

**Still to do by hand: run `python src/train.py selftest` in the container
once.** It should print the same eleven groups and pass. That is the one check
no static analysis covers.

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

## Stage 5 — `src/slurm/` hygiene — DONE 2026-10-27

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

Landed: the four moved with `git mv` (history preserved) into
`src/slurm/archive/`, which has a README saying what each ran and why it is
spent. `src/slurm/` is 26 -> 22 files. Every archived script still RUNS from
its new location — each uses absolute `$WORK/src/slurm/...` paths internally,
which was checked, not assumed.

Two scripts I had listed as candidates and kept instead: `smoke.sbatch` is a
container sanity check (numpy / torch / pykeops / pyg versions), not an
experiment; `region_test.sbatch` is superseded as a selector sweep but is
still referenced by `apo_test.sbatch` for how it trains the head, and is the
only way to regenerate the probability maps the sweep scores.

Nine stale path references were updated across `train.sbatch`,
`train_v3.sbatch`, the archived `train_v2.sbatch`'s own usage examples,
`RESULTS.md` (five) and `DESIGN.md`'s code map. A repo-wide grep confirms no
reference to the old paths remains.

## Stage 6 — a `data/` directory (PROPOSED, not started)

`$WORK` has ~16 top-level directories and the code is four of them. The user's
instinct is right, and so is the distinction they drew: `structures/` and
`systems/` are upstream PLINDER content, while `surfaces/`, `pockets/` and
`plip/` are things we computed. One correction to the premise: neither
`structures/` nor `systems/` is "all of PLINDER" - both hold only the
receptors and systems in OUR manifest. The genuinely PLINDER-wide copy is
`$PLINDER_DIR`, which already lives outside `$WORK`.

**A second axis matters more than upstream/derived, and `dataset_v2` is about
to make it urgent: what each artefact is KEYED by, because that says what a
rebuild costs.**

| artefact | keyed by | survives a dataset rebuild? |
|---|---|---|
| `structures/`, `systems/` | receptor_key / system_id | yes, and re-fetchable for free |
| `surfaces/`, `pockets/<finder>/` | `input_id` | **yes** - 19,293 files each, the expensive ones |
| `plip/` | `system_id` | yes, for rows that survive |
| `meta/`, `datasets/` | the dataset | no, rebuilt |
| ligand features (`runs/<n>/ligand_feats.npz`) | `fp_row` | **no** - `build_dataset` RE-INDEXES fp_row per dataset |

So the layout should make it obvious that a dataset_v2 rebuild does NOT touch
surfaces or pockets, and DOES invalidate ligand features. Proposed:

```
$WORK/
  src/ configs/ docs/ dMaSIF/            code, unchanged
  data/
    upstream/   structures/  systems/    re-fetchable; free to delete under disk pressure
    derived/    surfaces/  pockets/  plip/
    index/      meta/  datasets/
  runs/  reports/                        experiment outputs, not data
  tools/  envs/  keops_cache/            build and cache junk
```

`encoder_test/`, `apo_test/` and `probe/` are also experiment outputs and
could go under an `out/`; left out of this proposal because the user did not
ask and `runs/` alone is referenced 34 times in job scripts.

**The blocker, and why this splits in two.** `input_path` is not computed at
load time - it is a PERSISTED column holding a relative path
(`structures/<receptor_key>.cif`), written into `items.tsv` (`ITEM_COLS` in
probe_retrieval) and `index.parquet` (build_dataset), and resolved against the
cwd. **Moving `structures/` therefore invalidates every existing `items.tsv`
and `index.parquet`.** `systems/` is reached through `sdf_paths`, the same way.

The derived directories have no such problem: `surfaces/`, `pockets/` and
`plip/` appear only in job-script defaults and CLI flags, never inside a data
file.

So:

- **Now, cheap and safe: move the derived three.** `surfaces/` (7 references),
  `pockets/` (3), `plip/` (1), plus the `meta/` and `datasets/` defaults in the
  Python CLIs. Roughly a dozen default lines and a `git mv` on the HPC; nothing
  persisted changes.
- **With `dataset_v2`: move `upstream/`.** That rebuild regenerates
  `index.parquet` and `items.tsv` anyway, so the new `input_path` is written
  correct first time and the migration costs nothing extra. Doing it before
  then means either rewriting the parquet/TSV columns by hand or teaching
  `receptor_inputs.plinder_receptor` a configurable data root.

**Open question for the inventory to answer first: what is `feats/`?** It is
the `extract.py`-era surface cache (`feats/<tag>.npz`, per `ids.py` and
`labels.py`), from before `SurfaceStore` and `surfaces/`. Nothing current
writes it. If it is dead it should be deleted, not migrated, and it may be
sizeable. `src/inventory.py report` will say how many files and how large.
(Its description in `inventory.py` said "ligand feature blocks", which was
wrong and is fixed.)

## Not planned, and why

- **Renaming `probe_retrieval.py`.** It is referenced throughout
  `docs/STATUS.md` and in every result table. After stage 3 it is just the
  frozen-anchor probe experiment, which is what the name says.
- **Touching `dMaSIF/`.** Vendored upstream plus PyTorch-2 fixes. Leave it.
- **A `tests/` directory with pytest.** The self-tests run in the container
  with no test runner installed and are invoked as `<module>.py selftest`,
  which works from an sbatch. Adding pytest adds a dependency for no gain.
