"""Build a dataset of protein-ligand interactions from the manifest, and record how.

A dataset is a selection of manifest rows plus a receptor input policy, defined
by a spec file kept in git (configs/datasets/<name>.yaml). Building it writes
datasets/<name>/ under --root:

    index.parquet       one row per interaction: every manifest column, plus
                        fp_row (re-indexed into this dataset's fingerprints),
                        manifest_fp_row, and the policy's input_id /
                        input_path / input_chains
    ligand_fps.npy      ECFP4 bits for this dataset's ligands only
    inputs.txt          unique "path CHAINS" lines, ready for extract.py
    spec.yaml           the spec as used, defaults filled in
    provenance.json     git commit and dirty flag, manifest checksum, command,
                        per-step row counts, split and input counts

Nothing here touches structures beyond optional existence checks, so a build
takes seconds and datasets are cheap to vary. What goes into dMaSIF is decided
by receptor_input.policy (see receptor_inputs.py) and is recorded per row, so
changing it later only means a new spec, not new selection code.

Selection runs in this order, each step logged with its row count:
    splits  ->  require  ->  exclude  ->  query  ->  dedup

Example:
    python src/build_dataset.py configs/datasets/dataset_v1.yaml --root $WORK --dry_run
    python src/build_dataset.py configs/datasets/dataset_v1.yaml --root $WORK --check_files
"""

import argparse
import getpass
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import receptor_inputs  # noqa: E402

DEFAULTS = {
    "description": "",
    "manifest": "meta/manifest.parquet",
    "fingerprints": "meta/ligand_fps.npy",
    "selection": {
        "splits": ["train", "val", "test"],
        "require": [],    # boolean manifest columns that must be true
        "exclude": [],    # boolean manifest columns that must be false
        "query": None,    # pandas query string, e.g. "ligand_num_heavy_atoms <= 40"
        "dedup": None,    # list of columns; keeps the first row by system_id
    },
    "receptor_input": {"policy": "plinder_receptor", "params": {}},
}


def load_spec(path):
    raw = yaml.safe_load(Path(path).read_text()) or {}
    # The file name IS the dataset name; a separate `name:` may only repeat it.
    stem = Path(path).stem
    raw.setdefault("name", stem)
    if raw["name"] != stem:
        raise SystemExit(f"{path}: name {raw['name']!r} does not match the file name {stem!r}")
    unknown = set(raw) - set(DEFAULTS) - {"name"}
    if unknown:
        raise SystemExit(f"{path}: unknown keys {sorted(unknown)}")
    spec = {"name": raw["name"]}
    for key, default in DEFAULTS.items():
        val = raw.get(key, default)
        if isinstance(default, dict):
            bad = set(val or {}) - set(default)
            if bad:
                raise SystemExit(f"{path}: unknown {key} keys {sorted(bad)}")
            val = {**default, **(val or {})}
        spec[key] = val
    return spec


# ----------------------------------------------------------------------------
# Selection
# ----------------------------------------------------------------------------
def _bool_cols(df, cols, what):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        flags = sorted(c for c in df.columns if c.startswith("pass_")) + ["strict_rep"]
        raise SystemExit(f"selection.{what}: no column {missing}. Available flags: {flags}")


def select(df, sel):
    """Apply the selection steps in order -> (rows, attrition log)."""
    log = []

    def step(name, d):
        log.append({"step": name, "rows": int(len(d)),
                    "receptors": int(d["receptor_key"].nunique())})
        return d

    df = step("manifest", df)
    df = step(f"splits {sel['splits']}", df[df["split"].isin(sel["splits"])])
    _bool_cols(df, sel["require"], "require")
    for c in sel["require"]:
        df = step(f"require {c}", df[df[c].fillna(False).astype(bool)])
    _bool_cols(df, sel["exclude"], "exclude")
    for c in sel["exclude"]:
        df = step(f"exclude {c}", df[~df[c].fillna(False).astype(bool)])
    if sel["query"]:
        df = step(f"query {sel['query']}", df.query(sel["query"]))
    if sel["dedup"]:
        missing = [c for c in sel["dedup"] if c not in df.columns]
        if missing:
            raise SystemExit(f"selection.dedup: no column {missing}")
        df = step(f"dedup {sel['dedup']}",
                  df.sort_values("system_id").drop_duplicates(sel["dedup"], keep="first"))
    return df.copy(), log


# ----------------------------------------------------------------------------
# Provenance
# ----------------------------------------------------------------------------
def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def git_state():
    def git(*args):
        r = subprocess.run(["git", "-C", str(HERE), *args], capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else None

    status = git("status", "--porcelain", "--untracked-files=no")
    return {"commit": git("rev-parse", "HEAD"), "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(status) if status is not None else None}


def file_record(path):
    st = path.stat()
    return {"path": str(path), "bytes": st.st_size, "sha256": sha256(path),
            "mtime": datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat()}


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("spec", help="configs/datasets/<name>.yaml")
    ap.add_argument("--root", default=os.environ.get("WORK", "."),
                    help="project root holding meta/, structures/, datasets/ (default $WORK)")
    ap.add_argument("--out", help="output directory (default <root>/datasets/<name>)")
    ap.add_argument("--dry_run", action="store_true", help="print attrition and counts; write nothing")
    ap.add_argument("--check_files", action="store_true",
                    help="add input_exists / system_exists columns by checking the disk")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()

    spec = load_spec(a.spec)
    root = Path(a.root)
    out = Path(a.out) if a.out else root / "datasets" / spec["name"]
    if out.exists() and not a.overwrite and not a.dry_run:
        raise SystemExit(f"{out} exists; pass --overwrite to rebuild it")

    manifest_path = root / spec["manifest"]
    fps_path = root / spec["fingerprints"]
    df = pd.read_parquet(manifest_path)
    df, log = select(df, spec["selection"])
    if not len(df):
        raise SystemExit("selection is empty")

    pol = spec["receptor_input"]
    inputs = receptor_inputs.resolve(df, pol["policy"], pol["params"], root)
    df = df.join(inputs)
    df.insert(0, "input_policy", pol["policy"])

    if a.check_files:
        df["input_exists"] = df["input_path"].map(lambda p: Path(p).exists())
        systems = root / "systems"
        df["system_exists"] = df["system_id"].map(
            lambda s: (systems / s / "system.cif").exists())

    print(f"\n{'step':60s} {'rows':>9s} {'receptors':>10s}")
    for s in log:
        print(f"{s['step'][:60]:60s} {s['rows']:>9d} {s['receptors']:>10d}")
    by_split = {k: int(v) for k, v in df["split"].value_counts().items()}
    n_inputs = int(df["input_id"].nunique())
    print(f"\nsplits: {by_split}")
    print(f"{len(df)} interactions -> {n_inputs} dMaSIF inputs under policy {pol['policy']!r} "
          f"({len(df) / n_inputs:.2f} interactions per input)")
    if a.check_files:
        print(f"inputs on disk:  {int(df.drop_duplicates('input_id')['input_exists'].sum())}/{n_inputs}")
        print(f"systems on disk: {int(df['system_exists'].sum())}/{len(df)}")
    if a.dry_run:
        print("\n--dry_run: nothing written")
        return

    # Fingerprints for this dataset only, so it does not depend on meta/ staying put.
    fps = np.load(fps_path)
    rows = np.sort(df["fp_row"].unique())
    df["manifest_fp_row"] = df["fp_row"]
    df["fp_row"] = df["fp_row"].map({r: i for i, r in enumerate(rows)}).astype(np.int32)

    out.mkdir(parents=True, exist_ok=True)
    df = df.reset_index(drop=True)
    df.to_parquet(out / "index.parquet", index=False)
    np.save(out / "ligand_fps.npy", fps[rows])
    uniq = df.drop_duplicates("input_id").sort_values("input_id")
    (out / "inputs.txt").write_text("".join(
        f"{p} {c}\n" if c else f"{p}\n" for p, c in zip(uniq["input_path"], uniq["input_chains"])))
    (out / "spec.yaml").write_text(yaml.safe_dump(spec, sort_keys=False))
    provenance = {
        "name": spec["name"],
        "built_at": datetime.now(timezone.utc).isoformat(),
        "built_by": getpass.getuser(),
        "host": platform.node(),
        "command": " ".join([Path(sys.executable).name] + sys.argv),
        "spec_file": str(Path(a.spec).resolve()),
        "git": git_state(),
        "manifest": file_record(manifest_path),
        "fingerprints": file_record(fps_path),
        "attrition": log,
        "counts": {"interactions": len(df), "inputs": n_inputs,
                   "receptors": int(df["receptor_key"].nunique()),
                   "ligands": len(rows), "splits": by_split},
    }
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    if provenance["git"]["dirty"]:
        print("[warn] repo has uncommitted changes; provenance commit does not fully "
              "describe the code that built this dataset")
    print(f"\nwrote {out}/  (index.parquet, ligand_fps.npy, inputs.txt, spec.yaml, provenance.json)")


if __name__ == "__main__":
    main()
