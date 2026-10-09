"""Read-only report on the PLINDER data on disk: meta/, structures/, systems/.

Answers "what do we actually have" without assuming the pipeline finished:

    meta/        manifest columns, types, nulls, flag counts, splits;
                 fingerprint matrix; receptors.txt vs the manifest
    structures/  how many receptor files, leftovers from interrupted runs,
                 coverage of the manifest (all rows and strict_rep);
                 a few sampled files opened: chains, residues, atoms, HETATMs
    systems/     how many folders, how many complete (system.cif present),
                 coverage; sampled folders: files, chains, ligand residues,
                 and whether they match the manifest's ligand chain and CCD code

Writes nothing. Run it in the CPU venv and keep the output:
    python src/inspect_data.py --root $WORK | tee inspect.txt
"""

import argparse
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from ids import ligand_chain  # noqa: E402


def section(name):
    print(f"\n{'=' * 78}\n{name}\n{'=' * 78}")


def pct(n, d):
    return f"{n}/{d} ({100 * n / d:.1f}%)" if d else f"{n}/0"


# ----------------------------------------------------------------------------
def inspect_meta(meta):
    section(f"META  {meta}")
    for p in sorted(meta.iterdir()):
        print(f"  {p.name:28s} {p.stat().st_size / 1e6:9.1f} MB")

    df = pd.read_parquet(meta / "manifest.parquet")
    print(f"\nmanifest.parquet: {len(df)} rows x {len(df.columns)} columns")
    print(f"  unique system_id {df['system_id'].nunique()}   "
          f"receptor_key {df['receptor_key'].nunique()}   "
          f"ligand SMILES {df['ligand_rdkit_canonical_smiles'].nunique() if 'ligand_rdkit_canonical_smiles' in df else '-'}")

    print(f"\n  {'column':55s} {'dtype':10s} {'null%':>6s}  summary")
    for c in df.columns:
        s = df[c]
        null = 100 * s.isna().mean()
        if s.dtype == bool:
            summ = f"true {int(s.sum())}"
        elif pd.api.types.is_numeric_dtype(s):
            q = s.dropna()
            summ = (f"min {q.min():.4g}  median {q.median():.4g}  max {q.max():.4g}"
                    if len(q) else "all null")
        else:
            top = s.astype(str).value_counts().head(2)
            summ = f"{s.nunique()} unique, e.g. " + ", ".join(f"{k[:28]} ({v})" for k, v in top.items())
        print(f"  {c[:55]:55s} {str(s.dtype)[:10]:10s} {null:6.1f}  {summ}")

    print("\nsplit x strict_rep:")
    if "strict_rep" in df:
        print(pd.crosstab(df["split"], df["strict_rep"], margins=True).to_string())
    else:
        print(df["split"].value_counts().to_string())

    fps_p = meta / "ligand_fps.npy"
    if fps_p.exists():
        fps = np.load(fps_p, mmap_mode="r")
        bits = np.asarray(fps.sum(axis=1)).ravel()
        print(f"\nligand_fps.npy: shape {fps.shape} dtype {fps.dtype}; bits set per ligand "
              f"min {bits.min()} median {int(np.median(bits))} max {bits.max()}; "
              f"all-zero rows {int((bits == 0).sum())}")
        if "fp_row" in df:
            print(f"  manifest fp_row range {df['fp_row'].min()}..{df['fp_row'].max()} "
                  f"(matrix has {fps.shape[0]} rows)")

    rec_p = meta / "receptors.txt"
    if rec_p.exists():
        keys = [ln.split("\t")[0] for ln in rec_p.read_text().splitlines() if ln.strip()]
        man = set(df["receptor_key"])
        print(f"\nreceptors.txt: {len(keys)} lines; "
              f"in manifest {pct(len(set(keys) & man), len(set(keys)))}; "
              f"manifest receptors missing from it {len(man - set(keys))}")
    z = meta / "system_zips.txt"
    if z.exists():
        print(f"system_zips.txt: {len(z.read_text().split())} archives")
    return df


# ----------------------------------------------------------------------------
def scan(d):
    """One pass over a big directory -> (names of entries, Counter of kinds)."""
    t = time.time()
    names, kinds, size = set(), Counter(), 0
    with os.scandir(d) as it:
        for e in it:
            if e.is_dir(follow_symlinks=False):
                kinds["dir"] += 1
            else:
                kinds[Path(e.name).suffix or "(none)"] += 1
                size += e.stat(follow_symlinks=False).st_size
            names.add(e.name)
    print(f"  scanned {len(names)} entries in {time.time() - t:.0f}s: {dict(kinds)}; "
          f"files total {size / 1e9:.2f} GB")
    return names


def read_cif(path):
    """-> {chain: (n_residues, n_atoms)}, Counter of HETATM resnames, n hydrogens."""
    from Bio.PDB import MMCIFParser

    model = next(iter(MMCIFParser(QUIET=True).get_structure("x", str(path))))
    chains, het, n_h = {}, Counter(), 0
    for ch in model:
        res = list(ch)
        chains[ch.id] = (len(res), sum(len(r) for r in res))
        for r in res:
            if r.id[0] != " ":
                het[r.get_resname().strip()] += 1
            n_h += sum(1 for a in r if a.element == "H")
    return chains, het, n_h


def inspect_structures(d, df, n, rng):
    section(f"STRUCTURES  {d}")
    if not d.is_dir():
        print("  missing")
        return
    names = scan(d)
    cifs = {x[:-4] for x in names if x.endswith(".cif")}
    parts = [x for x in names if x.endswith(".part")]
    print(f"  receptor .cif files {len(cifs)}; leftover .part files {len(parts)}; "
          f".tmp dir {'present' if '.tmp' in names else 'absent'}")
    ei = d / "extract_inputs.txt"
    if ei.exists():
        print(f"  extract_inputs.txt: {len(ei.read_text().splitlines())} lines")

    all_keys = set(df["receptor_key"])
    print(f"  coverage, all manifest receptors: {pct(len(all_keys & cifs), len(all_keys))}")
    if "strict_rep" in df:
        sk = set(df.loc[df["strict_rep"], "receptor_key"])
        print(f"  coverage, strict_rep receptors:   {pct(len(sk & cifs), len(sk))}")
    print(f"  files not in manifest: {len(cifs - all_keys)}")

    for key in rng.sample(sorted(cifs), min(n, len(cifs))):
        p = d / f"{key}.cif"
        chains, het, n_h = read_cif(p)
        print(f"\n  {key}.cif  {p.stat().st_size / 1e3:.0f} kB  hydrogens {n_h}")
        print(f"    chains {{chain: (residues, atoms)}} {chains}")
        print(f"    HETATM residues {dict(het.most_common(8)) or 'none'}")


def inspect_systems(d, df, n, rng):
    section(f"SYSTEMS  {d}")
    if not d.is_dir():
        print("  missing")
        return
    names = scan(d)
    print("  checking which folders hold system.cif (one stat per folder)...")
    t = time.time()
    complete = {s for s in names if (d / s / "system.cif").exists()}
    print(f"  folders {len(names)}; complete {pct(len(complete), len(names))} "
          f"({time.time() - t:.0f}s)")

    all_sys = set(df["system_id"])
    print(f"  coverage, all manifest systems: {pct(len(all_sys & complete), len(all_sys))}")
    if "strict_rep" in df:
        ss = set(df.loc[df["strict_rep"], "system_id"])
        print(f"  coverage, strict_rep systems:   {pct(len(ss & complete), len(ss))}")
    print(f"  folders not in manifest: {len(names - all_sys)}")

    rows = df.drop_duplicates("system_id").set_index("system_id")
    pool = sorted(complete & all_sys)
    for sid in rng.sample(pool, min(n, len(pool))):
        r = rows.loc[sid]
        files = sorted(str(p.relative_to(d / sid)) for p in (d / sid).rglob("*") if p.is_file())
        chains, het, _ = read_cif(d / sid / "system.cif")
        want = ligand_chain(r.get("ligand_id"))
        print(f"\n  {sid}   manifest ligand {r.get('ligand_ccd_code')} on {sorted(want)}  "
              f"split {r.get('split')}  strict_rep {r.get('strict_rep')}")
        print(f"    files {files}")
        print(f"    chains {chains}")
        print(f"    HETATM residues {dict(het.most_common(8)) or 'none'}")
        print(f"    ligand chain present: {bool(want) and want <= set(chains)}   "
              f"CCD code among HETATMs: {r.get('ligand_ccd_code') in het}")


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=os.environ.get("WORK", "."))
    ap.add_argument("--n", type=int, default=4, help="files to open per folder")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    root = Path(a.root)
    rng = random.Random(a.seed)
    df = inspect_meta(root / "meta")
    inspect_structures(root / "structures", df, a.n, rng)
    inspect_systems(root / "systems", df, a.n, rng)

    log = root / "fetch_110950.log"
    if log.exists():
        section(f"FETCH LOG  {log.name} (last 5 lines)")
        print("\n".join(log.read_text(errors="replace").splitlines()[-5:]))


if __name__ == "__main__":
    main()
