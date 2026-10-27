"""What data exists on disk, is it complete, and is any of it stale?

Why. The tree under $WORK has grown an artefact per experiment - structures,
surfaces, fpocket pockets, PLIP interactions, ligand features, manifests,
datasets, run directories - all gitignored, all built by different jobs, and
several of them INVALIDATED by code changes nobody records against the files.
docs/DATA.md says what each one is for; this says what is actually there.

Three things it answers that `du` and `ls` cannot:

  coverage     a surface / pocket file / PLIP file per row of an items.tsv.
               A missing surface is sampled lazily and costs time; a missing
               pocket file silently drops that receptor's regions (training
               falls back to the ball and `fb` rises); a missing PLIP file is
               only a gap in a label set not yet wired in. The counts say
               which.
  staleness    ligand_feats.npz built before the RemoveAllHs fix kept isotopic
               H, H2 and hydrides as GRAPH atoms, so `n_heavy` disagrees with
               the graph's own atom counts. That is an exact test, not a date
               heuristic, and docs/STATUS.md records probe/ and train_check/
               as affected. Any mismatch here means rebuild before trusting a
               run that used it.
  runs         every runs/<name>/ summarised from history.json alone (no torch,
               no checkpoint load): evaluations, last step, best val MRR and
               where, last cosdiff and tau. This is the index of what each
               experiment directory actually contains, which otherwise lives
               only in docs/STATUS.md prose.

Read-only. It opens parquet, .npz headers and history.json; it never writes,
moves or deletes anything, so it is safe to run while jobs are training.

Example:
    # CPU venv (pandas for the parquet sections, numpy otherwise; no torch)
    python src/inventory.py report --items runs/v1/items.tsv
    python src/inventory.py report --root $WORK --items runs/v1/items.tsv --tsv inv.tsv
    python src/inventory.py selftest          # planted tree, no real data

`--items` is optional: without it the coverage checks are skipped and
everything else still reports.

Depends on: numpy; pandas only for the manifest / dataset sections (absent
            pandas degrades to "unreadable" rather than failing). No torch.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# directory -> (glob, what it is). The layout docs/STATUS.md "Layout" records.
TREES = [
    ("structures", "*.cif", "PLINDER receptor.cif per receptor_key"),
    ("surfaces", "*.npz", "cached dMaSIF surfaces, one per input_id"),
    ("systems", "*", "PLINDER system directories (ligand SDFs)"),
    ("plip", "*.npz", "PLIP interactions per training row"),
    ("feats", "*", "ligand feature blocks"),
    ("reports", "*", "retrieval_report dumps and TSVs"),
]


def human(n):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.0f}{unit}" if unit == "B" else f"{n:.1f}{unit}"
        n /= 1024


def dir_stats(path, pattern="*"):
    """-> (n_files, total_bytes, newest_mtime) without following into subdirs."""
    path = Path(path)
    if not path.is_dir():
        return None
    n, size, newest = 0, 0, 0.0
    for p in path.glob(pattern):
        try:
            st = p.stat()
        except OSError:
            continue
        n += 1
        size += st.st_size if p.is_file() else 0
        newest = max(newest, st.st_mtime)
    return n, size, newest


def read_items(path):
    """items.tsv -> list of dicts. A plain csv read on purpose.

    train_data.read_items DROPS rows whose ligand failed featurisation, which
    is right for training and wrong for an inventory: a dropped row's surface
    still sits on disk and still has to be accounted for.
    """
    with open(path, newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def coverage(rows, directory, key, suffix=".npz"):
    """-> (n_distinct_keys, n_present, [missing examples]) for one artefact dir."""
    d = Path(directory)
    want = {r[key] for r in rows if r.get(key)}
    if not d.is_dir():
        return len(want), 0, sorted(want)[:3]
    have = {p.name[:-len(suffix)] for p in d.glob(f"*{suffix}")}
    missing = sorted(want - have)
    return len(want), len(want & have), missing[:3]


def ligand_feats(path):
    """-> dict describing a ligand_feats.npz, including the staleness test.

    Stale test: `n_heavy` is written by ligand_features from RDKit's heavy-atom
    count, while atom_ptr delimits the GRAPH's atoms. Before RemoveAllHs the
    graph kept isotopic H / H2 / hydrides, so the two disagree for those
    ligands. Equality is the post-fix invariant.
    """
    p = Path(path)
    if not p.is_file():
        return {"path": str(p), "present": False}
    out = {"path": str(p), "present": True, "bytes": p.stat().st_size}
    try:
        with np.load(p, allow_pickle=False) as f:
            keys = set(f.files)
            out["n_ligands"] = int(len(f["fp"])) if "fp" in keys else None
            out["failed"] = int(f["failed"].sum()) if "failed" in keys else None
            out["desc_dim"] = int(f["desc"].shape[1]) if "desc" in keys else None
            if {"n_heavy", "atom_ptr"} <= keys:
                graph = np.diff(f["atom_ptr"])
                bad = int((f["n_heavy"] != graph).sum())
                out["graph_vs_heavy_mismatch"] = bad
                out["stale"] = bad > 0
            else:
                out["stale"] = None       # cannot tell from this file
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
    return out


def run_summary(d):
    """runs/<name>/ -> dict from history.json and the files present. No torch."""
    d = Path(d)
    out = {"name": d.name}
    for f in ("best.pt", "last.pt", "items.tsv", "ligand_feats.npz"):
        p = d / f
        out[f] = p.stat().st_size if p.is_file() else 0
    h = d / "history.json"
    if not h.is_file():
        out["evals"] = 0
        return out
    try:
        hist = json.loads(h.read_text())
    except Exception as e:
        out["error"] = f"history.json unreadable: {type(e).__name__}"
        return out
    fin = lambda v: isinstance(v, (int, float)) and np.isfinite(v)
    mrrs = [(e.get("mrr"), e.get("step")) for e in hist if fin(e.get("mrr"))]
    out["evals"] = len(hist)
    out["last_step"] = hist[-1].get("step") if hist else None
    if mrrs:
        best, at = max(mrrs)
        out["best_mrr"], out["best_at"] = best, at
    for k, name in (("cos_diff", "cosdiff"), ("tau", "tau"),
                    ("fallback_rate", "fb")):
        vals = [e.get(k) for e in hist if fin(e.get(k))]
        if vals:
            out[name] = vals[-1]
    return out


def collect(a):
    """-> the whole inventory as a dict of sections (what `selftest` asserts on)."""
    root = Path(a.root)
    inv = {"root": str(root), "trees": [], "runs": [], "coverage": [],
           "feats": [], "datasets": [], "notes": []}

    for name, pattern, what in TREES:
        st = dir_stats(root / name, pattern)
        inv["trees"].append({"dir": name, "what": what,
                             "present": st is not None,
                             "n": None if st is None else st[0],
                             "bytes": None if st is None else st[1]})
    # pockets/ holds one subdirectory per finder
    pk = root / "pockets"
    if pk.is_dir():
        for sub in sorted(p for p in pk.iterdir() if p.is_dir()):
            st = dir_stats(sub, "*.npz")
            failed = sum(len(f.read_text().splitlines())
                         for f in sub.glob("failed*.tsv"))
            inv["trees"].append({"dir": f"pockets/{sub.name}",
                                 "what": "alpha spheres per receptor, by pocket",
                                 "present": True, "n": st[0], "bytes": st[1],
                                 "failed": failed})
    else:
        inv["trees"].append({"dir": "pockets", "what": "fpocket output",
                             "present": False, "n": None, "bytes": None})

    # meta / datasets, the only sections needing pandas
    meta = root / "meta"
    for f, what in (("manifest.parquet", "every usable (system, ligand) row"),
                    ("ligand_fps.npy", "ECFP4 bits per distinct ligand"),
                    ("receptors.txt", "receptor_key -> surface tag")):
        p = meta / f
        rec = {"file": f"meta/{f}", "what": what, "present": p.is_file(),
               "bytes": p.stat().st_size if p.is_file() else 0}
        if p.is_file() and f.endswith(".parquet"):
            try:
                import pandas as pd
                df = pd.read_parquet(p, columns=None)
                rec["rows"] = len(df)
                if "strict_rep" in df.columns:
                    rec["strict_rep"] = int(df["strict_rep"].fillna(False).sum())
            except Exception as e:
                rec["error"] = f"{type(e).__name__}: {e}"
        elif p.is_file() and f.endswith(".npy"):
            try:
                rec["shape"] = list(np.load(p, mmap_mode="r").shape)
            except Exception as e:
                rec["error"] = f"{type(e).__name__}: {e}"
        inv["datasets"].append(rec)
    ds = root / "datasets"
    if ds.is_dir():
        for sub in sorted(p for p in ds.iterdir() if p.is_dir()):
            idx = sub / "index.parquet"
            rec = {"file": f"datasets/{sub.name}/index.parquet",
                   "what": "a built dataset", "present": idx.is_file(),
                   "bytes": idx.stat().st_size if idx.is_file() else 0}
            if idx.is_file():
                try:
                    import pandas as pd
                    df = pd.read_parquet(idx)
                    rec["rows"] = len(df)
                    for c, k in (("input_id", "receptors"), ("fp_row", "ligands")):
                        if c in df.columns:
                            rec[k] = int(df[c].nunique())
                    if "split" in df.columns:
                        rec["splits"] = {str(k): int(v) for k, v in
                                         df["split"].value_counts().items()}
                except Exception as e:
                    rec["error"] = f"{type(e).__name__}: {e}"
            inv["datasets"].append(rec)

    # ligand features: the run dirs' own, plus any given explicitly
    seen = set()
    cands = [Path(p) for p in (a.lig_feats or [])]
    runs_dir = root / "runs"
    if runs_dir.is_dir():
        cands += sorted(runs_dir.glob("*/ligand_feats.npz"))
    for p in cands:
        if p in seen:
            continue
        seen.add(p)
        rec = ligand_feats(p)
        inv["feats"].append(rec)
        if rec.get("stale"):
            inv["notes"].append(
                f"STALE ligand features: {p} has {rec['graph_vs_heavy_mismatch']} "
                "ligands whose graph atoms != heavy atoms, i.e. built before the "
                "RemoveAllHs fix. Rebuild with ligand_features.py before trusting "
                "a run that used it.")

    # coverage against one items.tsv
    if a.items:
        rows = read_items(a.items)
        inv["items"] = {"path": a.items, "rows": len(rows),
                        "input_ids": len({r["input_id"] for r in rows
                                          if r.get("input_id")})}
        checks = [("surfaces", root / "surfaces", "input_id")]
        if pk.is_dir():
            for sub in sorted(p for p in pk.iterdir() if p.is_dir()):
                checks.append((f"pockets/{sub.name}", sub, "input_id"))
        if (root / "plip").is_dir():
            checks.append(("plip", root / "plip", "system_id"))
        for label, d, key in checks:
            if not rows or key not in rows[0]:
                continue
            want, have, missing = coverage(rows, d, key)
            inv["coverage"].append({"artefact": label, "key": key, "want": want,
                                    "have": have, "missing_examples": missing})
            if have < want:
                inv["notes"].append(
                    f"{label}: {want - have} of {want} {key}s have no file"
                    + (f" (e.g. {', '.join(missing)})" if missing else ""))

    if runs_dir.is_dir():
        for sub in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
            inv["runs"].append(run_summary(sub))
    return inv


def report(a):
    inv = collect(a)
    print(f"root {inv['root']}\n")

    print("== data trees ==")
    for t in inv["trees"]:
        if not t["present"]:
            print(f"  {t['dir']:<22s} ABSENT          {t['what']}")
            continue
        extra = f"  ({t['failed']} failed)" if t.get("failed") else ""
        print(f"  {t['dir']:<22s} {t['n']:>8,} files {human(t['bytes']):>9s}"
              f"{extra}   {t['what']}")

    print("\n== manifests and datasets ==")
    for r in inv["datasets"]:
        if not r["present"]:
            print(f"  {r['file']:<40s} ABSENT")
            continue
        bits = [human(r["bytes"])]
        for k in ("rows", "strict_rep", "receptors", "ligands", "shape"):
            if r.get(k) is not None:
                bits.append(f"{k} {r[k]:,}" if isinstance(r[k], int)
                            else f"{k} {r[k]}")
        if r.get("splits"):
            bits.append("splits " + "/".join(f"{k} {v:,}" for k, v in r["splits"].items()))
        print(f"  {r['file']:<40s} " + ", ".join(bits))
        if r.get("error"):
            print(f"      [warn] {r['error']}")

    if inv["feats"]:
        print("\n== ligand features ==")
        for r in inv["feats"]:
            if not r["present"]:
                print(f"  {r['path']:<46s} ABSENT")
                continue
            tag = ("STALE" if r.get("stale") else
                   "ok" if r.get("stale") is False else "unknown")
            print(f"  {r['path']:<46s} {r.get('n_ligands', 0):,} ligands, "
                  f"{r.get('failed', 0)} failed, desc {r.get('desc_dim')}, "
                  f"graph/heavy {tag}")

    if inv["coverage"]:
        print(f"\n== coverage against {inv['items']['path']} "
              f"({inv['items']['rows']:,} rows, "
              f"{inv['items']['input_ids']:,} receptors) ==")
        for c in inv["coverage"]:
            pct = 100 * c["have"] / max(c["want"], 1)
            print(f"  {c['artefact']:<22s} {c['have']:>7,} / {c['want']:>7,} "
                  f"{c['key']}s  ({pct:5.1f}%)")

    if inv["runs"]:
        print("\n== runs ==")
        hdr = ("run", "evals", "last", "best mrr", "at", "cosdiff", "tau", "ckpt")
        print("  " + "".join(f"{h:>12s}" for h in hdr))
        for r in inv["runs"]:
            fmt = lambda v, w=12, p=3: (f"{v:>{w}.{p}f}" if isinstance(v, float)
                                        else f"{v:>{w}}" if v is not None else
                                        f"{'-':>{w}}")
            ck = "both" if r["best.pt"] and r["last.pt"] else (
                "last" if r["last.pt"] else "best" if r["best.pt"] else "-")
            print(f"  {r['name']:>12s}{fmt(r.get('evals'))}"
                  f"{fmt(r.get('last_step'))}{fmt(r.get('best_mrr'))}"
                  f"{fmt(r.get('best_at'))}{fmt(r.get('cosdiff'))}"
                  f"{fmt(r.get('tau'))}{ck:>12s}")
            if r.get("error"):
                print(f"      [warn] {r['error']}")

    if inv["notes"]:
        print("\n== things to act on ==")
        for n in inv["notes"]:
            print(f"  - {n}")
    else:
        print("\nnothing flagged.")

    if a.tsv:
        lines = ["\t".join(["section", "key", "field", "value"])]
        for sec in ("trees", "datasets", "feats", "coverage", "runs"):
            for rec in inv[sec]:
                name = str(rec.get("dir") or rec.get("file") or rec.get("path")
                           or rec.get("artefact") or rec.get("name"))
                for k, v in rec.items():
                    lines.append(f"{sec}\t{name}\t{k}\t{v}")
        Path(a.tsv).write_text("\n".join(lines) + "\n")
        print(f"\nmachine-readable: {a.tsv}")


# ---------------------------------------------------------------------------
def selftest(a):
    """A planted tree whose right answers are known. numpy only."""
    import shutil
    from types import SimpleNamespace

    fails = []

    def check(name, ok, detail=""):
        print(f"  {'ok  ' if ok else 'FAIL'} {name}" + (f"   {detail}" if detail else ""))
        if not ok:
            fails.append(name)

    tmp = Path(a.tmp)
    shutil.rmtree(tmp, ignore_errors=True)
    (tmp / "surfaces").mkdir(parents=True)
    (tmp / "pockets" / "fpocket").mkdir(parents=True)
    (tmp / "plip").mkdir(parents=True)
    (tmp / "runs" / "r1").mkdir(parents=True)

    print("[1] a planted tree: 3 receptors, 2 surfaces, 1 pocket file")
    ids = ["rec0", "rec1", "rec2"]
    for i in ids[:2]:
        np.savez(tmp / "surfaces" / f"{i}.npz", xyz=np.zeros((4, 3), np.float32))
    np.savez(tmp / "pockets" / "fpocket" / "rec0.npz", sphere_r=np.ones(2, np.float32))
    (tmp / "pockets" / "fpocket" / "failed_0of1.tsv").write_text("rec9\tboom\n")
    np.savez(tmp / "plip" / "sys0.npz", kind=np.zeros(1, np.int32))
    items = tmp / "items.tsv"
    items.write_text("system_id\tinput_id\tfp_row\n"
                     + "".join(f"sys{j}\t{i}\t{j}\n" for j, i in enumerate(ids)))

    # one stale and one clean ligand feature file
    def feats(path, stale):
        n = 4
        ptr = np.array([0, 3, 7, 10, 14], np.int64)
        heavy = np.diff(ptr).astype(np.int32)
        if stale:
            heavy[1] -= 1          # a graph atom that is not a heavy atom
        np.savez(path, fp=np.zeros((n, 8), np.uint8), desc=np.zeros((n, 5), np.float32),
                 atom_ptr=ptr, n_heavy=heavy, failed=np.zeros(n, bool))
    feats(tmp / "runs" / "r1" / "ligand_feats.npz", stale=True)
    feats(tmp / "clean_feats.npz", stale=False)

    json.dump([{"step": 0, "mrr": 0.02, "cos_diff": 0.99, "tau": 0.07},
               {"step": 125, "mrr": 0.11, "cos_diff": 0.82, "tau": 0.068},
               {"step": 250, "mrr": 0.09, "cos_diff": 0.80, "tau": 0.066,
                "fallback_rate": 0.058}],
              open(tmp / "runs" / "r1" / "history.json", "w"))
    (tmp / "runs" / "r1" / "last.pt").write_bytes(b"x" * 10)

    args = SimpleNamespace(root=str(tmp), items=str(items), tsv="",
                           lig_feats=[str(tmp / "clean_feats.npz")])
    inv = collect(args)

    trees = {t["dir"]: t for t in inv["trees"]}
    check("counts files per tree and reports an absent one",
          trees["surfaces"]["n"] == 2 and trees["plip"]["n"] == 1
          and trees["structures"]["present"] is False,
          f"surfaces {trees['surfaces']['n']}, plip {trees['plip']['n']}")
    check("finds the pocket subdirectory and counts its failures",
          trees["pockets/fpocket"]["n"] == 1
          and trees["pockets/fpocket"]["failed"] == 1)

    print("\n[2] coverage is per ARTEFACT key, and gaps are named")
    cov = {c["artefact"]: c for c in inv["coverage"]}
    check("surfaces: 2 of 3 input_ids, the missing one named",
          cov["surfaces"]["want"] == 3 and cov["surfaces"]["have"] == 2
          and cov["surfaces"]["missing_examples"] == ["rec2"],
          str(cov["surfaces"]["missing_examples"]))
    check("pockets: 1 of 3", cov["pockets/fpocket"]["have"] == 1)
    check("plip keys on system_id, not input_id",
          cov["plip"]["key"] == "system_id" and cov["plip"]["have"] == 1)
    check("every gap raises a note", len(inv["notes"]) >= 3,
          f"{len(inv['notes'])} notes")

    print("\n[3] the staleness test is exact, not a date heuristic")
    byp = {Path(r["path"]).name: r for r in inv["feats"]}
    check("a graph atom that is not a heavy atom reads as STALE",
          byp["ligand_feats.npz"]["stale"] is True
          and byp["ligand_feats.npz"]["graph_vs_heavy_mismatch"] == 1)
    check("a clean file reads as ok", byp["clean_feats.npz"]["stale"] is False)
    check("the stale file is called out by name",
          any("ligand_feats.npz" in n and "STALE" in n for n in inv["notes"]))

    print("\n[4] runs are summarised from history.json alone")
    r1 = inv["runs"][0]
    check("best MRR is the max over evaluations, with its step",
          r1["best_mrr"] == 0.11 and r1["best_at"] == 125 and r1["evals"] == 3
          and r1["last_step"] == 250,
          f"best {r1['best_mrr']} at {r1['best_at']}, last {r1['last_step']}")
    check("the last finite cosdiff / tau are reported, not the first",
          r1["cosdiff"] == 0.80 and r1["tau"] == 0.066)
    check("checkpoint presence is reported without loading torch",
          r1["last.pt"] == 10 and r1["best.pt"] == 0)

    print("\n[5] a run whose history has no finite mrr does not crash")
    (tmp / "runs" / "r2").mkdir()
    json.dump([{"step": 0, "mrr": float("nan")}],
              open(tmp / "runs" / "r2" / "history.json", "w"))
    inv2 = collect(args)
    r2 = [r for r in inv2["runs"] if r["name"] == "r2"][0]
    check("a NaN-only history gives no best, and still counts the evaluation",
          "best_mrr" not in r2 and r2["evals"] == 1)
    (tmp / "runs" / "r3").mkdir()
    (tmp / "runs" / "r3" / "history.json").write_text("{ not json")
    r3 = [r for r in collect(args)["runs"] if r["name"] == "r3"][0]
    check("unreadable history is reported, not raised", "error" in r3)

    shutil.rmtree(tmp, ignore_errors=True)
    print("\nSELFTEST " + ("PASSED" if not fails else f"FAILED: {fails}"))
    if fails:
        raise SystemExit(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("report", help="what is on disk, complete, and stale")
    p.add_argument("--root", default=".", help="$WORK; defaults to the cwd")
    p.add_argument("--items", default="",
                   help="items.tsv to check surface / pocket / PLIP coverage "
                        "against (optional)")
    p.add_argument("--lig_feats", action="append",
                   help="extra ligand_feats.npz to test for staleness; the "
                        "run directories' own are always checked")
    p.add_argument("--tsv", default="", help="also write a machine-readable TSV")

    p = sub.add_parser("selftest", help="planted tree, no real data")
    p.add_argument("--tmp", default="inventory_selftest")

    a = ap.parse_args()
    {"report": report, "selftest": selftest}[a.cmd](a)


if __name__ == "__main__":
    main()
