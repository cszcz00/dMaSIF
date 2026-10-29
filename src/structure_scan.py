"""Per-row structural measurements over the whole manifest, for gating dataset_v2.

Why. The manifest's filters are proxies - a global resolution, a binary
"cofactor" / "has a second ligand chain" flag - and the first funnel run
(RESULTS, 2026-10-29) showed they hide most of PLINDER's targets: 24,327 train
groups in the manifest, 2,605 in dataset_v1. What actually makes a row a bad
training example for THIS model is local and measurable from the files we
already have:

  - a companion molecule next to the ligand. extract.py keeps amino acids
    only, so a companion leaves a VOID in the surface; an ion leaves a
    dimple, a NAD leaves a cofactor-sized cavity merged into the pocket.
    What matters is its SIZE x DISTANCE, so strip-and-gate needs both.
  - a fragment that is not in a pocket. Small ligands are fine when buried.
  - a pocket that is missing atoms or unusually mobile, which a global
    resolution cutoff only guesses at.

One row per manifest (system_id, ligand_id):

    scan_status         ok | no_receptor | no_ligand_sdf | error:<type>
    lig_ha              ligand heavy atoms (the row's own SDF)
    burial45            share of ligand heavy atoms with a receptor heavy atom
                        within 4.5 A
    contacts4           receptor heavy atoms within 4 A, per ligand heavy atom
    pocket_res          receptor residues with an atom within 6 A of the ligand
    pocket_atoms        receptor heavy atoms within 6 A
    pocket_missing      heavy atoms missing from those residues against the
                        standard amino-acid composition (no OXT)
    pocket_b_ratio      mean B of the pocket atoms / mean B of the receptor;
                        NaN where B is all zero (some cryo-EM, models)
    pocket_min_occ      lowest occupancy among pocket atoms
    pocket_altloc_frac  share of pocket atoms that had alternate locations
    comp_n10            companions (non-water, non-receptor, non-own-ligand
                        groups in system.cif) with an atom within 10 A
    comp_min_dist       their minimum heavy-atom distance to the ligand (A);
                        NaN when there is none within 10 A
    comp_maxha_4/8      heavy atoms of the LARGEST companion with an atom
                        within 4 / 8 A (0 when none)
    comp_ha_8           all companion heavy atoms of groups within 8 A

The receptor is structures/<receptor_key>.cif - the same file the surface is
built from - with extract.py's rule: amino-acid residues plus MSE, first
model, no hydrogens, one conformer. Companions come from
systems/<system_id>/system.cif without relying on its chain naming: every
heavy atom that is not water, not a receptor atom and not the row's own
ligand (both matched by position, < 0.1 A), grouped by (chain, subchain).

Sharding: receptors in a fixed sorted order, every n-th one per array task,
so each receptor file is parsed once and the shards are disjoint. A shard
whose part file exists is skipped, so a rerun only redoes failed tasks.

Modes:
    run       one shard -> <out>/part_<i>of<n>.parquet  (src/slurm/scan.sbatch)
    merge     parts -> one table, coverage and distributions printed
    selftest  planted structures written with gemmi, checked against hand
              computed values

Example (CPU venv, `pip install gemmi` once):
    sbatch src/slurm/scan.sbatch                      # 64 shards
    python src/structure_scan.py merge --parts data/meta/scan \\
        --manifest data/meta/manifest.parquet --out data/meta/structure_scan.parquet
    python src/dataset_funnel.py report --manifest data/meta/manifest.parquet \\
        --scan data/meta/structure_scan.parquet --per_entry

Depends on: numpy, pandas, gemmi; ids (ligand files), test_encoder
            (read_sdf_heavy, numpy only). No torch.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

POCKET_R, BURIAL_R, CONTACT_R, COMP_R = 6.0, 4.5, 4.0, 10.0
MATCH_TOL = 0.1
WATER = {"HOH", "DOD", "WAT", "H2O"}
# Heavy atoms per standard residue, without OXT.
RES_HEAVY = {"ALA": 5, "ARG": 11, "ASN": 8, "ASP": 8, "CYS": 6, "GLN": 9, "GLU": 9,
             "GLY": 4, "HIS": 10, "ILE": 8, "LEU": 8, "LYS": 9, "MET": 8, "PHE": 11,
             "PRO": 7, "SER": 6, "THR": 7, "TRP": 14, "TYR": 12, "VAL": 7, "MSE": 8}
COLUMNS = ["system_id", "ligand_id", "scan_status", "lig_ha", "burial45", "contacts4",
           "pocket_res", "pocket_atoms", "pocket_missing", "pocket_b_ratio",
           "pocket_min_occ", "pocket_altloc_frac", "comp_n10", "comp_min_dist",
           "comp_maxha_4", "comp_maxha_8", "comp_ha_8"]


# ---------------------------------------------------------------------------
# Structure reading
# ---------------------------------------------------------------------------
def _is_heavy(atom):
    return atom.element.name not in ("H", "D")


def read_receptor(path):
    """Receptor heavy atoms as the surface sees them -> dict of arrays."""
    import gemmi

    st = gemmi.read_structure(str(path))
    st.setup_entities()
    model = st[0]
    xyz, b, occ, alt, res = [], [], [], [], []
    seen = set()
    for ch in model:
        for r in ch:
            info = gemmi.find_tabulated_residue(r.name)
            if r.name != "MSE" and not (info and info.is_amino_acid() and r.het_flag != "H"):
                continue
            rid = (ch.name, str(r.seqid), r.name)
            names_alt = {a.name for a in r if a.altloc != "\0"}
            for a in r:
                if not _is_heavy(a):
                    continue
                key = (rid, a.name)
                if key in seen:          # first conformer only
                    continue
                seen.add(key)
                p = a.pos
                xyz.append((p.x, p.y, p.z))
                b.append(a.b_iso)
                occ.append(a.occ)
                alt.append(a.name in names_alt)
                res.append(rid)
    if not xyz:
        raise ValueError("no protein atoms")
    uniq = {r: i for i, r in enumerate(dict.fromkeys(res))}
    return {"xyz": np.asarray(xyz, np.float32), "b": np.asarray(b, np.float32),
            "occ": np.asarray(occ, np.float32), "alt": np.asarray(alt, bool),
            "res": np.asarray([uniq[r] for r in res], np.int64),
            "resname": [r[2] for r in uniq]}


def read_system_groups(path):
    """system.cif -> list of (n_heavy, xyz) per (chain, subchain) group, waters out."""
    import gemmi

    st = gemmi.read_structure(str(path))
    groups = {}
    for ch in st[0]:
        for r in ch:
            if r.name in WATER:
                continue
            key = (ch.name, r.subchain or str(r.seqid))
            pts = [(a.pos.x, a.pos.y, a.pos.z) for a in r if _is_heavy(a)]
            if pts:
                groups.setdefault(key, []).extend(pts)
    return [np.asarray(v, np.float32) for v in groups.values()]


def _min_dists(a, b):
    """For each point of a, the distance to the nearest point of b (chunked)."""
    if not len(b):
        return np.full(len(a), np.inf, np.float32)
    out = np.empty(len(a), np.float32)
    for s in range(0, len(a), 2048):
        d = ((a[s:s + 2048, None] - b[None]) ** 2).sum(-1)
        out[s:s + 2048] = np.sqrt(d.min(1))
    return out


def _near(points, centre_pts, r):
    """Mask of `points` within r of any of centre_pts, bounding-box cropped first."""
    lo, hi = centre_pts.min(0) - r, centre_pts.max(0) + r
    box = np.all((points >= lo) & (points <= hi), axis=1)
    m = np.zeros(len(points), bool)
    idx = np.nonzero(box)[0]
    if len(idx):
        m[idx] = _min_dists(points[idx], centre_pts) <= r
    return m


# ---------------------------------------------------------------------------
# Measurements
# ---------------------------------------------------------------------------
def measure(rec, lig, groups):
    """rec: read_receptor dict; lig: (k, 3) ligand heavy atoms; groups: list
    of system.cif groups or None -> dict of the COLUMNS measurements."""
    out = {"lig_ha": len(lig)}
    X = rec["xyz"]
    pocket = _near(X, lig, POCKET_R)
    d_lig = _min_dists(lig, X[pocket]) if pocket.any() else np.full(len(lig), np.inf)
    out["burial45"] = float((d_lig <= BURIAL_R).mean())
    near4 = _near(X[pocket], lig, CONTACT_R) if pocket.any() else np.zeros(0, bool)
    out["contacts4"] = float(near4.sum() / max(len(lig), 1))
    res = np.unique(rec["res"][pocket])
    out["pocket_res"] = len(res)
    out["pocket_atoms"] = int(pocket.sum())
    present = np.bincount(rec["res"], minlength=len(rec["resname"]))
    out["pocket_missing"] = int(sum(max(0, RES_HEAVY[rec["resname"][i]] - present[i])
                                    for i in res if rec["resname"][i] in RES_HEAVY))
    mb = float(rec["b"].mean())
    out["pocket_b_ratio"] = (float(rec["b"][pocket].mean()) / mb
                             if pocket.any() and mb > 0 else np.nan)
    out["pocket_min_occ"] = float(rec["occ"][pocket].min()) if pocket.any() else np.nan
    out["pocket_altloc_frac"] = float(rec["alt"][pocket].mean()) if pocket.any() else np.nan

    if groups is None:
        out.update({k: np.nan for k in ("comp_n10", "comp_min_dist", "comp_maxha_4",
                                         "comp_maxha_8", "comp_ha_8")})
        return out
    n10, dmin, max4, max8, sum8 = 0, np.inf, 0, 0, 0
    for g in groups:
        near = _near(g, lig, COMP_R)
        if not near.any():
            continue
        gx = g[near]
        # drop the row's own ligand and any receptor atom present in system.cif
        own = _min_dists(gx, lig) < MATCH_TOL
        recv = _min_dists(gx, X[_near(X, gx, 1.0)]) < MATCH_TOL
        keep = ~own & ~recv
        if keep.mean() < 0.5:            # this group IS the ligand or the receptor
            continue
        d = float(_min_dists(gx[keep], lig).min())
        n_heavy = int(len(g) - (own.sum() + recv.sum()))
        n10 += 1
        dmin = min(dmin, d)
        if d <= 4.0:
            max4 = max(max4, n_heavy)
        if d <= 8.0:
            max8 = max(max8, n_heavy)
            sum8 += n_heavy
    out.update({"comp_n10": n10, "comp_min_dist": dmin if n10 else np.nan,
                "comp_maxha_4": max4, "comp_maxha_8": max8, "comp_ha_8": sum8})
    return out


def scan_rows(rows, structures, systems):
    """rows: DataFrame of one receptor's manifest rows -> list of result dicts."""
    from ids import ligand_chain, ligand_sdf_paths
    from test_encoder import read_sdf_heavy

    rk = rows["receptor_key"].iloc[0]
    out = []
    try:
        rec = read_receptor(Path(structures) / f"{rk}.cif")
    except Exception as e:
        status = "no_receptor" if not (Path(structures) / f"{rk}.cif").exists() \
            else f"error:{type(e).__name__}"
        return [{"system_id": r.system_id, "ligand_id": r.ligand_id, "scan_status": status}
                for r in rows.itertuples()]
    for r in rows.itertuples():
        row = {"system_id": r.system_id, "ligand_id": r.ligand_id}
        try:
            sdfs = ligand_sdf_paths(systems, r.system_id, ligand_chain(r.ligand_id))
            if not sdfs:
                row["scan_status"] = "no_ligand_sdf"
                out.append(row)
                continue
            lig = np.concatenate([read_sdf_heavy(p) for p in sdfs])
            sc = Path(systems) / r.system_id / "system.cif"
            groups = read_system_groups(sc) if sc.exists() else None
            row.update(measure(rec, lig, groups))
            row["scan_status"] = "ok"
        except Exception as e:
            row["scan_status"] = f"error:{type(e).__name__}"
        out.append(row)
    return out


# ---------------------------------------------------------------------------
def run(a):
    import pandas as pd

    i, n = (int(x) for x in a.shard.split("/"))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    part = out / f"part_{i:03d}of{n:03d}.parquet"
    if part.exists() and not a.overwrite:
        print(f"{part} exists; skipping (pass --overwrite to redo)")
        return
    df = pd.read_parquet(a.manifest, columns=["system_id", "ligand_id", "receptor_key"])
    keys = np.sort(df["receptor_key"].unique())[i::n]
    if a.limit:
        keys = keys[:a.limit]
    sub = df[df["receptor_key"].isin(set(keys))]
    print(f"shard {i}/{n}: {len(keys):,} receptors, {len(sub):,} rows")
    res, t0 = [], time.time()
    for j, (rk, rows) in enumerate(sub.groupby("receptor_key", sort=True)):
        res += scan_rows(rows, a.structures, a.systems)
        if (j + 1) % 500 == 0:
            el = time.time() - t0
            print(f"  {j + 1:,}/{len(keys):,} receptors, {el / (j + 1):.2f} s each, "
                  f"~{el / (j + 1) * (len(keys) - j - 1) / 60:.0f} min left", flush=True)
    tab = pd.DataFrame(res).reindex(columns=COLUMNS)
    tmp = part.with_suffix(".tmp")
    tab.to_parquet(tmp, index=False)
    tmp.rename(part)
    print(f"wrote {part}: {tab['scan_status'].value_counts().to_dict()}  "
          f"({(time.time() - t0) / 60:.1f} min)")


def merge(a):
    import pandas as pd

    parts = sorted(Path(a.parts).glob("part_*of*.parquet"))
    if not parts:
        raise SystemExit(f"no part files in {a.parts}")
    ns = {p.stem.split("of")[1] for p in parts}
    if len(ns) != 1:
        raise SystemExit(f"parts from different shard counts: {sorted(ns)}")
    n = int(ns.pop())
    have = {int(p.stem.split("_")[1].split("of")[0]) for p in parts}
    if len(have) != n:
        print(f"[warn] {n - len(have)} of {n} shards missing: "
              f"{sorted(set(range(n)) - have)[:20]}; rerun the array (done shards skip)")
    tab = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
    dup = tab.duplicated(["system_id", "ligand_id"])
    if dup.any():
        raise SystemExit(f"{int(dup.sum())} duplicate rows across parts")
    if a.manifest:
        m = pd.read_parquet(a.manifest, columns=["system_id", "ligand_id"])
        print(f"coverage: {len(tab):,} of {len(m):,} manifest rows")
    print("scan_status:")
    for k, v in tab["scan_status"].value_counts().items():
        print(f"  {v:>9,}  {k}")
    ok = tab[tab["scan_status"] == "ok"]
    print("\ndistributions over ok rows (p5 / p25 / p50 / p75 / p95):")
    for c in COLUMNS[3:]:
        v = pd.to_numeric(ok[c], errors="coerce").dropna()
        if len(v):
            q = np.percentile(v, [5, 25, 50, 75, 95])
            print(f"  {c:20s} " + " / ".join(f"{x:7.2f}" for x in q)
                  + f"   (n={len(v):,})")
    print(f"\nrows with a companion within 8 A: {(ok['comp_maxha_8'] > 0).mean():.1%}; "
          f"of those, largest companion <= 1 heavy atom: "
          f"{(ok.loc[ok['comp_maxha_8'] > 0, 'comp_maxha_8'] <= 1).mean():.1%}, "
          f"<= 6: {(ok.loc[ok['comp_maxha_8'] > 0, 'comp_maxha_8'] <= 6).mean():.1%}")
    tab.to_parquet(a.out, index=False)
    print(f"wrote {a.out}")


# ---------------------------------------------------------------------------
def selftest():
    import tempfile

    import gemmi
    import pandas as pd

    fails = []

    def check(name, ok, detail=""):
        print(f"  {'ok  ' if ok else 'FAIL'} {name}" + (f"   {detail}" if detail else ""))
        if not ok:
            fails.append(name)

    def residue(name, seq, atoms, het="A"):
        r = gemmi.Residue()
        r.name, r.seqid, r.het_flag = name, gemmi.SeqId(seq, " "), het
        for an, el, xyz, *rest in atoms:
            a = gemmi.Atom()
            a.name, a.element, a.pos = an, gemmi.Element(el), gemmi.Position(*xyz)
            a.b_iso = rest[0] if rest else 20.0
            a.occ = rest[1] if len(rest) > 1 else 1.0
            a.altloc = rest[2] if len(rest) > 2 else "\0"
            r.add_atom(a)
        return r

    def gly(seq, x, b=20.0):
        return residue("GLY", seq, [("N", "N", (x, 0, 0), b), ("CA", "C", (x + 1, 0, 0), b),
                                    ("C", "C", (x + 1, 1, 0), b), ("O", "O", (x + 1, 2, 0), b)])

    def write(path, chains):
        st = gemmi.Structure()
        m = gemmi.Model("1")
        for name, residues in chains:
            ch = gemmi.Chain(name)
            for r in residues:
                ch.add_residue(r)
            m.add_chain(ch)
        st.add_model(m)
        st.setup_entities()
        st.make_mmcif_document().write_file(str(path))

    def sdf(path, pts, el="C"):
        lines = ["lig", "  test", "", f"{len(pts):3d}  0  0  0  0  0  0  0  0  0999 V2000"]
        lines += [f"{x:10.4f}{y:10.4f}{z:10.4f} {el:<3} 0  0  0  0  0  0  0  0  0  0  0  0"
                  for x, y, z in pts]
        Path(path).write_text("\n".join(lines + ["M  END", "$$$$", ""]))

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "structures").mkdir()
        sid = "1abc__1__1.A__1.B_1.C_1.D"
        ldir = td / "systems" / sid / "ligand_files"
        ldir.mkdir(parents=True)
        # receptor: 4 complete glycines along x at 0, 4, 8, 12 (mobile pocket
        # B 40 near the ligand, 20 elsewhere) + a glycine missing its O, + one
        # residue with an altloc atom, far away along y
        near_gly = [gly(1, 0.0, 40.0), gly(2, 4.0, 40.0)]
        far_gly = [gly(3, 30.0), gly(4, 34.0)]
        broken = residue("GLY", 5, [("N", "N", (2, 4, 0), 40.0), ("CA", "C", (3, 4, 0), 40.0),
                                    ("C", "C", (3, 5, 0), 40.0)])
        alt = residue("ALA", 6, [("N", "N", (60, 0, 0)), ("CA", "C", (61, 0, 0)),
                                 ("C", "C", (61, 1, 0)), ("O", "O", (61, 2, 0)),
                                 ("CB", "C", (62, 0, 0), 20.0, 0.5, "A"),
                                 ("CB", "C", (62, 0.3, 0), 20.0, 0.5, "B")])
        rec_res = near_gly + far_gly + [broken, alt]
        write(td / "structures" / "1abc__1__1.A.cif", [("A", rec_res)])
        # ligand: 3 atoms sitting 3 A above the near glycines
        lig = [(1.0, 1.0, 3.0), (2.0, 1.0, 3.0), (3.0, 1.0, 3.0)]
        sdf(ldir / "1.B.sdf", lig)
        # system.cif: receptor copy + ligand + Mg 3 A from the ligand + a
        # 6-atom glycerol-like group 6 A away + a 12-atom group 20 A away + water
        mg = residue("MG", 1, [("MG", "Mg", (2.0, 1.0, 6.0))], het="H")
        gol = residue("GOL", 1, [(f"C{i}", "C", (2.0 + 0.5 * i, 7.0, 3.0)) for i in range(6)],
                      het="H")
        big = residue("NAD", 1, [(f"C{i}", "C", (2.0 + 0.3 * i, 21.0, 3.0)) for i in range(12)],
                      het="H")
        hoh = residue("HOH", 1, [("O", "O", (2.0, 1.0, 4.5))], het="H")
        own = residue("LIG", 1, [(f"C{i}", "C", p) for i, p in enumerate(lig)], het="H")
        write(td / "systems" / sid / "system.cif",
              [("A", rec_res), ("B", [own]), ("C", [mg]), ("D", [gol]), ("E", [big]),
               ("W", [hoh])])

        rows = pd.DataFrame([{"system_id": sid, "ligand_id": "1abc__1__1.B",
                              "receptor_key": "1abc__1__1.A"}])
        res = scan_rows(rows, td / "structures", td / "systems")[0]
        check("status ok", res.get("scan_status") == "ok", str(res.get("scan_status")))
        if res.get("scan_status") == "ok":
            check("ligand heavy atoms", res["lig_ha"] == 3)
            check("pocket = the two near glycines + the broken one",
                  res["pocket_res"] == 3, f"{res['pocket_res']}")
            check("pocket_missing counts the absent O", res["pocket_missing"] == 1,
                  f"{res['pocket_missing']}")
            check("pocket_b_ratio > 1 for the mobile pocket",
                  res["pocket_b_ratio"] > 1.2, f"{res['pocket_b_ratio']:.2f}")
            check("no altlocs in the pocket", res["pocket_altloc_frac"] == 0.0)
            check("burial: every ligand atom within 4.5 A of the receptor",
                  res["burial45"] == 1.0, f"{res['burial45']}")
            check("companions: Mg and the 6-atom group within 10 A, water and the "
                  "20 A group out, own ligand and receptor copy excluded",
                  res["comp_n10"] == 2, f"{res['comp_n10']}")
            check("comp_min_dist is the Mg at 3 A", abs(res["comp_min_dist"] - 3.0) < 1e-3,
                  f"{res['comp_min_dist']:.3f}")
            check("comp_maxha_4 = 1 (Mg only), comp_maxha_8 = 6, comp_ha_8 = 7",
                  (res["comp_maxha_4"], res["comp_maxha_8"], res["comp_ha_8"]) == (1, 6, 7),
                  f"{res['comp_maxha_4']}, {res['comp_maxha_8']}, {res['comp_ha_8']}")
        rec = read_receptor(td / "structures" / "1abc__1__1.A.cif")
        check("receptor keeps one conformer per atom (ALA has 5 heavy atoms)",
              len(rec["xyz"]) == 4 * 4 + 3 + 5 and rec["alt"].sum() == 1,
              f"{len(rec['xyz'])} atoms, {int(rec['alt'].sum())} altloc")
        bad = pd.DataFrame([{"system_id": "9zzz__1__1.A__1.B", "ligand_id": "9zzz__1__1.B",
                             "receptor_key": "9zzz__1__1.A"}])
        check("a missing receptor is reported, not raised",
              scan_rows(bad, td / "structures", td / "systems")[0]["scan_status"]
              == "no_receptor")

    print("\nPASS" if not fails else f"\nFAIL: {fails}")
    raise SystemExit(1 if fails else 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("--manifest", default="data/meta/manifest.parquet")
    p.add_argument("--structures", default="structures")
    p.add_argument("--systems", default="systems")
    p.add_argument("--out", default="data/meta/scan")
    p.add_argument("--shard", default="0/1", help="i/n: every n-th receptor from i")
    p.add_argument("--limit", type=int, help="receptors per shard, for a quick test")
    p.add_argument("--overwrite", action="store_true")
    p = sub.add_parser("merge")
    p.add_argument("--parts", default="data/meta/scan")
    p.add_argument("--manifest", help="report coverage against it")
    p.add_argument("--out", default="data/meta/structure_scan.parquet")
    sub.add_parser("selftest")
    a = ap.parse_args()
    {"run": run, "merge": merge, "selftest": lambda a: selftest()}[a.cmd](a)


if __name__ == "__main__":
    main()
