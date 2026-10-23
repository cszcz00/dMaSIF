"""PLIP interaction labels: which protein atoms each training ligand actually contacts, and how.

Why. Pair pooling's site-type head starts near-uniform, so its eight types carry
no chemistry until the contrastive loss teaches them some, and in runs/v3 that
never happened (docs/STATUS.md). The crystal complexes already say which protein
atoms H-bond, salt-bridge, pi-stack or pack against the ligand. PLIP (the
Protein-Ligand Interaction Profiler, Salentin et al. 2015 / Adasme et al. 2021)
detects those interactions from geometry. This module runs it on every training
row and turns its output into per-surface-point SITE labels in the chemotype
vocabulary (donor, acceptor, cation, anion, aromatic, hydrophobic), for an
auxiliary loss on the site types. The crystal structure is used as training
LABELS on the protein side only; nothing at inference needs it.

    run       container + the PLIP venv (src/slurm/setup_plip.sh). items.tsv ->
              <out>/<system_id>.npz, shardable like `train.py cache`. Per row:
              the receptor's protein atoms exactly as the surface sees them
              (extract.load_protein_atoms, same chain filter, no hydrogens) plus
              the row's own ligand from its SDF, merged into one complex, PLIP
              run on it, every interaction kept with its protein-side and
              ligand-side coordinates.
    demo      the same pipeline on one PDB file with a named ligand residue,
              next to PLIP on the original file - the end-to-end check
              setup_plip.sh runs on fpocket's 1ATP sample.
    check     numpy (CPU venv). Interactions per row by kind, and the site
              labels mapped onto cached surfaces: points per class, and how
              many of them fall inside the 4 A pocket label (they should, nearly
              all).
    selftest  numpy: kind -> class mapping, the surface mapping, npz round trip.

What is stored, per row: `kind` (k,) int8 indices into KINDS, `prot_xyz` and
`lig_xyz` (k, 3) float32. Coordinates, not atom or point indices, so the file
serves any resampling of the surface (labels.py's warning about saved indices).
The protein-side coordinate is an atom for H-bonds, hydrophobic contacts and
halogen bonds, and a ring or charge-group CENTRE for pi-stacking, pi-cation and
salt bridges; `site_labels` gives centres a larger radius.

    kind               protein side is ...                class
    hbond_pdon         the H-bond donor                   donor
    hbond_ldon         the H-bond acceptor                acceptor
    hydrophobic        a carbon in contact                hydrophobic
    pistack            an aromatic ring (centre)          aromatic
    pication_pring     an aromatic ring, ligand charged   aromatic
    pication_pcharge   a cationic group, ligand ring      cation
    saltbridge_ppos    the positive group (centre)        cation
    saltbridge_pneg    the negative group (centre)        anion
    halogen            the halogen-bond acceptor          acceptor
    water_pdon/_ldon   donor / acceptor via a water       donor / acceptor

What is NOT in the complex, by design: waters (our receptor input drops them,
so PLIP finds no water bridges; the kinds exist for completeness), metal ions
and cofactors (same reason; dataset_v1 has no companion ligands anyway), and
hydrogens from the structure (PLIP protonates the complex itself with
OpenBabel, which needs the ligand's bond orders - hence the ligand is read from
its SDF, not from coordinates alone).

Example:
    bash src/slurm/setup_plip.sh            # once: venv over the container
    sbatch src/slurm/plip.sbatch            # 16 CPU tasks -> plip/
    python src/plip_labels.py check --items runs/v1/items.tsv \\
        --plip plip --surface_cache surfaces --n 500

Depends on: numpy; extract, ids, external_pockets.write_pdb, openbabel, plip
            (run / demo only); test_encoder.read_sdf_heavy (check only).
"""

import argparse
import os
import shutil
import signal
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

KINDS = ("hbond_pdon", "hbond_ldon", "hydrophobic", "pistack", "pication_pring",
         "pication_pcharge", "saltbridge_ppos", "saltbridge_pneg", "halogen",
         "water_pdon", "water_ldon")
# class per kind, as indices into chemotype.GROUPS
CLASSES = ("donor", "acceptor", "cation", "anion", "aromatic", "hydrophobic")
KIND_CLASS = np.asarray([0, 1, 5, 4, 4, 2, 2, 3, 1, 0, 1], np.int64)
# kinds whose protein-side coordinate is a ring or charge-group centre
CENTRE_KINDS = np.asarray([k in ("pistack", "pication_pring", "pication_pcharge",
                                 "saltbridge_ppos", "saltbridge_pneg") for k in KINDS])
PLIP_KEYS = ("kind", "prot_xyz", "lig_xyz")
LIG_RESNAME = "LIG"


# ---------------------------------------------------------------------------
# labels on a surface (numpy only)
# ---------------------------------------------------------------------------
def load_plip(path):
    with np.load(path, allow_pickle=False) as d:
        return {k: d[k] for k in PLIP_KEYS}


def save_plip(path, rec):
    """Write-then-rename through a handle (np.savez appends .npz to a bare path)."""
    path = Path(path)
    tmp = path.with_suffix(".npz.tmp")
    with open(tmp, "wb") as fh:
        np.savez(fh, **{k: rec[k] for k in PLIP_KEYS})
    tmp.replace(path)


def site_labels(xyz, rec, atom_radius=3.0, centre_radius=4.0, lig_radius=None,
                chunk=8192):
    """Surface points -> (N, 6) bool site labels in CLASSES order.

    A point gets class c when it lies within `atom_radius` of the protein-side
    atom of an interaction of class c (`centre_radius` for ring / charge-group
    centres, which sit inside the residue rather than on it). The dMaSIF surface
    sits ~1.5-2 A out from atom centres, so 3 A takes the cap of surface over
    that atom. With `lig_radius`, the point must also lie within that distance of
    the interaction's ligand partner, which keeps the cap's side that faces the
    ligand.
    """
    xyz = np.asarray(xyz, np.float32)
    out = np.zeros((len(xyz), len(CLASSES)), bool)
    if not len(rec["kind"]):
        return out
    kind = rec["kind"].astype(np.int64)
    rad = np.where(CENTRE_KINDS[kind], centre_radius, atom_radius).astype(np.float32)
    cls = KIND_CLASS[kind]
    P, L = rec["prot_xyz"].astype(np.float32), rec["lig_xyz"].astype(np.float32)
    for s in range(0, len(xyz), chunk):
        x = xyz[s:s + chunk]
        near = ((x[:, None] - P[None]) ** 2).sum(-1) <= rad[None] ** 2     # (n, k)
        if lig_radius is not None:
            near &= ((x[:, None] - L[None]) ** 2).sum(-1) <= lig_radius ** 2
        for c in range(len(CLASSES)):
            m = cls == c
            if m.any():
                out[s:s + chunk, c] = near[:, m].any(1)
    return out


# ---------------------------------------------------------------------------
# the complex and PLIP (container + PLIP venv)
# ---------------------------------------------------------------------------
def complex_pdb(prot, sdf_paths, path):
    """Protein atoms + the row's ligand(s) -> one PDB file PLIP can read -> the
    ligand's chain character.

    The protein goes through external_pockets.write_pdb (real residue and atom
    names, one character per chain label). The ligand is converted from its
    SDF by OpenBabel, which writes its bond orders as repeated CONECT entries -
    PLIP protonates the complex itself and needs them - and is then merged AS
    TEXT: its HETATM residue fields become LIG on a chain character no protein
    chain uses (one residue number per SDF), and its atom serials, in HETATM
    and CONECT alike, are offset past the protein's. Text rather than
    OpenBabel's residue API, which on a merged molecule attached the ligand
    atoms to a protein residue.
    """
    from openbabel import pybel

    from external_pockets import _CHAIN_CHARS, write_pdb

    path = Path(path)
    n_chains = len(dict.fromkeys(prot["atom_chain"].tolist()))
    free = [c for c in reversed(_CHAIN_CHARS) if c not in _CHAIN_CHARS[:n_chains]]
    lig_chain = free[0] if free else "Z"
    prot_path = path.with_suffix(".prot.pdb")
    write_pdb(prot, prot_path)
    lines = [ln for ln in prot_path.read_text().splitlines() if ln.startswith("ATOM")]
    prot_path.unlink(missing_ok=True)
    offset = len(lines)
    conect = []
    for k, sdf in enumerate(sdf_paths, 1):
        block = next(pybel.readfile("sdf", str(sdf))).write("pdb").splitlines()
        het = [ln for ln in block if ln.startswith(("HETATM", "ATOM"))]
        if offset + len(het) > 99999:
            raise ValueError(f"{offset + len(het)} atoms: PDB serials would wrap and "
                             "the CONECT records could not name the ligand atoms")
        old_to_new = {}
        for i, ln in enumerate(het, 1):
            ln = ln.ljust(80)
            old_to_new[int(ln[6:11])] = offset + i
            el = ln[76:78].strip() or "X"
            name = f"{el}{i}"[:4]
            name = f" {name:<3s}" if len(el) == 1 and len(name) < 4 else f"{name:<4s}"
            lines.append(f"HETATM{offset + i:5d} {name} {LIG_RESNAME:>3s} {lig_chain}"
                         f"{k:4d}    {ln[30:54]}{ln[54:66]}          {el:>2s}")
        for ln in block:
            if not ln.startswith("CONECT"):
                continue
            nums = [int(ln[s:s + 5]) for s in range(6, len(ln.rstrip()), 5)
                    if ln[s:s + 5].strip()]
            nums = [old_to_new[n] for n in nums if n in old_to_new]
            if len(nums) > 1:
                conect.append("CONECT" + "".join(f"{n:5d}" for n in nums))
        offset += len(het)
    path.write_text("\n".join(lines + conect + ["END"]) + "\n")
    return lig_chain


def _plip_quiet():
    import logging

    from plip.basic import config

    for name, val in (("NOFIXFILE", True), ("QUIET", True), ("SILENT", True)):
        if hasattr(config, name):
            setattr(config, name, val)
    logging.getLogger("plip").setLevel(logging.ERROR)


def interactions(site):
    """One PLIP binding site -> list of (kind index, protein xyz, ligand xyz)."""
    out = []

    def add(kind, p, l):
        out.append((KINDS.index(kind), tuple(p), tuple(l)))

    for h in list(site.hbonds_pdon) + list(site.hbonds_ldon):
        if h.protisdon:
            add("hbond_pdon", h.d.coords, h.a.coords)
        else:
            add("hbond_ldon", h.a.coords, h.d.coords)
    for c in site.hydrophobic_contacts:
        add("hydrophobic", c.bsatom.coords, c.ligatom.coords)
    for s in site.pistacking:
        add("pistack", s.proteinring.center, s.ligandring.center)
    for p in list(site.pication_paro) + list(site.pication_laro):
        if p.protcharged:
            add("pication_pcharge", p.charge.center, p.ring.center)
        else:
            add("pication_pring", p.ring.center, p.charge.center)
    for s in list(site.saltbridge_lneg) + list(site.saltbridge_pneg):
        if s.protispos:
            add("saltbridge_ppos", s.positive.center, s.negative.center)
        else:
            add("saltbridge_pneg", s.negative.center, s.positive.center)
    for x in site.halogen_bonds:
        add("halogen", x.acc.o.coords, x.don.x.coords)
    for w in site.water_bridges:
        if w.protisdon:
            add("water_pdon", w.d.coords, w.a.coords)
        else:
            add("water_ldon", w.a.coords, w.d.coords)
    return out


def run_plip(pdb_path, resname, chain=None):
    """PLIP on a PDB file -> (record dict, number of binding sites merged) for
    every binding site of ligand residue `resname` (on `chain`, if given)."""
    from plip.structure.preparation import PDBComplex

    _plip_quiet()
    mol = PDBComplex()
    mol.output_path = str(Path(pdb_path).parent)
    mol.load_pdb(str(pdb_path))
    mol.analyze()
    rows, n = [], 0
    for key, site in mol.interaction_sets.items():
        name, ch = key.split(":")[:2]
        if name != resname or (chain is not None and ch != chain):
            continue
        n += 1
        rows += interactions(site)
    rec = {"kind": np.asarray([r[0] for r in rows], np.int8),
           "prot_xyz": np.asarray([r[1] for r in rows], np.float32).reshape(-1, 3),
           "lig_xyz": np.asarray([r[2] for r in rows], np.float32).reshape(-1, 3)}
    return rec, n


class _Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise _Timeout()


def summarize(recs):
    """Counts by kind over records -> printable lines."""
    tot = np.zeros(len(KINDS), np.int64)
    has = np.zeros(len(CLASSES), np.int64)
    for r in recs:
        tot += np.bincount(r["kind"].astype(np.int64), minlength=len(KINDS))
        if len(r["kind"]):
            has[np.unique(KIND_CLASS[r["kind"].astype(np.int64)])] += 1
    n = max(len(recs), 1)
    lines = [f"{len(recs)} records; interactions per record by kind:"]
    lines += [f"  {k:18s} {tot[i] / n:6.2f}" for i, k in enumerate(KINDS)]
    lines += ["records with at least one site of class:"]
    lines += [f"  {c:12s} {has[i] / n:6.2f}" for i, c in enumerate(CLASSES)]
    return lines


def run(a):
    from extract import load_protein_atoms
    from ids import parse_chains
    from train_data import read_items

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = sorted(read_items(a.items, a.limit), key=lambda r: r["system_id"])
    dup = len(rows) - len({r["system_id"] for r in rows})
    if dup:
        # Files are keyed by system_id; dataset_v1 has one ligand chain per
        # system so this is unique. A table with several ligands per system
        # needs a richer key - refuse rather than overwrite.
        raise SystemExit(f"{dup} rows share a system_id; plip files are keyed by it")
    tag = ""
    if a.shard:
        i, n = (int(x) for x in a.shard.split("/"))
        rows = rows[i::n]
        tag = f"_{i}of{n}"
        print(f"shard {i}/{n}: {len(rows)} rows")
    work = Path(tempfile.mkdtemp(prefix="plip_", dir=a.workdir or os.environ.get("TMPDIR", "/tmp")))
    signal.signal(signal.SIGALRM, _alarm)
    t0, built, failed, recs = time.time(), 0, [], []
    try:
        for j, row in enumerate(rows, 1):
            dest = out / f"{row['system_id']}.npz"
            if dest.exists() and not a.overwrite:
                continue
            pdb = work / "complex.pdb"
            try:
                signal.alarm(a.timeout)
                prot = load_protein_atoms(row["input_path"], parse_chains(row["input_chains"]),
                                          keep_hydrogens=False)
                chain = complex_pdb(prot, row["sdf_paths"].split(","), pdb)
                rec, n_sites = run_plip(pdb, LIG_RESNAME, chain)
                signal.alarm(0)
                if n_sites == 0:
                    raise RuntimeError("PLIP found no binding site for the ligand")
                save_plip(dest, rec)
                built += 1
                recs.append(rec)
            except _Timeout:
                failed.append((row["system_id"], f"timeout after {a.timeout}s"))
            except Exception as e:
                signal.alarm(0)
                failed.append((row["system_id"], f"{type(e).__name__}: {e}"))
            if j % 50 == 0 or j == len(rows):
                print(f"  {j}/{len(rows)}  {built} built  {len(failed)} failed  "
                      f"{(time.time() - t0) / j:.2f}s/row", flush=True)
    finally:
        signal.alarm(0)
        shutil.rmtree(work, ignore_errors=True)
    print(f"\n{built} PLIP files written to {out}, {len(failed)} failed")
    if recs:
        print("\n".join(summarize(recs)))
    if failed:
        log = out / f"failed{tag}.tsv"
        log.write_text("".join(f"{k}\t{m}\n" for k, m in failed))
        for k, m in failed[:10]:
            print(f"    {k}: {m[:160]}")
        print(f"all failures: {log}")


def demo(a):
    """Our pipeline on one PDB (protein atoms + the ligand residue re-read from
    an SDF) next to PLIP on the original file. Agreement on H-bonds and salt
    bridges says the complex we build is one PLIP reads correctly."""
    from openbabel import openbabel as ob
    from openbabel import pybel

    from extract import load_protein_atoms

    work = Path(tempfile.mkdtemp(prefix="plip_demo_"))
    try:
        # the ligand residue alone, as an SDF, the form PLINDER ships
        full = next(pybel.readfile("pdb", a.pdb)).OBMol
        lig = ob.OBMol(full)
        drop = [atom for atom in ob.OBMolAtomIter(lig)
                if atom.GetResidue() is None or atom.GetResidue().GetName() != a.ligand]
        for atom in reversed(drop):
            lig.DeleteAtom(atom)
        if not lig.NumAtoms():
            raise SystemExit(f"no residue {a.ligand} in {a.pdb}")
        sdf = work / "lig.sdf"
        pybel.Molecule(lig).write("sdf", str(sdf), overwrite=True)
        prot = load_protein_atoms(a.pdb, keep_hydrogens=False)
        chain = complex_pdb(prot, [sdf], work / "complex.pdb")
        ours, n1 = run_plip(work / "complex.pdb", LIG_RESNAME, chain)
        ref, n2 = run_plip(a.pdb, a.ligand)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    print(f"{a.pdb}, ligand {a.ligand}: {n1} site(s) in our complex, {n2} in the original")
    print(f"{'kind':18s} {'ours':>6s} {'original':>9s}")
    for i, k in enumerate(KINDS):
        print(f"{k:18s} {int((ours['kind'] == i).sum()):6d} {int((ref['kind'] == i).sum()):9d}")
    print("Expect H-bonds and salt bridges to agree; the original can have water "
          "bridges and metal contacts our complex leaves out by design.")
    hb = [KINDS.index("hbond_pdon"), KINDS.index("hbond_ldon")]
    if not np.isin(ours["kind"], hb).any():
        raise SystemExit("FAIL: no H-bonds in our complex")


# ---------------------------------------------------------------------------
# check: the labels on real surfaces (numpy only)
# ---------------------------------------------------------------------------
def check(a):
    from test_encoder import read_sdf_heavy
    from train_data import read_items

    rows = read_items(a.items)
    pdir, sdir = Path(a.plip), Path(a.surface_cache)
    have = [r for r in rows if (pdir / f"{r['system_id']}.npz").exists()]
    print(f"{len(rows)} rows; {len(have)} have a PLIP file")
    if not have:
        raise SystemExit("nothing to check")
    print("\n".join(summarize([load_plip(pdir / f"{r['system_id']}.npz") for r in have])))
    rng = np.random.default_rng(a.seed)
    sample = [r for r in have if (sdir / f"{r['input_id']}.npz").exists()]
    if a.n and len(sample) > a.n:
        sample = [sample[i] for i in np.sort(rng.choice(len(sample), a.n, replace=False))]
    pts, inside, pocket_hit, n_rows = np.zeros(6), np.zeros(6), [], 0
    for r in sample:
        with np.load(sdir / f"{r['input_id']}.npz", allow_pickle=False) as d:
            xyz = d["xyz"].astype(np.float32)
        lig = np.concatenate([read_sdf_heavy(p) for p in r["sdf_paths"].split(",")])
        dmin = np.full(len(xyz), np.inf, np.float32)
        for s in range(0, len(lig), 64):
            dmin = np.minimum(dmin, np.sqrt(
                ((xyz[:, None] - lig[None, s:s + 64]) ** 2).sum(-1)).min(1))
        pocket = dmin < 4.0
        lab = site_labels(xyz, load_plip(pdir / f"{r['system_id']}.npz"), a.atom_radius,
                          a.centre_radius, a.lig_radius)
        pts += lab.sum(0)
        inside += (lab & pocket[:, None]).sum(0)
        if pocket.any():
            pocket_hit.append(float(lab.any(1)[pocket].mean()))
        n_rows += 1
    print(f"\nsite labels on {n_rows} cached surfaces (atom r {a.atom_radius}, centre r "
          f"{a.centre_radius}, ligand r {a.lig_radius}):")
    print(f"{'class':12s} {'points/row':>10s} {'in 4 A pocket':>14s}")
    for i, c in enumerate(CLASSES):
        print(f"{c:12s} {pts[i] / max(n_rows, 1):10.1f} {inside[i] / max(pts[i], 1):14.2f}")
    print(f"share of the 4 A pocket carrying any site label: {np.mean(pocket_hit):.2f}")
    print("A class's 'in 4 A pocket' should be high (most labelled points face the "
          "ligand); a low value means the radius reaches round the back of residues.")


# ---------------------------------------------------------------------------
def selftest(a):
    fails = []

    def ok(name, cond, detail=""):
        print(f"  {'ok  ' if cond else 'FAIL'} {name}" + (f"   {detail}" if detail else ""))
        if not cond:
            fails.append(name)

    print("[1] vocabulary")
    ok("every kind has a class", len(KIND_CLASS) == len(KINDS) == len(CENTRE_KINDS))
    import chemotype
    ok("classes are chemotype.GROUPS, same order", tuple(chemotype.GROUPS) == CLASSES)

    print("[2] site labels on a synthetic surface")
    g = np.random.default_rng(0)
    v = g.normal(size=(5000, 3))
    xyz = (20 * v / np.linalg.norm(v, axis=1, keepdims=True)).astype(np.float32)
    don = np.asarray([0, 0, 18.5], np.float32)      # an atom just under the north pole
    ring = np.asarray([0, 0, -18.0], np.float32)    # a ring centre under the south pole
    rec = {"kind": np.asarray([KINDS.index("hbond_pdon"), KINDS.index("pistack")], np.int8),
           "prot_xyz": np.stack([don, ring]),
           "lig_xyz": np.asarray([[0, 0, 22.0], [0, 0, -23.5]], np.float32)}
    lab = site_labels(xyz, rec)
    d_don = np.linalg.norm(xyz - don, axis=1)
    d_ring = np.linalg.norm(xyz - ring, axis=1)
    ok("donor points = within 3 A of the donor atom",
       np.array_equal(lab[:, 0], d_don <= 3.0), f"{int(lab[:, 0].sum())} points")
    ok("aromatic points = within 4 A of the ring centre (a centre kind)",
       np.array_equal(lab[:, 4], d_ring <= 4.0), f"{int(lab[:, 4].sum())} points")
    ok("no other class set", not lab[:, [1, 2, 3, 5]].any())
    lab_l = site_labels(xyz, rec, lig_radius=4.0)
    ok("lig_radius keeps only the side facing the ligand partner",
       lab_l[:, 0].sum() <= lab[:, 0].sum()
       and np.all(np.linalg.norm(xyz[lab_l[:, 0]] - rec["lig_xyz"][0], axis=1) <= 4.0))
    empty = {"kind": np.zeros(0, np.int8), "prot_xyz": np.zeros((0, 3), np.float32),
             "lig_xyz": np.zeros((0, 3), np.float32)}
    ok("no interactions -> no labels", not site_labels(xyz, empty).any())

    print("[3] storage")
    tmp = Path(tempfile.mkdtemp())
    try:
        save_plip(tmp / "s.npz", rec)
        back = load_plip(tmp / "s.npz")
        ok("npz round trip", all(np.array_equal(back[k], rec[k]) for k in PLIP_KEYS)
           and not list(tmp.glob("*.tmp")))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("[4] summary")
    lines = summarize([rec, empty])
    ok("summary counts per record", any("hbond_pdon" in ln and "0.50" in ln for ln in lines))
    print(f"\n{'PASS' if not fails else 'FAIL: ' + ', '.join(fails)}")
    sys.exit(1 if fails else 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run", help="container + PLIP venv: PLIP on every row")
    p.add_argument("--items", required=True)
    p.add_argument("--out", required=True, help="directory of <system_id>.npz")
    p.add_argument("--workdir", default="", help="scratch; default $TMPDIR or /tmp")
    p.add_argument("--timeout", type=int, default=300, help="seconds per row")
    p.add_argument("--shard", default="", help="i/n, for an array job")
    p.add_argument("--limit", type=int)
    p.add_argument("--overwrite", action="store_true")
    p = sub.add_parser("demo", help="container + PLIP venv: one PDB, ours vs original")
    p.add_argument("--pdb", required=True)
    p.add_argument("--ligand", required=True, help="ligand residue name, e.g. ATP")
    p = sub.add_parser("check", help="numpy: counts and labels on cached surfaces")
    p.add_argument("--items", required=True)
    p.add_argument("--plip", required=True)
    p.add_argument("--surface_cache", required=True)
    p.add_argument("--n", type=int, default=500)
    p.add_argument("--atom_radius", type=float, default=3.0)
    p.add_argument("--centre_radius", type=float, default=4.0)
    p.add_argument("--lig_radius", type=float, default=None)
    p.add_argument("--seed", type=int, default=0)
    sub.add_parser("selftest", help="numpy: mapping and storage")
    a = ap.parse_args()
    {"run": run, "demo": demo, "check": check, "selftest": selftest}[a.cmd](a)


if __name__ == "__main__":
    main()
