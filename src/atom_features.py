"""Per-atom chemistry for the learned typing branch: 16 numbers per heavy atom.

Why. AtomNet sees a 6-way ELEMENT one-hot, which cannot tell a carbonyl O from
a hydroxyl O or an amine N from an amide N. chemotype.py fixes that by hand -
fixed pharmacophore flags aggregated onto surface points by fixed rules. This
module is the learned alternative: describe each atom by what it IS chemically,
and let an AtomNet-style network (SurfaceEncoder.chemnet, cfg.atom_feats) learn
the typing and the aggregation from those descriptions.

The residue and atom name are used ONLY as keys into the table below, to
recover chemistry a PDB file does not spell out (hybridisation, charge, H
count). The model sees the 16 values and never the names, and the table is
many-to-one onto residues: 175 (residue, atom) entries of the 20 amino acids
plus MSE collapse to 30 descriptors, the commonest shared by 15-21 residues.
The 13 single-residue ones are chemistry that really is unique (guanidinium
N, amine N, thiol S, the indole NH and its CH, proline N, glycine CA,
threonine's CH-OH carbon) plus histidine's side chain, which carries its own
flag by design. `selftest` reports these counts. No sequence order,
residue index or per-(residue, atom) partial charge goes in: partial charges
are nearly unique per entry, i.e. residue identity by another name.

Features, in order (DIMS = 16):

    0-3    element               C, N, O, S/Se (one-hot; H and unknown: none)
    4-6    hybridisation         sp3, sp2, aromatic (one-hot)
    7      attached H count      0-3, divided by 3
    8      formal charge         spread over equivalent atoms: carboxylate O
                                 -0.5 each (C-terminal O / OXT too), guanidinium
                                 N +1/3 each, amine NZ +1
    9      H-bond donor          0 / 1
    10     H-bond acceptor       0 / 1
    11     carbon bonded to N/O  0 / 1 (CA, C, carboxyl and amide C, ...)
    12     backbone              N, CA, C, O, OXT
    13     histidine             His side-chain atoms. Its ring N are given
                                 donor + acceptor, charge 0, H count 0; the flag
                                 says they are tautomer- and charge-ambiguous
                                 instead of encoding that in fractions
    14-15  burial                log1p(heavy atoms within 6 / 10 A) / 5

Sulfur (Cys SG, Met SD, MSE SE) gets no donor / acceptor flag: element plus
learned context. No N-terminal charge. Hydrogens, if kept, are all zero. A (residue, atom) not in the table keeps its element and, if it is a
backbone name, the backbone rules; everything else 0.

Example:
    python src/atom_features.py          # self-test, numpy only

Depends on: numpy (scipy's cKDTree for burial when it imports).
Imported by: train_data.py (SurfaceStore), surface_encoder.py (DIMS).
"""

import sys
from collections import defaultdict
from functools import lru_cache

import numpy as np

DIMS = 16
NAMES = ("C", "N", "O", "S", "sp3", "sp2", "aromatic", "n_h", "charge", "donor",
         "acceptor", "polar_c", "backbone", "histidine", "burial6", "burial10")
STATIC = 14                      # everything before burial: a function of the name
BURIAL_RADII = (6.0, 10.0)
BURIAL_SCALE = 5.0               # log1p(count) / 5 stays near O(1), as chemotype's

# extract.ELE2NUM order (chemotype.ELEMENT_ORDER) -> our element column
BACKBONE = ("N", "CA", "C", "O", "OXT")


def _a(hyb, n_h, q=0.0, don=0, acc=0, pc=0):
    """hyb: 3 = sp3, 2 = sp2, "ar" = aromatic."""
    return (hyb, n_h, q, don, acc, pc)


def _backbone(res):
    pro, gly = res == "PRO", res == "GLY"
    return {"N": _a(2, 0 if pro else 1, don=0 if pro else 1),
            "CA": _a(3, 2 if gly else 1, pc=1),
            "C": _a(2, 0, pc=1),
            "O": _a(2, 0, acc=1),
            "OXT": _a(2, 0, -0.5, acc=1)}


_AR = "ar"
SIDECHAIN = {
    "ALA": {"CB": _a(3, 3)},
    "ARG": {"CB": _a(3, 2), "CG": _a(3, 2), "CD": _a(3, 2, pc=1),
            "NE": _a(2, 1, 1 / 3, don=1), "CZ": _a(2, 0, pc=1),
            "NH1": _a(2, 2, 1 / 3, don=1), "NH2": _a(2, 2, 1 / 3, don=1)},
    "ASN": {"CB": _a(3, 2), "CG": _a(2, 0, pc=1), "OD1": _a(2, 0, acc=1),
            "ND2": _a(2, 2, don=1)},
    "ASP": {"CB": _a(3, 2), "CG": _a(2, 0, pc=1), "OD1": _a(2, 0, -0.5, acc=1),
            "OD2": _a(2, 0, -0.5, acc=1)},
    "CYS": {"CB": _a(3, 2), "SG": _a(3, 1)},
    "GLN": {"CB": _a(3, 2), "CG": _a(3, 2), "CD": _a(2, 0, pc=1),
            "OE1": _a(2, 0, acc=1), "NE2": _a(2, 2, don=1)},
    "GLU": {"CB": _a(3, 2), "CG": _a(3, 2), "CD": _a(2, 0, pc=1),
            "OE1": _a(2, 0, -0.5, acc=1), "OE2": _a(2, 0, -0.5, acc=1)},
    "GLY": {},
    "HIS": {"CB": _a(3, 2), "CG": _a(_AR, 0), "ND1": _a(_AR, 0, don=1, acc=1),
            "CD2": _a(_AR, 1, pc=1), "CE1": _a(_AR, 1, pc=1),
            "NE2": _a(_AR, 0, don=1, acc=1)},
    "ILE": {"CB": _a(3, 1), "CG1": _a(3, 2), "CG2": _a(3, 3), "CD1": _a(3, 3)},
    "LEU": {"CB": _a(3, 2), "CG": _a(3, 1), "CD1": _a(3, 3), "CD2": _a(3, 3)},
    "LYS": {"CB": _a(3, 2), "CG": _a(3, 2), "CD": _a(3, 2), "CE": _a(3, 2, pc=1),
            "NZ": _a(3, 3, 1.0, don=1)},
    "MET": {"CB": _a(3, 2), "CG": _a(3, 2), "SD": _a(3, 0), "CE": _a(3, 3)},
    "MSE": {"CB": _a(3, 2), "CG": _a(3, 2), "SE": _a(3, 0), "CE": _a(3, 3)},
    "PHE": {"CB": _a(3, 2), "CG": _a(_AR, 0),
            **{k: _a(_AR, 1) for k in ("CD1", "CD2", "CE1", "CE2", "CZ")}},
    "PRO": {"CB": _a(3, 2), "CG": _a(3, 2), "CD": _a(3, 2, pc=1)},
    "SER": {"CB": _a(3, 2, pc=1), "OG": _a(3, 1, don=1, acc=1)},
    "THR": {"CB": _a(3, 1, pc=1), "OG1": _a(3, 1, don=1, acc=1), "CG2": _a(3, 3)},
    "TRP": {"CB": _a(3, 2), "CG": _a(_AR, 0), "CD1": _a(_AR, 1, pc=1),
            "CD2": _a(_AR, 0), "NE1": _a(_AR, 1, don=1), "CE2": _a(_AR, 0, pc=1),
            **{k: _a(_AR, 1) for k in ("CE3", "CZ2", "CZ3", "CH2")}},
    "TYR": {"CB": _a(3, 2), "CG": _a(_AR, 0),
            **{k: _a(_AR, 1) for k in ("CD1", "CD2", "CE1", "CE2")},
            "CZ": _a(_AR, 0, pc=1), "OH": _a(3, 1, don=1, acc=1)},
    "VAL": {"CB": _a(3, 1), "CG1": _a(3, 3), "CG2": _a(3, 3)},
}


def _element_of(name):
    return "SE" if name.startswith("SE") else name[0]


_EL_COL = {"C": 0, "N": 1, "O": 2, "S": 3, "SE": 3}
_TYPE_SYMBOL = ("C", "H", "O", "N", "S", "SE")      # extract.ELE2NUM order


@lru_cache(maxsize=None)
def static_features(resname, name, element):
    """(residue, atom name, element symbol) -> the 14 name-determined values."""
    v = np.zeros(STATIC, np.float32)
    col = _EL_COL.get(element)
    if col is not None:
        v[col] = 1.0
    if name in BACKBONE:
        d = _backbone(resname).get(name)
        v[12] = 1.0
    else:
        d = SIDECHAIN.get(resname, {}).get(name)
        if resname == "HIS":
            v[13] = 1.0
    if d is None or col is None:
        return v
    hyb, n_h, q, don, acc, pc = d
    v[4 + {3: 0, 2: 1, _AR: 2}[hyb]] = 1.0
    v[7], v[8], v[9], v[10], v[11] = n_h / 3.0, q, don, acc, pc
    return v


def burial(xyz, radii=BURIAL_RADII, chunk=2048):
    """(M, 3) heavy-atom coordinates -> (M, len(radii)) log1p(neighbours) / 5,
    neighbours = other atoms within each radius."""
    xyz = np.asarray(xyz, np.float32)
    out = np.zeros((len(xyz), len(radii)), np.float32)
    if not len(xyz):
        return out
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        cKDTree = None
    if cKDTree is not None:
        tree = cKDTree(xyz)
        for j, r in enumerate(radii):
            out[:, j] = tree.query_ball_point(xyz, r, return_length=True) - 1
    else:
        r2 = np.asarray(radii, np.float32) ** 2
        for s in range(0, len(xyz), chunk):
            d2 = ((xyz[s:s + chunk, None] - xyz[None]) ** 2).sum(-1)
            out[s:s + chunk] = (d2[..., None] <= r2).sum(1) - 1
    return np.log1p(out) / BURIAL_SCALE


def atom_features(prot):
    """load_protein_atoms() dict -> (M, DIMS) float32.

    Burial counts heavy atoms only, so the same receptor reads the same with
    and without hydrogens; hydrogen rows (kept only with keep_hydrogens) are
    all zero."""
    resn = np.char.upper(np.char.strip(np.asarray(prot["atom_resname"]).astype(str)))
    name = np.char.upper(np.char.strip(np.asarray(prot["atom_name"]).astype(str)))
    types = np.asarray(prot["atom_type"])
    out = np.zeros((len(types), DIMS), np.float32)
    for i in range(len(types)):
        if int(types[i]) == 1:          # hydrogen
            continue
        # the element as parsed (extract.ELE2NUM order), not guessed from the name
        out[i, :STATIC] = static_features(resn[i], name[i], _TYPE_SYMBOL[int(types[i])])
    # C-terminal carboxylate: the residue's O shares OXT's charge
    if "atom_chain" in prot and (name == "OXT").any():
        key = np.char.add(np.char.add(np.asarray(prot["atom_chain"]).astype(str), "|"),
                          np.asarray(prot["atom_resnum"]).astype(str))
        if "atom_icode" in prot:
            key = np.char.add(key, np.asarray(prot["atom_icode"]).astype(str))
        term = set(key[name == "OXT"])
        out[(name == "O") & np.isin(key, list(term)), 8] = -0.5
    heavy = types != 1
    out[heavy, STATIC:] = burial(np.asarray(prot["atom_xyz"])[heavy])
    return out


def annotate(prot):
    """Add atom_feats to a load_protein_atoms() dict, in place -> the dict."""
    prot["atom_feats"] = atom_features(prot)
    return prot


# ---------------------------------------------------------------------------
def descriptor_table():
    """-> {descriptor tuple: set of residues} over every table entry."""
    by = defaultdict(set)
    for res, sc in SIDECHAIN.items():
        names = list(_backbone(res)) + list(sc)
        for nm in names:
            if nm == "OXT":
                continue
            by[tuple(static_features(res, nm, _element_of(nm)).round(4))].add(res)
    return by


def selftest():
    fails = []

    def check(label, ok, detail=""):
        print(f"  {'ok  ' if ok else 'FAIL'} {label}" + (f"   {detail}" if detail else ""))
        if not ok:
            fails.append(label)

    import chemotype

    check("element order matches extract / chemotype",
          _TYPE_SYMBOL == tuple(chemotype.ELEMENT_ORDER))

    # coverage: every standard side-chain atom chemotype knows has an entry
    missing = [(r, n) for r, s in chemotype.SIDECHAIN.items() for n in s.split()
               if n not in SIDECHAIN.get(r, {})]
    check("every standard side-chain heavy atom is in the table", not missing,
          str(missing[:5]))
    extra = [(r, n) for r, s in SIDECHAIN.items() for n in s
             if n not in chemotype.SIDECHAIN.get(r, "").split()]
    check("and nothing outside chemotype's list", not extra, str(extra[:5]))

    by = descriptor_table()
    n_entries = sum(len(s) + 4 for s in SIDECHAIN.values())
    uniq = {d: r for d, r in by.items() if len(r) == 1}
    widest = max(len(r) for r in by.values())
    print(f"       {n_entries} (residue, atom) entries -> {len(by)} descriptors; "
          f"{len(uniq)} name a single residue; the commonest is shared by {widest}")
    for d, r in sorted(uniq.items(), key=lambda x: sorted(x[1])):
        print(f"         {sorted(r)[0]}: " + " ".join(
            f"{NAMES[i]}={v:g}" for i, v in enumerate(d) if v))
    single = {next(iter(r)) for r in uniq.values()}
    check("single-residue descriptors are only the chemically unique groups + HIS",
          single <= {"ARG", "LYS", "CYS", "PRO", "GLY", "THR", "TRP", "HIS"},
          str(sorted(single)))
    check("many-to-one: >= 5x fewer descriptors than entries", n_entries >= 5 * len(by))

    # charges: residue totals are the textbook ones
    def total(res):
        return sum(float(static_features(res, n, _element_of(n))[8])
                   for n in list(_backbone(res))[:4] + list(SIDECHAIN[res]))
    tots = {r: round(total(r), 6) for r in ("ASP", "GLU", "LYS", "ARG", "HIS", "SER")}
    check("net charge: ASP / GLU -1, LYS / ARG +1, HIS / SER 0",
          tots == {"ASP": -1, "GLU": -1, "LYS": 1, "ARG": 1, "HIS": 0, "SER": 0}, str(tots))

    # spot values
    og = static_features("SER", "OG", "O")
    check("SER OG: O, sp3, 1 H, donor + acceptor, not backbone",
          og[2] == 1 and og[4] == 1 and abs(og[7] - 1 / 3) < 1e-6 and og[9] == og[10] == 1
          and og[12] == 0)
    pn = static_features("PRO", "N", "N")
    check("PRO N: backbone, no H, not a donor", pn[12] == 1 and pn[7] == 0 and pn[9] == 0)
    hs = static_features("HIS", "NE2", "N")
    check("HIS NE2: aromatic, donor + acceptor, histidine flag, charge 0",
          hs[6] == 1 and hs[9] == hs[10] == hs[13] == 1 and hs[8] == 0)
    hb = static_features("HIS", "CA", "C")
    check("HIS backbone atoms do not carry the histidine flag", hb[13] == 0 and hb[12] == 1)
    sg = static_features("CYS", "SG", "S")
    check("CYS SG: no donor / acceptor flag", sg[3] == 1 and sg[9] == sg[10] == 0)
    se = static_features("MSE", "SE", "SE")
    check("MSE SE reads as sulfur", se[3] == 1 and se[4] == 1)
    unk = static_features("XYZ", "C7", "C")
    check("unknown residue: element only", unk[0] == 1 and unk[1:].sum() == 0)
    unkb = static_features("XYZ", "CA", "C")
    check("unknown residue backbone: backbone rules apply", unkb[12] == 1 and unkb[4] == 1)

    # a two-residue fragment through atom_features, with OXT and an H
    prot = {"atom_resname": np.array(["SER"] * 6 + ["GLY"] * 5 + ["GLY"]),
            "atom_name": np.array(["N", "CA", "C", "O", "CB", "OG",
                                   "N", "CA", "C", "O", "OXT", "H"]),
            "atom_type": np.array([3, 0, 0, 2, 0, 2, 3, 0, 0, 2, 2, 1]),
            "atom_chain": np.array(["A"] * 12), "atom_resnum": np.array([1] * 6 + [2] * 6),
            "atom_icode": np.array([""] * 12),
            "atom_xyz": np.random.default_rng(0).normal(size=(12, 3)).astype(np.float32) * 3}
    f = atom_features(prot)
    check("atom_features: shape and dtype", f.shape == (12, DIMS) and f.dtype == np.float32)
    check("C-terminal O and OXT carry -0.5 each; the other residue's O 0",
          f[9, 8] == -0.5 and f[10, 8] == -0.5 and f[3, 8] == 0)
    check("hydrogen: static part all zero", f[11, :STATIC].sum() == 0)
    check("burial is finite and non-negative", np.isfinite(f[:, STATIC:]).all()
          and (f[:, STATIC:] >= 0).all())
    # burial: chunked numpy agrees with a direct count
    rng = np.random.default_rng(1)
    pts = rng.uniform(0, 30, size=(500, 3)).astype(np.float32)
    d = np.sqrt(((pts[:, None] - pts[None]) ** 2).sum(-1))
    want = np.log1p(np.stack([(d <= r).sum(1) - 1 for r in BURIAL_RADII], 1)) / BURIAL_SCALE
    check("burial matches a direct neighbour count", np.allclose(burial(pts, chunk=97), want,
                                                                 atol=1e-6))
    check("burial: rigid-motion invariant",
          np.allclose(burial(pts @ _rot().T + 5.0), burial(pts), atol=1e-6))

    print("\nPASS" if not fails else f"\nFAIL: {fails}")
    return 1 if fails else 0


def _rot():
    q = np.random.default_rng(2).normal(size=(3, 3))
    u, _, vt = np.linalg.svd(q)
    r = u @ vt
    return (r * np.sign(np.linalg.det(r))).astype(np.float32)


if __name__ == "__main__":
    sys.exit(selftest())
