"""Per-atom chemistry and polar directions for the learned typing branch.

Why. AtomNet sees a 6-way ELEMENT one-hot, which cannot tell a carbonyl O from
a hydroxyl O or an amine N from an amide N. chemotype.py fixes that by hand -
fixed pharmacophore flags aggregated onto surface points by fixed rules. This
module is the learned alternative: describe each atom by what it IS chemically
(13 numbers) and where its polar group POINTS (a unit vector), and let
dir_atomnet.DirAtomNet (SurfaceEncoder.chemnet, cfg.atom_feats) learn the
typing and the aggregation from those.

What is in and what is left out. The model gets what heavy-atom coordinates
cannot show - charge, whether an N or O carries an H (donor) or a lone pair
(acceptor), aromaticity, density - plus things that are inferable only with
difficulty (sp2 / sp3 from 0.2 A bond-length differences, H count, the
backbone pattern). Left out because one round of message passing recovers
them: "carbon bonded to N/O" (a neighbour ~1.4 A away whose element is N/O),
a histidine flag (aromatic N that is donor + acceptor is His), and burial at
a second radius (correlated with the first).

The residue and atom name are used ONLY as keys into the table below. The
model never sees them, and the table is many-to-one onto residues: 175
(residue, atom) entries of the 20 amino acids plus MSE collapse to 23
descriptors. The 8 single-residue ones are chemical groups that really are
unique - guanidinium N (2), amine N, thiol S, imidazole N, indole NH, proline
N, glycine CA - and `selftest` reports them. No sequence order, residue
index or per-(residue, atom) partial charge goes in: partial charges are
nearly unique per entry, i.e. residue identity by another name.

Features, in order (DIMS = 13):

    0-3    element               C, N, O, S/Se (one-hot; H and unknown: none)
    4-6    hybridisation         sp3, sp2, aromatic (one-hot)
    7      attached H count      0-3, divided by 3
    8      formal charge         spread over equivalent atoms: carboxylate O
                                 -0.5 each (C-terminal O / OXT too), guanidinium
                                 N +1/3 each, amine NZ +1; histidine 0
    9      H-bond donor          0 / 1 (His ring N: donor and acceptor)
    10     H-bond acceptor       0 / 1
    11     backbone              N, CA, C, O, OXT
    12     burial                log1p(heavy atoms within 10 A) / 5

Polar direction, per atom (atom_direction), cached beside the features as
`atom_dir` (M, 3): for a polar atom (donor, acceptor or charged), the unit
sum of unit vectors pointing AWAY from its covalently bonded heavy
neighbours (closer than 1.9 A). That is the N-H / lone-pair bisector for a
backbone N, ring N or amide N, the C->O axis for carbonyl, carboxylate and
hydroxyl O, the C->N axis for an amine. Zero for non-polar atoms and atoms
with no bonded neighbour. Divided by max(|sum|, 0.5) rather than |sum|, so
it is a unit vector for every real protein atom (one bond: |sum| = 1; two at
~120 deg: ~0.97) and, in geometry where the bond vectors cancel, shrinks
continuously to 0 instead of becoming a noisy unit vector - no threshold,
no jump, gain at most 2. It is a VECTOR in the structure's own frame,
never a feature: DirAtomNet uses it only in a dot product with the
atom-to-surface-point vector, so everything the network sees is invariant.

Sulfur (Cys SG, Met SD, MSE SE) gets no donor / acceptor flag: element plus
learned context. No N-terminal charge. Hydrogens, if kept, are all zero. A
(residue, atom) not in the table keeps its element and, if it is a backbone
name, the backbone rules; everything else 0.

Example:
    python src/atom_features.py          # self-test, numpy only

Depends on: numpy (scipy's cKDTree for burial when it imports).
Imported by: train_data.py (SurfaceStore), surface_encoder.py, dir_atomnet.py.
"""

import sys
from collections import defaultdict
from functools import lru_cache

import numpy as np

DIMS = 13
NAMES = ("C", "N", "O", "S", "sp3", "sp2", "aromatic", "n_h", "charge", "donor",
         "acceptor", "backbone", "burial")
STATIC = 12                      # everything before burial: a function of the name
BURIAL_RADIUS = 10.0
BURIAL_SCALE = 5.0               # log1p(count) / 5 stays near O(1), as chemotype's
BOND_MAX = 1.9                   # A; covalent N/O bonds are 1.2-1.5, the next
                                 # heavy-atom shell (bond angles) starts ~2.3
DIR_FLOOR = 0.5                  # direction = sum / max(|sum|, DIR_FLOOR): unit for
                                 # every real protein atom (|sum| >= ~0.9), shrinking
                                 # continuously to 0 where bond vectors cancel

BACKBONE = ("N", "CA", "C", "O", "OXT")


def _a(hyb, n_h, q=0.0, don=0, acc=0):
    """hyb: 3 = sp3, 2 = sp2, "ar" = aromatic."""
    return (hyb, n_h, q, don, acc)


def _backbone(res):
    pro, gly = res == "PRO", res == "GLY"
    return {"N": _a(2, 0 if pro else 1, don=0 if pro else 1),
            "CA": _a(3, 2 if gly else 1),
            "C": _a(2, 0),
            "O": _a(2, 0, acc=1),
            "OXT": _a(2, 0, -0.5, acc=1)}


_AR = "ar"
SIDECHAIN = {
    "ALA": {"CB": _a(3, 3)},
    "ARG": {"CB": _a(3, 2), "CG": _a(3, 2), "CD": _a(3, 2),
            "NE": _a(2, 1, 1 / 3, don=1), "CZ": _a(2, 0),
            "NH1": _a(2, 2, 1 / 3, don=1), "NH2": _a(2, 2, 1 / 3, don=1)},
    "ASN": {"CB": _a(3, 2), "CG": _a(2, 0), "OD1": _a(2, 0, acc=1),
            "ND2": _a(2, 2, don=1)},
    "ASP": {"CB": _a(3, 2), "CG": _a(2, 0), "OD1": _a(2, 0, -0.5, acc=1),
            "OD2": _a(2, 0, -0.5, acc=1)},
    "CYS": {"CB": _a(3, 2), "SG": _a(3, 1)},
    "GLN": {"CB": _a(3, 2), "CG": _a(3, 2), "CD": _a(2, 0),
            "OE1": _a(2, 0, acc=1), "NE2": _a(2, 2, don=1)},
    "GLU": {"CB": _a(3, 2), "CG": _a(3, 2), "CD": _a(2, 0),
            "OE1": _a(2, 0, -0.5, acc=1), "OE2": _a(2, 0, -0.5, acc=1)},
    "GLY": {},
    "HIS": {"CB": _a(3, 2), "CG": _a(_AR, 0), "ND1": _a(_AR, 0, don=1, acc=1),
            "CD2": _a(_AR, 1), "CE1": _a(_AR, 1),
            "NE2": _a(_AR, 0, don=1, acc=1)},
    "ILE": {"CB": _a(3, 1), "CG1": _a(3, 2), "CG2": _a(3, 3), "CD1": _a(3, 3)},
    "LEU": {"CB": _a(3, 2), "CG": _a(3, 1), "CD1": _a(3, 3), "CD2": _a(3, 3)},
    "LYS": {"CB": _a(3, 2), "CG": _a(3, 2), "CD": _a(3, 2), "CE": _a(3, 2),
            "NZ": _a(3, 3, 1.0, don=1)},
    "MET": {"CB": _a(3, 2), "CG": _a(3, 2), "SD": _a(3, 0), "CE": _a(3, 3)},
    "MSE": {"CB": _a(3, 2), "CG": _a(3, 2), "SE": _a(3, 0), "CE": _a(3, 3)},
    "PHE": {"CB": _a(3, 2), "CG": _a(_AR, 0),
            **{k: _a(_AR, 1) for k in ("CD1", "CD2", "CE1", "CE2", "CZ")}},
    "PRO": {"CB": _a(3, 2), "CG": _a(3, 2), "CD": _a(3, 2)},
    "SER": {"CB": _a(3, 2), "OG": _a(3, 1, don=1, acc=1)},
    "THR": {"CB": _a(3, 1), "OG1": _a(3, 1, don=1, acc=1), "CG2": _a(3, 3)},
    "TRP": {"CB": _a(3, 2), "CG": _a(_AR, 0), "CD1": _a(_AR, 1),
            "CD2": _a(_AR, 0), "NE1": _a(_AR, 1, don=1), "CE2": _a(_AR, 0),
            **{k: _a(_AR, 1) for k in ("CE3", "CZ2", "CZ3", "CH2")}},
    "TYR": {"CB": _a(3, 2), "CG": _a(_AR, 0),
            **{k: _a(_AR, 1) for k in ("CD1", "CD2", "CE1", "CE2")},
            "CZ": _a(_AR, 0), "OH": _a(3, 1, don=1, acc=1)},
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
        v[11] = 1.0
    else:
        d = SIDECHAIN.get(resname, {}).get(name)
    if d is None or col is None:
        return v
    hyb, n_h, q, don, acc = d
    v[4 + {3: 0, 2: 1, _AR: 2}[hyb]] = 1.0
    v[7], v[8], v[9], v[10] = n_h / 3.0, q, don, acc
    return v


def burial(xyz, radii=(BURIAL_RADIUS,), chunk=2048):
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


def atom_direction(xyz, feats, bond_max=BOND_MAX, chunk=1024):
    """(M, 3) coordinates + (M, DIMS) features -> (M, 3) float32 polar
    directions (module docstring). Hydrogen rows (all-zero features) and
    non-polar atoms get 0."""
    xyz = np.asarray(xyz, np.float32)
    feats = np.asarray(feats)
    out = np.zeros_like(xyz)
    heavy = feats[:, :4].any(1)
    polar = heavy & ((feats[:, 9] > 0) | (feats[:, 10] > 0) | (feats[:, 8] != 0))
    pi, hx = np.nonzero(polar)[0], xyz[heavy]
    for s in range(0, len(pi), chunk):
        rows = pi[s:s + chunk]
        v = xyz[rows, None] - hx[None]                          # (n, H, 3)
        d = np.sqrt((v ** 2).sum(-1))
        bonded = (d > 0.5) & (d < bond_max)                    # 0.5: not itself
        u = (v / np.maximum(d, 1e-6)[..., None] * bonded[..., None]).sum(1)
        out[rows] = u / np.maximum(np.linalg.norm(u, axis=1), DIR_FLOOR)[:, None]
    return out


def annotate(prot):
    """Add atom_feats and atom_dir to a load_protein_atoms() dict, in place ->
    the dict."""
    prot["atom_feats"] = atom_features(prot)
    prot["atom_dir"] = atom_direction(prot["atom_xyz"], prot["atom_feats"])
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
    single = sorted(next(iter(r)) for r in uniq.values())
    check("single-residue descriptors are only chemically unique groups",
          set(single) <= {"ARG", "LYS", "CYS", "PRO", "GLY", "TRP", "HIS"}, str(single))
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
          and og[11] == 0)
    pn = static_features("PRO", "N", "N")
    check("PRO N: backbone, no H, not a donor", pn[11] == 1 and pn[7] == 0 and pn[9] == 0)
    hs = static_features("HIS", "NE2", "N")
    check("HIS NE2: aromatic, donor + acceptor, charge 0",
          hs[6] == 1 and hs[9] == hs[10] == 1 and hs[8] == 0)
    sg = static_features("CYS", "SG", "S")
    check("CYS SG: no donor / acceptor flag", sg[3] == 1 and sg[9] == sg[10] == 0)
    se = static_features("MSE", "SE", "SE")
    check("MSE SE reads as sulfur", se[3] == 1 and se[4] == 1)
    unk = static_features("XYZ", "C7", "C")
    check("unknown residue: element only", unk[0] == 1 and unk[1:].sum() == 0)
    unkb = static_features("XYZ", "CA", "C")
    check("unknown residue backbone: backbone rules apply", unkb[11] == 1 and unkb[4] == 1)

    # ---- polar directions ----
    def feats_of(names):
        return np.stack([np.concatenate([static_features(r, n, _element_of(n)), [0.0]])
                         for r, n in names]).astype(np.float32)

    def unit(v):
        return v / np.linalg.norm(v)

    # a peptide N between C(i-1) (1.33 A) and CA (1.46 A), 121.7 deg apart,
    # with a carbonyl C=O (1.23 A) and a far-away methyl
    ang = np.radians(121.7)
    xyz = np.array([[0, 0, 0],                                   # N
                    [1.33, 0, 0],                                # C(i-1)
                    [1.46 * np.cos(ang), 1.46 * np.sin(ang), 0],  # CA
                    [1.33 + 0.62, 1.06, 0],                      # O of C(i-1), 1.23 A
                    [9.0, 9.0, 9.0]], np.float32)                # CB elsewhere
    fe = feats_of([("ALA", "N"), ("ALA", "C"), ("ALA", "CA"), ("ALA", "O"), ("ALA", "CB")])
    u = atom_direction(xyz, fe)
    nh = -unit(unit(xyz[1] - xyz[0]) + unit(xyz[2] - xyz[0]))
    co = unit(xyz[3] - xyz[1])
    check("backbone N: points along the N-H bisector, ~119 deg from both bonds",
          np.allclose(u[0], nh, atol=1e-5)
          and all(abs(np.degrees(np.arccos(u[0] @ unit(xyz[k] - xyz[0]))) - 119.15) < 0.5
                  for k in (1, 2)))
    check("carbonyl O: points along C->O", np.allclose(u[3], co, atol=1e-5))
    check("non-polar atoms (C, CA, CB): no direction", not u[[1, 2, 4]].any())
    check("unit length where defined", abs(np.linalg.norm(u[0]) - 1) < 1e-5)
    R, t = _rot(), np.array([3.0, -7.0, 11.0], np.float32)
    ur = atom_direction(xyz @ R.T + t, fe)
    check("equivariant: rotating the atoms rotates the directions, translating "
          "leaves them", np.allclose(ur, u @ R.T, atol=1e-5))
    lone = atom_direction(np.array([[0, 0, 0], [5, 0, 0]], np.float32),
                          feats_of([("SER", "OG"), ("SER", "CB")]))
    check("a polar atom with no bonded neighbour: zero, not NaN",
          np.isfinite(lone).all() and not lone.any())
    sym = atom_direction(np.array([[0, 0, 0], [1.4, 0, 0], [-1.4, 0, 0]], np.float32),
                         feats_of([("SER", "OG"), ("SER", "CB"), ("SER", "CA")]))
    check("bond vectors that cancel: zero, not a noisy direction",
          np.isfinite(sym).all() and not sym[0].any())
    near = atom_direction(np.array([[0, 0, 0], [1.4, 0, 0], [-1.4, 0.05, 0]], np.float32),
                          feats_of([("SER", "OG"), ("SER", "CB"), ("SER", "CA")]))
    resid = np.linalg.norm(unit(np.array([-1.4, 0, 0])) + unit(np.array([1.4, -0.05, 0])))
    check("almost cancelling: a short vector of length |sum| / 0.5, not a unit one "
          "(continuous, no blow-up)",
          np.isfinite(near).all() and abs(np.linalg.norm(near[0]) - resid / 0.5) < 1e-4,
          f"|u| {np.linalg.norm(near[0]):.3f}")

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
    a2 = dict(prot)
    annotate(a2)
    check("annotate adds atom_feats and atom_dir; hydrogen has no direction",
          a2["atom_dir"].shape == (12, 3) and not a2["atom_dir"][11].any())
    check("burial is finite and non-negative", np.isfinite(f[:, STATIC:]).all()
          and (f[:, STATIC:] >= 0).all())
    # burial: chunked numpy agrees with a direct count
    rng = np.random.default_rng(1)
    pts = rng.uniform(0, 30, size=(500, 3)).astype(np.float32)
    d = np.sqrt(((pts[:, None] - pts[None]) ** 2).sum(-1))
    want = np.log1p(np.stack([(d <= r).sum(1) - 1 for r in (BURIAL_RADIUS,)], 1)) / BURIAL_SCALE
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
