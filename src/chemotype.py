"""Per-surface-point chemical descriptors: the pocket's pharmacophore.

The encoder's only chemical vocabulary today is a 6-way ELEMENT one-hot
(C/H/O/N/S/Se) fed to AtomNet_MP. That cannot distinguish a carbonyl O from a
hydroxyl O, a backbone N from a lysine NZ, or an aromatic C from an aliphatic
one - and it is being asked to align with a ligand branch that encodes ECFP
substructures and explicit donor/acceptor/charge/aromatic flags. This module
closes that asymmetry by typing the protein surface in the SAME vocabulary the
ligand side uses: the six standard pharmacophore types.

It also adds the thing AtomNet structurally cannot provide. AtomNet means over
the 16 NEAREST atoms, which normalises density away, so the current input has
no enclosure signal at all - and enclosure is most of what makes a pocket a
pocket. `buriedness` below is that signal.

Two steps:

    1. type the atoms      (resname, atom_name, element) -> 6 group flags
       Both columns are already parsed by extract.load_protein_atoms and then
       discarded. Heavy atoms only: donors are inferred from the heavy atom's
       identity, never from an attached H, because PLINDER structures disagree
       about hydrogens (the encoder tests strip them).

    2. aggregate onto surface points, three complementary views:

       composition  f_g(p) = Σ_{a∈g} w / Σ_{all a} w        bounded [0, 1]
       buriedness   ρ(p,σ) = log(1 + Σ_{all a} w) / 5       enclosure
       proximity    π_g(p) = exp(−min_{a∈g} d / σ₀)         nearest member

       with w = exp(−d²/2σ²) over atoms within `cutoff`. Proximity alone looks
       at EVERY atom of the protein, not just those inside the cutoff: masking
       it would map "the nearest donor is 15 A away" and "this protein has no
       donor at all" both to exactly 0, and the exponential already drives far
       members to ~0 on its own.

       Composition and buriedness are separate on purpose: one says WHAT
       chemistry is nearby, the other HOW enclosed the point is. Conflating
       them is also a scaling problem, since raw counts are unbounded and would
       dominate the bounded curvature channels inside the conv's first Linear.
       Proximity is kept because it is sharper: "an acceptor at 2.9 A" is an
       H-bond, which a Gaussian sum blurs away.

       Group membership OVERLAPS by design - a tyrosine OH is donor and
       acceptor, a phenyl carbon is aromatic and hydrophobic. So the six
       composition fractions each lie in [0, 1] but do NOT sum to 1.

Channels (12), in order:

    0-5   composition at σ = 3 A: donor acceptor cation anion aromatic hydrophobic
    6-8   buriedness at σ = 3, 6, 9 A
    9-11  proximity to the nearest donor / acceptor / hydrophobic atom

Everything here is DETERMINISTIC given (atoms, surface points), so it carries
no parameters, needs no gradient, and caches with the surface - the same status
as the 10 curvature channels it sits beside.

Rotation and translation invariant: only pairwise distances are used. That is a
hard requirement, not a nicety - apo_eval relies on the encoder being invariant
("placement only matters for the labels"), and at screening time a structure
arrives in an arbitrary frame.

torch is imported inside surface_features(), so the typing tables can be
checked in any environment.

Example:
    python src/chemotype.py --tables     # typing only, no torch needed
    python src/chemotype.py              # full self-test (needs torch)

Depends on: numpy; torch (surface_features only).
Imported by: surface_encoder.py, test_encoder.py, apo_eval.py, probe_retrieval.py.
"""

import argparse
from functools import lru_cache

import numpy as np

GROUPS = ("donor", "acceptor", "cation", "anion", "aromatic", "hydrophobic")
N_GROUPS = len(GROUPS)

# Mirrors extract.ELE2NUM. Duplicated rather than imported because extract.py
# pulls in torch at module level; selftest() asserts the two agree.
ELEMENT_ORDER = ("C", "H", "O", "N", "S", "SE")

BACKBONE = {"N", "CA", "C", "O", "OXT"}

# --- sidechain typing, by residue ------------------------------------------
# Hydroxyls (SER/THR/TYR) and histidine ring N appear as BOTH donor and
# acceptor: that is chemically right, not a duplicated entry.
DONOR = {
    "ARG": {"NE", "NH1", "NH2"}, "ASN": {"ND2"}, "GLN": {"NE2"},
    "HIS": {"ND1", "NE2"}, "LYS": {"NZ"}, "TRP": {"NE1"},
    "SER": {"OG"}, "THR": {"OG1"}, "TYR": {"OH"},
    "CYS": {"SG"},                                    # weak
}
ACCEPTOR = {
    "ASP": {"OD1", "OD2"}, "GLU": {"OE1", "OE2"}, "ASN": {"OD1"},
    "GLN": {"OE1"}, "HIS": {"ND1", "NE2"},
    "SER": {"OG"}, "THR": {"OG1"}, "TYR": {"OH"},
    "MET": {"SD"}, "MSE": {"SE"}, "CYS": {"SG"},      # weak
}
# Charged at physiological pH. Histidine is included although its pKa is ~6,
# so it is cationic only part of the time; excluding it entirely would lose a
# group that genuinely lines many pockets.
CATION = {
    "LYS": {"NZ"},
    "ARG": {"NE", "NH1", "NH2", "CZ"},   # CZ carries the delocalised charge
    "HIS": {"ND1", "NE2"},
}
ANION = {"ASP": {"OD1", "OD2"}, "GLU": {"OE1", "OE2"}}
AROMATIC = {
    "PHE": {"CG", "CD1", "CD2", "CE1", "CE2", "CZ"},
    "TYR": {"CG", "CD1", "CD2", "CE1", "CE2", "CZ"},
    "TRP": {"CG", "CD1", "NE1", "CD2", "CE2", "CE3", "CZ2", "CZ3", "CH2"},
    "HIS": {"CG", "ND1", "CD2", "CE1", "NE2"},
}
# Sidechain carbons that carry or neighbour the polar/charged group, so they are
# NOT hydrophobic. Everything else made of C/S/Se outside the backbone is.
# Bonds are unavailable here, so this exclusion list stands in for "carbon not
# bonded to N/O".
POLAR_CARBON = {
    "ASP": {"CG"}, "GLU": {"CD"}, "ASN": {"CG"}, "GLN": {"CD"},
    "ARG": {"CZ", "CD"}, "LYS": {"CE"},
}

# Standard sidechain heavy atoms, for the coverage check in check_tables().
# Hydrophobic membership comes from a FALLBACK rule rather than a table, so
# without this list a diagnostic that walks only the tables above reports
# leucine and valine as untyped, which is wrong.
SIDECHAIN = {
    "ALA": "CB", "ARG": "CB CG CD NE CZ NH1 NH2", "ASN": "CB CG OD1 ND2",
    "ASP": "CB CG OD1 OD2", "CYS": "CB SG", "GLN": "CB CG CD OE1 NE2",
    "GLU": "CB CG CD OE1 OE2", "GLY": "", "HIS": "CB CG ND1 CD2 CE1 NE2",
    "ILE": "CB CG1 CG2 CD1", "LEU": "CB CG CD1 CD2", "LYS": "CB CG CD CE NZ",
    "MET": "CB CG SD CE", "PHE": "CB CG CD1 CD2 CE1 CE2 CZ", "PRO": "CB CG CD",
    "SER": "CB OG", "THR": "CB OG1 CG2",
    "TRP": "CB CG CD1 CD2 NE1 CE2 CE3 CZ2 CZ3 CH2",
    "TYR": "CB CG CD1 CD2 CE1 CE2 CZ OH", "VAL": "CB CG1 CG2",
    "MSE": "CB CG SE CE",
}

# Sidechain atoms that are deliberately in NO group: the carboxyl, amide and
# guanidinium carbons, which are excluded from hydrophobic because they bear the
# polar group, and are not themselves donors or acceptors. Their oxygens and
# nitrogens carry the typing instead.
UNTYPED = {("ASP", "CG"), ("GLU", "CD"), ("ASN", "CG"), ("GLN", "CD"),
           ("ARG", "CD"), ("LYS", "CE")}


def _element_of(name):
    """Element from a standard PDB atom name (SE before the S/C/N/O initial)."""
    return "SE" if name.startswith("SE") else name[0]


@lru_cache(maxsize=None)
def group_mask(resname, name, element):
    """(resname, atom_name, element) -> 6-tuple of 0/1 flags, order GROUPS.

    Cached: a structure has tens of thousands of atoms but only a few hundred
    distinct (residue, atom, element) combinations.
    """
    m = [0] * N_GROUPS
    if name in BACKBONE:
        # Backbone N donates in every residue except proline, whose N is
        # tertiary and has no hydrogen to give.
        if name == "N" and resname != "PRO":
            m[0] = 1
        if name in ("O", "OXT"):
            m[1] = 1
        if name == "OXT":          # C-terminal carboxylate
            m[3] = 1
        return tuple(m)
    if name in DONOR.get(resname, ()):
        m[0] = 1
    if name in ACCEPTOR.get(resname, ()):
        m[1] = 1
    if name in CATION.get(resname, ()):
        m[2] = 1
    if name in ANION.get(resname, ()):
        m[3] = 1
    if name in AROMATIC.get(resname, ()):
        m[4] = 1
    if element in ("C", "S", "SE") and name not in POLAR_CARBON.get(resname, ()):
        m[5] = 1
    return tuple(m)


def atom_groups(atom_resname, atom_name, atom_type):
    """-> (M, 6) uint8 group membership for a structure's atoms.

    atom_type is the element index as extract.load_protein_atoms writes it
    (ELEMENT_ORDER).
    """
    out = np.zeros((len(atom_resname), N_GROUPS), dtype=np.uint8)
    for i in range(len(atom_resname)):
        el = ELEMENT_ORDER[int(atom_type[i])]
        out[i] = group_mask(str(atom_resname[i]).strip().upper(),
                            str(atom_name[i]).strip().upper(), el)
    return out


def annotate(prot):
    """Add atom_groups to a load_protein_atoms() dict, in place -> the dict.

    Call this once per structure at load time, not per training step: the
    typing is string work and the result is fixed for the file.
    """
    prot["atom_groups"] = atom_groups(prot["atom_resname"], prot["atom_name"],
                                      prot["atom_type"])
    return prot


# ---------------------------------------------------------------------------
# surface aggregation
# ---------------------------------------------------------------------------
SIGMAS = (3.0, 6.0, 9.0)         # buriedness scales; sigmas[0] also sets composition
PROX_GROUPS = (0, 1, 5)          # donor, acceptor, hydrophobic
BUR_SCALE = 5.0                  # keeps log1p(count) near O(1)


def dims(sigmas=SIGMAS):
    """Channel count: composition + one buriedness per scale + proximity.

    Derived from the sigma count rather than hardcoded, so a config that
    changes the scales cannot silently disagree with in_channels.
    """
    return N_GROUPS + len(sigmas) + len(PROX_GROUPS)


DIMS = dims()                    # 12 at the default scales


def feature_names(sigmas=SIGMAS):
    return ([f"comp_{g}_s{sigmas[0]:g}" for g in GROUPS]
            + [f"buried_s{s:g}" for s in sigmas]
            + [f"prox_{GROUPS[g]}" for g in PROX_GROUPS])


def surface_features(xyz, batch, atom_xyz, atom_groups_t, batch_atoms,
                     sigmas=SIGMAS, cutoff=12.0, chunk=1024):
    """-> (N, 12) float, on xyz's device. See the module docstring for channels.

    Runs per protein, NOT over the concatenated batch. That is a correctness
    requirement rather than an optimisation: proteins are concatenated while
    each keeps its own crystal frame, so two of them can overlap arbitrarily in
    coordinate space and one protein's atoms would otherwise contribute to
    another's surface points.

    Memory scales as chunk x atoms: d2 and w are each (chunk, M), and M reaches
    ~20k atoms on a large assembly, so `chunk` is the knob if this ever runs
    out of room.
    """
    import torch

    with torch.no_grad():
        n_bur = len(sigmas)
        prox0 = N_GROUPS + n_bur          # first proximity column
        out = torch.zeros(len(xyz), dims(sigmas), dtype=xyz.dtype, device=xyz.device)
        c2 = float(cutoff) ** 2
        n_prot = int(batch.max()) + 1 if len(batch) else 0
        for b in range(n_prot):
            pts = (batch == b).nonzero().squeeze(1)
            ats = (batch_atoms == b).nonzero().squeeze(1)
            if not len(pts) or not len(ats):
                continue
            A = atom_xyz[ats]
            G = atom_groups_t[ats].to(xyz.dtype)
            has = [bool(G[:, g].any()) for g in PROX_GROUPS]
            for s in range(0, len(pts), chunk):
                idx = pts[s:s + chunk]
                d2 = torch.cdist(xyz[idx], A) ** 2
                near = (d2 < c2).to(xyz.dtype)
                for k, sg in enumerate(sigmas):
                    w = torch.exp(-d2 / (2.0 * sg * sg)) * near
                    total = w.sum(1)
                    out[idx, N_GROUPS + k] = torch.log1p(total) / BUR_SCALE
                    if k == 0:
                        out[idx, :N_GROUPS] = (w @ G) / total[:, None].clamp_min(1e-6)
                # proximity: nearest member of the group, 0 when absent
                for j, g in enumerate(PROX_GROUPS):
                    if not has[j]:
                        continue
                    sel = G[:, g] > 0
                    dmin = d2[:, sel].min(1).values.clamp_min(0).sqrt()
                    out[idx, prox0 + j] = torch.exp(-dmin / sigmas[0])
        return out


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------
def check_tables():
    """Typing sanity, no torch required."""
    cases = [
        # (resname, atom, element, expected groups)
        ("LYS", "NZ", "N", {"donor", "cation", }),
        ("ASP", "OD1", "O", {"acceptor", "anion"}),
        ("ASP", "CG", "C", set()),                       # carboxyl C, not hydrophobic
        ("ASP", "CB", "C", {"hydrophobic"}),
        ("TYR", "OH", "O", {"donor", "acceptor"}),       # hydroxyl is both
        ("TYR", "CZ", "C", {"aromatic", "hydrophobic"}),  # ring C is both
        ("PHE", "CB", "C", {"hydrophobic"}),
        ("ARG", "NH1", "N", {"donor", "cation"}),
        ("ARG", "CD", "C", set()),                       # next to NE
        ("LEU", "CD1", "C", {"hydrophobic"}),
        ("MET", "SD", "S", {"acceptor", "hydrophobic"}),
        ("MSE", "SE", "SE", {"acceptor", "hydrophobic"}),
        ("HIS", "ND1", "N", {"donor", "acceptor", "cation", "aromatic"}),
        ("TRP", "NE1", "N", {"donor", "aromatic"}),
        ("GLY", "N", "N", {"donor"}),                    # backbone amide
        ("PRO", "N", "N", set()),                        # tertiary, no H
        ("ALA", "O", "O", {"acceptor"}),                 # backbone carbonyl
        ("ALA", "OXT", "O", {"acceptor", "anion"}),      # C-terminus
        ("ALA", "CA", "C", set()),                       # backbone, not typed
        ("SER", "OG", "O", {"donor", "acceptor"}),
    ]
    bad = []
    for resname, name, el, want in cases:
        got = {g for g, f in zip(GROUPS, group_mask(resname, name, el)) if f}
        if got != want:
            bad.append(f"  {resname} {name} ({el}): got {sorted(got)}, want {sorted(want)}")
    print(f"{len(cases) - len(bad)}/{len(cases)} typing cases pass")
    for line in bad:
        print(line)

    # Coverage: walk every standard sidechain heavy atom and confirm it lands
    # in a group, or is one of the handful deliberately left untyped. Without
    # this, a residue class can be silently invisible to the features.
    print("\nper-residue sidechain coverage:")
    stray = []
    for aa in sorted(SIDECHAIN):
        atoms = SIDECHAIN[aa].split()
        hits, none = set(), []
        for n in atoms:
            g = {k for k, f in zip(GROUPS, group_mask(aa, n, _element_of(n))) if f}
            if g:
                hits |= g
            else:
                none.append(n)
                if (aa, n) not in UNTYPED:
                    stray.append(f"{aa} {n}")
        note = f"   untyped: {' '.join(none)}" if none else ""
        print(f"  {aa}  {', '.join(sorted(hits)) or '(none)'}{note}")

    if stray:
        print(f"\n[FAIL] sidechain atoms in no group and not in UNTYPED: {stray}")
        bad.append("coverage")
    missing = sorted(aa for aa in SIDECHAIN if SIDECHAIN[aa] and not any(
        any(group_mask(aa, n, _element_of(n))) for n in SIDECHAIN[aa].split()))
    if missing:
        print(f"[FAIL] residues with no typed sidechain atom at all: {missing}")
        bad.append("residue coverage")
    if bad:
        raise SystemExit("typing table is wrong")
    print(f"\nall sidechain atoms typed except the {len(UNTYPED)} in UNTYPED, by design")
    return True


def selftest():
    import torch

    check_tables()

    # the element order must match the loader that produces atom_type
    from extract import ELE2NUM
    assert tuple(sorted(ELE2NUM, key=ELE2NUM.get)) == ELEMENT_ORDER, \
        f"ELEMENT_ORDER {ELEMENT_ORDER} disagrees with extract.ELE2NUM {ELE2NUM}"
    print("\nELEMENT_ORDER matches extract.ELE2NUM")

    rng = np.random.default_rng(0)

    def fake(n_atoms, centre, seed):
        r = np.random.default_rng(seed)
        A = torch.from_numpy((r.normal(scale=6.0, size=(n_atoms, 3)) + centre).astype(np.float32))
        G = torch.from_numpy(r.integers(0, 2, (n_atoms, N_GROUPS)).astype(np.uint8))
        return A, G

    A1, G1 = fake(400, np.zeros(3), 1)
    A2, G2 = fake(300, np.zeros(3), 2)        # DELIBERATELY the same centre
    P1 = torch.from_numpy(rng.normal(scale=8.0, size=(150, 3)).astype(np.float32))
    P2 = torch.from_numpy(rng.normal(scale=8.0, size=(120, 3)).astype(np.float32))

    def run(pts_list, atoms_list, groups_list):
        xyz = torch.cat(pts_list)
        batch = torch.cat([torch.full((len(p),), i, dtype=torch.long)
                           for i, p in enumerate(pts_list)])
        A = torch.cat(atoms_list)
        G = torch.cat(groups_list)
        ba = torch.cat([torch.full((len(a),), i, dtype=torch.long)
                        for i, a in enumerate(atoms_list)])
        return surface_features(xyz, batch, A, G, ba)

    f_alone1 = run([P1], [A1], [G1])
    f_alone2 = run([P2], [A2], [G2])
    f_batch = run([P1, P2], [A1, A2], [G1, G2])
    assert f_batch.shape == (len(P1) + len(P2), DIMS), f_batch.shape
    assert len(feature_names()) == DIMS, (len(feature_names()), DIMS)
    # no channel may be structurally dead: every column must be written
    # somewhere across these proteins
    assert (f_batch.abs().sum(0) > 0).all(), \
        f"all-zero channels: {[feature_names()[i] for i in (f_batch.abs().sum(0) == 0).nonzero().flatten().tolist()]}"
    assert torch.isfinite(f_batch).all(), "non-finite features"
    d1 = float((f_batch[:len(P1)] - f_alone1).abs().max())
    d2 = float((f_batch[len(P1):] - f_alone2).abs().max())
    print(f"batch isolation (overlapping proteins): max diff {max(d1, d2):.2e}")
    assert max(d1, d2) < 1e-5, "one protein's atoms leaked into another's features"

    # rotation + translation invariance
    q = rng.normal(size=(3, 3))
    Q, _ = np.linalg.qr(q)
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    R = torch.from_numpy(Q.astype(np.float32))
    t = torch.from_numpy(rng.normal(size=3).astype(np.float32))
    f_moved = run([P1 @ R.T + t], [A1 @ R.T + t], [G1])
    delta = float((f_moved - f_alone1).abs().max())
    print(f"rotation + translation invariance: max diff {delta:.2e}")
    assert delta < 1e-4, "features are not rigid-motion invariant"

    # buriedness must order: a point among many atoms vs one far outside
    dense = torch.zeros(1, 3)
    far = torch.full((1, 3), 40.0)
    f = run([torch.cat([dense, far])], [A1], [G1])
    print(f"buriedness (sigma 9) buried {float(f[0, N_GROUPS + 2]):.3f} "
          f"vs isolated {float(f[1, N_GROUPS + 2]):.3f}")
    assert f[0, N_GROUPS + 2] > f[1, N_GROUPS + 2], "buriedness does not order"

    # composition: only-donor atoms -> donor fraction 1, others 0
    Gd = torch.zeros(len(A1), N_GROUPS, dtype=torch.uint8)
    Gd[:, 0] = 1
    f = run([dense], [A1], [Gd])
    assert abs(float(f[0, 0]) - 1.0) < 1e-4, float(f[0, 0])
    assert float(f[0, 1:N_GROUPS].abs().max()) < 1e-6
    print("composition on an all-donor cloud: donor 1.000, others 0")

    # proximity must fall with distance
    one = torch.zeros(1, 3)
    Ad = torch.tensor([[2.0, 0.0, 0.0]])
    Af = torch.tensor([[9.0, 0.0, 0.0]])
    g1 = torch.zeros(1, N_GROUPS, dtype=torch.uint8)
    g1[0, 0] = 1
    prox0 = N_GROUPS + len(SIGMAS)
    near = float(run([one], [Ad], [g1])[0, prox0])
    farp = float(run([one], [Af], [g1])[0, prox0])
    print(f"donor proximity at 2 A {near:.3f} vs 9 A {farp:.3f}")
    assert near > farp > 0

    print(f"\nchannels ({DIMS}): {', '.join(feature_names())}")
    print("selftest passed")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tables", action="store_true",
                    help="check the typing tables only; no torch required")
    a = ap.parse_args()
    if a.tables:
        check_tables()
    else:
        selftest()
