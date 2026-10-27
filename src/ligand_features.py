"""Ligand featurisation: SMILES -> fingerprints, descriptors and a 2D graph.

Runs in the CPU venv (rdkit), NOT in the GPU container, which has no rdkit.
Everything the model needs is packed into one .npz of plain arrays, so the
container only ever reads numbers. ligand_encoder.py imports the SCHEMA
constants from this module; the rdkit imports are inside the functions, so
that import costs nothing and works without rdkit installed.

Three representations, written side by side so arms can be compared on exactly
the same molecules (see probe_retrieval.py):

    fp      ECFP4, 2048 bits, uint8. The same settings as
            build_manifest.ligand_fingerprints, recomputed here so a dataset's
            features do not depend on data/meta/ staying put.
    desc    RDKit physchem descriptors (DESCRIPTORS below), float32. Size,
            polar surface area, flexibility and shape - the properties a
            surface patch's shape and chemistry can plausibly be complementary
            to. Raw and unstandardised: a consumer fits the standardisation on
            its OWN training split (ligand_encoder.DescEncoder.set_scaler).
    graph   heavy-atom graph: per-atom and per-bond one-hot features, packed
            CSR-style with atom_ptr / bond_ptr offsets.

2D, deliberately. The crystal pose is available for a training ligand and
would leak the answer: at screening time a candidate molecule has no pose, so
nothing downstream may depend on one. Hydrogens are dropped, matching the
protein side (encoder tests strip H because PLINDER structures disagree about
them).

A molecule that rdkit cannot parse is kept as a row of zeros with failed=True,
never silently dropped: a consumer that would train on it must say so (the
probe drops them and prints the count). Descriptor values that come back NaN
or infinite are zeroed and counted.

Output, in --out:

    ligand_feats.npz
        smiles       (n,)        the input SMILES, the join key
        fp           (n, 2048)   uint8 0/1
        desc         (n, D)      float32, raw
        desc_names   (D,)        descriptor names, in column order
        atom_feat    (A, 47)     uint8, concatenated over molecules
        atom_ptr     (n + 1,)    molecule i owns atom_feat[ptr[i]:ptr[i+1]]
        bond_feat    (B, 6)      uint8; each bond appears ONCE
        bond_index   (2, B)      atom indices LOCAL to the molecule
        bond_ptr     (n + 1,)
        n_heavy      (n,)        heavy atoms per molecule
        failed       (n,)        bool; True rows are all zero
        schema       json        feature names and widths

Example:
    python src/ligand_features.py --index data/datasets/dataset_v1/index.parquet \\
        --out data/datasets/dataset_v1/ --check
    python src/ligand_features.py --index data/datasets/dataset_v1/index.parquet \\
        --out data/datasets/dataset_v1/

Depends on: rdkit, numpy, pandas (--index only).
Imported by: ligand_encoder.py (schema only), probe_retrieval.py.
"""

import argparse
import json
from pathlib import Path

import numpy as np

FP_BITS = 2048
FP_RADIUS = 2  # ECFP4, as in build_manifest.py

# ---------------------------------------------------------------------------
# Graph schema. Every field is one-hot or binary, so atom_feat / bond_feat fit
# in uint8 and a consumer can slice a group out by name. "other" buckets keep
# the width fixed for anything unusual.
# ---------------------------------------------------------------------------
ELEMENTS = ["C", "N", "O", "S", "F", "Cl", "Br", "I", "P", "B", "Si", "Se"]
DEGREES = [0, 1, 2, 3, 4, 5]          # heavy-atom degree
CHARGES = [-2, -1, 0, 1, 2]
NUM_HS = [0, 1, 2, 3, 4]              # implicit + explicit H on a heavy atom
HYBRIDS = ["SP", "SP2", "SP3", "SP3D", "SP3D2"]
RING_SIZES = [3, 4, 5, 6, 7]

ATOM_GROUPS = [
    ("element", len(ELEMENTS) + 1),
    ("degree", len(DEGREES) + 1),
    ("charge", len(CHARGES) + 1),
    ("num_h", len(NUM_HS) + 1),
    ("hybrid", len(HYBRIDS) + 1),
    ("aromatic", 1),
    ("in_ring", 1),
    ("donor", 1),
    ("acceptor", 1),
    ("ring_size", len(RING_SIZES)),
]
ATOM_DIM = sum(w for _, w in ATOM_GROUPS)  # 47

BOND_TYPES = ["SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"]
BOND_GROUPS = [("bond_type", len(BOND_TYPES)), ("conjugated", 1), ("in_ring", 1)]
BOND_DIM = sum(w for _, w in BOND_GROUPS)  # 6

# Descriptors, by rdkit module and attribute name. Chosen to be cheap, defined
# for every valid molecule, and about SIZE / POLARITY / FLEXIBILITY / SHAPE
# rather than about substructure identity, which ECFP already covers. Nothing
# here needs a conformer. Partial-charge descriptors are deliberately absent:
# they return NaN on common elements.
DESCRIPTORS = [
    ("Descriptors", "MolWt"),
    ("Descriptors", "HeavyAtomCount"),
    ("Descriptors", "MolLogP"),
    ("Descriptors", "MolMR"),
    ("Descriptors", "TPSA"),
    ("Descriptors", "LabuteASA"),
    ("Descriptors", "NumHAcceptors"),
    ("Descriptors", "NumHDonors"),
    ("Descriptors", "NumRotatableBonds"),
    ("Descriptors", "NumHeteroatoms"),
    ("Descriptors", "NHOHCount"),
    ("Descriptors", "NOCount"),
    ("Descriptors", "RingCount"),
    ("Descriptors", "NumAromaticRings"),
    ("Descriptors", "NumAliphaticRings"),
    ("Descriptors", "NumSaturatedRings"),
    ("Descriptors", "FractionCSP3"),
    ("Descriptors", "HallKierAlpha"),
    ("Descriptors", "Kappa1"),
    ("Descriptors", "Kappa2"),
    ("Descriptors", "Kappa3"),
    ("Descriptors", "Chi0v"),
    ("Descriptors", "Chi1v"),
    ("Descriptors", "Chi2v"),
    ("Descriptors", "Chi3v"),
    ("Descriptors", "Chi4v"),
    ("Descriptors", "BalabanJ"),
    ("Descriptors", "BertzCT"),
    ("Descriptors", "NumValenceElectrons"),
    ("rdMolDescriptors", "CalcNumAmideBonds"),
    ("rdMolDescriptors", "CalcNumBridgeheadAtoms"),
    ("rdMolDescriptors", "CalcNumSpiroAtoms"),
    ("rdMolDescriptors", "CalcNumRings"),
]
DESC_DIM = len(DESCRIPTORS)


def schema():
    """The feature layout, stored in the npz so a consumer can check it."""
    return {
        "atom_groups": ATOM_GROUPS, "atom_dim": ATOM_DIM,
        "bond_groups": BOND_GROUPS, "bond_dim": BOND_DIM,
        "elements": ELEMENTS, "bond_types": BOND_TYPES,
        "desc_names": [n for _, n in DESCRIPTORS], "desc_dim": DESC_DIM,
        "fp_bits": FP_BITS, "fp_radius": FP_RADIUS,
    }


# ---------------------------------------------------------------------------
# one molecule
# ---------------------------------------------------------------------------
def _onehot(value, choices):
    """One-hot with a trailing "other" slot, so an unlisted value is still
    represented rather than dropped."""
    out = [0] * (len(choices) + 1)
    out[choices.index(value) if value in choices else len(choices)] = 1
    return out


def atom_features(atom, ring_info, donors, acceptors):
    hyb = str(atom.GetHybridization())
    return (
        _onehot(atom.GetSymbol(), ELEMENTS)
        + _onehot(atom.GetDegree(), DEGREES)
        + _onehot(atom.GetFormalCharge(), CHARGES)
        + _onehot(atom.GetTotalNumHs(), NUM_HS)
        + _onehot(hyb, HYBRIDS)
        + [int(atom.GetIsAromatic()), int(atom.IsInRing()),
           int(atom.GetIdx() in donors), int(atom.GetIdx() in acceptors)]
        + [int(ring_info.IsAtomInRingOfSize(atom.GetIdx(), s)) for s in RING_SIZES]
    )


def bond_features(bond):
    return (_onehot(str(bond.GetBondType()), BOND_TYPES)[:len(BOND_TYPES)]
            + [int(bond.GetIsConjugated()), int(bond.IsInRing())])


def hbond_atoms(mol):
    """Donor and acceptor atom indices, by rdkit's own Lipinski SMARTS.

    Explicit because the element one-hot cannot express them: a carbonyl O and
    a hydroxyl O are both "O", and the difference is exactly what a surface
    patch's hydrogen-bonding chemistry would pair with.
    """
    from rdkit.Chem import Lipinski

    donors = {m[0] for m in mol.GetSubstructMatches(Lipinski.HDonorSmarts)}
    acceptors = {m[0] for m in mol.GetSubstructMatches(Lipinski.HAcceptorSmarts)}
    return donors, acceptors


def featurise_one(smiles, fp_gen, desc_fns):
    """-> dict of arrays for one molecule, or None if rdkit cannot parse it."""
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    # Heavy-atom graph, as on the protein side. RemoveAllHs, not RemoveHs:
    # RemoveHs keeps isotopic H ([2H], deuterated ligands), H2 and hydrides,
    # which then became graph nodes and ECFP atoms (173 of dataset_v1's 8,378
    # ligands, flagged by report()'s heavy != graph atom count check).
    mol = Chem.RemoveAllHs(mol)
    if mol.GetNumAtoms() == 0:  # nothing but hydrogen, e.g. [H][H]
        return None

    ring_info = mol.GetRingInfo()
    donors, acceptors = hbond_atoms(mol)
    atoms = np.asarray([atom_features(a, ring_info, donors, acceptors)
                        for a in mol.GetAtoms()], dtype=np.uint8)

    pairs, bfeat = [], []
    for b in mol.GetBonds():
        pairs.append((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))
        bfeat.append(bond_features(b))
    bond_index = (np.asarray(pairs, dtype=np.int32).T if pairs
                  else np.zeros((2, 0), dtype=np.int32))
    bond_feat = (np.asarray(bfeat, dtype=np.uint8) if bfeat
                 else np.zeros((0, BOND_DIM), dtype=np.uint8))

    fp = np.zeros(FP_BITS, dtype=np.uint8)
    fp[list(fp_gen.GetFingerprint(mol).GetOnBits())] = 1

    desc = np.zeros(DESC_DIM, dtype=np.float64)
    for i, fn in enumerate(desc_fns):
        try:
            desc[i] = fn(mol)
        except Exception:
            desc[i] = np.nan
    return {"atom_feat": atoms, "bond_index": bond_index, "bond_feat": bond_feat,
            "fp": fp, "desc": desc, "n_heavy": mol.GetNumHeavyAtoms()}


# ---------------------------------------------------------------------------
# a list of molecules
# ---------------------------------------------------------------------------
def featurise(smiles_list, verbose=True):
    """-> the dict of packed arrays described in the module docstring."""
    from rdkit import RDLogger
    from rdkit.Chem import Descriptors, rdFingerprintGenerator, rdMolDescriptors

    RDLogger.DisableLog("rdApp.*")
    modules = {"Descriptors": Descriptors, "rdMolDescriptors": rdMolDescriptors}
    desc_fns = [getattr(modules[m], n) for m, n in DESCRIPTORS]
    fp_gen = rdFingerprintGenerator.GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_BITS)

    n = len(smiles_list)
    fp = np.zeros((n, FP_BITS), dtype=np.uint8)
    desc = np.zeros((n, DESC_DIM), dtype=np.float64)
    n_heavy = np.zeros(n, dtype=np.int32)
    failed = np.zeros(n, dtype=bool)
    atom_chunks, bond_chunks, index_chunks = [], [], []
    atom_ptr = np.zeros(n + 1, dtype=np.int64)
    bond_ptr = np.zeros(n + 1, dtype=np.int64)

    for i, smi in enumerate(smiles_list):
        rec = featurise_one(smi, fp_gen, desc_fns)
        if rec is None:
            failed[i] = True
        else:
            fp[i] = rec["fp"]
            desc[i] = rec["desc"]
            n_heavy[i] = rec["n_heavy"]
            atom_chunks.append(rec["atom_feat"])
            bond_chunks.append(rec["bond_feat"])
            index_chunks.append(rec["bond_index"])
        atom_ptr[i + 1] = atom_ptr[i] + (0 if rec is None else len(rec["atom_feat"]))
        bond_ptr[i + 1] = bond_ptr[i] + (0 if rec is None else rec["bond_index"].shape[1])
        if verbose and (i + 1) % 2000 == 0:
            print(f"  {i + 1}/{n}", flush=True)

    bad = ~np.isfinite(desc)
    if bad.any():
        desc[bad] = 0.0
    return {
        "smiles": np.asarray([str(s) for s in smiles_list]),
        "fp": fp,
        "desc": desc.astype(np.float32),
        "desc_names": np.asarray([n for _, n in DESCRIPTORS]),
        "atom_feat": (np.concatenate(atom_chunks) if atom_chunks
                      else np.zeros((0, ATOM_DIM), dtype=np.uint8)),
        "atom_ptr": atom_ptr,
        "bond_feat": (np.concatenate(bond_chunks) if bond_chunks
                      else np.zeros((0, BOND_DIM), dtype=np.uint8)),
        "bond_index": (np.concatenate(index_chunks, axis=1) if index_chunks
                       else np.zeros((2, 0), dtype=np.int32)),
        "bond_ptr": bond_ptr,
        "n_heavy": n_heavy,
        "failed": failed,
        "n_nonfinite_desc": int(bad.sum()),
        "schema": json.dumps(schema()),
    }


def report(feats):
    n = len(feats["smiles"])
    n_bad = int(feats["failed"].sum())
    ok = ~feats["failed"]
    print(f"\n{n} molecules, {n_bad} unparseable "
          f"({'kept as zero rows, flagged failed' if n_bad else 'none'})")
    if feats["n_nonfinite_desc"]:
        print(f"{feats['n_nonfinite_desc']} non-finite descriptor values zeroed")
    if not ok.any():
        return
    heavy = feats["n_heavy"][ok]
    bits = feats["fp"][ok].sum(1)
    atoms = np.diff(feats["atom_ptr"])[ok]
    bonds = np.diff(feats["bond_ptr"])[ok]
    print(f"\n{'':18s}" + "".join(f"{f'p{p}':>9s}" for p in (1, 25, 50, 75, 99)))
    for name, v in [("heavy atoms", heavy), ("graph atoms", atoms),
                    ("graph bonds", bonds), ("ECFP4 bits on", bits)]:
        print(f"{name:18s}" + "".join(f"{np.percentile(v, p):9.1f}"
                                      for p in (1, 25, 50, 75, 99)))
    if not np.array_equal(heavy, atoms):
        n_diff = int((heavy != atoms).sum())
        print(f"[warn] {n_diff} molecules where heavy-atom count != graph atom count")
    zero = int((bits == 0).sum())
    if zero:
        print(f"[warn] {zero} parseable molecules with an all-zero fingerprint")
    d = feats["desc"][ok]
    const = [str(feats["desc_names"][i]) for i in range(d.shape[1]) if d[:, i].std() == 0]
    if const:
        print(f"[warn] descriptors with zero variance here: {list(const)}")
    print(f"\ndescriptor ranges (raw; standardise on YOUR train split):")
    for i, name in enumerate(feats["desc_names"]):
        print(f"  {str(name):26s} mean {d[:, i].mean():12.3f}  std {d[:, i].std():12.3f}")


# ---------------------------------------------------------------------------
def read_smiles(a):
    """-> the list of SMILES to featurise, one per output row.

    From a dataset index, the ligands are taken in fp_row order so that row i
    of the output is fp_row i, lining up with the dataset's own
    ligand_fps.npy and with index.parquet's fp_row column.
    """
    if a.smiles:
        return [s.strip() for s in Path(a.smiles).read_text().splitlines() if s.strip()]
    import pandas as pd

    df = pd.read_parquet(a.index, columns=["fp_row", "ligand_rdkit_canonical_smiles"])
    uniq = df.drop_duplicates("fp_row").sort_values("fp_row")
    rows = uniq["fp_row"].to_numpy()
    if not np.array_equal(rows, np.arange(len(rows))):
        raise SystemExit(
            f"fp_row is not 0..{len(rows) - 1} in {a.index} (got "
            f"{rows.min()}..{rows.max()}, {len(rows)} distinct); this index was "
            "not written by build_dataset.py")
    print(f"{a.index}: {len(df)} interactions -> {len(uniq)} distinct ligands")
    return uniq["ligand_rdkit_canonical_smiles"].tolist()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--index", help="data/datasets/<name>/index.parquet; ligands in fp_row order")
    src.add_argument("--smiles", help="text file of SMILES, one per line")
    ap.add_argument("--out", required=True, help="directory for ligand_feats.npz")
    ap.add_argument("--name", default="ligand_feats.npz")
    ap.add_argument("--check", action="store_true",
                    help="featurise --n_check molecules, print the report, write nothing")
    ap.add_argument("--n_check", type=int, default=500)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()

    smiles = read_smiles(a)
    if a.check:
        smiles = smiles[:a.n_check]
        print(f"--check: featurising the first {len(smiles)}")
    out = Path(a.out) / a.name
    if out.exists() and not a.overwrite and not a.check:
        raise SystemExit(f"{out} exists; pass --overwrite")

    feats = featurise(smiles)
    report(feats)
    if a.check:
        print("\n--check: nothing written")
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **feats)
    print(f"\nwrote {out}  ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
