"""Do frozen dMaSIF embeddings distinguish a ligand pocket from the rest of the surface?

Runs on one or more extracted .npz files against the ligand in a structure file.
The point of comparing an asymmetric-unit run to a biological-assembly run is
that many pockets are completed by a neighbouring subunit: 1STP's biotin site
takes a conserved Trp120 from the adjacent monomer, so the ASU surface has a
hole where that wall should be.

Cohen's d per feature dimension: |mean_pocket - mean_rest| / pooled SD.
Roughly, 0.2 is negligible, 0.5 moderate, 0.8+ strong. With ~4% positives a
large d does NOT imply usable precision - that needs the held-out probe over
many proteins, not this script.

Example:
    python probe_pocket.py --ligand test_pdbs/1STP.pdb --ligand_resname BTN \\
        --feats feats/test/1STP.npz feats/test/1STP_assembly.npz \\
        --labels ASU assembly
"""

import argparse
from pathlib import Path

import numpy as np


def ligand_coords(path, resname=None, exclude=("HOH", "DOD")):
    """Heavy atoms of the ligand. Without --ligand_resname, takes the largest
    non-water HETATM residue, which is usually the ligand of interest."""
    from Bio.PDB import MMCIFParser, PDBParser

    path = Path(path)
    parser = (MMCIFParser if path.suffix.lower() in (".cif", ".mmcif") else PDBParser)(QUIET=True)
    model = next(iter(parser.get_structure(path.stem, str(path))))

    candidates = {}
    for chain in model:
        for res in chain:
            if res.id[0] == " ":
                continue
            name = res.get_resname().strip()
            if name in exclude:
                continue
            if resname and name != resname:
                continue
            xyz = [a.get_coord() for a in res if (a.element or "").upper() != "H"]
            if xyz:
                candidates.setdefault(name, []).append(np.asarray(xyz, dtype=np.float32))

    if not candidates:
        raise ValueError(f"No ligand found in {path} (resname={resname})")
    if resname is None and len(candidates) > 1:
        best = max(candidates, key=lambda k: sum(len(c) for c in candidates[k]))
        print(f"[note] HETATM residues {sorted(candidates)}; using {best}")
        candidates = {best: candidates[best]}
    name, copies = next(iter(candidates.items()))
    return name, np.concatenate(copies), len(copies)


def replicate_over_models(path, lig, resname):
    """Map the ligand onto every symmetry copy in an assembly file.

    An assembly holds N copies of the protein but usually one copy of the
    ligand, so the other N-1 binding sites end up labelled as NEGATIVES -
    genuine pockets counted as background, which caps AUC and inflates the
    variance Cohen's d divides by. Superposing model 1 onto model i gives the
    transform that places the ligand in site i.
    """
    from Bio.PDB import PDBParser, MMCIFParser, Superimposer

    path = Path(path)
    parser = (MMCIFParser if path.suffix.lower() in (".cif", ".mmcif") else PDBParser)(QUIET=True)
    models = list(parser.get_structure(path.stem, str(path)))
    if len(models) < 2:
        return lig, 1

    def backbone(model):
        return [a for ch in model for r in ch
                if r.id[0] == " " for a in r if a.get_id() == "CA"]

    ref = backbone(models[0])
    copies = [lig]
    sup = Superimposer()
    for m in models[1:]:
        mob = backbone(m)
        n = min(len(ref), len(mob))
        if n < 3:
            continue
        # fixed=this model, moving=model 1: the transform maps model 1 -> model i
        sup.set_atoms(mob[:n], ref[:n])
        rot, tran = sup.rotran
        rmsd_note = sup.rms
        if rmsd_note > 2.0:
            print(f"  [warn] model superposition RMSD {rmsd_note:.2f} A; copies may be off")
        copies.append(lig @ rot + tran)
    return np.concatenate(copies).astype(np.float32), len(copies)


def cohens_d(X, y):
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    sd = np.sqrt(0.5 * (X[y].var(0) + X[~y].var(0))) + 1e-8
    return np.abs(X[y].mean(0) - X[~y].mean(0)) / sd


def auc(scores, y):
    """Rank-based AUC, direction-agnostic.

    An embedding dimension may separate pocket from non-pocket with either
    sign, so a raw AUC of 0.01 means the same separation quality as 0.99.
    Returns max(a, 1 - a) and the sign, since only |separation| is meaningful
    for an unsupervised feature.
    """
    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan"), 0
    a = (ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    return (a, 1) if a >= 0.5 else (1.0 - a, -1)


def report(npz_path, lig, cutoff, keys):
    d = np.load(npz_path, allow_pickle=True)
    xyz = d["xyz"]
    dist = np.linalg.norm(xyz[:, None, :] - lig[None, :, :], axis=2).min(axis=1)
    y = dist < cutoff

    n_atoms = len(d["atom_type"])
    chains = sorted(set(d["atom_chain"].tolist()))
    print(f"  {n_atoms} atoms, {len(xyz)} surface points, chains {chains}")
    print(f"  {y.sum()} points within {cutoff} A of ligand ({100 * y.mean():.1f}%)")
    if y.sum() < 10:
        print("  [warn] too few positives to interpret; is the ligand in the same frame?")

    rows = {}
    for k in keys:
        if k not in d.files:
            continue
        X = d[k]
        dv = cohens_d(X, y)
        best = int(np.argmax(dv))
        a, sign = auc(X[:, best], y)
        rows[k] = (dv.max(), dv.mean(), a)
        print(f"  {k:12s} max|d|={dv.max():5.2f}  mean|d|={dv.mean():5.2f}  "
              f"AUC(dim {best}{'+' if sign > 0 else '-'})={a:.3f}  "
              f"top dims={np.argsort(-dv)[:3]}")
    return rows, y.mean()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ligand", required=True, help="structure file containing the ligand")
    ap.add_argument("--ligand_resname", help="e.g. BTN; default = largest non-water HETATM")
    ap.add_argument("--feats", nargs="+", required=True, help=".npz files to compare")
    ap.add_argument("--labels", nargs="*", help="names for each --feats entry")
    ap.add_argument("--cutoff", type=float, default=4.0)
    ap.add_argument(
        "--replicate_over",
        help="assembly file: copy the ligand onto every symmetry-related site, "
             "so the other copies are not counted as background",
    )
    ap.add_argument("--keys", nargs="*", default=["input_feats", "emb1", "emb2"])
    a = ap.parse_args()

    name, lig, n_copies = ligand_coords(a.ligand, a.ligand_resname)
    print(f"ligand {name}: {len(lig)} heavy atoms in {n_copies} cop{'y' if n_copies == 1 else 'ies'}")
    if a.replicate_over:
        lig, n_sites = replicate_over_models(a.replicate_over, lig, a.ligand_resname)
        print(f"replicated onto {n_sites} symmetry-related sites -> {len(lig)} atoms")
    print()

    labels = a.labels or [Path(f).stem for f in a.feats]
    results = {}
    for path, label in zip(a.feats, labels):
        print(f"[{label}] {path}")
        results[label], _ = report(path, lig, a.cutoff, a.keys)
        print()

    if len(results) > 1:
        print("max|d| by feature set:")
        print(f"  {'':12s} " + "".join(f"{l:>12s}" for l in labels))
        for k in a.keys:
            if all(k in r for r in results.values()):
                print(f"  {k:12s} " + "".join(f"{results[l][k][0]:12.2f}" for l in labels))
        print("\nCompare AUC rather than |d|: AUC is invariant to the class balance,")
        print("which changes between an ASU and an assembly. Unlabelled symmetry-")
        print("related sites depress both, so use --replicate_over on assemblies.")


if __name__ == "__main__":
    main()