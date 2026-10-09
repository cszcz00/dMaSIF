"""Pocket labels: which cached surface points sit against a system's ligand.

The surface in feats/<tag>.npz was built from structures/<receptor_key>.cif,
which holds the receptor ONLY ... load_protein_atoms drops every HETATM. The
ligand lives alongside it under systems/<system_id>/, in the same frame. This
module pairs the two and records which surface points fall within --cutoff of a
ligand heavy atom.

LIGAND SELECTION IS NOT A SEARCH. A PLINDER system_id names its ligand chains in
field 4 ("1a00__1__1.A__1.E" -> 1.E), and build_manifest.py keeps only systems
with exactly one proper ligand chain, so the ligand is given, not inferred. The
order is: the SDF in ligand_files/ named for that chain; failing that, the cif's
HETATM records on that chain; failing that, HETATM residues matching the
manifest's CCD code; failing that, RAISE. There is deliberately no
largest-HETATM fallback - picking some other ligand when the named one is
missing yields a confident label on the wrong pocket, which is worse than a
crash because it trains.

Output, one per system, in --out:

    <system_id>.npz
        idx        (P,) int32    surface point indices within cutoff
        lig_xyz    (L, 3)        ligand heavy atoms, so idx can be RECOMPUTED
        n_points   ()            the surface's point count when labelled
        cutoff     ()

lig_xyz and n_points exist because idx alone is fragile. extract.py seeds once
before its loop, so a protein's random surface offsets depend on which proteins
shared its batch and on where it sat in the concatenated atom array - and the
resume check changes batching between runs. Same --seed, different point cloud.
Saved indices then address unrelated locations, silently. n_points lets a
consumer detect that loudly; lig_xyz lets it recover without re-walking the
structures.

Also written on a full pass:
    pocket_qc.parquet       one row per system - the real product, see below
    receptor_union.npz      receptor_key -> union of pocket indices over ALL its
                            systems. A receptor with two bound ligands appears
                            as two systems sharing one receptor_key, so training
                            on one puts the other's genuine pocket in the
                            negatives unless something records it.

QC decides usability; nothing is dropped on a threshold here. Parsing tens of
thousands of structures is the expensive pass, so thresholds belong downstream
where they can change without redoing it.

    min_dist        ligand to nearest surface point. A few A is normal. Tens of
                    A means the structures are NOT in the same frame and every
                    label for that system is garbage. This is the gate.
    lig_coverage    fraction of ligand heavy atoms with a surface point within
                    cutoff. Catches shallow or surface-skimming ligands that a
                    healthy n_pocket would hide.
    n_pocket        positives. Near zero on holo means the ligand missed; on apo
                    (later) it means the cleft closed, which is signal.
    pocket_extent   max pairwise distance among pocket points. Its upper tail is
                    what decides whether max-over-patches leaves signal on the
                    table for extended sites.

Example:
    python labels.py --manifest meta/manifest.parquet --feats feats/plinder \\
        --systems systems/ --receptors meta/receptors.txt --out labels/ --check
    python labels.py --manifest meta/manifest.parquet --feats feats/plinder \\
        --systems systems/ --receptors meta/receptors.txt --out labels/
"""

import argparse
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from ids import (ligand_chain, ligand_chains, ligand_sdf_paths,
                 read_receptors, system_cif)

WATERS = {"HOH", "DOD", "WAT"}


# ----------------------------------------------------------------------------
# ligand coordinates
# ----------------------------------------------------------------------------
def ligand_from_sdf(paths):
    """Heavy-atom coordinates from one or more SDF files.

    sanitize=False and removeHs=False, with hydrogens filtered by atomic number
    afterwards: sanitization can reject a perfectly good set of coordinates over
    valence bookkeeping, and nothing here needs a chemically valid molecule -
    only positions and elements.
    """
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    xyz, names = [], []
    for path in paths:
        supplier = Chem.SDMolSupplier(str(path), removeHs=False, sanitize=False)
        for mol in supplier:
            if mol is None or mol.GetNumConformers() == 0:
                continue
            pos = mol.GetConformer().GetPositions()
            heavy = [i for i, at in enumerate(mol.GetAtoms()) if at.GetAtomicNum() > 1]
            if heavy:
                xyz.append(pos[heavy])
                names.append(mol.GetProp("_Name").strip()
                             if mol.HasProp("_Name") else Path(path).stem)
    if not xyz:
        raise ValueError(f"no usable conformer in {[str(p) for p in paths]}")
    return "/".join(dict.fromkeys(names)), np.concatenate(xyz).astype(np.float32)


def ligand_from_cif(cif_path, chains, ccd_code=None):
    """Heavy-atom coordinates of the NAMED ligand, plus the receptor's atoms.

    Chain match first, CCD code second, then raise. The CCD fallback is safe -
    it keeps the right chemistry when the parser reports chain IDs differently
    from the system_id - whereas a largest-HETATM fallback would not be.
    """
    from Bio.PDB import MMCIFParser, PDBParser

    cif_path = Path(cif_path)
    parser = (MMCIFParser if cif_path.suffix.lower() in (".cif", ".mmcif")
              else PDBParser)(QUIET=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = next(iter(parser.get_structure(cif_path.stem, str(cif_path))))

    hits, protein = defaultdict(list), []
    for chain in model:
        for res in chain:
            if res.id[0] == " ":
                protein.extend(a.get_coord() for a in res
                               if (a.element or "").upper() != "H")
                continue
            name = res.get_resname().strip()
            if name in WATERS:
                continue
            xyz = [a.get_coord() for a in res if (a.element or "").upper() != "H"]
            if xyz:
                hits[(name, chain.id)].append(np.asarray(xyz, np.float32))

    sel = {k: v for k, v in hits.items() if chains and k[1] in chains}
    if not sel and ccd_code:
        sel = {k: v for k, v in hits.items() if k[0] == str(ccd_code).strip()}
    if not sel:
        raise ValueError(
            f"ligand chain(s) {sorted(chains) if chains else '?'} "
            f"(ccd={ccd_code}) absent from {cif_path.name}; "
            f"HETATM present: {sorted({f'{n}@{c}' for n, c in hits})[:8]}"
        )

    lig = np.concatenate([c for copies in sel.values() for c in copies])
    name = "/".join(dict.fromkeys(n for n, _ in sel))
    prot = np.asarray(protein, np.float32) if protein else np.zeros((0, 3), np.float32)
    return name, lig.astype(np.float32), prot


# ----------------------------------------------------------------------------
# labelling
# ----------------------------------------------------------------------------
def pocket_indices(xyz, lig, cutoff, chunk=20000):
    """-> (indices within cutoff, min point-ligand distance, per-ligand-atom min).

    Chunked over surface points: the dense (N, L) matrix is the one thing here
    that can blow up on a large assembly surface.
    """
    keep, dmin, per_atom = [], [], None
    for i in range(0, len(xyz), chunk):
        d = np.linalg.norm(xyz[i:i + chunk, None, :] - lig[None, :, :], axis=2)
        dmin.append(d.min(1))
        a = d.min(0)
        per_atom = a if per_atom is None else np.minimum(per_atom, a)
        keep.append(np.nonzero(d.min(1) < cutoff)[0] + i)
    return (np.concatenate(keep).astype(np.int32),
            float(np.concatenate(dmin).min()), per_atom)


def extent(points, cap=4000, seed=0):
    """Max pairwise distance, subsampled above `cap` points (it is O(n^2))."""
    if len(points) < 2:
        return 0.0
    if len(points) > cap:
        points = points[np.random.default_rng(seed).choice(len(points), cap, False)]
    return float(np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2).max())


def label_one(npz_path, systems_root, system_id, chains, ccd_code, cutoff,
              chain_source="ligand_id"):
    d = np.load(npz_path, allow_pickle=True)
    xyz = d["xyz"]

    sdfs = ligand_sdf_paths(systems_root, system_id, chains)
    prot = None
    if sdfs:
        lig_source = "sdf"
        name, lig = ligand_from_sdf(sdfs)
    else:
        cif = system_cif(systems_root, system_id)
        if cif is None:
            raise FileNotFoundError(f"no system.cif or ligand_files for {system_id}")
        lig_source = "cif"
        name, lig, prot = ligand_from_cif(cif, chains, ccd_code)

    idx, min_dist, per_atom = pocket_indices(xyz, lig, cutoff)

    # Frame sanity, independent of min_dist: the receptor atoms we embedded
    # should coincide with the receptor atoms in the system file. Only available
    # on the cif path, since an SDF holds no protein - there min_dist is the
    # whole check.
    offset = float("nan")
    if prot is not None and len(prot):
        offset = float(np.linalg.norm(d["atom_xyz"].mean(0) - prot.mean(0)))

    return idx, lig, {
        "n_points": int(len(xyz)),
        "n_pocket": int(len(idx)),
        "frac_pocket": float(len(idx) / max(len(xyz), 1)),
        "min_dist": min_dist,
        "centroid_offset": offset,
        "lig_source": lig_source,
        "lig_chain": ",".join(sorted(chains)) if chains else "",
        "chain_source": chain_source,
        "lig_ccd": str(ccd_code) if ccd_code is not None else "",
        # What the file itself called it: the SDF's _Name (which repeats the
        # chain) or the cif's residue name. Disagreeing with lig_ccd means the
        # chain resolved to a different ligand than the manifest row describes.
        "lig_name": name,
        "n_lig_atoms": int(len(lig)),
        "lig_coverage": float((per_atom < cutoff).mean()),
        "lig_extent": extent(lig),
        "pocket_extent": extent(xyz[idx]) if len(idx) else 0.0,
        "pocket_centroid": (xyz[idx].mean(0).tolist() if len(idx) else None),
        "variant": "holo",
    }


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--feats", required=True, help="directory of extract.py .npz files")
    ap.add_argument("--systems", required=True,
                    help="root holding <system_id>/system.cif and <system_id>/ligand_files/")
    ap.add_argument("--receptors", required=True, help="meta/receptors.txt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cutoff", type=float, default=4.0)
    ap.add_argument("--check", action="store_true",
                    help="label --n_check systems, print QC, write nothing")
    ap.add_argument("--n_check", type=int, default=50)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()

    df = pd.read_parquet(a.manifest)
    tags = read_receptors(a.receptors)
    feats, out = Path(a.feats), Path(a.out)

    rows = df[["system_id", "receptor_key", "split"]].copy()
    rows["ccd"] = df["ligand_ccd_code"] if "ligand_ccd_code" in df.columns else None
    rows["lid"] = df["ligand_id"] if "ligand_id" in df.columns else None
    if rows["lid"].isna().all():
        print("[warn] manifest has no ligand_id; falling back to system_id field 4, "
              "which names EVERY ligand chain in the system. On PLINDER that is "
              "wrong for about a third of systems - rebuild the manifest with "
              "ligand_id instead.")
    else:
        n_multi = int(df["system_id"].str.split("__").str[3].str.contains("_").sum())
        print(f"{n_multi}/{len(df)} systems name >1 ligand chain; ligand_id "
              "selects this row's chain in each")
    if a.check:
        rows = rows.sample(min(a.n_check, len(rows)), random_state=0)
    else:
        out.mkdir(parents=True, exist_ok=True)

    qc, union = [], defaultdict(list)
    n_missing = 0
    for i, r in enumerate(rows.itertuples(), 1):
        dest = out / f"{r.system_id}.npz"
        if not a.check and dest.exists() and not a.overwrite:
            continue
        tag = tags.get(r.receptor_key)
        npz = feats / f"{tag}.npz" if tag else None
        if npz is None or not npz.exists():
            n_missing += 1
            continue
        chains, chain_source = ligand_chain(r.lid), "ligand_id"
        if not chains:
            chains, chain_source = ligand_chains(r.system_id), "system_id"
        try:
            idx, lig, rec = label_one(npz, a.systems, r.system_id, chains,
                                      r.ccd, a.cutoff, chain_source)
        except Exception as e:
            qc.append({"system_id": r.system_id, "receptor_key": r.receptor_key,
                       "split": r.split, "error": f"{type(e).__name__}: {e}"[:200]})
            continue
        rec.update(system_id=r.system_id, receptor_key=r.receptor_key,
                   split=r.split, error=None)
        qc.append(rec)
        union[r.receptor_key].append(idx)
        if not a.check:
            np.savez_compressed(dest, idx=idx, lig_xyz=lig,
                                n_points=rec["n_points"], cutoff=a.cutoff)
        if i % 500 == 0:
            print(f"  {i}/{len(rows)}", flush=True)

    q = pd.DataFrame(qc)
    if not len(q):
        raise SystemExit(f"nothing labelled; {n_missing} surfaces missing from {feats}")
    ok = q[q["error"].isna()]
    print(f"\n{len(q)} attempted, {len(q) - len(ok)} errored, "
          f"{n_missing} surfaces missing from {feats}")

    if len(q) > len(ok):
        print("\nmost common errors:")
        for msg, n in q.loc[q["error"].notna(), "error"].str.slice(0, 60) \
                       .value_counts().head(5).items():
            print(f"  {n:6d}  {msg}")

    if len(ok):
        print("\nQC percentiles:")
        for c in ["n_pocket", "frac_pocket", "min_dist", "centroid_offset",
                  "lig_coverage", "pocket_extent", "lig_extent"]:
            v = ok[c].dropna() if c in ok.columns else []
            if len(v):
                print(f"  {c:17s} " + "  ".join(
                    f"p{p}={np.percentile(v, p):8.2f}" for p in (1, 25, 50, 75, 99)))
        print("\nligand source: " + str(ok["lig_source"].value_counts().to_dict())
              + "   chain from: " + str(ok["chain_source"].value_counts().to_dict()))
        print(f"\n  {int((ok['min_dist'] > 5.0).sum())} systems with min_dist > 5 A "
              "(frame mismatch -> labels unusable, do NOT train on these)")
        print(f"  {int((ok['lig_coverage'] < 0.5).sum())} with ligand coverage < 0.5 "
              "(shallow or surface-skimming)")
        print(f"  {int((ok['n_pocket'] < 10).sum())} with < 10 pocket points")
        print(f"  {int((ok['pocket_extent'] > 15).sum())} with pocket extent > 15 A "
              "(extended sites; drives the max-vs-LSE choice)")
        multi = {k: v for k, v in union.items() if len(v) > 1}
        print(f"  {len(multi)} receptors carry more than one labelled system "
              "(their other sites would otherwise train as negatives)")

    if not a.check:
        q.to_parquet(out / "pocket_qc.parquet", index=False)
        np.savez_compressed(out / "receptor_union.npz",
                            **{k: np.unique(np.concatenate(v)).astype(np.int32)
                               for k, v in union.items()})
        print(f"\n{out}/pocket_qc.parquet and receptor_union.npz written")


if __name__ == "__main__":
    main()