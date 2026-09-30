"""Build a training manifest from PLINDER for contrastive protein-ligand retrieval.

Reads a column subset of index/annotation_table.parquet (745 columns, ~1 GB on
disk - never load it whole) and merges the split labels from splits/split.parquet.

Three modes:

    --inspect   print the schema of all three tables and exit
    --counts    per-filter attrition and the headline numbers; writes nothing
    (default)   write manifest.parquet, receptors.txt, ligand_fps.npy

The cluster columns (pli_unique_qcov__95__strong__component and friends) are
denormalized into the annotation table, so dedup, leakage-aware grouping and
batch negative masking need neither clusters/ nor scores/. Only pairwise
similarity VALUES need scores/, and this script no longer requires it.

Example:
    python build_manifest.py --plinder $PLINDER_DIR --out meta/ --inspect
    python build_manifest.py --plinder $PLINDER_DIR --out meta/ --counts
    python build_manifest.py --plinder $PLINDER_DIR --out meta/
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

FP_BITS = 2048
FP_RADIUS = 2  # ECFP4

# Missing columns are reported, not fatal: these names come from a 745-column
# table and some may differ from what the docs imply.
WANTED = [
    "system_id",
    # composition
    "system_proper_num_ligand_chains",
    "system_proper_ligand_max_molecular_weight",
    "system_proper_num_interactions",
    "system_proper_num_pocket_residues",
    # ligand class - these are ROW level (one row per system-ligand), unlike the
    # system_proper_* columns which describe the whole system
    "ligand_is_proper",
    "ligand_is_ion",
    "ligand_is_cofactor",
    "ligand_is_artifact",
    "ligand_is_covalent",
    "ligand_is_oligo",
    "ligand_is_invalid",
    "ligand_is_rdkit_loadable",
    "ligand_molecular_weight",
    "ligand_num_heavy_atoms",
    "ligand_num_interactions",
    "ligand_num_unique_interactions",
    "ligand_positions_correct",
    "ligand_rdkit_canonical_smiles",
    "ligand_ccd_code",
    "ligand_id",
    # the pocket must be in the coordinates, and not made of symmetry mates
    "ligand_fraction_atoms_with_crystal_contacts",
    "ligand_num_missing_pli_interface_residues",
    "all_protein_chains_present",
    # how well resolved the pocket is - the surface is built from these atoms
    "entry_resolution",
    "system_pocket_validation_num_unresolved_heavy_atoms",
    "system_pocket_validation_max_alt_count",
    "system_pocket_validation_average_occupancy",
    # grouping
    "pli_unique_qcov__95__strong__component",
    "pli_unique_qcov__50__community",
    "pocket_qcov__50__community",
    "system_pocket_Pfam",
    "system_pocket_UniProt",
]

# (label, predicate). Applied one at a time so attrition is visible.
def _flag(col, want=True):
    """Row-level boolean test that treats a null as failing."""
    return lambda d: d[col].fillna(not want).astype(bool) == want


# (label, predicate), applied one at a time so attrition is visible.
#
# ROW-level filters come first and are the ones that matter for correctness:
# annotation_table has one row per (system, ligand), so a system_* column says
# nothing about WHICH ligand this row is. Filtering only on system_* columns
# lets a magnesium ion ride along in a system whose proper ligand is drug-like,
# because system_proper_ligand_max_molecular_weight is the max over proper
# ligands and the MG row inherits it.
FILTERS = [
    # -- row level: is THIS ligand a training example? --
    ("ligand is proper", _flag("ligand_is_proper")),
    ("not an ion", _flag("ligand_is_ion", False)),
    ("not a cofactor", _flag("ligand_is_cofactor", False)),
    ("not an artifact", _flag("ligand_is_artifact", False)),
    ("not an oligomer", _flag("ligand_is_oligo", False)),
    ("not covalently bound", _flag("ligand_is_covalent", False)),
    ("valid / rdkit loadable", _flag("ligand_is_rdkit_loadable")),
    ("ligand MW 200-800", lambda d: d["ligand_molecular_weight"].between(200, 800)),
    ("ligand positions correct", _flag("ligand_positions_correct")),
    ("3-50 interactions", lambda d: d["ligand_num_interactions"].between(3, 50)),
    # -- system level: is the pocket well defined? --
    ("system has 1 proper ligand", lambda d: d["system_proper_num_ligand_chains"] == 1),
    ("pocket 5-100 residues", lambda d: d["system_proper_num_pocket_residues"].between(5, 100)),
    ("crystal contacts < 25%",
     lambda d: d["ligand_fraction_atoms_with_crystal_contacts"].fillna(0) < 0.25),
    ("no missing pocket residues",
     lambda d: d["ligand_num_missing_pli_interface_residues"].fillna(0) == 0),
    ("all protein chains present", _flag("all_protein_chains_present")),
    ("resolution <= 2.5 A", lambda d: d["entry_resolution"] <= 2.5),
    ("pocket fully resolved",
     lambda d: d["system_pocket_validation_num_unresolved_heavy_atoms"].fillna(0) == 0),
    ("no alt conformations",
     lambda d: d["system_pocket_validation_max_alt_count"].fillna(1) <= 1),
]

DEDUP_CLUSTER = "pli_unique_qcov__95__strong__component"


def receptor_key(system_id):
    """`pdbid__biounit__receptorchains__ligandchains` -> receptor identity.

    Systems differing only in which ligand they describe share a receptor and
    so share one dMaSIF extraction. Chains are dot-labelled within the
    biological assembly (1.A, 2.A), matching extract.py --merge_models.
    """
    return "__".join(system_id.split("__")[:3])


def receptor_chains(system_id):
    """The receptor chain labels, comma separated for extract.py."""
    parts = system_id.split("__")
    return ",".join(parts[2].split("_")) if len(parts) > 2 else ""


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------
def paths(plinder):
    p = Path(plinder)
    return (
        p / "index" / "annotation_table.parquet",
        p / "splits" / "split.parquet",
        p / "fingerprints" / "ligands_per_system.parquet",
    )


def inspect(plinder):
    import pyarrow.parquet as pq

    ann_p, split_p, lig_p = paths(plinder)

    f = pq.ParquetFile(ann_p)
    have = set(f.schema.names)
    print(f"=== annotation_table.parquet: {f.metadata.num_rows} rows, "
          f"{f.metadata.num_columns} columns ===")
    print("\nwanted columns:")
    for c in WANTED:
        print(f"  {'OK     ' if c in have else 'MISSING'}  {c}")
    missing = [c for c in WANTED if c not in have]
    if missing:
        print(f"\n{len(missing)} missing; near-matches by suffix:")
        for c in missing:
            stem = c.split("_")[-1]
            print(f"  {c}\n    -> {[n for n in f.schema.names if stem in n][:5]}")

    for name, path in [("split.parquet", split_p), ("ligands_per_system.parquet", lig_p)]:
        if not path.exists():
            print(f"\n=== {name}: NOT DOWNLOADED ===")
            continue
        df = pd.read_parquet(path)
        print(f"\n=== {name}: {len(df)} rows ===")
        for c in df.columns:
            print(f"  {c:55s} {df[c].dtype}")
        if "split" in df.columns:
            print("\nsplit counts:\n", df["split"].value_counts())


def load(plinder):
    """Annotation subset + split labels. Never reads all 745 columns."""
    import pyarrow.parquet as pq

    ann_p, split_p, lig_p = paths(plinder)
    have = set(pq.ParquetFile(ann_p).schema.names)
    cols = [c for c in WANTED if c in have]
    missing = [c for c in WANTED if c not in have]
    if missing:
        print(f"[warn] absent, filters using them will be skipped: {missing}")
    ann = pd.read_parquet(ann_p, columns=cols)
    print(f"annotation_table: {len(ann)} rows, {len(cols)} columns read")

    split = pd.read_parquet(split_p, columns=["system_id", "split", "uniqueness"])
    ann = ann.merge(split, on="system_id", how="inner")
    print(f"{len(ann)} rows after merging split labels")

    if "ligand_rdkit_canonical_smiles" not in ann.columns and lig_p.exists():
        lig = pd.read_parquet(lig_p)
        keep = [c for c in ["system_id", "ligand_rdkit_canonical_smiles", "inchikeys",
                            "ligand_ccd_code"] if c in lig.columns]
        ann = ann.merge(lig[keep].drop_duplicates("system_id"), on="system_id", how="left")
        print("joined ligand SMILES from fingerprints/")
    return ann


# ----------------------------------------------------------------------------
# Filtering
# ----------------------------------------------------------------------------
def apply_filters(df, splits=("train", "val", "test")):
    """Apply filters one at a time, reporting what each removes.

    Several overlap heavily - a low-resolution entry often also has unresolved
    pocket atoms - so the individual drops will not sum to the total. Read the
    running remainder, not the per-filter numbers.
    """
    df = df[df["split"].isin(splits)].copy()
    print(f"\n{'filter':35s} {'drops':>9s} {'remaining':>10s}")
    print(f"{'(start, train/val/test only)':35s} {'':>9s} {len(df):>10d}")

    mask = pd.Series(True, index=df.index)
    for label, fn in FILTERS:
        try:
            m = fn(df).fillna(False)
        except KeyError as e:
            print(f"{label:35s} {'SKIPPED':>9s}  (missing {e})")
            continue
        drops = int((mask & ~m).sum())
        mask &= m
        print(f"{label:35s} {drops:>9d} {int(mask.sum()):>10d}")

    out = df[mask].copy()
    dup = int(out["system_id"].duplicated().sum())
    if dup:
        print(f"\n[warn] {dup} rows share a system_id with another surviving row. "
              "A system should contribute one training pair; check that the "
              "row-level ligand filters are applied.")
        print(out[out["system_id"].duplicated(keep=False)]
              .sort_values("system_id")[["system_id", "ligand_ccd_code"]].head(10).to_string())
    out["receptor_key"] = out["system_id"].map(receptor_key)
    return out


def deduplicate(df, cluster_col=DEDUP_CLUSTER):
    """One representative per (interaction cluster, ligand).

    Near-duplicates are mislabelled negatives in a contrastive batch, which is
    a stronger reason to dedup than in a docking pipeline. The cluster column
    is preferred over `uniqueness` because the threshold is explicit.
    """
    key = []
    if cluster_col in df.columns:
        key.append(cluster_col)
    elif "uniqueness" in df.columns:
        print(f"[warn] {cluster_col} absent; falling back to `uniqueness`")
        key.append("uniqueness")
    for c in ("inchikeys", "ligand_ccd_code"):
        if c in df.columns:
            key.append(c)
            break
    before = len(df)
    out = df.sort_values("system_id").drop_duplicates(subset=key, keep="first")
    print(f"\ndedup on {key}: {before} -> {len(out)}")
    return out


def summarize(df):
    print(f"\n    {'split':8s}{'pairs':>9s}{'receptors':>11s}{'ligands':>9s}{'Pfam':>8s}")
    for sp in ["train", "val", "test"]:
        d = df[df["split"] == sp]
        if not len(d):
            continue
        lig = (d["ligand_rdkit_canonical_smiles"].nunique()
               if "ligand_rdkit_canonical_smiles" in d.columns else -1)
        pf = d["system_pocket_Pfam"].nunique() if "system_pocket_Pfam" in d.columns else -1
        print(f"    {sp:8s}{len(d):>9d}{d['receptor_key'].nunique():>11d}{lig:>9d}{pf:>8d}")
    n_rec = df["receptor_key"].nunique()
    print(f"\n{len(df)} pairs over {n_rec} receptors "
          f"({len(df) / max(n_rec, 1):.2f} pairs per dMaSIF extraction)")
    print(f"feature storage at ~0.5 MB/receptor: {n_rec * 0.5 / 1024:.1f} GB")


# ----------------------------------------------------------------------------
# Outputs
# ----------------------------------------------------------------------------
def ligand_fingerprints(smiles_list):
    """Packed ECFP4 bits. Binary bits make batch Tanimoto a single matmul."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator

    RDLogger.DisableLog("rdApp.*")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_BITS)
    fps = np.zeros((len(smiles_list), FP_BITS), dtype=np.uint8)
    bad = 0
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
        if mol is None:
            bad += 1
            continue
        fps[i, list(gen.GetFingerprint(mol).GetOnBits())] = 1
    if bad:
        print(f"[warn] {bad} SMILES failed to parse; those rows are all-zero")
    return fps


def write_receptors(df, out):
    """`receptor_key<TAB>chains<TAB>two_char_code<TAB>system_id`, one per receptor.

    The system_id is a representative one sharing this receptor; it is what
    names the folder inside systems/{two_char_code}.zip, so fetch_structures.py
    needs it to find receptor.cif.

    The structures live inside systems/{two_char_code}.zip, so a separate step
    resolves these to receptor.cif paths. system_zips.txt lists only the
    archives actually needed, which is the selective download the official
    tooling does not offer.
    """
    rec = (
        df[["receptor_key", "system_id"]]
        .drop_duplicates("receptor_key")
        .assign(
            chains=lambda d: d["system_id"].map(receptor_chains),
            two_char=lambda d: d["system_id"].str[1:3],
        )
        .sort_values("receptor_key")
    )
    (out / "receptors.txt").write_text(
        "\n".join(f"{r.receptor_key}\t{r.chains}\t{r.two_char}\t{r.system_id}"
                  for r in rec.itertuples()) + "\n"
    )
    zips = sorted(rec["two_char"].unique())
    (out / "system_zips.txt").write_text("\n".join(f"{z}.zip" for z in zips) + "\n")
    print(f"receptors.txt: {len(rec)} receptors across {len(zips)} systems/*.zip archives")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plinder", required=True, help="PLINDER release/iteration directory")
    ap.add_argument("--out", required=True)
    ap.add_argument("--inspect", action="store_true", help="print schemas and exit")
    ap.add_argument("--counts", action="store_true", help="filter attrition only; no writes")
    ap.add_argument("--dedup_cluster", default=DEDUP_CLUSTER)
    ap.add_argument("--no_dedup", action="store_true")
    a = ap.parse_args()

    if a.inspect:
        inspect(a.plinder)
        return

    df = load(a.plinder)
    df = apply_filters(df)
    if not a.no_dedup:
        df = deduplicate(df, a.dedup_cluster)
    summarize(df)

    if a.counts:
        print("\n--counts: nothing written")
        return

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    smi_col = "ligand_rdkit_canonical_smiles"
    if smi_col in df.columns:
        uniq = df[smi_col].drop_duplicates().reset_index(drop=True)
        df["fp_row"] = df[smi_col].map({s: i for i, s in enumerate(uniq)}).astype(np.int32)
        fps = ligand_fingerprints(uniq.tolist())
        np.save(out / "ligand_fps.npy", fps)
        print(f"ligand_fps.npy: {fps.shape}")
    else:
        print("[warn] no SMILES column; skipping fingerprints")

    df = df.reset_index(drop=True)
    df["sys_row"] = np.arange(len(df), dtype=np.int32)
    write_receptors(df, out)
    df.to_parquet(out / "manifest.parquet", index=False)
    print(f"manifest.parquet: {len(df)} rows")


if __name__ == "__main__":
    main()