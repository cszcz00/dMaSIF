"""Build a training manifest from PLINDER for contrastive protein-ligand retrieval.

Reads a column subset of index/annotation_table.parquet
and merges the split labels from splits/split.parquet.

Three modes:

    --inspect   print the schema of all three tables and exit
    --counts    per-filter attrition and the headline numbers; writes nothing
    (default)   write manifest.parquet, receptors.txt, ligand_fps.npy

Loose by default: only the HARD filters (is the ligand a usable small molecule
at all?) remove rows. Every other filter is recorded as a boolean pass_<name>
column, so the dataset can be trimmed later from the manifest alone, without
re-reading PLINDER. `pass_all` marks rows passing every filter, and
`strict_rep` marks exactly the rows the strict pipeline (all filters + dedup)
would have kept. --strict restores that pipeline as the output.

The cluster columns (pli_unique_qcov__95__strong__component and friends) are
denormalized into the annotation table, so dedup, leakage-aware grouping and
batch negative masking need neither clusters/ nor scores/. Only pairwise
similarity VALUES need scores/, and this script no longer requires it.

Example:
    python build_manifest.py --plinder $PLINDER_DIR --out meta/ --inspect
    python build_manifest.py --plinder $PLINDER_DIR --out meta/ --counts
    python build_manifest.py --plinder $PLINDER_DIR --out meta/
    python build_manifest.py --plinder $PLINDER_DIR --out meta_strict/ --strict

Trimming later, e.g. back to the strict set:
    df = pd.read_parquet("meta/manifest.parquet"); df = df[df.strict_rep]
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

def _flag(col, want=True):
    """Row-level boolean test that treats a null as failing."""
    return lambda d: d[col].fillna(not want).astype(bool) == want


# (name, label, predicate). Each becomes a pass_<name> column in the manifest.
#
# ROW-level filters come first and are the ones that matter for correctness:
# annotation_table has one row per (system, ligand), so a system_* column says
# nothing about WHICH ligand this row is. Filtering only on system_* columns
# lets a magnesium ion ride along in a system whose proper ligand is drug-like,
# because system_proper_ligand_max_molecular_weight is the max over proper
# ligands and the MG row inherits it.
FILTERS = [
    # -- row level: is THIS ligand a training example? --
    ("proper", "ligand is proper", _flag("ligand_is_proper")),
    ("not_ion", "not an ion", _flag("ligand_is_ion", False)),
    ("not_cofactor", "not a cofactor", _flag("ligand_is_cofactor", False)),
    ("not_artifact", "not an artifact", _flag("ligand_is_artifact", False)),
    ("not_oligo", "not an oligomer", _flag("ligand_is_oligo", False)),
    ("not_covalent", "not covalently bound", _flag("ligand_is_covalent", False)),
    ("rdkit", "valid / rdkit loadable", _flag("ligand_is_rdkit_loadable")),
    ("mw", "ligand MW 200-800", lambda d: d["ligand_molecular_weight"].between(200, 800)),
    ("positions", "ligand positions correct", _flag("ligand_positions_correct")),
    ("interactions", "3-50 interactions", lambda d: d["ligand_num_interactions"].between(3, 50)),
    # -- system level: is the pocket well defined? --
    ("single_ligand", "system has 1 proper ligand",
     lambda d: d["system_proper_num_ligand_chains"] == 1),
    ("pocket_size", "pocket 5-100 residues",
     lambda d: d["system_proper_num_pocket_residues"].between(5, 100)),
    ("crystal_contacts", "crystal contacts < 25%",
     lambda d: d["ligand_fraction_atoms_with_crystal_contacts"].fillna(0) < 0.25),
    ("no_missing_pocket", "no missing pocket residues",
     lambda d: d["ligand_num_missing_pli_interface_residues"].fillna(0) == 0),
    ("all_chains", "all protein chains present", _flag("all_protein_chains_present")),
    ("resolution", "resolution <= 2.5 A", lambda d: d["entry_resolution"] <= 2.5),
    ("pocket_resolved", "pocket fully resolved",
     lambda d: d["system_pocket_validation_num_unresolved_heavy_atoms"].fillna(0) == 0),
    ("no_altlocs", "no alt conformations",
     lambda d: d["system_pocket_validation_max_alt_count"].fillna(1) <= 1),
]

# Applied even in loose mode: rows failing these are not small-molecule
# ligands with a fingerprint, so no amount of later trimming makes them usable.
HARD = {"proper", "not_ion", "not_artifact", "rdkit"}

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
def apply_filters(df, splits=("train", "val", "test"), strict=False):
    """Evaluate every filter, record it as pass_<name>, drop rows failing HARD.

    `fails` is how many rows fail that filter on its own. `strict left` is the
    running remainder if every filter were applied in order - the attrition
    the strict pipeline would show. Filters overlap heavily (a low-resolution
    entry often also has unresolved pocket atoms), so read the running
    remainder, not the per-filter numbers.
    """
    df = df[df["split"].isin(splits)].copy()
    print(f"\n{'filter':30s} {'fails':>9s} {'strict left':>12s}  applied")
    print(f"{'(start, ' + '/'.join(splits) + ')':30s} {'':>9s} {len(df):>12d}")

    keep = pd.Series(True, index=df.index)
    strict_mask = pd.Series(True, index=df.index)
    for name, label, fn in FILTERS:
        try:
            m = fn(df).fillna(False).astype(bool)
        except KeyError as e:
            print(f"{label:30s} {'SKIPPED':>9s}  (missing {e})")
            continue
        df[f"pass_{name}"] = m
        strict_mask &= m
        applied = strict or name in HARD
        if applied:
            keep &= m
        print(f"{label:30s} {int((~m).sum()):>9d} {int(strict_mask.sum()):>12d}  "
              f"{'yes' if applied else '-'}")

    df["pass_all"] = strict_mask
    out = df[keep].copy()
    print(f"\nkept {len(out)} rows ({int(out['pass_all'].sum())} pass every filter)")

    multi = int(out["system_id"].duplicated().sum())
    if multi:
        # Expected in loose mode: a system with two proper ligands gives two rows.
        # In strict mode single_ligand should prevent it, so it points at a bug.
        note = "check the row-level filters" if strict else "systems with several ligands"
        print(f"[{'warn' if strict else 'info'}] {multi} rows share a system_id ({note})")
    out["receptor_key"] = out["system_id"].map(receptor_key)
    return out


def mark_strict_reps(df, cluster_col=DEDUP_CLUSTER):
    """strict_rep = the rows the strict pipeline (all filters + dedup) keeps.

    Dedup runs on the pass_all subset with the same key and ordering as the
    strict pipeline, so df[df.strict_rep] reproduces it exactly.
    """
    print("\nstrict subset:", end="")
    reps = deduplicate(df[df["pass_all"]], cluster_col).index
    df["strict_rep"] = df.index.isin(reps)
    return df


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


def summarize(df, title="manifest"):
    print(f"\n[{title}]")
    print(f"    {'split':8s}{'pairs':>9s}{'receptors':>11s}{'ligands':>9s}{'Pfam':>8s}")
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
    """ECFP4 bits, one uint8 (0/1) per bit. Batch Tanimoto is then a single matmul."""
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
    ap.add_argument("--no_dedup", action="store_true",
                    help="with --strict: skip dedup (loose output is never deduplicated)")
    ap.add_argument("--strict", action="store_true",
                    help="apply every filter + dedup to the output (the original behaviour); "
                         "default applies only HARD filters and records the rest as columns")
    a = ap.parse_args()

    if a.inspect:
        inspect(a.plinder)
        return

    df = load(a.plinder)
    df = apply_filters(df, strict=a.strict)
    df = mark_strict_reps(df, a.dedup_cluster)
    if a.strict and not a.no_dedup:
        df = df[df["strict_rep"]].copy()
    summarize(df)
    if not a.strict:
        summarize(df[df["strict_rep"]], "strict subset (strict_rep)")

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