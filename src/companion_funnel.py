"""What does `n_ligand_chains == 1` cost us, and would relaxing it be safe?

Why. `configs/datasets/dataset_v1.yaml` keeps a PLINDER system only if it has
exactly ONE ligand chain of ANY kind:

    query: "n_ligand_chains == 1"
    # n_ligand_chains counts every ligand chain in the system, ions and other
    # non-proper ligands included. strict_rep only guarantees ONE PROPER ligand.

So a structure with a drug-like ligand plus a magnesium, or plus a glycerol
from the cryoprotectant 30 A from the site, is thrown away whole. The row-level
`not_cofactor` / `not_ion` filters are a different thing: they stop a cofactor
being the TARGET, which we want and keep.

That restriction is defensible for one reason only - `extract.load_protein_atoms`
keeps amino acids and drops everything else, so a companion molecule leaves a
VOID in the surface, fpocket fills that void with alpha spheres, and the void
can then merge into the site region the model pools over. The question this
script answers is how much data the restriction costs and how much of that
data is actually affected by the void, which is a DISTANCE question: a
companion far from the proper ligand cannot touch its site region, so those
systems are free to admit with the current extraction untouched.

Two stages, because they need different inputs:

    funnel    the manifest alone (pandas). How many strict_rep rows each
              `n_ligand_chains` value holds, how many receptors and target
              groups they bring, and what the companions ARE (ion / cofactor /
              artifact / oligo), from the sibling rows of the same system.
              This is the "how much data" half.
    distance  the manifest plus PLINDER's system files (numpy + the SDF / CIF
              readers). For a sample of the rows funnel would ADD, the minimum
              distance from any companion heavy atom to the proper ligand, and
              how many fall inside the 4 A pocket label and the ~8 A region a
              site pocket spans. This is the "is it safe" half, and it decides
              whether dataset_v2 needs a distance gate or can take every
              companion system.

How to read it. Three outcomes, with what each implies:
  few rows added                 leave dataset_v1's query alone; the
                                 restriction is not what limits us.
  many rows, companions far      relax to `n_proper_ligand_chains == 1` and
                                 keep the current extraction. Cheapest path:
                                 existing surfaces and pocket files stay
                                 valid, only the NEW receptors need sampling
                                 and fpocket, and both jobs skip work done.
  many rows, companions close    relax WITH a distance gate (admit a system
                                 only if every companion is beyond --gate A of
                                 the proper ligand), or pay for the expensive
                                 path: keep metals and cofactors as receptor
                                 atoms, which invalidates all 19,293 surfaces,
                                 every pocket file, the PLIP labels and the
                                 comparability of every run in docs/STATUS.md.

Note the expensive path also costs AlphaFold compatibility: predicted
structures carry no cofactors or metals, and the apo test's finding that
predicted inputs lose only 6% of AP was measured with everything stripped on
both sides. Stripping is what makes holo / apo / predicted comparable.

Example:
    # CPU venv, both stages
    python src/companion_funnel.py funnel --manifest data/meta/manifest.parquet
    python src/companion_funnel.py distance --manifest data/meta/manifest.parquet \\
        --systems systems --n 400

Depends on: pandas, numpy (funnel); extract.load_protein_atoms and
            test_encoder.read_sdf_heavy (distance). No torch.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# build_manifest records these per (system, ligand) row; a companion is a row
# of the same system that is not the proper ligand we would train on.
KIND_FLAGS = ("ligand_is_ion", "ligand_is_cofactor", "ligand_is_artifact",
              "ligand_is_oligo", "ligand_is_covalent")


def _strict(df):
    """The strict_rep rows, i.e. what dataset_v1 starts from."""
    if "strict_rep" not in df.columns:
        raise SystemExit("manifest has no strict_rep column; build it with "
                         "src/build_manifest.py (loose mode is fine)")
    return df[df["strict_rep"].fillna(False).astype(bool)]


def kind_of(row):
    """A companion row -> one word, for the composition table."""
    for f in KIND_FLAGS:
        if f in row and bool(row[f]):
            return f.replace("ligand_is_", "")
    return "other"


def funnel(a):
    import pandas as pd

    df = pd.read_parquet(a.manifest)
    s = _strict(df)
    if "n_ligand_chains" not in s.columns:
        raise SystemExit("manifest has no n_ligand_chains column; "
                         "dataset_v1.yaml's query needs it, so this is a "
                         "different manifest build")
    print(f"{a.manifest}: {len(df):,} rows, {len(s):,} strict_rep\n")

    n = s["n_ligand_chains"].fillna(0).astype(int)
    keep, add = s[n == 1], s[n > 1]
    tab = n.value_counts().sort_index()
    print("strict_rep rows by n_ligand_chains (dataset_v1 keeps == 1):")
    for v, c in tab.items():
        mark = "  <- kept" if v == 1 else ""
        print(f"  {v:>3d} chains  {c:>7,} rows  ({c / len(s):5.1%}){mark}")

    def census(d, label):
        rec = d["system_id"].map(lambda x: "__".join(str(x).split("__")[:3]))
        grp = (d[a.group_col] if a.group_col in d.columns
               else pd.Series(["?"] * len(d), index=d.index))
        print(f"  {label:<22s} {len(d):>7,} rows  "
              f"{rec.nunique():>7,} receptors  {grp.nunique():>7,} groups")

    print("\nwhat the query keeps and what it discards:")
    census(keep, "kept (== 1)")
    census(add, "discarded (> 1)")
    if len(keep):
        print(f"\n  relaxing to one PROPER ligand would add {len(add) / len(keep):.0%} "
              f"more rows ({len(add):,}), on top of {len(keep):,}")

    # What the companions are. Group the discarded systems' OTHER rows by kind.
    if "system_id" in df.columns:
        extra = df[df["system_id"].isin(set(add["system_id"]))]
        comp = extra[~extra.index.isin(add.index)]
        if len(comp):
            kinds = comp.apply(kind_of, axis=1).value_counts()
            print("\ncompanion ligand chains in the discarded systems, by kind:")
            for k, c in kinds.items():
                print(f"  {k:<12s} {c:>7,}  ({c / len(comp):5.1%})")
            print("  (a companion can satisfy several flags; the first match "
                  "above wins, in ion/cofactor/artifact/oligo/covalent order)")
            print("\nartifact = crystallisation additive (glycerol, PEG, buffer). "
                  "\nThose are the ones most likely to sit far from the site, so "
                  "\nrun `distance` before deciding.")
        else:
            print("\n[warn] found no companion rows for the discarded systems; "
                  "the manifest may hold only one row per system, in which case "
                  "the kinds have to come from the annotation table instead")


def distance(a):
    import pandas as pd

    from extract import load_protein_atoms            # noqa: F401  (import check)
    from test_encoder import read_sdf_heavy

    df = pd.read_parquet(a.manifest)
    s = _strict(df)
    n = s["n_ligand_chains"].fillna(0).astype(int)
    add = s[n > 1]
    if not len(add):
        print("nothing discarded by the query; nothing to measure")
        return
    rng = np.random.default_rng(a.seed)
    take = add.iloc[rng.permutation(len(add))[:a.n]]
    root = Path(a.systems)
    print(f"{len(take)} of {len(add):,} discarded rows sampled; reading ligand "
          f"files under {root}\n")

    mins, n_ok, failed = [], 0, []
    for _, r in take.iterrows():
        sid = str(r["system_id"])
        d = root / sid / "ligand_files"
        if not d.is_dir():
            failed.append((sid, "no ligand_files directory"))
            continue
        sdfs = sorted(d.glob("*.sdf"))
        if len(sdfs) < 2:
            failed.append((sid, f"{len(sdfs)} sdf files, need the companions"))
            continue
        # The proper ligand is the row's own chain; everything else is a
        # companion. PLINDER names each file after its ligand chain.
        own = str(r.get("ligand_instance_chain", "")) or None
        try:
            mine, others = None, []
            for f in sdfs:
                xyz = read_sdf_heavy(f)
                if own and f.stem.startswith(own):
                    mine = xyz
                else:
                    others.append(xyz)
            if mine is None:                  # fall back to the largest
                sizes = [len(read_sdf_heavy(f)) for f in sdfs]
                j = int(np.argmax(sizes))
                mine = read_sdf_heavy(sdfs[j])
                others = [read_sdf_heavy(f) for i, f in enumerate(sdfs) if i != j]
            if not len(mine) or not others:
                failed.append((sid, "no companion coordinates"))
                continue
            d2 = min(float(np.sqrt(((mine[:, None] - o[None]) ** 2).sum(-1)).min())
                     for o in others if len(o))
            mins.append(d2)
            n_ok += 1
        except Exception as e:
            failed.append((sid, f"{type(e).__name__}: {e}"))

    if not mins:
        print("no distances measured; first failures:")
        for sid, m in failed[:10]:
            print(f"  {sid}: {m[:120]}")
        print("\nPLINDER's layout may differ from the assumed "
              "<systems>/<system_id>/ligand_files/<chain>.sdf; check one "
              "directory by hand and pass --systems accordingly")
        return
    m = np.asarray(mins)
    print(f"minimum companion-to-ligand heavy-atom distance, {n_ok} systems "
          f"({len(failed)} unreadable):")
    for q in (5, 25, 50, 75, 95):
        print(f"  p{q:<3d} {np.percentile(m, q):6.2f} A")
    for cut, why in ((4.0, "inside the 4 A pocket label"),
                     (8.0, "inside roughly a site region's span"),
                     (a.gate, f"inside the proposed --gate {a.gate:g} A")):
        print(f"  < {cut:4.1f} A  {(m < cut).mean():5.1%}  {why}")
    print(f"\nRead: the share beyond --gate {a.gate:g} A is the data a relaxed "
          f"query could take\nwith the CURRENT extraction untouched - those "
          "companions cannot reach the site\nregion. The share under 4 A is "
          "where stripping puts a void in the binding site\nitself; gate those "
          "out, or pay for keeping companion atoms as receptor atoms.")
    if failed:
        print(f"\nfirst unreadable systems:")
        for sid, msg in failed[:5]:
            print(f"  {sid}: {msg[:120]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("funnel", help="manifest only: how much data the query costs")
    p.add_argument("--manifest", default="data/meta/manifest.parquet")
    p.add_argument("--group_col", default="pli_unique_qcov__95__strong__component",
                   help="column counted as a target group")

    p = sub.add_parser("distance", help="how far companions sit from the ligand")
    p.add_argument("--manifest", default="data/meta/manifest.parquet")
    p.add_argument("--systems", default="systems",
                   help="PLINDER systems directory (<system_id>/ligand_files/)")
    p.add_argument("--n", type=int, default=400, help="systems to sample")
    p.add_argument("--gate", type=float, default=8.0,
                   help="proposed minimum companion distance for dataset_v2")
    p.add_argument("--seed", type=int, default=0)

    a = ap.parse_args()
    {"funnel": funnel, "distance": distance}[a.cmd](a)


if __name__ == "__main__":
    main()
