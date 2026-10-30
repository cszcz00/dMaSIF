"""Pick a few rows of items.tsv and list every HPC file they need, for rsync.

Why. The figures want real complexes, but the trees they live in are far too
big to copy (structures/ 69.5 GB, systems/ 307k directories). One row needs
only a handful of small files, all derivable from its ids:

    systems/<system_id>/                  PLINDER system: system.cif (receptor
                                          WITH ligands), ligand_files/*.sdf, ...
    structures/<receptor_key>.cif         the receptor dMaSIF was given
    data/surfaces/<input_id>.npz          cached surface: xyz, normals, atoms
    data/pockets/fpocket/<input_id>.npz   fpocket's pockets on that receptor
    data/plip/<system_id>.npz             PLIP protein-ligand interactions

The ligand's features need no copy: they are rows of the run's
ligand_feats.npz (fp_row), which is already local.

Writes, under --out (default plots/subsets/<name>):
    <name>.tsv     the chosen rows, all items.tsv columns
    <name>.files   one path per line, relative to $WORK, for --files-from

Then one rsync fetches the lot into the same relative paths, so every other
script finds the files where the cluster has them:

    rsync -avr --files-from=plots/subsets/demo.files alpha.empire-ai.org:/mnt/home/cjs2301/dmasif/ .

(-r is needed because --files-from turns off the recursion -a implies, and
systems/<id>/ is a directory. A missing file, e.g. the one receptor fpocket
failed on, makes rsync exit 23 after copying everything else.)

Choosing rows - filters first, then a sample of what is left:
    --system_ids ID ...      exactly these rows
    --groups up:P00918 ...   rows of these target groups
    --smiles_contains SUB    ligand SMILES containing this substring
    --split train|val|test
    --n N                    then N rows, at most --per_group per target group,
                             biggest groups first unless --random

Examples:
    # 12 rows from the biggest targets, 2 each
    python plots/fetch_subset.py --items out/runs/dataset_v2/items.tsv --name big_targets --n 12 --per_group 2
    # every biotin conjugate (streptavidin binders)
    python plots/fetch_subset.py --items out/runs/v1/items.tsv --name biotin \\
        --smiles_contains "SC[C@@H]2NC(=O)N[C@@H]2" --n 20
    python plots/fetch_subset.py selftest

Depends on: pandas.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd


def row_files(r):
    """One items.tsv row -> its files, relative to $WORK."""
    # input_path is persisted ABSOLUTE in some tables (/mnt/home/.../structures/x.cif);
    # the receptor file is always structures/<receptor_key>.cif either way
    return [f"systems/{r.system_id}/",
            f"structures/{r.receptor_key}.cif",
            f"data/surfaces/{r.input_id}.npz",
            f"data/pockets/fpocket/{r.input_id}.npz",
            f"data/plip/{r.system_id}.npz"]


def choose(df, a):
    if a.system_ids:
        df = df[df.system_id.isin(a.system_ids)]
    if a.groups:
        df = df[df.group.isin(a.groups)]
    if a.smiles_contains:
        df = df[df.smiles.astype(str).str.contains(a.smiles_contains, regex=False)]
    if a.split:
        df = df[df.split == a.split]
    if not a.n or len(df) <= a.n and not a.per_group:
        return df
    df = df.sample(frac=1, random_state=a.seed)            # random rows within a group
    df = df.groupby("group", sort=False).head(a.per_group or len(df))
    if not a.random:
        size = df.group.map(df.group.value_counts())
        df = df.assign(_size=size).sort_values("_size", ascending=False, kind="stable")
        df = df.drop(columns="_size")
    return df.head(a.n)


def run(a):
    df = pd.read_csv(a.items, sep="\t", dtype=str)
    sel = choose(df, a)
    if sel.empty:
        raise SystemExit("no rows match")
    out = Path(a.out or f"plots/subsets/{a.name}")
    out.mkdir(parents=True, exist_ok=True)
    sel.to_csv(out / f"{a.name}.tsv", sep="\t", index=False)
    files = list(dict.fromkeys(f for r in sel.itertuples() for f in row_files(r)))
    (out / f"{a.name}.files").write_text("\n".join(files) + "\n")

    print(f"{a.items}: {len(df)} rows -> {len(sel)} chosen, {sel.group.nunique()} target "
          f"groups, {sel.receptor_key.nunique()} receptors")
    for r in sel.itertuples():
        print(f"  {r.system_id:32s} {r.group:16s} {str(r.smiles)[:60]}")
    print(f"\nwrote {out / (a.name + '.tsv')} and {out / (a.name + '.files')} "
          f"({len(files)} paths)\nfetch with:\n"
          f"  rsync -avr --files-from={out / (a.name + '.files')} "
          f"alpha.empire-ai.org:/mnt/home/cjs2301/dmasif/ .")


def selftest():
    rows = [dict(system_id=f"s{i}__1__1.A__1.B", input_id=f"r{i % 4}_1A",
                 receptor_key=f"r{i % 4}", split="train" if i < 8 else "val",
                 input_path=f"/abs/structures/r{i % 4}.cif", input_chains="1.A",
                 sdf_paths=f"systems/s{i}/ligand_files/1.B.sdf", fp_row=str(i),
                 group=f"g{min(i, 3)}", smiles="CCO" if i % 2 else "c1ccccc1")
            for i in range(10)]
    df = pd.DataFrame(rows)
    base = dict(system_ids=None, groups=None, smiles_contains="", split="", n=0,
                per_group=0, random=False, seed=0)
    ns = lambda **k: argparse.Namespace(**{**base, **k})
    assert len(choose(df, ns())) == 10
    assert set(choose(df, ns(split="val")).system_id) == {"s8__1__1.A__1.B", "s9__1__1.A__1.B"}
    assert (choose(df, ns(smiles_contains="ccc")).smiles == "c1ccccc1").all()
    # g3 holds rows 3..9 (7 rows): biggest first, capped per group
    got = choose(df, ns(n=3, per_group=2))
    assert len(got) == 3 and (got.group.values[:2] == "g3").all(), got.group.tolist()
    assert choose(df, ns(n=4, per_group=1)).group.is_unique
    f = row_files(next(df.iloc[:1].itertuples()))
    assert f == ["systems/s0__1__1.A__1.B/", "structures/r0.cif", "data/surfaces/r0_1A.npz",
                 "data/pockets/fpocket/r0_1A.npz", "data/plip/s0__1__1.A__1.B.npz"], f
    print("PASS")


def main():
    if sys.argv[1:2] == ["selftest"]:
        return selftest()
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--items", required=True, help="a run's items.tsv")
    ap.add_argument("--name", required=True, help="names the subset's files")
    ap.add_argument("--out", default="", help="default plots/subsets/<name>")
    ap.add_argument("--system_ids", nargs="+")
    ap.add_argument("--groups", nargs="+")
    ap.add_argument("--smiles_contains", default="")
    ap.add_argument("--split", default="")
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--per_group", type=int, default=0)
    ap.add_argument("--random", action="store_true", help="random groups, not biggest first")
    ap.add_argument("--seed", type=int, default=0)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
