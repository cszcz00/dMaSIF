"""One fixed validation set for every run, across datasets.

Why. `train.py` carves validation by target group out of whatever items it is
given, so a run on dataset_v2 would validate on different rows from every run
on dataset_v1, and "did more data help" could not be told apart from "the
validation set changed". This fixes one set of rows and makes both sides use
it:

  - the TARGETS are dataset_v1's carved validation groups, re-derived exactly
    as every v1-based run carved them (same items, the same rows dropped for
    unfeaturisable ligands as train.load_rows drops, split_by_group at the
    same val_frac / split_seed). So no checkpoint trained so far has seen
    them, and old runs can be re-scored on the new set without retraining.
  - the ROWS are a group-balanced draw of those targets' rows that exist in
    BOTH datasets (same system_id, same ligand, featurisable in both): the
    per-entry dedup dropped some dataset_v1 rows from dataset_v2, and a row
    must be scorable by a v1 and a v2 checkpoint alike, against the same
    candidate ligands.
  - a dataset_v2 run trains on every row whose target is NOT one of
    dataset_v1's validation targets (the whole pool, not only the drawn rows).

Written to a directory:
    val_set.tsv          system_id, group, smiles - the rows
    holdout_groups.txt   every target group a run must not train on
    provenance.json      inputs, seeds, counts

Used by `train.py train --val_set DIR` (validation = these rows, training =
everything outside the holdout groups) and `train.py eval --val_set DIR`
(score a checkpoint, old or new, on exactly these rows).

    python src/valset.py make --items_a out/runs/v1/items.tsv \\
        --lig_feats_a out/runs/v1/ligand_feats.npz \\
        --items_b out/runs/dataset_v2/items.tsv \\
        --lig_feats_b out/runs/dataset_v2/ligand_feats.npz --out out/valset_v1v2
    python src/valset.py selftest

Depends on: numpy; train_data.read_items, probe_retrieval.split_by_group
            (both numpy only). No torch: runs in the CPU venv.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def load_items(items, lig_feats):
    """items.tsv rows minus those whose ligand failed featurisation - the same
    rows, in the same order, as train.load_rows keeps."""
    from train_data import read_items

    rows = read_items(items)
    with np.load(lig_feats, allow_pickle=False) as d:
        failed = d["failed"] if "failed" in d.files else np.zeros(len(d["n_heavy"]), bool)
    return [r for r in rows if not failed[r["fp_row"]]]


def balanced(rows, groups, n, seed):
    """Round-robin over groups in random order: train.subsample_val(balance=True),
    copied so this module stays torch-free."""
    rows = np.asarray(rows)
    rng = np.random.default_rng(seed)
    g = np.asarray(groups)[rows]
    buckets = [rng.permutation(rows[g == u]) for u in rng.permutation(np.unique(g))]
    out, depth = [], 0
    while len(out) < n and any(depth < len(b) for b in buckets):
        out += [b[depth] for b in buckets if depth < len(b)][:n - len(out)]
        depth += 1
    return np.sort(np.asarray(out, dtype=np.int64))


def choose(rows_a, rows_b, val_frac=0.1, split_seed=0, n=400):
    """-> (val rows as dicts from A, holdout groups, counts)."""
    from probe_retrieval import split_by_group

    groups_a = np.asarray([r["group"] for r in rows_a])
    _, pool = split_by_group(groups_a, val_frac, split_seed)
    holdout = sorted(set(groups_a[pool]))
    by_sid = {}
    for r in rows_b:
        if r["system_id"] in by_sid:
            raise SystemExit(f"items_b: system_id {r['system_id']} appears twice")
        by_sid[r["system_id"]] = r
    both = [i for i in pool
            if rows_a[i]["system_id"] in by_sid
            and by_sid[rows_a[i]["system_id"]]["smiles"] == rows_a[i]["smiles"]]
    pick = balanced(both, groups_a, n, split_seed)
    counts = {"pool_rows": int(len(pool)), "pool_groups": len(holdout),
              "rows_in_both": len(both),
              "groups_in_both": len({groups_a[i] for i in both}),
              "val_rows": int(len(pick)),
              "val_groups": len({groups_a[i] for i in pick})}
    return [rows_a[i] for i in pick], holdout, counts


def write(out, val_rows, holdout, provenance):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "val_set.tsv").write_text(
        "system_id\tgroup\tsmiles\n"
        + "".join(f"{r['system_id']}\t{r['group']}\t{r['smiles']}\n" for r in val_rows))
    (out / "holdout_groups.txt").write_text("".join(g + "\n" for g in holdout))
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


def read(path):
    """-> (val system_ids in file order, holdout group set)."""
    path = Path(path)
    lines = (path / "val_set.tsv").read_text().splitlines()
    sids = [ln.split("\t")[0] for ln in lines[1:] if ln.strip()]
    holdout = {ln.strip() for ln in (path / "holdout_groups.txt").read_text().splitlines()
               if ln.strip()}
    return sids, holdout


def apply(rows, path):
    """A loaded table's rows (train.load_rows order) + a val set directory ->
    (train_idx, val_idx, missing system_ids). Training is every row whose
    group is neither held out nor a validation row's group."""
    sids, holdout = read(path)
    where = {}
    for i, r in enumerate(rows):
        where.setdefault(r["system_id"], i)
    val = np.asarray(sorted(where[s] for s in sids if s in where), dtype=np.int64)
    missing = [s for s in sids if s not in where]
    blocked = holdout | {rows[i]["group"] for i in val}
    train = np.asarray([i for i, r in enumerate(rows) if r["group"] not in blocked],
                       dtype=np.int64)
    return train, val, missing


def make(a):
    rows_a = load_items(a.items_a, a.lig_feats_a)
    rows_b = load_items(a.items_b, a.lig_feats_b)
    val_rows, holdout, counts = choose(rows_a, rows_b, a.val_frac, a.split_seed, a.n)
    prov = {"items_a": str(a.items_a), "lig_feats_a": str(a.lig_feats_a),
            "items_b": str(a.items_b), "lig_feats_b": str(a.lig_feats_b),
            "val_frac": a.val_frac, "split_seed": a.split_seed, "n": a.n,
            "rows_a": len(rows_a), "rows_b": len(rows_b), **counts}
    write(a.out, val_rows, holdout, prov)
    tr_b, va_b, miss = apply(rows_b, a.out)
    tr_a, va_a, _ = apply(rows_a, a.out)
    print(f"dataset A: {len(rows_a):,} rows; its carved val pool {counts['pool_rows']:,} "
          f"rows / {counts['pool_groups']:,} groups, of which {counts['rows_in_both']:,} "
          f"rows / {counts['groups_in_both']:,} groups also in B")
    print(f"val set: {counts['val_rows']} rows over {counts['val_groups']} groups "
          f"-> {a.out}/")
    print(f"dataset B ({len(rows_b):,} rows) under it: {len(tr_b):,} train rows over "
          f"{len({rows_b[i]['group'] for i in tr_b}):,} groups, {len(va_b)} val rows, "
          f"{len(miss)} missing")
    print(f"dataset A under it: {len(tr_a):,} train rows, {len(va_a)} val rows")
    if miss:
        raise SystemExit("val rows missing from B; this should be impossible by "
                         "construction - is items_b the file the set was made from?")


# ---------------------------------------------------------------------------
def selftest():
    import tempfile

    fails = []

    def check(name, ok, detail=""):
        print(f"  {'ok  ' if ok else 'FAIL'} {name}" + (f"   {detail}" if detail else ""))
        if not ok:
            fails.append(name)

    rng = np.random.default_rng(0)

    def row(i, g, smi=None):
        return {"system_id": f"s{i}", "group": f"up:P{g}", "smiles": smi or f"C{i}",
                "fp_row": i}

    rows_a = [row(i, i % 60) for i in range(600)]
    # B: A minus every 3rd row (the per-entry dedup), one changed ligand, plus new rows
    rows_b = [r for i, r in enumerate(rows_a) if i % 3]
    rows_b[0] = dict(rows_b[0], smiles="DIFFERENT")
    rows_b += [row(1000 + i, 100 + i % 40) for i in range(400)]
    rng.shuffle(rows_b)
    val, holdout, c = choose(rows_a, rows_b, val_frac=0.1, split_seed=0, n=40)
    sid_b = {r["system_id"]: r for r in rows_b}
    check("val rows exist in both datasets with the same ligand",
          all(r["system_id"] in sid_b and sid_b[r["system_id"]]["smiles"] == r["smiles"]
              for r in val) and len(val) == 40)
    check("val rows come only from A's carved pool, spread over its groups",
          {r["group"] for r in val} <= set(holdout) and c["val_groups"] == min(40, c["groups_in_both"]),
          f"{c}")
    from probe_retrieval import split_by_group
    _, pool = split_by_group(np.asarray([r["group"] for r in rows_a]), 0.1, 0)
    check("holdout = exactly A's carved val groups (what every v1 run held out)",
          set(holdout) == {rows_a[i]["group"] for i in pool})

    with tempfile.TemporaryDirectory() as td:
        write(td, val, holdout, {"selftest": True})
        tr, va, miss = apply(rows_b, td)
        check("on B: every val row found, no train row in a held-out group",
              len(va) == 40 and not miss
              and not ({rows_b[i]["group"] for i in tr} & set(holdout)))
        check("on B: everything outside the holdout trains (new targets included)",
              len(tr) == sum(r["group"] not in set(holdout) for r in rows_b))
        tra, vaa, _ = apply(rows_a, td)
        check("on A: the same system_ids validate, train is disjoint by group",
              sorted(rows_a[i]["system_id"] for i in vaa)
              == sorted(rows_b[i]["system_id"] for i in va)
              and not ({rows_a[i]["group"] for i in tra} & {rows_a[i]["group"] for i in vaa}))
        sids, ho = read(td)
        check("files round-trip", len(sids) == 40 and ho == set(holdout))

        # load_items drops failed ligands exactly like train.load_rows
        from train_data import ITEM_COLS
        items = Path(td) / "items.tsv"
        items.write_text("\t".join(ITEM_COLS) + "\n" + "".join(
            "\t".join({"system_id": f"s{i}", "fp_row": str(i), "group": "up:P1",
                       "smiles": "C"}.get(c, "x") for c in ITEM_COLS) + "\n"
            for i in range(5)))
        np.savez(Path(td) / "lig.npz", n_heavy=np.ones(5), failed=np.array([0, 1, 0, 0, 1], bool))
        kept = load_items(items, Path(td) / "lig.npz")
        check("load_items drops rows whose ligand failed featurisation",
              [r["system_id"] for r in kept] == ["s0", "s2", "s3"])

    print("\nPASS" if not fails else f"\nFAIL: {fails}")
    raise SystemExit(1 if fails else 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("make")
    p.add_argument("--items_a", required=True, help="the dataset the old runs carved from")
    p.add_argument("--lig_feats_a", required=True)
    p.add_argument("--items_b", required=True, help="the new dataset")
    p.add_argument("--lig_feats_b", required=True)
    p.add_argument("--val_frac", type=float, default=0.1, help="as the old runs carved")
    p.add_argument("--split_seed", type=int, default=0, help="as the old runs carved")
    p.add_argument("--n", type=int, default=400, help="val rows, group-balanced")
    p.add_argument("--out", required=True)
    sub.add_parser("selftest")
    a = ap.parse_args()
    selftest() if a.cmd == "selftest" else make(a)


if __name__ == "__main__":
    main()
