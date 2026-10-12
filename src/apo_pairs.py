"""Pick holo / apo / AlphaFold pairs for the apo-vs-holo encoder test.

PLINDER links each holo system to up to 5 apo PDB chains and to AlphaFold
models of the same protein (links/kind={apo,pred}/links.parquet, joined on
reference_system_id = our system_id). Most apo pockets barely differ from holo
(median pocket lDDT 94), so a random sample would test easy cases only. Pairs
are drawn by stratum of how much the apo pocket changed:

    A_near     apo pocket lDDT >= 95
    B_mild     85 <= pocket lDDT < 95
    C_large    pocket lDDT < 85
    D_closed   the holo ligand, placed in the apo structure, collides with it
               (posebusters_minimum_distance_to_protein fails); overrides A-C
    eval       every val / test system with a link, one random link each - an
               unbiased reference rather than a stratified one

Restrictions, so differences come from conformation and not from the setup:
    - single-chain receptors (apo and AlphaFold files are single chains)
    - pocket_fident >= --min_fident and every receptor chain mapped
    - at most one pair per target protein (system_pocket_UniProt) per stratum
    - receptors in --exclude (the items the head was trained on) are skipped

Each row also carries one AlphaFold link of the same system when there is one.
Links are chosen at random (seeded): PLINDER's sort_score has a different
meaning per kind, and picking the best-matching link would bias toward easy
pairs.

Writes <out>/pairs.tsv, one row per (system, apo link); a system in several sets
appears once with all its set names in `sets`.

Example (CPU venv):
    python src/apo_pairs.py --links $PLINDER_DIR/links \\
        --index datasets/dataset_v1/index.parquet \\
        --exclude encoder_test/items.tsv --out apo_test/
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

LINK_COLS = ["reference_system_id", "id", "pocket_fident", "pocket_lddt",
             "fraction_reference_proteins_mapped", "num_reference_proteins",
             "num_model_proteins", "posebusters_minimum_distance_to_protein",
             "lddt_pli_ave", "bisy_rmsd_ave"]
OUT_COLS = ["system_id", "split", "sets", "input_path", "input_chains", "sdf_paths",
            "uniprot", "apo_id", "apo_pocket_lddt", "apo_clash", "pred_id", "pred_pocket_lddt"]


def load_links(path, idx, min_fident):
    df = pd.read_parquet(path, columns=LINK_COLS)
    df = df[df.reference_system_id.isin(idx.system_id)]
    keep = ((df.pocket_fident >= min_fident) & (df.fraction_reference_proteins_mapped == 1)
            & (df.num_reference_proteins == 1) & (df.num_model_proteins == 1))
    df = df[keep].rename(columns={"reference_system_id": "system_id"})
    # posebusters_minimum_distance_to_protein: 1 = passes (no clash), 0 = clash
    df["clash"] = df.posebusters_minimum_distance_to_protein.fillna(1) < 0.5
    return df.drop_duplicates(["system_id", "id"])


def stratum(row):
    if row.clash:
        return "D_closed"
    if row.pocket_lddt >= 95:
        return "A_near"
    if row.pocket_lddt >= 85:
        return "B_mild"
    return "C_large"


def one_per(df, key, rng):
    """One random row per value of `key`."""
    df = df.sample(frac=1, random_state=rng.integers(1 << 31))
    return df.drop_duplicates(key)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--links", required=True, help="PLINDER links/ directory")
    ap.add_argument("--index", required=True, help="dataset index.parquet")
    ap.add_argument("--systems", default="systems")
    ap.add_argument("--exclude", default="", help="items.tsv whose receptors to skip")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=100, help="pairs per stratum")
    ap.add_argument("--min_fident", type=float, default=95.0)
    ap.add_argument("--eval_splits", default="val,test")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    from ids import ligand_chain, ligand_sdf_paths

    rng = np.random.default_rng(a.seed)
    idx = pd.read_parquet(a.index)
    n_rec = idx.system_id.map(lambda s: len(s.split("__")[2].split("_")))
    idx = idx[n_rec == 1]
    if a.exclude:
        seen = set(pd.read_csv(a.exclude, sep="\t")["input_path"])
        idx = idx[~idx.input_path.isin(seen)]
    idx = idx.drop_duplicates("system_id").copy()
    idx["uniprot"] = (idx["system_pocket_UniProt"].fillna("").astype(str)
                      if "system_pocket_UniProt" in idx else "")
    idx.loc[idx.uniprot == "", "uniprot"] = idx.receptor_key  # unknown target: own group

    apo = load_links(Path(a.links) / "kind=apo" / "links.parquet", idx, a.min_fident)
    pred = load_links(Path(a.links) / "kind=pred" / "links.parquet", idx, a.min_fident)
    apo = apo.merge(idx[["system_id", "split", "uniprot"]], on="system_id")
    apo["stratum"] = apo.apply(stratum, axis=1)
    print(f"apo links usable: {len(apo)} on {apo.system_id.nunique()} systems; "
          f"pred links usable: {len(pred)} on {pred.system_id.nunique()} systems")
    print("usable apo links by stratum:", apo.stratum.value_counts().to_dict())

    chosen = {}  # (system_id, apo_id) -> set of set names
    for name in ("A_near", "B_mild", "C_large", "D_closed"):
        pool = apo[apo.stratum == name]
        pool = one_per(one_per(pool, "system_id", rng), "uniprot", rng)
        pick = pool.sample(n=min(a.n, len(pool)), random_state=rng.integers(1 << 31))
        for s, i in zip(pick.system_id, pick.id):
            chosen.setdefault((s, i), set()).add(name)
        print(f"{name:9s} {len(pick):4d} pairs (from {len(pool)} distinct targets)")

    evals = idx[idx.split.isin(a.eval_splits.split(","))]
    ev = one_per(apo[apo.system_id.isin(evals.system_id)], "system_id", rng)
    for s, i in zip(ev.system_id, ev.id):
        chosen.setdefault((s, i), set()).add("eval")
    # eval systems with an AlphaFold link but no apo link still count
    pred_only = set(pred.system_id[pred.system_id.isin(evals.system_id)]) - set(ev.system_id)
    for s in pred_only:
        chosen.setdefault((s, ""), set()).add("eval")
    print(f"eval      {len(ev)} with apo, {len(pred_only)} with AlphaFold only "
          f"({a.eval_splits})")

    pred_pick = one_per(pred, "system_id", rng).set_index("system_id")
    apo_by = apo.set_index(["system_id", "id"])
    info = idx.set_index("system_id")
    rows, n_nosdf = [], 0
    for (s, i), sets in sorted(chosen.items()):
        r = info.loc[s]
        sdfs = ligand_sdf_paths(a.systems, s, ligand_chain(r.ligand_id))
        if not sdfs:
            n_nosdf += 1
            continue
        lk = apo_by.loc[(s, i)] if i else None
        pr = pred_pick.loc[s] if s in pred_pick.index else None
        rows.append([s, r.split, ",".join(sorted(sets)), r.input_path, r.input_chains,
                     ",".join(map(str, sdfs)), r.uniprot,
                     i, "" if lk is None else f"{lk.pocket_lddt:g}",
                     "" if lk is None else str(int(lk.clash)),
                     "" if pr is None else pr.id,
                     "" if pr is None else f"{pr.pocket_lddt:g}"])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=OUT_COLS).to_csv(out / "pairs.tsv", sep="\t", index=False)
    n_pred = sum(1 for r in rows if r[10])
    print(f"\n{out / 'pairs.tsv'}: {len(rows)} systems, {sum(1 for r in rows if r[7])} with apo, "
          f"{n_pred} with AlphaFold" + (f"; {n_nosdf} skipped without a ligand SDF" if n_nosdf else ""))


if __name__ == "__main__":
    main()
