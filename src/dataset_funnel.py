"""Where dataset_v1 loses its data, and what each relaxation would buy.

Why. `v3c+` put the ceiling at MEMORISATION: C plateaus at ~0.18 known-site
MRR after ~69 visits per target group, with train nce still falling. More
distinct TARGETS is the lever, not more steps or more rows on targets we
already have. dataset_v1 is 21,209 rows out of a 341,945-row manifest, and
nobody has measured which of the 18 manifest filters, the dedup or the
`n_ligand_chains == 1` query is responsible, or how many NEW target groups
each would bring back. This does, from the manifest alone (pandas, seconds).

Everything is counted against a BASELINE that is dataset_v1's selection
(every build_manifest filter at its v1 threshold, `n_ligand_chains == 1`,
dedup on the interaction cluster + ligand). A VARIANT changes one criterion -
drops it, or loosens its threshold - and is reported as what it ADDS:

    rows          train rows (the rows `train.py prepare --split train` takes)
    groups        train target groups (UniProt, falling back to the 95%
                  interaction component - probe_retrieval.group_of, so these
                  are the groups the sampler balances over)
    +groups       groups the baseline does not have at all. THE number.
    +rows/new     train rows landing in those new groups
    +inputs       receptor surfaces not already in the baseline: the cost,
                  in surface sampling + fpocket + PLIP (all of which skip work
                  already done)
    val / test    PLINDER's own splits, which v1 has far too few of
    visits        steps x batch / groups at 5,000 x 32: v3c+'s ~69 is the
                  number this must bring down

Dedup prefers baseline rows (`strict_rep` first, then resolution, then
system_id), so every variant is a SUPERSET of the baseline: the existing
surfaces, pocket files and PLIP labels stay valid, and a v2 run is directly
comparable to every run so far. `--per_entry` adds a second dedup on (PDB
entry, ligand, UniProt), the fix for symmetry copies (TRP: 507 rows from 57
entries). UniProt is in the key so the same ligand bound to two DIFFERENT
proteins of one complex stays two examples; without it the first run lost
236 train groups to this dedup. It removes rows, so under --per_entry the
baseline itself is per-entry deduplicated and both baselines are printed.

Modes:
    report    one line per single-criterion variant (leave-one-out and
              loosened thresholds), sorted by +groups; then the cumulative
              LADDER (--ladder, applied left to right) with the running
              totals; then the biggest groups of the final ladder step, since
              relaxing quality filters can pile rows onto the targets that
              already dominate (PDE10A, Mpro, BACE1).
    selftest  a planted manifest, pandas only.

How to read it.
  - Sort by +groups, not by rows. A filter that adds 10,000 rows on PDE10A
    and Mpro does nothing for the memorisation ceiling; one that adds 300
    targets does.
  - `ligand_chains` (dataset_v1's `n_ligand_chains == 1`) can only be relaxed
    safely by distance - a companion molecule leaves a void in the surface
    because extract.py keeps amino acids only. Its +groups here is the upper
    bound; `companion_funnel.py distance` says how much of it is close.
  - Quality filters (resolution, altlocs, unresolved pocket atoms) cost
    surface accuracy, not label correctness. dMaSIF samples a smoothed
    surface at ~1 point per A^2 with 12 A convolutions, so a 2.8 A structure
    or one missing a pocket side-chain tip is a modest perturbation; a
    missing pocket RESIDUE is a hole and stays strict by default.
  - The ladder's last line is the proposed dataset_v2, before the distance
    gate. Write it into configs/datasets/dataset_v2.yaml only after this.

Example (CPU venv):
    python src/dataset_funnel.py report --manifest data/meta/manifest.parquet
    python src/dataset_funnel.py report --manifest data/meta/manifest.parquet \\
        --ladder "no_altlocs=drop,resolution<=3.0,pocket_resolved<=5" --per_entry
    python src/dataset_funnel.py selftest

Depends on: pandas, numpy; build_manifest (FILTERS), build_dataset
            (add_derived), receptor_inputs (input_id), probe_retrieval's
            grouping rule (re-implemented vectorised here, checked against it
            in the self-test). No torch.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

DEDUP_KEY = ["pli_unique_qcov__95__strong__component", "ligand_ccd_code"]
ENTRY_KEY = ["entry", "ligand_ccd_code", "system_pocket_UniProt"]
# Cost per new receptor, from inventory.py on dataset_v1 (2026-10-27):
# 5.8 GB / 19,293 surfaces, 310 MB / 19,293 fpocket files.
SURFACE_MB, POCKET_MB = 0.30, 0.016
STEPS, BATCH = 5000, 32


# ---------------------------------------------------------------------------
# Criteria
# ---------------------------------------------------------------------------
def v1_criteria():
    """name -> predicate(df) -> bool Series, at dataset_v1's thresholds.

    build_manifest's non-HARD filters (HARD ones already removed rows from the
    manifest) plus dataset_v1's query, named `ligand_chains`.
    """
    from build_manifest import FILTERS, HARD

    crit = {name: fn for name, _, fn in FILTERS if name not in HARD}
    crit["ligand_chains"] = lambda d: d["n_ligand_chains"] == 1
    # Never relaxed by the default ladder: a covalent sugar-only ligand is
    # glycosylation. A no-op on dataset_v1, which excludes covalent ligands.
    crit["not_glycosylation"] = lambda d: ~(d["ligand_is_covalent"].fillna(False).astype(bool)
                                            & d["ligand_is_glycan"])
    crit["not_detergent"] = lambda d: ~d["ligand_is_detergent"]
    return crit


def _col(d, c, fill):
    return d[c].fillna(fill)


# Loosened thresholds worth measuring, beyond plain "drop". Each is a
# (label, predicate) replacing the v1 predicate of that name.
LOOSER = {
    "resolution": [("<=3.0", lambda d: d["entry_resolution"] <= 3.0),
                   ("<=3.5", lambda d: d["entry_resolution"] <= 3.5),
                   ("<=4.0", lambda d: d["entry_resolution"] <= 4.0)],
    "no_altlocs": [("<=2", lambda d: _col(d, "system_pocket_validation_max_alt_count", 1) <= 2)],
    "pocket_resolved": [
        ("<=5", lambda d: _col(d, "system_pocket_validation_num_unresolved_heavy_atoms", 0) <= 5),
        ("<=20", lambda d: _col(d, "system_pocket_validation_num_unresolved_heavy_atoms", 0) <= 20)],
    "no_missing_pocket": [
        ("<=1", lambda d: _col(d, "ligand_num_missing_pli_interface_residues", 0) <= 1)],
    "crystal_contacts": [
        ("<0.5", lambda d: _col(d, "ligand_fraction_atoms_with_crystal_contacts", 0) < 0.5)],
    "mw": [("150-900", lambda d: d["ligand_molecular_weight"].between(150, 900)),
           ("100-1000", lambda d: d["ligand_molecular_weight"].between(100, 1000))],
    "interactions": [("2-80", lambda d: d["ligand_num_interactions"].between(2, 80))],
    "pocket_size": [("5-150", lambda d: d["system_proper_num_pocket_residues"].between(5, 150))],
    "ligand_chains": [("<=2", lambda d: d["n_ligand_chains"] <= 2),
                      ("<=4", lambda d: d["n_ligand_chains"] <= 4),
                      # strip-and-gate, from structure_scan.py (--scan): admit a
                      # system when no companion within 8 A of the ligand is
                      # bigger than an ion (1 heavy atom) / a small additive
                      # (6: sulfate, glycerol, acetate), so its void is a dimple
                      ("gate_ion8", lambda d: (d["n_ligand_chains"] == 1)
                       | (d["comp_maxha_8"] <= 1)),
                      ("gate_small8", lambda d: (d["n_ligand_chains"] == 1)
                       | (d["comp_maxha_8"] <= 6))],
    # small oligomers (short peptides, di/tri-saccharides): an oligo ligand is
    # admitted up to this many heavy atoms (~7.5 per peptide residue)
    "not_oligo": [("<=HA40", lambda d: ~d["ligand_is_oligo"].fillna(False).astype(bool)
                   | (d["ligand_num_heavy_atoms"] <= 40)),
                  ("<=HA60", lambda d: ~d["ligand_is_oligo"].fillna(False).astype(bool)
                   | (d["ligand_num_heavy_atoms"] <= 60))],
}
# Scan-based variants of existing criteria, evaluable only with --scan.
LOOSER["mw"].append(
    # fragments below 200 Da only when buried in a real pocket
    ("buried", lambda d: d["ligand_molecular_weight"].between(200, 900)
     | (d["ligand_molecular_weight"].between(120, 200) & (d["burial45"] >= 0.9))))
LOOSER["resolution"].append(
    # past 3.5 A only with a complete, not unusually mobile pocket
    ("<=4.5q", lambda d: (d["entry_resolution"] <= 3.5)
     | ((d["entry_resolution"] <= 4.5) & (d["pocket_missing"] <= 2)
        & (d["pocket_b_ratio"] <= 1.5))))

# The cofactor cap: cofactor-as-target rows admitted, at most N train rows per
# ligand (CCD code), spread over as many targets as the cap allows. Shared
# with build_dataset.cap_rows so the funnel and the build agree row for row;
# a dataset spec writes the same dict under selection.cap.
COFACTOR_CAP = {"where": "ligand_is_cofactor == True", "by": "ligand_ccd_code",
                "spread": "system_pocket_UniProt", "splits": ["train"]}
# The same cap on EVERY ligand: no ligand is the positive for more than N train
# rows. The third run's hubs were mostly NOT flagged cofactors - ANP (439 rows /
# 266 targets), GNP, ACP, AGS (non-hydrolysable nucleotides, admitted with
# their Mg2+), NAG, GLC-GLC, GLU - so a cofactor-only cap misses them.
LIGAND_CAP = {"by": "ligand_ccd_code", "spread": "system_pocket_UniProt",
              "splits": ["train"]}
CAPS = (25, 50, 100)


def parse_change(tok, crit):
    """'name=drop' | 'name<=3.0' | 'not_cofactor=cap50' | 'all_ligands=cap50'
    -> (name, label, predicate or None for "drop", cap or None)."""
    tok = tok.strip()
    if tok.startswith("all_ligands=cap"):
        return "all_ligands", tok.split("=")[1], None, int(tok[len("all_ligands=cap"):])
    if tok.startswith("not_cofactor=cap"):
        return "not_cofactor", tok.split("=")[1], None, int(tok[len("not_cofactor=cap"):])
    if tok.endswith("=drop"):
        name = tok[:-5]
        if name not in crit:
            raise SystemExit(f"ladder: no criterion {name!r}; have {sorted(crit)}")
        return name, "drop", None, None
    for name, opts in LOOSER.items():
        if tok.startswith(name):
            label = tok[len(name):]
            for lab, fn in opts:
                if lab == label:
                    return name, lab, fn, None
            raise SystemExit(f"ladder: {name} has no variant {label!r}; "
                             f"have {[l for l, _ in opts]} or {name}=drop")
    raise SystemExit(f"ladder: cannot parse {tok!r} (want name=drop or e.g. resolution<=3.0)")


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------
def groups_of(d):
    """Vectorised probe_retrieval.group_of: UniProt, else 95% component, else system."""
    import pandas as pd

    up = d.get("system_pocket_UniProt", pd.Series(index=d.index, dtype=object))
    up = up.where(up.map(lambda v: isinstance(v, str) and bool(v.strip())))
    comp = d.get("pli_unique_qcov__95__strong__component",
                 pd.Series(index=d.index, dtype=object))
    comp_ok = comp.map(lambda v: v is not None and str(v) not in ("", "nan", "None"))
    out = "sys:" + d["system_id"].astype(str)
    out = out.where(~comp_ok, "pli:" + comp.astype(str))
    return out.where(up.isna(), "up:" + up.astype(str).str.strip())


def prepare_frame(df):
    """Derived columns once: n_ligand_chains, entry, group, input_id."""
    from build_dataset import add_derived
    from receptor_inputs import plinder_receptor

    df = add_derived(df.copy())                 # n_ligand_chains, entry
    df["group"] = groups_of(df)
    df["input_id"] = plinder_receptor(df, {}, Path("."))["input_id"]
    if "strict_rep" not in df.columns:
        df["strict_rep"] = False
    return df


def dedup(d, key, per_entry=False):
    """One row per key, preferring baseline rows; optional (entry, ligand) pass."""
    order = d.assign(_rep=~d["strict_rep"].fillna(False).astype(bool),
                     _res=d["entry_resolution"].fillna(99.0))
    order = order.sort_values(["_rep", "_res", "system_id"], kind="mergesort")
    key = [k for k in key if k in d.columns]
    out = order.drop_duplicates(key, keep="first") if key else order
    if per_entry:
        out = out.drop_duplicates([k for k in ENTRY_KEY if k in out.columns], keep="first")
    return out.drop(columns=["_rep", "_res"])


def select(df, crit, per_entry=False, cap=None, lig_cap=None):
    """Rows passing every criterion in `crit` (name -> predicate), deduplicated,
    then the cofactor cap (`cap` rows per cofactor ligand) and the cap on every
    ligand (`lig_cap`), each if given."""
    from build_dataset import cap_rows

    m = np.ones(len(df), bool)
    for fn in crit.values():
        m &= fn(df).fillna(False).to_numpy(bool)
    out = dedup(df[m], DEDUP_KEY, per_entry)
    if cap:
        out = cap_rows(out, max=cap, **COFACTOR_CAP)
    if lig_cap:
        out = cap_rows(out, max=lig_cap, **LIGAND_CAP)
    return out


def _true(d):
    import pandas as pd
    return pd.Series(True, index=d.index)


def stats(sel, base=None):
    tr = sel[sel["split"] == "train"]
    g = tr["group"].value_counts()
    s = {"rows": len(tr), "groups": len(g), "inputs": tr["input_id"].nunique(),
         "ligs": tr["ligand_ccd_code"].nunique(),
         "val": int((sel["split"] == "val").sum()),
         "test": int((sel["split"] == "test").sum()),
         "visits": STEPS * BATCH / max(len(g), 1),
         "top": g.head(5)}
    if base is not None:
        btr = base[base["split"] == "train"]
        bg, bi = set(btr["group"]), set(btr["input_id"])
        new = ~tr["group"].isin(bg)
        s.update({"+groups": int(tr.loc[new, "group"].nunique()),
                  "+rows/new": int(new.sum()),
                  "+inputs": int((~tr["input_id"].drop_duplicates().isin(bi)).sum()),
                  "superset": bool(base.index.isin(sel.index).all())})
    return s


HEAD = (f"{'variant':30s} {'rows':>8s} {'groups':>7s} {'ligs':>7s} {'+groups':>8s} "
        f"{'+rows/new':>10s} {'+inputs':>8s} {'val':>6s} {'test':>6s} {'visits':>7s}")


def line(name, s):
    return (f"{name[:30]:30s} {s['rows']:>8,} {s['groups']:>7,} {s['ligs']:>7,} "
            f"{s.get('+groups', 0):>8,} "
            f"{s.get('+rows/new', 0):>10,} {s.get('+inputs', 0):>8,} "
            f"{s['val']:>6,} {s['test']:>6,} {s['visits']:>7.1f}")


# ---------------------------------------------------------------------------
# Ordered from least to most change to what a training example IS: quality,
# then ligand classes. The scan ladder swaps in the measured gates: resolution
# past 3.5 A only with a complete pocket, fragments only when buried, and
# companions up to an additive's size (every companion sits within ~4 A of
# the ligand, so size is the only gate that means anything).
# The cap goes last, on EVERY ligand (it subsumes the cofactor cap), so its
# step shows what capping the hubs costs in targets.
LADDER = ("no_altlocs=drop,resolution<=3.5,mw150-900,not_covalent=drop,"
          "not_oligo<=HA60,not_cofactor=drop,all_ligands=cap50")
LADDER_SCAN = ("no_altlocs=drop,resolution<=4.5q,mwburied,not_covalent=drop,"
               "not_oligo<=HA60,ligand_chainsgate_small8,not_cofactor=drop,"
               "all_ligands=cap50")


def report(a):
    import pandas as pd

    if a.ladder is None:
        a.ladder = LADDER_SCAN if a.scan else LADDER

    from build_dataset import join_tables

    df = pd.read_parquet(a.manifest)
    if a.scan:
        df = join_tables(df, [a.scan])
        print(f"joined {a.scan}: {int(df['scan_status'].eq('ok').sum()):,} rows "
              f"scanned ok of {len(df):,}")
    df = prepare_frame(df)
    crit = v1_criteria()
    missing = [n for n, fn in crit.items() if not _works(fn, df)]
    for n in missing:
        print(f"[warn] criterion {n} cannot be evaluated on this manifest; dropped")
        crit.pop(n)
    print(f"{a.manifest}: {len(df):,} rows, {df['receptor_key'].nunique():,} receptors, "
          f"{df.loc[df['split'] == 'train', 'group'].nunique():,} train groups in the "
          "whole (loose) manifest - the ceiling for any selection\n")

    base = select(df, crit)
    bs = stats(base)
    v1 = df[df["strict_rep"].fillna(False).astype(bool) & (df["n_ligand_chains"] == 1)]
    print(f"baseline (dataset_v1's criteria, dedup preferring strict_rep): "
          f"{len(base):,} rows; dataset_v1 itself (strict_rep & n_ligand_chains == 1) "
          f"{len(v1):,}. The difference is cluster keys whose strict_rep row had "
          "companions,\nwhere the baseline admits a sibling instead. v1 subset of "
          f"baseline: {bool(v1.index.isin(base.index).all())}\n")
    v1_line = stats(base)
    if a.per_entry:
        base = select(df, crit, per_entry=True)
        bs = stats(base)
        print(f"per-entry dedup on {ENTRY_KEY}: train rows {v1_line['rows']:,} -> "
              f"{bs['rows']:,}, train groups {v1_line['groups']:,} -> {bs['groups']:,}. "
              "Every line below is per-entry deduplicated and counted against "
              "this smaller baseline.")
        tr0 = select(df, crit)
        lost = set(tr0.loc[tr0["split"] == "train", "group"]) - set(
            base.loc[base["split"] == "train", "group"])
        kinds = {}
        for g in lost:
            kinds[g.split(":")[0]] = kinds.get(g.split(":")[0], 0) + 1
        print(f"  the {len(lost):,} train groups it removes, by kind: {kinds} "
              "(pli: groups are rows without a UniProt, one per interaction "
              "component, so symmetry copies of one entry can count as several)\n")

    # Attrition, in build_manifest's order, so the biggest single losses are visible.
    print("strict attrition on the manifest's train/val/test rows:")
    m = np.ones(len(df), bool)
    for n, fn in crit.items():
        k = fn(df).fillna(False).to_numpy(bool)
        print(f"  {n:20s} fails {int((~k).sum()):>9,}   left {int((m & k).sum()):>9,}")
        m &= k
    print(f"  {'dedup':20s}                 left {len(base):>9,}\n")

    cofactor_census(df, crit, a.per_entry)

    variants = []
    for n in crit:
        variants.append((f"{n}=drop", {**crit, n: _true}, None))
        for lab, fn in LOOSER.get(n, []):
            if _works(fn, df):          # scan variants need --scan
                variants.append((f"{n}{lab}", {**crit, n: fn}, None))
    if "not_cofactor" in crit:
        for k in CAPS:
            variants.append((f"not_cofactor=cap{k}", {**crit, "not_cofactor": _true}, k))
    rows = [(name, stats(select(df, c, a.per_entry, cap), base))
            for name, c, cap in variants]
    # the all-ligand cap on its own: how much of the BASELINE is hub ligands
    rows += [(f"all_ligands=cap{k}", stats(select(df, crit, a.per_entry, lig_cap=k), base))
             for k in CAPS]
    rows.sort(key=lambda t: -t[1]["+groups"])
    print("single-criterion variants, sorted by NEW train target groups:")
    print(HEAD)
    if a.per_entry:
        print(line("dataset_v1 rules", v1_line))
    print(line("baseline", bs))
    for name, s in rows:
        print(line(name, s))

    if a.ladder:
        print(f"\ncumulative ladder{' (+ per-entry dedup)' if a.per_entry else ''}:")
        print(HEAD)
        if a.per_entry:
            print(line("dataset_v1 rules", v1_line))
        print(line("baseline", bs))
        cur, cap, lig_cap = dict(crit), None, None
        last = None
        for tok in a.ladder.split(","):
            name, lab, fn, k = parse_change(tok, crit)
            if name == "all_ligands":
                lig_cap = k
            else:
                cur[name] = fn or _true
                cap = k if name == "not_cofactor" else cap
            prev = last if last is not None else base
            last = select(df, cur, a.per_entry, cap, lig_cap)
            print(line("+ " + tok.strip(), stats(last, base)))
            print("    " + added_ligands(prev, last))
        s = stats(last, base)
        gb = s["+inputs"] * (SURFACE_MB + POCKET_MB) / 1024
        print(f"\nfinal step: +{s['+inputs']:,} receptor inputs to sample + fpocket "
              f"(~{gb:.1f} GB on top of the existing 6.1 GB); baseline kept "
              f"whole: {s['superset']}")
        print("largest train groups at the final step (rows):")
        for g, c in s["top"].items():
            print(f"  {c:>6,}  {g}")
        tr = last[last["split"] == "train"]
        lig = tr.groupby("ligand_ccd_code").agg(rows=("system_id", "size"),
                                               groups=("group", "nunique"))
        print("most frequent train ligands at the final step (rows / groups):")
        for c, r in lig.sort_values("rows", ascending=False).head(10).iterrows():
            print(f"  {r['rows']:>6,} / {r['groups']:>5,}  {c}")
        if a.out:
            out = Path(a.out)
            last.to_parquet(out) if out.suffix == ".parquet" else last[
                ["system_id", "split", "group", "input_id"]].to_csv(out, sep="\t", index=False)
            print(f"\nwrote the final selection to {out}")


FLAG_COLS = (("covalent", "ligand_is_covalent"), ("oligo", "ligand_is_oligo"),
             ("cofactor", "ligand_is_cofactor"), ("glycan", "ligand_is_glycan"))


def added_ligands(prev, cur, k=6):
    """What a ladder step adds: its most frequent new train ligands and the
    share of added rows carrying each ligand-class flag."""
    tr = cur[(cur["split"] == "train") & ~cur.index.isin(prev.index)]
    if not len(tr):
        return "adds no train rows"
    top = tr["ligand_ccd_code"].value_counts().head(k)
    flags = ", ".join(f"{n} {tr[c].fillna(False).astype(bool).mean():.0%}"
                      for n, c in FLAG_COLS if c in tr.columns)
    return (f"adds {len(tr):,} train rows: " + ", ".join(f"{c} {v}" for c, v in top.items())
            + f"  [{flags}]")


def cofactor_census(df, crit, per_entry):
    """What not_cofactor=drop would admit, by ligand: the case for a cap."""
    if "not_cofactor" not in crit:
        return
    add = select(df, {**crit, "not_cofactor": _true}, per_entry)
    add = add[(add["split"] == "train")
              & add["ligand_is_cofactor"].fillna(False).astype(bool)]
    if not len(add):
        return
    lig = (add.groupby("ligand_ccd_code")
           .agg(rows=("system_id", "size"), groups=("group", "nunique"))
           .sort_values("rows", ascending=False))
    cum = lig["rows"].cumsum() / len(add)
    print(f"cofactor census: not_cofactor=drop admits {len(add):,} cofactor-target "
          f"train rows over {lig.shape[0]:,} distinct ligands and "
          f"{add['group'].nunique():,} groups. Top ligands (rows / groups / "
          "cumulative share of rows):")
    for c, r in lig.head(25).iterrows():
        print(f"  {r['rows']:>6,} / {r['groups']:>5,} / {cum[c]:5.1%}  {c}")
    for k in CAPS:
        kept = lig["rows"].clip(upper=k).sum()
        print(f"  cap {k:>3d}: keeps {kept:,} of {len(add):,} rows")
    print()


def _works(fn, df):
    try:
        fn(df.head(5))
        return True
    except (KeyError, AttributeError):
        return False


# ---------------------------------------------------------------------------
def selftest():
    import pandas as pd

    from probe_retrieval import group_of

    fails = []

    def check(name, ok, detail=""):
        print(f"  {'ok  ' if ok else 'FAIL'} {name}" + (f"   {detail}" if detail else ""))
        if not ok:
            fails.append(name)

    def row(sid, up, comp, ccd, split="train", res=2.0, alts=1, unres=0, rep=False):
        return {"system_id": sid, "receptor_key": "__".join(sid.split("__")[:3]),
                "system_pocket_UniProt": up, "pli_unique_qcov__95__strong__component": comp,
                "ligand_ccd_code": ccd, "split": split, "entry_resolution": res,
                "system_pocket_validation_max_alt_count": alts,
                "system_pocket_validation_num_unresolved_heavy_atoms": unres,
                "ligand_molecular_weight": 400.0, "ligand_num_interactions": 10,
                "system_proper_num_ligand_chains": 1, "system_proper_num_pocket_residues": 20,
                "ligand_fraction_atoms_with_crystal_contacts": 0.0,
                "ligand_num_missing_pli_interface_residues": 0,
                "all_protein_chains_present": True, "ligand_is_cofactor": False,
                "ligand_is_oligo": False, "ligand_is_covalent": False,
                "ligand_positions_correct": True, "strict_rep": rep}

    rows = [
        # baseline: two targets, P1 with a symmetry copy (same entry, ligand)
        row("1abc__1__1.A__1.B", "P1", "c1", "LIG", rep=True),
        row("1abc__1__1.C__1.D", "P1", "c1b", "LIG", rep=True),
        row("2xyz__1__1.A__1.B", "P2", "c2", "AAA", split="val", rep=True),
        # a duplicate of P1's cluster+ligand at worse resolution: dedup must keep the rep
        row("0aaa__1__1.A__1.B", "P1", "c1", "LIG", res=2.4),
        # new target behind resolution 2.8
        row("3new__1__1.A__1.B", "P3", "c3", "BBB", res=2.8),
        # new target behind altlocs, plus more rows on P1 behind altlocs
        row("4alt__1__1.A__1.B", "P4", "c4", "CCC", alts=2),
        row("5alt__1__1.A__1.B", "P1", "c5", "DDD", alts=3),
        # new target with a companion chain (two ligand chains in the system id)
        row("6cmp__1__1.A__1.B_1.C", None, "c6", "EEE"),
        # same entry and ligand as 1abc but bound to a DIFFERENT protein
        row("1abc__1__1.E__1.F", "P9", "c9", "LIG", rep=True),
    ]
    df = prepare_frame(pd.DataFrame(rows))
    check("vectorised groups == probe_retrieval.group_of",
          list(df["group"]) == [group_of(r) for r in df.to_dict("records")],
          f"{list(df['group'])}")
    crit = v1_criteria()
    base = select(df, crit)
    check("baseline = the strict_rep, single-chain rows",
          sorted(base["system_id"]) == sorted(r["system_id"] for r in rows[:3] + rows[-1:]))
    bs = stats(base)
    check("baseline counts train rows / groups only", (bs["rows"], bs["groups"], bs["val"]) == (3, 2, 1))

    s = stats(select(df, {**crit, "resolution": LOOSER["resolution"][0][1]}), base)
    check("resolution<=3.0 adds P3 as one new group, not the 2.4 A duplicate",
          (s["+groups"], s["+rows/new"], s["rows"]) == (1, 1, 4) and s["superset"],
          f"{s['+groups']} / {s['+rows/new']} / {s['rows']}")
    s = stats(select(df, {**crit, "no_altlocs": LOOSER["no_altlocs"][0][1]}), base)
    check("no_altlocs<=2 admits 2 alts but not 3", (s["+groups"], s["rows"]) == (1, 4))
    drop = {**crit, "no_altlocs": lambda d: pd.Series(True, index=d.index)}
    s = stats(select(df, drop), base)
    check("no_altlocs=drop: +1 group, and the P1 row counts as rows but not groups",
          (s["+groups"], s["+rows/new"], s["rows"]) == (1, 1, 5))
    s = stats(select(df, {**crit, "ligand_chains": LOOSER["ligand_chains"][0][1]}), base)
    check("ligand_chains<=2 admits the companion system as a pli: group",
          s["+groups"] == 1 and "pli:c6" in set(select(df, {**crit, "ligand_chains":
                                                         LOOSER["ligand_chains"][0][1]})["group"]))
    pe = select(df, crit, per_entry=True)
    check("per-entry dedup folds the symmetry copy, keeping a strict_rep row, "
          "and keeps the same ligand on a different protein",
          len(pe) == 3 and pe["strict_rep"].all() and set(pe["group"]) == {"up:P1", "up:P2", "up:P9"})

    name, lab, fn, _ = parse_change("resolution<=3.0", crit)
    check("ladder tokens parse", (name, lab) == ("resolution", "<=3.0") and fn is not None
          and parse_change("no_altlocs=drop", crit)[2] is None
          and parse_change("not_cofactor=cap50", crit)[3] == 50
          and parse_change("not_oligo<=HA60", crit)[1] == "<=HA60")

    # cofactor cap: 6 ATP rows over 4 targets, cap 4 -> one per target first
    atp = pd.DataFrame([row(f"{i}atp__1__1.A__1.B", f"K{i % 4}", f"k{i}", "ATP")
                        for i in range(6)] + [row("9nad__1__1.A__1.B", "D1", "d1", "NAD")])
    atp["ligand_is_cofactor"] = True
    atp = prepare_frame(atp)
    open_ = {**crit, "not_cofactor": _true}
    full, capped = select(atp, open_), select(atp, open_, cap=4)
    ka = capped[capped["ligand_ccd_code"] == "ATP"]
    check("cofactor cap keeps <= cap rows per ligand, spread over distinct targets, "
          "and leaves other ligands alone",
          len(full) == 7 and len(ka) == 4 and ka["group"].nunique() == 4
          and (capped["ligand_ccd_code"] == "NAD").sum() == 1,
          f"{len(full)} -> {len(capped)}, ATP groups {ka['group'].nunique()}")
    oligo = prepare_frame(pd.DataFrame([
        {**row("1pep__1__1.A__1.B", "Q1", "q1", "PEP"), "ligand_is_oligo": True,
         "ligand_num_heavy_atoms": 35},
        {**row("2pep__1__1.A__1.B", "Q2", "q2", "BIG"), "ligand_is_oligo": True,
         "ligand_num_heavy_atoms": 55}]))
    n40 = len(select(oligo, {**crit, "not_oligo": LOOSER["not_oligo"][0][1]}))
    n60 = len(select(oligo, {**crit, "not_oligo": LOOSER["not_oligo"][1][1]}))
    check("small-oligomer variants admit by heavy atoms", (n40, n60) == (1, 2))
    sug = prepare_frame(pd.DataFrame([
        {**row("1gly__1__1.A__1.B", "G1", "g1", "NAG-NAG"), "ligand_is_covalent": True},
        {**row("2gly__1__1.A__1.B", "G2", "g2", "NAG"), "ligand_is_covalent": True},
        {**row("3mal__1__1.A__1.B", "G3", "g3", "GLC-GLC")},
        {**row("4cov__1__1.A__1.B", "G4", "g4", "XYZ"), "ligand_is_covalent": True}]))
    got = set(select(sug, {**crit, "not_covalent": _true})["ligand_ccd_code"])
    hub = prepare_frame(pd.DataFrame(
        [row(f"{i}anp__1__1.A__1.B", f"K{i % 5}", f"a{i}", "ANP") for i in range(8)]
        + [row("9det__1__1.A__1.B", "D9", "d9", "DMU"), row("8drg__1__1.A__1.B", "D8", "d8", "XYZ")]))
    hc = select(hub, crit, lig_cap=5)
    ka = hc[hc["ligand_ccd_code"] == "ANP"]
    check("all-ligand cap: a non-cofactor hub (ANP) is capped, one row per target first; "
          "the detergent is out, a drug-like ligand stays",
          len(ka) == 5 and ka["group"].nunique() == 5
          and "DMU" not in set(hc["ligand_ccd_code"]) and "XYZ" in set(hc["ligand_ccd_code"])
          and parse_change("all_ligands=cap50", crit)[::3] == ("all_ligands", 50),
          f"ANP {len(ka)} rows / {ka['group'].nunique()} groups")
    check("covalent sugars (glycosylation) stay out when covalent ligands are admitted; "
          "a non-covalent sugar and a covalent drug-like ligand come in",
          got == {"GLC-GLC", "XYZ"}, f"{sorted(got)}")

    print("\nPASS" if not fails else f"\nFAIL: {fails}")
    raise SystemExit(1 if fails else 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("report")
    p.add_argument("--manifest", default="data/meta/manifest.parquet")
    # Ordered from least to most change to what a training example IS: quality
    # filters first, then ligand classes the first run showed to be the
    # biggest sources of new targets (not_cofactor +1,888 groups, mw +1,246).
    p.add_argument("--ladder", default=None,
                   help="default: LADDER, or LADDER_SCAN when --scan is given. "
                        "Comma-separated cumulative changes, e.g. "
                        "'no_altlocs=drop,resolution<=3.0'; '' to skip")
    p.add_argument("--scan", help="structure_scan.py merge output, joined on "
                   "(system_id, ligand_id); enables the gate / buried / q variants")
    p.add_argument("--per_entry", action="store_true",
                   help="also dedup on (PDB entry, ligand): the symmetry-copy fix")
    p.add_argument("--out", help="write the final ladder selection (.parquet or .tsv)")
    sub.add_parser("selftest")
    a = ap.parse_args()
    selftest() if a.cmd == "selftest" else report(a)


if __name__ == "__main__":
    main()
