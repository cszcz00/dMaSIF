"""Paired comparison of two checkpoints on the same validation rows.

Why. Window means of in-training evaluations say whether two runs differ "in
every window", but the windows come from the same two runs and are not
independent, and one evaluation of 400 rows moves by ~+-0.015 MRR. Scoring
both checkpoints on the SAME rows and comparing row by row removes the
between-row variation, which is most of that noise: the question becomes
"on a given ligand, does B rank its target higher than A does?".

Input: two `train.py eval --per_row FILE` outputs (one line per scored row:
row, system_id, group, rank, n_cand). Rows are paired by `row`; a row only one
file scored (no usable anchor in the other run) is dropped and counted.

Output, for B - A:
    dMRR      mean of 1/rank_B - 1/rank_A, with a 95% interval
    dtop1 / dtop10   the same for the hit rates
    P(d <= 0) the share of bootstrap resamples with no improvement
    wins / ties / losses   rows where B ranks the target higher / equal / lower

The bootstrap resamples TARGET GROUPS, not rows: rows of one group share a
protein (or homologues), so they are not independent, and resampling rows
would make the interval too narrow. 240 groups on the fixed val set.

Example (CPU venv or container; numpy only):
    python src/paired_eval.py A.tsv B.tsv
    python src/paired_eval.py selftest

How to produce the inputs (GPU, eval.sbatch, same rows for both):
    CKPTS=".../v4/last.pt" EVAL_ARGS="--val_set $WORK/out/valset_v1v2 \\
        --pocket_select site --per_row $WORK/out/per_row/v4_last.tsv" sbatch ...

Depends on: numpy.
"""

import argparse
import sys

import numpy as np


def read(path):
    """-> dict row -> (group, rank, n_cand), and the file's comment line."""
    rows, note, cols = {}, "", None
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith("#"):
                note = line[1:].strip()
                continue
            parts = line.split("\t")
            if cols is None:
                cols = parts
                continue
            d = dict(zip(cols, parts))
            rows[int(d["row"])] = (d["group"], int(d["rank"]), int(d["n_cand"]))
    return rows, note


def paired(a, b, n_boot=10000, seed=0):
    """Two read() dicts -> summary dict of B - A."""
    common = sorted(set(a) & set(b))
    if not common:
        raise ValueError("no rows in common")
    ga = np.asarray([a[r][0] for r in common])
    ra = np.asarray([a[r][1] for r in common], float)
    rb = np.asarray([b[r][1] for r in common], float)
    metrics = {"mrr": (1 / rb) - (1 / ra),
               "top1": (rb <= 1).astype(float) - (ra <= 1),
               "top10": (rb <= 10).astype(float) - (ra <= 10)}
    groups, gid = np.unique(ga, return_inverse=True)
    G = len(groups)
    # per-group sums and counts, so one resample is a weighted sum over groups
    cnt = np.bincount(gid, minlength=G).astype(float)
    sums = {k: np.bincount(gid, weights=v, minlength=G) for k, v in metrics.items()}
    rng = np.random.default_rng(seed)
    w = rng.multinomial(G, np.full(G, 1.0 / G), size=n_boot).astype(float)  # (n_boot, G)
    out = {"n": len(common), "only_a": len(set(a) - set(b)),
           "only_b": len(set(b) - set(a)), "groups": G,
           "mrr_a": float((1 / ra).mean()), "mrr_b": float((1 / rb).mean()),
           "wins": int((rb < ra).sum()), "ties": int((rb == ra).sum()),
           "losses": int((rb > ra).sum())}
    for k, v in metrics.items():
        boot = (w @ sums[k]) / (w @ cnt)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        out[f"d{k}"] = float(v.mean())
        out[f"d{k}_ci"] = (float(lo), float(hi))
        out[f"d{k}_p_le0"] = float((boot <= 0).mean())
    return out


def report(s, name_a="A", name_b="B"):
    print(f"{s['n']} rows paired over {s['groups']} target groups "
          f"({s['only_a']} only in {name_a}, {s['only_b']} only in {name_b}, dropped)")
    print(f"MRR  {name_a} {s['mrr_a']:.4f}   {name_b} {s['mrr_b']:.4f}")
    for k in ("mrr", "top1", "top10"):
        lo, hi = s[f"d{k}_ci"]
        print(f"  d{k:6s} {s[f'd{k}']:+.4f}   95% CI [{lo:+.4f}, {hi:+.4f}]   "
              f"P(d <= 0) {s[f'd{k}_p_le0']:.3f}")
    print(f"  rows: {name_b} better {s['wins']}, tied {s['ties']}, worse {s['losses']}")


def selftest():
    fails = []

    def check(label, ok, detail=""):
        print(f"  {'ok  ' if ok else 'FAIL'} {label}" + (f"   {detail}" if detail else ""))
        if not ok:
            fails.append(label)

    rng = np.random.default_rng(0)
    n, G = 400, 240
    grp = rng.integers(0, G, n)
    base = rng.integers(1, 200, n)
    a = {i: (f"g{grp[i]}", int(base[i]), 398) for i in range(n)}
    same = paired(a, dict(a), n_boot=2000)
    check("identical files: d = 0 exactly, interval [0, 0]",
          same["dmrr"] == 0 and same["dmrr_ci"] == (0.0, 0.0) and same["ties"] == n)
    # B halves every rank: a large, certain improvement
    b = {i: (g, max(1, r // 2), c) for i, (g, r, c) in a.items()}
    s = paired(a, b, n_boot=2000)
    check("better B: dMRR > 0, interval above 0, P(d <= 0) = 0",
          s["dmrr"] > 0 and s["dmrr_ci"][0] > 0 and s["dmrr_p_le0"] == 0)
    # noise only: B = A with ranks jittered symmetrically -> interval covers 0
    jit = {i: (g, max(1, r + int(rng.integers(-3, 4))), c) for i, (g, r, c) in a.items()}
    z = paired(a, jit, n_boot=4000)
    check("symmetric noise: the interval covers 0",
          z["dmrr_ci"][0] <= 0 <= z["dmrr_ci"][1], f"{z['dmrr_ci']}")
    # unpaired rows are dropped and counted
    b2 = dict(b)
    del b2[0], b2[1]
    b2[9999] = ("g0", 1, 398)
    s2 = paired(a, b2, n_boot=500)
    check("rows in one file only are dropped and counted",
          s2["n"] == n - 2 and s2["only_a"] == 2 and s2["only_b"] == 1)
    # group bootstrap is wider than a row bootstrap when groups are correlated
    gb = {i: (f"g{grp[i] % 20}", max(1, a[i][1] - (30 if grp[i] % 20 < 10 else -30)), 398)
          for i in range(n)}
    ag = {i: (f"g{grp[i] % 20}", a[i][1], 398) for i in range(n)}
    sg = paired(ag, gb, n_boot=3000)
    rows_only = {i: (f"r{i}", *gb[i][1:]) for i in range(n)}
    sr = paired({i: (f"r{i}", *ag[i][1:]) for i in range(n)}, rows_only, n_boot=3000)
    wg = sg["dmrr_ci"][1] - sg["dmrr_ci"][0]
    wr = sr["dmrr_ci"][1] - sr["dmrr_ci"][0]
    check("resampling groups gives a wider interval than resampling rows "
          "when the effect is per group", wg > wr, f"{wg:.4f} vs {wr:.4f}")
    print("\nPASS" if not fails else f"\nFAIL: {fails}")
    return 1 if fails else 0


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "selftest":
        sys.exit(selftest())
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a", help="per-row TSV of the baseline")
    ap.add_argument("b", help="per-row TSV of the candidate")
    ap.add_argument("--n_boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    ra, na = read(a.a)
    rb, nb = read(a.b)
    print(f"A: {a.a}" + (f"  ({na})" if na else ""))
    print(f"B: {a.b}" + (f"  ({nb})" if nb else ""))
    report(paired(ra, rb, a.n_boot, a.seed))


if __name__ == "__main__":
    main()
