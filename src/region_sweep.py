"""Compare pocket-region selectors (anchors.py) on saved probability maps.

Region selection has no gradient and no learned parameters, so all three
options can be scored against the SAME trained pocket head: train once with

    test_encoder.py overfit ... --dump maps.pt

then run every selector setting here over those maps. CPU only.

    watershed   option 1: watershed + persistence merging. persistence 0 is
                the current selector (the row marked *); persistence >=
                top_fraction merges everything connected
    components  option 2: P2Rank-style single-linkage clusters, ranked by
                sum p^2
    ball        option 3: greedy NMS balls of fixed radius

Two oracle blocks come first, independent of any selector:
    pocket      the labelled pockets: points, share of the surface, extent
                (max pairwise distance), radius of gyration, and how many
                pieces they fall into at 2 / 3 A linkage. A label already in
                pieces cannot be covered by one linked region
    candidates  the union of all candidate points (top fraction after
                smoothing). Its cover is the ceiling for watershed and
                components, which only ever group candidates

Per selector setting, over each protein's top --k_pos predicted regions:
    #reg, size    regions per protein, mean points per region
    iou, cover, prec   of the region that best matches the pocket (by IoU)
    wcov, wprec   the same weighted by raw pocket probability, i.e. as
                  probability-weighted pooling sees that region
    hit, top1     some region / the top-ranked region covers >= 0.5 of the
                  pocket at >= 0.3 precision (as in test_encoder.py)
    iou1          IoU of the top-ranked region
    dcc1, dccA    top-ranked / any region's probability-weighted centre within
                  4 A of the ligand centroid (P2Rank's DCC criterion)
    ms            selector time per protein, excluding neighbour lists and
                  smoothing (shared by every setting, built once per protein)

Example (container, or any env with torch + numpy):
    python src/region_sweep.py --maps encoder_test/maps.pt \\
        --out encoder_test/region_sweep.tsv --per_item encoder_test/region_items.tsv
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from anchors import (REGION_METHODS, _neighbors, candidates, pocket_regions,  # noqa: E402
                     smoothed, watershed_regions)

METRICS = ["n", "size", "iou", "cover", "prec", "wcov", "wprec",
           "hit", "top1", "iou1", "dcc1", "dccA", "ms"]
# the encoder's current selector (EncoderConfig defaults), marked * in tables
BASELINE = dict(method="watershed", top_fraction=0.05, smooth_radius=2.0,
                link_radius=2.0, persistence=0.0)


# ----------------------------------------------------------------------------
# settings
# ----------------------------------------------------------------------------
def grid(methods):
    out = []
    if "watershed" in methods:
        for sm in (2.0, 4.0):
            for link in (2.0, 3.0):
                for pers in (0.0, 0.0025, 0.005, 0.01, 0.02, 0.05):
                    out.append(dict(method="watershed", top_fraction=0.05, smooth_radius=sm,
                                    link_radius=link, persistence=pers))
    if "components" in methods:
        for tf in (0.03, 0.05, 0.08):
            for sm in (0.0, 2.0, 4.0):
                for link in (2.0, 3.0, 4.0):
                    out.append(dict(method="components", top_fraction=tf, smooth_radius=sm,
                                    link_radius=link))
    if "ball" in methods:
        for sm in (2.0, 4.0, 6.0):
            for rad in (6.0, 8.0, 10.0, 12.0):
                for shift in (0, 2):
                    out.append(dict(method="ball", top_fraction=0.05, smooth_radius=sm,
                                    ball_radius=rad, ball_shift=shift))
    return out


SHORT = {"top_fraction": "tf", "smooth_radius": "sm", "link_radius": "link",
         "persistence": "pers", "ball_radius": "R", "ball_shift": "shift"}


def name(s):
    star = "*" if s == BASELINE else " "
    return star + s["method"] + " " + " ".join(f"{SHORT[k]}={v:g}" for k, v in s.items()
                                               if k != "method")


# ----------------------------------------------------------------------------
# oracles
# ----------------------------------------------------------------------------
def pocket_oracle(it):
    xyz = it["xyz"][it["pocket"]]
    ones = torch.ones(len(xyz))
    d = torch.cdist(xyz, xyz)
    row = {"points": len(xyz), "share": len(xyz) / len(it["xyz"]), "extent": float(d.max()),
           "rg": float(((xyz - xyz.mean(0)) ** 2).sum(-1).mean().sqrt())}
    for link in (2.0, 3.0):
        pieces = watershed_regions(xyz, ones, 0.5, link, 1, 0.0, persistence=float("inf"))
        row[f"pieces@{link:g}"] = len(pieces)
        row[f"largest@{link:g}"] = max(len(p) for p in pieces) / len(xyz)
    return row


def candidate_oracle(it, cache):
    pocket = it["pocket"]
    row = {}
    for sm in (0.0, 2.0, 4.0, 6.0):
        p = smoothed(it["xyz"], it["prob"], sm, cache)
        for tf in (0.03, 0.05, 0.08, 0.10):
            m = candidates(p, None, tf)
            inter = float((m & pocket).sum())
            row[(sm, tf)] = (inter / max(int(pocket.sum()), 1), inter / max(int(m.sum()), 1))
    return row


# ----------------------------------------------------------------------------
# scoring
# ----------------------------------------------------------------------------
def score(regions, it, a):
    xyz, prob, pocket = it["xyz"], it["prob"], it["pocket"]
    regs = regions[:a.k_pos]
    st = dict.fromkeys(METRICS, 0.0)
    st.update(n=len(regs), size=np.nan)
    if not regs:
        return st
    n_pocket = float(pocket.sum())
    sizes = torch.tensor([len(r) for r in regs], dtype=torch.float)
    inter = torch.tensor([float(pocket[r].sum()) for r in regs])
    iou = inter / (sizes + n_pocket - inter)
    cover, prec = inter / n_pocket, inter / sizes
    w_in = torch.tensor([float(prob[r][pocket[r]].sum()) for r in regs])
    w_all = torch.tensor([float(prob[r].sum()) for r in regs])
    ok = (cover >= a.min_cover) & (prec >= a.min_prec)
    centre = torch.stack([(prob[r][:, None] * xyz[r]).sum(0) / (prob[r].sum() + 1e-12)
                          for r in regs])
    dcc = (centre - it["lig"].mean(0)).norm(dim=1)
    b = int(torch.argmax(iou))
    st.update(size=float(sizes.mean()), iou=float(iou[b]), cover=float(cover[b]),
              prec=float(prec[b]), wcov=float(w_in[b] / prob[pocket].sum()),
              wprec=float(w_in[b] / (w_all[b] + 1e-12)), hit=float(ok.any()),
              top1=float(ok[0]), iou1=float(iou[0]), dcc1=float(dcc[0] < a.dcc),
              dccA=float((dcc < a.dcc).any()))
    return st


def warm(it, settings, cache):
    """Build every neighbour list and smoothed map the settings need, once."""
    for s in settings:
        smoothed(it["xyz"], it["prob"], s["smooth_radius"], cache)
        if s["method"] != "ball":
            _neighbors(it["xyz"], s["link_radius"], cache)


# ----------------------------------------------------------------------------
# reporting
# ----------------------------------------------------------------------------
def fmt_row(label, vals):
    return (f"{label:44s} {vals['n']:5.1f} {vals['size']:6.0f} {vals['iou']:5.2f} "
            f"{vals['cover']:5.2f} {vals['prec']:5.2f} {vals['wcov']:5.2f} {vals['wprec']:5.2f} "
            f"{vals['hit']:5.2f} {vals['top1']:5.2f} {vals['iou1']:5.2f} {vals['dcc1']:5.2f} "
            f"{vals['dccA']:5.2f} {vals['ms']:6.0f}")


HEADER = (f"{'setting':44s} {'#reg':>5s} {'size':>6s} {'iou':>5s} {'cover':>5s} {'prec':>5s} "
          f"{'wcov':>5s} {'wprec':>5s} {'hit':>5s} {'top1':>5s} {'iou1':>5s} {'dcc1':>5s} "
          f"{'dccA':>5s} {'ms':>6s}")


def mean_stats(rows):
    return {m: float(np.nanmean([r[m] for r in rows])) if rows else np.nan for m in METRICS}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--maps", required=True, help="from test_encoder.py overfit --dump")
    ap.add_argument("--out", default="", help="TSV of per-setting means per split")
    ap.add_argument("--per_item", default="", help="TSV of every item x setting")
    ap.add_argument("--methods", default=",".join(REGION_METHODS))
    ap.add_argument("--splits", default="held,train")
    ap.add_argument("--k_pos", type=int, default=8)
    ap.add_argument("--min_points", type=int, default=20)
    ap.add_argument("--min_cover", type=float, default=0.5)
    ap.add_argument("--min_prec", type=float, default=0.3)
    ap.add_argument("--dcc", type=float, default=4.0, help="A; DCC success cutoff")
    ap.add_argument("--select_on", default="held",
                    help="split whose mean IoU picks each method's best setting")
    a = ap.parse_args()

    blob = torch.load(a.maps, map_location="cpu", weights_only=False)
    splits = a.splits.split(",")
    items = [it for it in blob["maps"] if it["split"] in splits and it["pocket"].any()]
    settings = grid(a.methods.split(","))
    print(f"{a.maps}: {len(items)} items "
          + ", ".join(f"{s} {sum(it['split'] == s for it in items)}" for s in splits)
          + f"; {len(settings)} settings; head trained with "
          + ("pretrained start" if blob["args"].get("pretrained") else "fresh weights")
          + f", {blob['args'].get('steps')} steps")

    pocket_rows, cand_rows = {s: [] for s in splits}, {s: [] for s in splits}
    results = {s: [[] for _ in settings] for s in splits}  # split -> setting -> per-item
    per_item = []
    t0 = time.time()
    for n, it in enumerate(items):
        cache = {}
        pocket_rows[it["split"]].append(pocket_oracle(it))
        cand_rows[it["split"]].append(candidate_oracle(it, cache))
        warm(it, settings, cache)
        for j, s in enumerate(settings):
            t = time.perf_counter()
            regions = pocket_regions(it["xyz"], it["prob"], min_points=a.min_points,
                                     cache=cache, **s)
            ms = 1e3 * (time.perf_counter() - t)
            st = score(regions, it, a)
            st["ms"] = ms
            results[it["split"]][j].append(st)
            per_item.append({"id": it["id"], "split": it["split"], "setting": name(s).strip(),
                             **st})
        print(f"  {n + 1}/{len(items)} {it['id']} ({len(it['xyz'])} points) "
              f"{time.time() - t0:.0f}s", flush=True)

    for sp in splits:
        if not pocket_rows[sp]:
            continue
        print(f"\n===== {sp}: {len(pocket_rows[sp])} items =====")
        pr = pocket_rows[sp]
        print("\nlabelled pocket (mean / median)")
        for k in pr[0]:
            v = np.array([r[k] for r in pr], dtype=float)
            print(f"  {k:12s} {v.mean():8.3f} {np.median(v):8.3f}")
        print("\ncandidate union: cover / prec (mean); cover is the ceiling for "
              "watershed and components")
        tfs = (0.03, 0.05, 0.08, 0.10)
        print(f"  {'smooth':>6s} " + " ".join(f"{'top ' + format(tf, '.0%'):>13s}" for tf in tfs))
        for sm in (0.0, 2.0, 4.0, 6.0):
            cells = []
            for tf in tfs:
                c = np.array([r[(sm, tf)] for r in cand_rows[sp]])
                cells.append(f"{c[:, 0].mean():5.2f} / {c[:, 1].mean():5.2f}")
            print(f"  {sm:6g} " + " ".join(f"{x:>13s}" for x in cells))

        print("\n" + HEADER)
        method = None
        for s, rows in zip(settings, results[sp]):
            if s["method"] != method:
                method = s["method"]
                print(f"--- {method}")
            print(fmt_row(name(s), mean_stats(rows)))

    sel = a.select_on if pocket_rows.get(a.select_on) else splits[0]
    print(f"\n===== best setting per method, by mean IoU on {sel} =====")
    print(f"{'':6s}" + HEADER)
    for m in a.methods.split(","):
        idx = [j for j, s in enumerate(settings) if s["method"] == m]
        if not idx:
            continue
        best = max(idx, key=lambda j: mean_stats(results[sel][j])["iou"])
        for sp in splits:
            if results[sp][best]:
                print(f"{sp:6s}" + fmt_row(name(settings[best]), mean_stats(results[sp][best])))
    if BASELINE in settings:
        j = settings.index(BASELINE)
        for sp in splits:
            if results[sp][j]:
                print(f"{sp:6s}" + fmt_row(name(BASELINE), mean_stats(results[sp][j])))

    if a.out:
        lines = ["\t".join(["split", "setting", "items"] + METRICS)]
        for sp in splits:
            for s, rows in zip(settings, results[sp]):
                if rows:
                    ms = mean_stats(rows)
                    lines.append("\t".join([sp, name(s).strip(), str(len(rows))]
                                           + [f"{ms[m]:.4f}" for m in METRICS]))
        Path(a.out).write_text("\n".join(lines) + "\n")
        print(f"\n{a.out}")
    if a.per_item:
        cols = ["id", "split", "setting"] + METRICS
        Path(a.per_item).write_text(
            "\t".join(cols) + "\n"
            + "".join("\t".join(str(r[c]) for c in cols) + "\n" for r in per_item))
        print(a.per_item)


if __name__ == "__main__":
    main()
