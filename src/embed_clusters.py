"""What organises the retrieval space? Clustering of pocket and ligand vectors.

Why. MRR says how well a checkpoint ranks; it does not say what the space it
ranks in is built around. A pocket vector could sit near other pockets of the
same protein FAMILY (resemblance, which a family-disjoint val set rewards only
by analogy), near pockets that bind the same KIND OF LIGAND (binding chemistry,
which is what retrieval needs), or near pockets of the same SIZE (the "size
matching" suspicion from the probe). These are different models with the same
MRR, and the neighbourhoods tell them apart.

Input: a `retrieval_report.py dump` (the train and val slices' anchors and
every ligand's vector, all in the 128-dim space the loss compares), plus the
items.tsv / ligand_feats.npz it was dumped from and, for protein families,
the manifest (system_pocket_Pfam). numpy only (CPU venv); rdkit is not needed.

Three analyses, each with a figure:

  proteins   one vector per receptor row (its site anchor). For each label -
             Pfam, the bound ligand's chemotype class, pocket size tertile,
             UniProt group - the share of each pocket's k nearest pockets that
             share it, against a label-permutation baseline (enrichment = x
             chance). Plus k-means on the val pockets and its NMI with each
             label, and for every val pocket its nearest TRAIN pocket: same
             family? same ligand class? (val targets are UniProt-disjoint
             from train, so this is how an unseen target is placed.)
  ligands    every distinct ligand's vector. Spearman between vector cosine
             and Tanimoto over random pairs; the same kNN agreement with the
             chemotype class and with a size tertile; which descriptors the
             top principal components follow.
  joint      pockets and ligands together. The modality gap (CLIP's: the two
             sets occupy offset cones), true-pair vs random-pair cosine, and
             class-level cross retrieval: how often a pocket's nearest
             ligands are of its bound ligand's class.

Ligand chemotype classes: Butina clustering of ECFP4 Tanimoto over every
distinct ligand (--class_sim, default 0.4: members are at least that similar
to the class centre). A pocket's class is its bound ligand's.

Figures: t-SNE (exact, numpy, on the cosine geometry) to 2D, written as
self-contained SVG - pockets coloured by Pfam and by ligand class; ligands by
class and by size; the joint set with each modality centred first (the raw
joint map is mostly the modality gap, which the printed numbers report).
Coordinates and labels also go to TSV.

Example:
    # GPU, once (src/slurm/report.sbatch defaults to C = v3c_site_pairs_bn/best.pt,
    # its own dataset_v1 val split, 400 val + 400 train receptors)
    SELECTS=site sbatch src/slurm/report.sbatch
    # CPU venv
    python src/embed_clusters.py report \\
        --dump out/reports/v3c_site_pairs_bn_site.npz \\
        --items out/runs/v1/items.tsv --lig_feats out/runs/v1/ligand_feats.npz \\
        --manifest data/meta/manifest.parquet --out out/clusters/v3c
    python src/embed_clusters.py selftest

Depends on: numpy; retrieval_report (load_ligand_arrays, tanimoto, numpy
only); pandas for --manifest only.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

PALETTE = ("#2f6db5", "#d1495b", "#3f9e6a", "#e0a030", "#7b4fa0", "#1fa3a8",
           "#c06a2b", "#8a8f2e", "#b03a8c", "#5b6b7a")
OTHER = "#c4c9d0"


# ---------------------------------------------------------------------------
# small numpy tools
# ---------------------------------------------------------------------------
def unit(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-9)


def knn_cos(X, k, Y=None):
    """k nearest rows of Y (default X, self excluded) by cosine -> (n, k)."""
    Xu = unit(X)
    Yu = Xu if Y is None else unit(Y)
    S = Xu @ Yu.T
    if Y is None:
        np.fill_diagonal(S, -np.inf)
    k = min(k, S.shape[1] - (Y is None))
    return np.argsort(-S, axis=1)[:, :k]


def knn_agreement(nbr, labels, n_perm=200, seed=0):
    """Share of neighbours carrying the query's label, rows with a label only;
    -> (observed, permutation mean, permutation sd). Missing label: None / ""."""
    lab = np.asarray([("" if v is None else str(v)) for v in labels], dtype=object)
    have = lab != ""
    if have.sum() < 3:
        return float("nan"), float("nan"), float("nan")

    def score(L):
        m = (L[nbr] == L[:, None]) & (L[nbr] != "")
        valid = (L[nbr] != "").sum(1)
        q = have & (valid > 0)
        return float((m.sum(1)[q] / valid[q]).mean())

    obs = score(lab)
    rng = np.random.default_rng(seed)
    perm = []
    for _ in range(n_perm):
        L = lab.copy()
        L[have] = rng.permutation(L[have])
        perm.append(score(L))
    return obs, float(np.mean(perm)), float(np.std(perm))


def nmi(a, b):
    """Normalised mutual information of two labelings (arithmetic mean norm)."""
    a, b = np.asarray(a), np.asarray(b)
    _, ai = np.unique(a, return_inverse=True)
    _, bi = np.unique(b, return_inverse=True)
    n = len(a)
    C = np.zeros((ai.max() + 1, bi.max() + 1))
    np.add.at(C, (ai, bi), 1)
    P = C / n
    pa, pb = P.sum(1), P.sum(0)
    nz = P > 0
    mi = (P[nz] * np.log(P[nz] / (pa[:, None] * pb[None, :])[nz])).sum()
    ha = -(pa[pa > 0] * np.log(pa[pa > 0])).sum()
    hb = -(pb[pb > 0] * np.log(pb[pb > 0])).sum()
    return float(2 * mi / (ha + hb)) if ha + hb > 0 else 0.0


def kmeans(X, k, n_init=5, iters=60, seed=0):
    """Spherical k-means (cosine), k-means++ init -> labels."""
    Xu = unit(X)
    rng = np.random.default_rng(seed)
    best, best_in = None, -np.inf
    for _ in range(n_init):
        c = [Xu[rng.integers(len(Xu))]]
        for _ in range(1, k):
            d = 1 - np.max(Xu @ np.stack(c).T, axis=1)
            p = np.maximum(d, 0) ** 2
            c.append(Xu[rng.choice(len(Xu), p=p / p.sum()) if p.sum() > 0
                        else rng.integers(len(Xu))])
        C = np.stack(c)
        for _ in range(iters):
            lab = np.argmax(Xu @ C.T, axis=1)
            newC = np.stack([unit(Xu[lab == j].sum(0, keepdims=True))[0]
                             if (lab == j).any() else C[j] for j in range(k)])
            if np.allclose(newC, C):
                break
            C = newC
        inertia = float((Xu * C[lab]).sum())
        if inertia > best_in:
            best, best_in = lab, inertia
    return best


def butina(fp, sim, tanimoto, chunk=1024):
    """Butina clustering of 0/1 fingerprints at Tanimoto >= sim -> class id
    per row (class 0 the largest)."""
    n = len(fp)
    nbrs = []
    for s in range(0, n, chunk):
        T = tanimoto(fp[s:s + chunk], fp)
        T = np.atleast_2d(T)
        for i in range(T.shape[0]):
            nbrs.append(np.nonzero(T[i] >= sim)[0])
    order = np.argsort([-len(x) for x in nbrs], kind="stable")
    lab = np.full(n, -1)
    c = 0
    for i in order:
        if lab[i] >= 0:
            continue
        members = [j for j in nbrs[i] if lab[j] < 0]
        lab[members] = c
        lab[i] = c
        c += 1
    return lab


def spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean()
    ry -= ry.mean()
    return float((rx * ry).sum() / np.sqrt((rx ** 2).sum() * (ry ** 2).sum()))


def tsne(X, perplexity=30.0, iters=750, seed=0):
    """Exact t-SNE on squared Euclidean distances of the unit vectors (i.e. the
    cosine geometry) -> (n, 2). Fine for ~1-2k points in numpy."""
    Xu = unit(X)
    n = len(Xu)
    D = np.maximum(2 - 2 * Xu @ Xu.T, 0)
    target = np.log(min(perplexity, (n - 1) / 3))
    P = np.zeros((n, n))
    for i in range(n):
        lo, hi, beta = 1e-10, 1e10, 1.0
        di = np.delete(D[i], i)
        for _ in range(60):
            p = np.exp(-di * beta)
            sp = p.sum()
            if sp == 0:
                beta /= 2
                continue
            h = np.log(sp) + beta * (di * p).sum() / sp
            if abs(h - target) < 1e-5:
                break
            if h > target:
                lo, beta = beta, beta * 2 if hi == 1e10 else (beta + hi) / 2
            else:
                hi, beta = beta, (beta + lo) / 2
        P[i, np.arange(n) != i] = p / max(sp, 1e-12)
    P = (P + P.T) / (2 * n)
    P = np.maximum(P, 1e-12)
    rng = np.random.default_rng(seed)
    Y = rng.normal(scale=1e-4, size=(n, 2))
    vel, gains = np.zeros_like(Y), np.ones_like(Y)
    for it in range(iters):
        ex = 12.0 if it < 250 else 1.0
        mom = 0.5 if it < 250 else 0.8
        sq = (Y ** 2).sum(1)
        num = 1 / (1 + sq[:, None] + sq[None, :] - 2 * Y @ Y.T)
        np.fill_diagonal(num, 0)
        Q = np.maximum(num / num.sum(), 1e-12)
        PQ = (ex * P - Q) * num
        grad = 4 * (np.diag(PQ.sum(1)) - PQ) @ Y
        gains = np.where(np.sign(grad) != np.sign(vel), gains + 0.2, gains * 0.8)
        gains = np.maximum(gains, 0.01)
        vel = mom * vel - 200.0 * gains * grad
        Y = Y + vel
        Y -= Y.mean(0)
    return Y


# ---------------------------------------------------------------------------
# SVG scatter (no plotting library)
# ---------------------------------------------------------------------------
def top_colours(labels, n=len(PALETTE)):
    """Most frequent n labels get palette colours, the rest grey."""
    labs = np.asarray([("" if v is None else str(v)) for v in labels], dtype=object)
    vals, cnt = np.unique(labs[labs != ""], return_counts=True)
    keep = [v for v, _ in sorted(zip(vals, cnt), key=lambda t: -t[1])][:n]
    cmap = {v: PALETTE[i] for i, v in enumerate(keep)}
    return [cmap.get(v, OTHER) for v in labs], cmap


def svg_panels(path, Y, panels, title, markers=None):
    """panels: list of (subtitle, colours, legend {label: colour}). One shared
    layout, side by side. markers: per point 'o' (circle) or '^' (triangle)."""
    W, H, pad, leg = 460, 460, 30, 150
    Ymin, Ymax = Y.min(0), Y.max(0)
    span = np.maximum(Ymax - Ymin, 1e-9)
    P = pad + (Y - Ymin) / span * (W - 2 * pad)
    P[:, 1] = H - P[:, 1]
    total_w = len(panels) * (W + leg)
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {total_w} {H + 50}" '
           f'font-family="Helvetica, Arial, sans-serif">',
           f'<rect width="{total_w}" height="{H + 50}" fill="#ffffff"/>',
           f'<text x="12" y="24" font-size="16" font-weight="700" fill="#1d2733">'
           f'{_esc(title)}</text>']
    for k, (sub, cols, legend) in enumerate(panels):
        ox = k * (W + leg)
        out.append(f'<g transform="translate({ox},40)">')
        out.append(f'<rect x="4" y="4" width="{W - 8}" height="{H - 8}" fill="none" '
                   f'stroke="#e1e5ea"/>')
        out.append(f'<text x="14" y="22" font-size="13" fill="#5b6775">{_esc(sub)}</text>')
        # grey first, so coloured points draw on top
        order = sorted(range(len(P)), key=lambda i: cols[i] != OTHER)
        for i in order:
            x, y = P[i]
            if markers is not None and markers[i] == "^":
                out.append(f'<path d="M{x:.1f},{y - 4:.1f} L{x + 3.5:.1f},{y + 2.5:.1f} '
                           f'L{x - 3.5:.1f},{y + 2.5:.1f} Z" fill="{cols[i]}" '
                           f'fill-opacity="0.8"/>')
            else:
                out.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="{cols[i]}" '
                           f'fill-opacity="0.8"/>')
        for j, (lab, c) in enumerate(legend.items()):
            ly = 40 + 18 * j
            out.append(f'<circle cx="{W + 10}" cy="{ly - 4}" r="5" fill="{c}"/>')
            out.append(f'<text x="{W + 20}" y="{ly}" font-size="11" fill="#1d2733">'
                       f'{_esc(str(lab)[:20])}</text>')
        if legend and OTHER in cols:
            ly = 40 + 18 * len(legend)
            out.append(f'<circle cx="{W + 10}" cy="{ly - 4}" r="5" fill="{OTHER}"/>')
            out.append(f'<text x="{W + 20}" y="{ly}" font-size="11" fill="#5b6775">'
                       f'other</text>')
        out.append("</g>")
    out.append("</svg>")
    Path(path).write_text("\n".join(out))


def _esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def load(a):
    from retrieval_report import load_ligand_arrays
    from train_data import ItemTable, read_items

    dump = {k: v for k, v in np.load(a.dump, allow_pickle=False).items()}
    table = ItemTable(read_items(a.items))
    ligands = load_ligand_arrays(a.lig_feats)
    # the same row filter the dump's training code applied
    bad = ligands.failed[table.fp_row]
    if bad.any():
        table = ItemTable([r for r, b in zip(table.rows, bad) if not b])
    pfam = {}
    if a.manifest and Path(a.manifest).exists():
        import pandas as pd
        m = pd.read_parquet(a.manifest, columns=["system_id", "system_pocket_Pfam"])
        m = m.dropna().drop_duplicates("system_id")
        pfam = dict(zip(m["system_id"].astype(str), m["system_pocket_Pfam"].astype(str)))
    elif a.manifest:
        print(f"[warn] no manifest at {a.manifest}: Pfam labels skipped")
    return dump, table, ligands, pfam


def pockets(dump, tag):
    """-> (rows, (n, D) site vectors) for one split: each row's first usable
    anchor, rows without one dropped."""
    an, us, rows = dump[f"{tag}_anchors"], dump[f"{tag}_usable"], dump[f"{tag}_rows"]
    keep = us.any(1)
    first = us.argmax(1)
    vec = an[np.arange(len(an)), first]
    return rows[keep], vec[keep]


def tertile(x):
    q = np.quantile(x, [1 / 3, 2 / 3])
    return np.asarray(["small", "medium", "large"], dtype=object)[np.searchsorted(q, x)]


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def line(name, obs, mu, sd):
    if not np.isfinite(obs):
        print(f"    {name:26s} (no labels)")
        return
    z = (obs - mu) / sd if sd and sd > 0 else float("nan")
    print(f"    {name:26s} {obs:6.3f}   chance {mu:6.3f}   x{obs / max(mu, 1e-9):5.2f}   "
          f"z {z:6.1f}")


def report(a):
    from retrieval_report import tanimoto

    dump, table, ligands, pfam = load(a)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    lig_vec = dump["lig_vec"]
    print(f"{a.dump}: step {int(dump['step'])}, pocket_select {str(dump['pocket_select'])}; "
          f"{len(lig_vec)} ligand vectors")

    # ligand chemotype classes over every distinct ligand in the table
    lig_ids = np.unique(table.fp_row)
    cls_of = np.full(len(lig_vec), -1)
    cls_of[lig_ids] = butina(ligands.fp[lig_ids], a.class_sim, tanimoto)
    n_cls = len(np.unique(cls_of[lig_ids]))
    sizes = np.bincount(cls_of[lig_ids])
    print(f"ligand classes: Butina at Tanimoto >= {a.class_sim}: {n_cls} classes over "
          f"{len(lig_ids)} ligands; largest {[int(x) for x in sorted(sizes)[-5:][::-1]]}, "
          f"{(sizes == 1).sum()} singletons")
    cls_name = {c: f"class {c} (n={sizes[c]})" for c in range(len(sizes))}

    # ---------------- proteins ----------------
    print("\n=== proteins (site anchors) ===")
    split = {}
    for tag in ("train", "val"):
        if f"{tag}_anchors" in dump:
            rows, vec = pockets(dump, tag)
            r = [table.rows[int(i)] for i in rows]
            split[tag] = {
                "rows": rows, "vec": vec,
                "pfam": [pfam.get(x["system_id"], "") for x in r],
                "lclass": [str(cls_of[int(x["fp_row"])]) for x in r],
                "group": [x["group"] for x in r],
                "size": (tertile(dump[f"{tag}_site_size"][dump[f"{tag}_usable"].any(1)])
                         if f"{tag}_site_size" in dump else [""] * len(rows)),
                "fp": np.asarray([int(x["fp_row"]) for x in r])}
    v = split["val"]
    print(f"  val {len(v['rows'])} pockets ({len(set(v['group']))} UniProt groups, "
          f"{len(set(p for p in v['pfam'] if p))} Pfam labels, "
          f"{len(set(v['lclass']))} ligand classes)"
          + (f"; train {len(split['train']['rows'])}" if "train" in split else ""))
    nb = knn_cos(v["vec"], a.k)
    print(f"  share of each val pocket's {nb.shape[1]} nearest val pockets with the same ...")
    for name, key in (("Pfam family", "pfam"), ("bound-ligand class", "lclass"),
                      ("pocket size tertile", "size"), ("UniProt group", "group")):
        line(name, *knn_agreement(nb, v[key]))
    km = kmeans(v["vec"], a.n_clusters)
    print(f"  k-means ({a.n_clusters} clusters) on val pockets, NMI with ...")
    for name, key in (("Pfam family", "pfam"), ("bound-ligand class", "lclass"),
                      ("pocket size tertile", "size")):
        lab = np.asarray(v[key], dtype=object)
        ok = lab != ""
        if ok.sum() < 3:
            print(f"    {name:26s} (no labels)")
            continue
        null = np.mean([nmi(km[ok], np.random.default_rng(s).permutation(lab[ok]))
                        for s in range(50)])
        print(f"    {name:26s} {nmi(km[ok], lab[ok]):6.3f}   shuffled {null:6.3f}")
    if "train" in split:
        t = split["train"]
        nn = knn_cos(v["vec"], 1, t["vec"])[:, 0]
        print("  each val pocket's nearest TRAIN pocket has the same ...")
        for name, key in (("Pfam family", "pfam"), ("bound-ligand class", "lclass")):
            vl = np.asarray(v[key], dtype=object)
            tl = np.asarray(t[key], dtype=object)
            ok = (vl != "") & (tl[nn] != "")
            if ok.sum() < 3:
                print(f"    {name:26s} (no labels)")
                continue
            obs = float((vl[ok] == tl[nn][ok]).mean())
            rng = np.random.default_rng(0)
            null = [float((vl[ok] == rng.permutation(tl)[nn][ok]).mean()) for _ in range(200)]
            line(name, obs, float(np.mean(null)), float(np.std(null)))

    # ---------------- ligands ----------------
    print("\n=== ligands (every distinct ligand) ===")
    L = lig_vec[lig_ids]
    rng = np.random.default_rng(0)
    i, j = rng.integers(0, len(lig_ids), size=(2, 20000))
    keep = i != j
    i, j = i[keep], j[keep]
    cos = (unit(L[i]) * unit(L[j])).sum(1)
    tani = (ligands.fp[lig_ids[i]] & ligands.fp[lig_ids[j]]).sum(1) / np.maximum(
        (ligands.fp[lig_ids[i]] | ligands.fp[lig_ids[j]]).sum(1), 1)
    print(f"  Spearman(cosine, Tanimoto) over {len(i)} random pairs: {spearman(cos, tani):.3f}")
    nbl = knn_cos(L, a.k)
    hv = ligands.n_heavy[lig_ids].astype(float)
    print(f"  share of each ligand's {nbl.shape[1]} nearest ligands with the same ...")
    line("chemotype class", *knn_agreement(nbl, [str(c) for c in cls_of[lig_ids]]))
    line("size tertile", *knn_agreement(nbl, tertile(hv)))
    Lc = L - L.mean(0)
    _, sv, Vt = np.linalg.svd(Lc, full_matrices=False)
    pcs = Lc @ Vt[:3].T
    expl = sv ** 2 / (sv ** 2).sum()
    print(f"  top principal components (variance {', '.join(f'{x:.1%}' for x in expl[:3])}) "
          "follow, |Spearman| with each descriptor:")
    names = list(ligands.desc_names)
    want = [n for n in ("HeavyAtomCount", "MolWt", "MolLogP", "TPSA", "NumHDonors",
                        "NumHAcceptors", "NumRotatableBonds", "NumAromaticRings") if n in names]
    print("    " + " ".join(f"{n[:12]:>13s}" for n in want))
    for c in range(3):
        print(f"    PC{c + 1} " + " ".join(
            f"{abs(spearman(pcs[:, c], ligands.desc_raw[lig_ids, names.index(n)])):13.2f}"
            for n in want))

    # ---------------- joint ----------------
    print("\n=== joint (val pockets and every ligand) ===")
    Pu, Lu = unit(v["vec"]), unit(L)
    mp, ml = Pu.mean(0), Lu.mean(0)
    print(f"  modality gap: |mean pocket - mean ligand| {np.linalg.norm(mp - ml):.3f}; "
          f"cos(mean pocket, mean ligand) {float(unit(mp[None])[0] @ unit(ml[None])[0]):.3f}")
    tv = unit(lig_vec[v["fp"]])
    true_cos = (Pu * tv).sum(1)
    rand_cos = (Pu * tv[rng.permutation(len(tv))]).sum(1)
    print(f"  pocket . its own ligand: mean cos {true_cos.mean():.3f}; "
          f"against a random val ligand {rand_cos.mean():.3f}")
    nbj = knn_cos(v["vec"], a.k, L)
    lc_of = cls_of[lig_ids]
    own = np.asarray([int(c) for c in v["lclass"]])
    obs = float((lc_of[nbj] == own[:, None]).mean())
    null = float(np.mean([(lc_of[nbj] == rng.permutation(own)[:, None]).mean()
                          for _ in range(200)]))
    print(f"  share of each pocket's {nbj.shape[1]} nearest ligands in its bound ligand's "
          f"class: {obs:.3f}   chance {null:.3f}   x{obs / max(null, 1e-9):.2f}")

    # ---------------- figures ----------------
    if a.no_figures:
        return
    print("\nt-SNE ...")
    Yp = tsne(v["vec"], seed=0)
    cP, lP = top_colours(v["pfam"])
    cC, lC = top_colours([cls_name[int(c)] for c in v["lclass"]])
    svg_panels(out.with_name(out.name + "_proteins.svg"), Yp,
               [("coloured by Pfam family", cP, lP),
                ("coloured by bound-ligand class", cC, lC)],
               f"Pocket vectors (val, {len(Yp)}): t-SNE")
    # ligands: the val and train bound ligands plus a sample of the rest
    show = np.unique(np.concatenate([v["fp"]] + ([split["train"]["fp"]]
                                                 if "train" in split else [])))
    if len(show) > a.max_plot:
        show = rng.choice(show, a.max_plot, replace=False)
    Yl = tsne(lig_vec[show], seed=0)
    cL, lL = top_colours([cls_name[int(c)] for c in cls_of[show]])
    cS, lS = top_colours(tertile(ligands.n_heavy[show].astype(float)))
    svg_panels(out.with_name(out.name + "_ligands.svg"), Yl,
               [("coloured by chemotype class", cL, lL),
                ("coloured by size (heavy-atom tertile)", cS, lS)],
               f"Ligand vectors ({len(Yl)} bound ligands): t-SNE")
    # joint: each modality centred, so the map shows alignment, not the gap
    lv = lig_vec[v["fp"]]
    J = np.concatenate([unit(v["vec"]) - mp, unit(lv) - unit(lv).mean(0)])
    Yj = tsne(J, seed=0)
    mk = ["o"] * len(v["vec"]) + ["^"] * len(lv)
    cJ, lJ = top_colours([cls_name[int(c)] for c in v["lclass"]] * 2)
    modal = ["#2f6db5"] * len(v["vec"]) + ["#d1495b"] * len(lv)
    svg_panels(out.with_name(out.name + "_joint.svg"), Yj,
               [("pockets (circles) and their ligands (triangles)", modal,
                 {"pocket": "#2f6db5", "ligand": "#d1495b"}),
                ("coloured by bound-ligand class", cJ, lJ)],
               "Pockets and ligands together (each modality centred): t-SNE", markers=mk)
    tsv = ["set\tx\ty\tlabel_a\tlabel_b"]
    tsv += [f"pocket\t{x:.3f}\t{y:.3f}\t{p}\t{c}" for (x, y), p, c
            in zip(Yp, v["pfam"], v["lclass"])]
    tsv += [f"ligand\t{x:.3f}\t{y:.3f}\t{cls_of[f]}\t{ligands.n_heavy[f]}"
            for (x, y), f in zip(Yl, show)]
    out.with_name(out.name + "_coords.tsv").write_text("\n".join(tsv) + "\n")
    print(f"wrote {out}_proteins.svg, {out}_ligands.svg, {out}_joint.svg, {out}_coords.tsv")


# ---------------------------------------------------------------------------
def selftest():
    from retrieval_report import tanimoto

    fails = []

    def check(name, ok, detail=""):
        print(f"  {'ok  ' if ok else 'FAIL'} {name}" + (f"   {detail}" if detail else ""))
        if not ok:
            fails.append(name)

    rng = np.random.default_rng(0)
    # planted: 6 ligand classes; pockets cluster by the class of what binds
    # them, NOT by their (random) family
    n, D, C = 240, 32, 6
    centres = unit(rng.normal(size=(C, D)))
    cls = rng.integers(0, C, n)
    fam = rng.integers(0, 8, n).astype(str)
    X = unit(centres[cls] + 0.08 * rng.normal(size=(n, D)))   # noise norm ~0.45
    nb = knn_cos(X, 10)
    o, mu, sd = knn_agreement(nb, cls.astype(str))
    check("planted class: kNN agreement far above chance", o > 0.8 and o > mu + 10 * sd,
          f"{o:.2f} vs {mu:.2f}")
    o2, mu2, sd2 = knn_agreement(nb, fam)
    check("random family: kNN agreement at chance", abs(o2 - mu2) < 4 * sd2 + 0.02,
          f"{o2:.2f} vs {mu2:.2f}")
    km = kmeans(X, C)
    check("k-means recovers the planted classes (NMI > 0.8), not the family",
          nmi(km, cls) > 0.8 and nmi(km, fam) < 0.2,
          f"class {nmi(km, cls):.2f}, family {nmi(km, fam):.2f}")
    check("nmi: identical labelings 1, independent near 0",
          abs(nmi(cls, cls) - 1) < 1e-9 and nmi(cls, rng.permutation(cls)) < 0.1)
    Y = tsne(X, perplexity=20, iters=500)
    o3, mu3, _ = knn_agreement(knn_cos(np.c_[Y, np.zeros(n)], 10), cls.astype(str))
    check("t-SNE keeps the planted classes apart in 2D", o3 > 0.7, f"{o3:.2f} vs {mu3:.2f}")
    # butina on planted fingerprints: 3 families of bit patterns
    base = rng.random((3, 256)) < 0.15
    fp = np.concatenate([base[k] ^ (rng.random((20, 256)) < 0.03) for k in range(3)])
    lab = butina(fp.astype(np.uint8), 0.4, tanimoto)
    truth = np.repeat(np.arange(3), 20)
    check("Butina recovers planted fingerprint families", nmi(lab, truth) > 0.9,
          f"NMI {nmi(lab, truth):.2f}, {len(np.unique(lab))} classes")
    check("spearman: monotone 1, reversed -1",
          abs(spearman(np.arange(10.0), np.arange(10.0) ** 3) - 1) < 1e-9
          and abs(spearman(np.arange(10.0), -np.arange(10.0)) + 1) < 1e-9)
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        svg_panels(Path(td) / "t.svg", Y, [("a", top_colours(cls.astype(str))[0],
                                             top_colours(cls.astype(str))[1])], "test")
        s = (Path(td) / "t.svg").read_text()
        check("SVG written, one mark per point", s.count("<circle") >= n)
    print("\nPASS" if not fails else f"\nFAIL: {fails}")
    return 1 if fails else 0


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "selftest":
        sys.exit(selftest())
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("report")
    p.add_argument("--dump", required=True, help="retrieval_report.py dump .npz")
    p.add_argument("--items", required=True)
    p.add_argument("--lig_feats", required=True)
    p.add_argument("--manifest", default="data/meta/manifest.parquet",
                   help="for Pfam labels; skipped if missing")
    p.add_argument("--out", required=True, help="prefix for the figures and TSV")
    p.add_argument("--k", type=int, default=10, help="neighbours for the agreement shares")
    p.add_argument("--n_clusters", type=int, default=12)
    p.add_argument("--class_sim", type=float, default=0.4,
                   help="Butina Tanimoto threshold for ligand chemotype classes")
    p.add_argument("--max_plot", type=int, default=1200)
    p.add_argument("--no_figures", action="store_true")
    sub.add_parser("selftest")
    a = ap.parse_args()
    report(a)


if __name__ == "__main__":
    main()
