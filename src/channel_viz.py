"""What the encoder's channels look like on real surfaces, and what they mean.

Why. Every number so far is a scalar about the RANKING. Nothing says what the
model's per-point channels encode:

  - the 16 input channels (10 curvatures, 6 AtomNet "chemistry" channels). The
    curvatures are fixed geometry; the AtomNet channels are LEARNED - the
    published dMaSIF weights, fine-tuned in every run since - so they have no
    fixed meaning and differ per checkpoint.
  - pair pooling's per-point site types a_i (8 + "not a site") and salience
    s_i, trained only through the contrastive loss. Never inspected: whether
    the 8 types differentiated at all, or collapsed into one, is unknown.

And before supervising any of them with PLIP (docs/DESIGN.md, PLIP labels),
how well they ALREADY line up with what ligands actually contact.

Three stages:

    dump      container, GPU. A checkpoint + N proteins -> one .npz: per
              surface point the input channels, the published checkpoint's
              AtomNet channels on the same points (what fine-tuning changed),
              chemotype.py's 12 interpretable channels as a reference (computed
              even when the run had --chemo off), the site types and salience,
              pocket logit / 4 A label / the site region; per protein the
              ligand and receptor atoms. Same rows across checkpoints with
              --rows_from.
    report    numpy (container or CPU venv). From one or more dumps:
                naming     each learned channel's strongest correlates among
                           chemotype, local element shares and curvatures
                pocket     AUROC of every channel for the 4 A pocket label,
                           which also gives each curvature's sign convention
                plip       AUROC of every channel per PLIP class (donor,
                           acceptor, cation, anion, aromatic, hydrophobic),
                           inside the site region, against plip_labels'
                           per-point labels; chemotype's own channels give
                           the reference a typed pocket reaches
                types      usage of the 8 site types in the site, effective
                           number of types, their correlation: the collapse
                           test
              plus an HTML viewer per protein (plotly, every channel in a
              dropdown, ligand drawn) and PyMOL files (B-factor = channel).
    selftest  numpy only: a planted dump with known answers.

How to read it.
  - types: effective types well under 8, or mean |r| between type masses near
    1, means the 8 types collapsed - pair pooling's 288 pair dims are then
    one histogram repeated, and supervising the types is the obvious fix.
  - plip: a type or salience AUROC near 0.5 for every class means the
    contrastive loss alone did not make it chemical; the chemotype reference
    row is roughly what supervision could reach.
  - naming: an AtomNet channel whose top correlate is a curvature or
    buriedness is shape, not chemistry - AtomNet means over the 16 nearest
    atoms, which can encode density.

Example:
    sbatch src/slurm/viz.sbatch          # C, v3d, v3f; dump + report
    # or by hand, CPU venv, on dumps already written:
    python src/channel_viz.py report --dumps out/viz/v3c.npz out/viz/v3d.npz \\
        --plip data/plip --out out/viz/report

Depends on: numpy (report, selftest); torch, KeOps and train.py's loaders
            (dump only); plip_labels (numpy) for the PLIP labels.
"""

import argparse
import html
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

ELEMENTS = ("C", "H", "O", "N", "S", "Se")      # extract.ELE2NUM order
ELEM_RADIUS = 6.0
PLIP_CLASSES = ("donor", "acceptor", "cation", "anion", "aromatic", "hydrophobic")
POINT_F16 = ("normals", "feats", "pre_atomnet", "chemo_ref", "point_emb",
             "pocket_logit", "types", "sal")


def feature_names(curvature_scales=(1.0, 2.0, 3.0, 5.0, 10.0), atom_dims=6,
                  chemo_names=()):
    """Input channel names in surface_encoder.input_features order:
    geometry_processing.curvatures interleaves (mean, Gaussian) per scale."""
    names = []
    for s in curvature_scales:
        names += [f"curv_H{s:g}", f"curv_K{s:g}"]
    names += [f"atomnet_{k}" for k in range(atom_dims)]
    return names + [f"chemo_{n}" for n in chemo_names]


# ---------------------------------------------------------------------------
# The dump format: ragged per-protein arrays concatenated with pointers
# ---------------------------------------------------------------------------
def pack(proteins, meta):
    """proteins: list of dicts (one per protein) -> flat dict for np.savez.

    Per point: xyz (n,3) and the POINT_F16 keys present; per protein: lig_xyz,
    atom_xyz / atom_type / atom_groups, pocket_label / site (n,) bool, and the
    scalars system_id, receptor_key, group, split, row, fallback."""
    out = {f"meta_{k}": np.asarray(v) for k, v in meta.items()}
    for key, ptr in (("xyz", "pt_ptr"), ("lig_xyz", "lig_ptr"), ("atom_xyz", "atom_ptr")):
        out[ptr] = np.cumsum([0] + [len(p[key]) for p in proteins]).astype(np.int64)
    out["xyz"] = np.concatenate([p["xyz"] for p in proteins]).astype(np.float32)
    for k in POINT_F16:
        if all(k in p and p[k] is not None for p in proteins):
            out[k] = np.concatenate([p[k] for p in proteins]).astype(np.float16)
    for k in ("pocket_label", "site"):
        out[k] = np.concatenate([p[k] for p in proteins]).astype(bool)
    out["lig_xyz"] = np.concatenate([p["lig_xyz"] for p in proteins]).astype(np.float32)
    out["atom_xyz"] = np.concatenate([p["atom_xyz"] for p in proteins]).astype(np.float32)
    out["atom_type"] = np.concatenate([p["atom_type"] for p in proteins]).astype(np.int8)
    if all("atom_groups" in p for p in proteins):
        out["atom_groups"] = np.concatenate([p["atom_groups"] for p in proteins]).astype(bool)
    for k in ("system_id", "receptor_key", "group", "split"):
        out[k] = np.asarray([str(p[k]) for p in proteins])
    out["row"] = np.asarray([int(p["row"]) for p in proteins], np.int64)
    out["fallback"] = np.asarray([bool(p["fallback"]) for p in proteins])
    return out


def unpack(path_or_dict):
    """-> (meta dict, list of per-protein dicts with float32 point arrays)."""
    d = dict(np.load(path_or_dict, allow_pickle=False)) if not isinstance(
        path_or_dict, dict) else path_or_dict
    meta = {k[5:]: d[k] for k in d if k.startswith("meta_")}
    meta = {k: (v.item() if v.ndim == 0 else v) for k, v in meta.items()}
    prots = []
    for i in range(len(d["system_id"])):
        a, b = d["pt_ptr"][i], d["pt_ptr"][i + 1]
        la, lb = d["lig_ptr"][i], d["lig_ptr"][i + 1]
        aa, ab = d["atom_ptr"][i], d["atom_ptr"][i + 1]
        p = {k: d[k][i].item() if hasattr(d[k][i], "item") else d[k][i]
             for k in ("system_id", "receptor_key", "group", "split", "row", "fallback")}
        p["xyz"] = d["xyz"][a:b]
        for k in POINT_F16:
            if k in d:
                p[k] = d[k][a:b].astype(np.float32)
        p["pocket_label"], p["site"] = d["pocket_label"][a:b], d["site"][a:b]
        p["lig_xyz"] = d["lig_xyz"][la:lb]
        p["atom_xyz"], p["atom_type"] = d["atom_xyz"][aa:ab], d["atom_type"][aa:ab]
        if "atom_groups" in d:
            p["atom_groups"] = d["atom_groups"][aa:ab]
        prots.append(p)
    return meta, prots


# ---------------------------------------------------------------------------
# dump (container, GPU)
# ---------------------------------------------------------------------------
def pick_rows(table, ckpt, saved, a):
    """The checkpoint's val rows, group-balanced, + named groups; or --rows_from."""
    from train import subsample_val

    if a.rows_from:
        sids = [str(s) for s in np.load(a.rows_from)["system_id"]]
        where = {r["system_id"]: i for i, r in enumerate(table.rows)}
        missing = [s for s in sids if s not in where]
        if missing:
            raise SystemExit(f"--rows_from: {len(missing)} system_ids not in "
                             f"{a.items}, e.g. {missing[:3]}")
        return np.asarray([where[s] for s in sids])
    va = np.asarray(ckpt["val_rows"])
    va = va[va < len(table)]
    rows = list(subsample_val(va, table.group, min(a.n, len(va)), saved.split_seed, True))
    for g in (a.groups.split(",") if a.groups else []):
        g = g if ":" in g else f"up:{g}"
        hit = [i for i in va if table.group[i] == g] or \
              [i for i in range(len(table)) if table.group[i] == g]
        if not hit:
            print(f"[warn] --groups {g}: no row in {a.items}")
        elif hit[0] not in rows:
            rows.append(hit[0])
    return np.asarray(rows)


def dump(a):
    import copy

    import torch

    import chemotype
    from train import (POCKET_DEFAULTS, anchor_block, build_batch, load_model_from_ckpt,
                       load_rows, pocket_source, use_neighbor_device)
    from train_data import LabelCache, SurfaceStore

    use_neighbor_device(a, a.device)
    model, ckpt, saved = load_model_from_ckpt(a.ckpt, a.device, overrides=a)
    enc = model.encoder
    cfg = enc.cfg
    a.n_decoys = 0                       # decoys change nothing a viewer shows
    table, _ = load_rows(a)
    for k, v in POCKET_DEFAULTS.items():
        if getattr(a, k, None) is None:
            setattr(a, k, getattr(saved, k, None) or v)
    store = SurfaceStore(a.surface_cache, ram=a.surface_ram, device=a.device,
                         keep_hydrogens=saved.keep_hydrogens,
                         pockets=pocket_source(a, cfg))
    labels = LabelCache(table, cutoff=a.cutoff, ram=a.label_ram)
    rows = pick_rows(table, ckpt, saved, a)
    print(f"{a.ckpt}: step {ckpt['step']}, pool_mode {cfg.pool_mode}, chemo "
          f"{cfg.chemo}; {len(rows)} proteins, pocket_select {a.pocket_select}")

    pre = None
    if not a.no_pretrained:
        try:
            pre = copy.deepcopy(enc)
            pre.load_pretrained(parts=("atomnet",))
            pre.eval()
        except Exception as e:           # published weights not where cfg.repo says
            print(f"[warn] no pretrained AtomNet comparison: {type(e).__name__}: {e}")
            pre = None
    chemo_names = chemotype.feature_names(tuple(cfg.chemo_sigmas)) if cfg.chemo else ()
    names = feature_names(cfg.curvature_scales, cfg.atom_dims, chemo_names)
    n_curv = 2 * len(cfg.curvature_scales)

    prots, gen = [], torch.Generator().manual_seed(a.seed)
    model.eval()
    with torch.no_grad():
        for s in range(0, len(rows), a.eval_batch):
            idx = rows[s:s + a.eval_batch]
            P, labs = build_batch(model, store, labels, table, idx, a.device, 0)
            blk = anchor_block(model, P, labs, a, generator=gen)
            out = blk["out"]
            feats = out["input_feats"]
            if feats.shape[1] != len(names):
                raise SystemExit(f"{feats.shape[1]} input channels, {len(names)} names")
            types = sal = None
            if enc.pair_pool is not None:
                types, sal = enc.pair_pool.site_types(
                    out["pool_emb"], feats if cfg.pair_feats else None)
            pre_chem = (pre.atomnet(out["xyz"], P["atom_xyz"], P["atomtypes"],
                                    out["batch"], P["batch_atoms"]) if pre is not None else None)
            if cfg.chemo:
                ref = feats[:, n_curv + cfg.atom_dims:]
            elif "atom_groups" in P:
                ref = chemotype.surface_features(
                    out["xyz"], out["batch"], P["atom_xyz"], P["atom_groups"],
                    P["batch_atoms"], sigmas=tuple(cfg.chemo_sigmas))
            else:
                ref = None
            for b, i in enumerate(idx):
                sel = (out["batch"] == b).nonzero().squeeze(1)
                asel = (P["batch_atoms"] == b).nonzero().squeeze(1)
                lab = labs[b]
                site = torch.zeros(len(sel), dtype=torch.bool)
                if (lab.get("n_site", 1) and bool(out["anchor_valid"][b, 0])
                        and bool(out["anchor_is_pos"][b, 0])
                        and a.pocket_select == "site"):
                    site[(out["anchor_regions"][b][0] - sel[0]).cpu()] = True
                cpu = lambda t: None if t is None else t[sel].float().cpu().numpy()
                row = table.rows[int(i)]
                p = {"xyz": out["xyz"][sel].cpu().numpy(),
                     "normals": cpu(out["normals"]), "feats": cpu(feats),
                     "pre_atomnet": cpu(pre_chem), "chemo_ref": cpu(ref),
                     "point_emb": cpu(out["point_emb"]),
                     "pocket_logit": cpu(out["pocket_logit"]),
                     "types": cpu(types), "sal": cpu(sal),
                     "pocket_label": np.asarray(lab["pocket"].cpu().numpy(), bool),
                     "site": site.numpy(),
                     "lig_xyz": np.asarray(lab["lig"], np.float32),
                     "atom_xyz": P["atom_xyz"][asel].cpu().numpy(),
                     "atom_type": P["atomtypes"][asel].argmax(1).cpu().numpy(),
                     "system_id": row["system_id"], "receptor_key": row["receptor_key"],
                     "group": row["group"], "split": row["split"], "row": int(i),
                     "fallback": bool(blk["fallback"][b])}
                if "atom_groups" in P:
                    p["atom_groups"] = P["atom_groups"][asel].cpu().numpy()
                prots.append(p)
            print(f"    {s + len(idx)}/{len(rows)}", end="\r", flush=True)
    meta = {"ckpt": str(a.ckpt), "step": int(ckpt["step"]), "tag": a.tag or Path(a.ckpt).parent.name,
            "feat_names": np.asarray(names),
            "chemo_ref_names": np.asarray(chemotype.feature_names(tuple(cfg.chemo_sigmas))),
            "pocket_select": str(a.pocket_select),
            "cfg_json": json.dumps({k: v for k, v in vars(cfg).items()
                                    if isinstance(v, (int, float, str, bool, tuple, list))})}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, **pack(prots, meta))
    n_pts = sum(len(p["xyz"]) for p in prots)
    print(f"\nwrote {a.out}: {len(prots)} proteins, {n_pts:,} points; site types "
          f"{'yes' if prots and prots[0]['types'] is not None else 'no'}, pretrained "
          f"AtomNet {'yes' if pre is not None else 'no'}")


# ---------------------------------------------------------------------------
# report (numpy)
# ---------------------------------------------------------------------------
def avg_ranks(x):
    order = np.argsort(x, kind="mergesort")
    s = x[order]
    starts = np.r_[0, np.nonzero(np.diff(s))[0] + 1]
    ends = np.r_[starts[1:], len(s)]
    r = np.empty(len(x), np.float64)
    r[order] = np.repeat((starts + ends + 1) / 2.0, ends - starts)
    return r


def auroc(score, label):
    """Mann-Whitney AUROC with ties averaged; NaN without both classes."""
    label = np.asarray(label, bool)
    n1, n0 = int(label.sum()), int((~label).sum())
    if not n1 or not n0:
        return np.nan
    r = avg_ranks(np.asarray(score, np.float64))
    return float((r[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def element_shares(xyz, atom_xyz, atom_type, radius=ELEM_RADIUS, chunk=2048):
    """Per point, the share of heavy atoms within `radius` that are C/O/N/S."""
    heavy = atom_type != 1
    A, T = atom_xyz[heavy], atom_type[heavy]
    out = np.zeros((len(xyz), 4), np.float32)
    for s in range(0, len(xyz), chunk):
        d2 = ((xyz[s:s + chunk, None] - A[None]) ** 2).sum(-1)
        near = d2 <= radius ** 2
        tot = np.maximum(near.sum(1), 1)
        for j, e in enumerate((0, 2, 3, 4)):
            out[s:s + chunk, j] = (near & (T == e)[None]).sum(1) / tot
    return out


def channels(meta, p, plip_rec=None, plip_kw=None):
    """One protein -> (subjects, references, plip (n,6) or None), name -> (n,).

    Subjects are what a model learned: AtomNet channels (this checkpoint and
    the published one), site-type masses m_t = s * a_t, salience, pocket
    probability. References are interpretable: chemotype channels, element
    shares, curvatures."""
    names = [str(n) for n in meta["feat_names"]]
    subj, ref = {}, {}
    for j, n in enumerate(names):
        (ref if n.startswith("curv_") else subj if n.startswith("atomnet_") else ref)[n] = \
            p["feats"][:, j]
    if "pre_atomnet" in p:
        for k in range(p["pre_atomnet"].shape[1]):
            subj[f"pre_atomnet_{k}"] = p["pre_atomnet"][:, k]
    if "types" in p:
        for t in range(p["types"].shape[1]):
            subj[f"type_{t}"] = p["sal"] * p["types"][:, t]
        subj["salience"] = p["sal"]
    subj["pocket_prob"] = 1 / (1 + np.exp(-p["pocket_logit"]))
    if "chemo_ref" in p and not any(n.startswith("chemo_") for n in names):
        for k, n in enumerate(meta["chemo_ref_names"]):
            ref[f"chemo_{n}"] = p["chemo_ref"][:, k]
    es = element_shares(p["xyz"], p["atom_xyz"], p["atom_type"])
    for j, e in enumerate(("C", "O", "N", "S")):
        ref[f"elem_{e}"] = es[:, j]
    lab = None
    if plip_rec is not None:
        from plip_labels import site_labels
        lab = site_labels(p["xyz"], plip_rec, **(plip_kw or {}))
    return subj, ref, lab


def _corr(x, y):
    x, y = x - x.mean(), y - y.mean()
    den = np.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / den) if den > 0 else np.nan


def analyse(meta, prots, plip_dir=None, plip_kw=None):
    """-> dict of tables (lists of rows) for one dump."""
    per = []
    for p in prots:
        rec = None
        if plip_dir:
            f = Path(plip_dir) / f"{p['system_id']}.npz"
            if f.exists():
                from plip_labels import load_plip
                rec = load_plip(f)
        per.append((p,) + channels(meta, p, rec, plip_kw))
    subj_names = list(per[0][1])
    ref_names = list(per[0][2])

    # naming: pooled correlation over every point of every protein
    pooled = lambda d, n: np.concatenate([x[d][n] for x in per])
    S = {n: pooled(1, n) for n in subj_names}
    R = {n: pooled(2, n) for n in ref_names}
    naming = []
    for n in subj_names:
        cs = sorted(((abs(_corr(S[n], R[r])), _corr(S[n], R[r]), r) for r in ref_names),
                    key=lambda t: -np.nan_to_num(t[0]))
        naming.append([n] + [f"{r} {c:+.2f}" for _, c, r in cs[:3]])

    # pocket: per-protein AUROC for the 4 A label, mean over proteins
    pocket = []
    for n in subj_names + ref_names:
        v = [auroc(x[1 if n in x[1] else 2][n], x[0]["pocket_label"]) for x in per]
        pocket.append([n, np.nanmean(v), np.nanstd(v), int(np.isfinite(v).sum())])

    # plip: inside the site region (fallback: the 4 A label), per class
    plip = []
    have = [x for x in per if x[3] is not None]
    if have:
        for n in subj_names + ref_names:
            row = [n]
            for c in range(len(PLIP_CLASSES)):
                v = []
                for x in have:
                    m = x[0]["site"] if x[0]["site"].sum() >= 10 else x[0]["pocket_label"]
                    lab = x[3][m, c]
                    if lab.sum() >= 3 and (~lab).sum() >= 3:
                        v.append(auroc((x[1] if n in x[1] else x[2])[n][m], lab))
                row.append(np.nanmean(v) if v else np.nan)
            v = []
            for x in have:
                m = x[0]["site"] if x[0]["site"].sum() >= 10 else x[0]["pocket_label"]
                lab = x[3][m].any(1)
                if lab.sum() >= 3 and (~lab).sum() >= 3:
                    v.append(auroc((x[1] if n in x[1] else x[2])[n][m], lab))
            plip.append(row + [np.nanmean(v) if v else np.nan])

    # types: usage and collapse inside the site region
    types = None
    if "types" in prots[0]:
        T = prots[0]["types"].shape[1]
        masses, ents = [], []
        for x in per:
            m = x[0]["site"] if x[0]["site"].sum() >= 10 else x[0]["pocket_label"]
            if not m.any():
                continue
            mt = np.stack([x[1][f"type_{t}"][m] for t in range(T)], 1)
            masses.append(mt)
            a = x[0]["types"][m]
            q = a / np.maximum(a.sum(1, keepdims=True), 1e-9)
            ents.append(-(q * np.log(np.maximum(q, 1e-12))).sum(1))
        M = np.concatenate(masses)
        use = M.sum(0) / max(M.sum(), 1e-12)
        eff = float(np.exp(-(use * np.log(np.maximum(use, 1e-12))).sum()))
        C = np.corrcoef(M.T) if M.shape[0] > 1 else np.full((T, T), np.nan)
        off = C[~np.eye(T, dtype=bool)]
        types = {"usage": use, "effective": eff, "corr": C,
                 "mean_abs_corr": float(np.nanmean(np.abs(off))),
                 "point_entropy": float(np.concatenate(ents).mean()),
                 "max_entropy": float(np.log(T)),
                 "site_salience": float(np.concatenate(
                     [x[0]["sal"][x[0]["site"]] for x in per if x[0]["site"].any()]
                     or [np.full(1, np.nan)]).mean())}
    return {"naming": naming, "pocket": pocket, "plip": plip, "types": types,
            "n_plip": len(have), "per": per}


def format_report(meta, res):
    L = [f"=== {meta['tag']} (step {meta['step']}, {meta['ckpt']}) ===",
         f"{len(res['per'])} proteins, {sum(len(x[0]['xyz']) for x in res['per']):,} points; "
         f"PLIP labels for {res['n_plip']}"]
    t = res["types"]
    if t:
        L += ["", "site types inside the site region (the collapse test):",
              "  usage share   " + "  ".join(f"t{i} {u:.2f}" for i, u in enumerate(t["usage"])),
              f"  effective types {t['effective']:.2f} of {len(t['usage'])}; mean |r| between "
              f"type masses {t['mean_abs_corr']:.2f}; per-point type entropy "
              f"{t['point_entropy']:.2f} (max {t['max_entropy']:.2f}); mean salience in "
              f"the site {t['site_salience']:.2f}"]
        # Usage and correlation alone miss a head that is near-uniform at every
        # point: all 8 types then get used and vary independently, yet each
        # point is a soft mixture and the pair block is ~one histogram. The
        # first real run (v3f: entropy 1.91 of 2.08) is why this is checked.
        soft = t["point_entropy"] > 0.85 * t["max_entropy"]
        verdict = ("COLLAPSED" if t["effective"] < 2 or t["mean_abs_corr"] > 0.9
                   else "partly redundant" if t["mean_abs_corr"] > 0.6
                   else "differentiated in usage but SOFT per point" if soft
                   else "differentiated")
        L.append(f"  -> {verdict}")
    L += ["", "naming: strongest pooled correlates of each learned channel",
          f"  {'channel':16s} {'1st':34s} {'2nd':34s} {'3rd':34s}"]
    L += [f"  {r[0]:16s} " + " ".join(f"{c:34s}" for c in r[1:]) for r in res["naming"]]
    L += ["", "pocket: AUROC for the 4 A pocket label, mean (sd) over proteins "
          "(< 0.5 = lower on pocket points)"]
    rows = sorted(res["pocket"], key=lambda r: -abs(np.nan_to_num(r[1]) - 0.5))
    L += [f"  {r[0]:24s} {r[1]:.3f} ({r[2]:.3f})  n={r[3]}" for r in rows]
    if res["plip"]:
        L += ["", "plip: AUROC per PLIP class inside the site region, mean over "
              "proteins with >= 3 labelled and unlabelled points",
              f"  {'channel':24s} " + " ".join(f"{c[:6]:>7s}" for c in PLIP_CLASSES)
              + f" {'any':>7s}"]
        L += [f"  {r[0]:24s} " + " ".join(f"{v:7.3f}" for v in r[1:]) for r in res["plip"]]
    return "\n".join(L)


# ---------------------------------------------------------------------------
# viewer (HTML, plotly) and PyMOL files
# ---------------------------------------------------------------------------
PLOTLY = "https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js"


def _bonds(lig, cut=1.9):
    d = np.sqrt(((lig[:, None] - lig[None]) ** 2).sum(-1))
    i, j = np.nonzero(np.triu(d < cut, 1))
    xs, ys, zs = [], [], []
    for a, b in zip(i, j):
        for v, w in ((xs, 0), (ys, 1), (zs, 2)):
            v += [float(lig[a, w]), float(lig[b, w]), None]
    return xs, ys, zs


# Mesh: the surface the model saw, triangulated, with features smoothed onto it.
# The sampled points carry dMaSIF's outward normals, so the zero level set of
# an oriented-point signed distance passes through them; naive surface nets
# triangulates it. Numpy only (no scikit-image / Open3D in the container or
# the venv). Between two pocket walls both normals point into the cleft, so
# the field is positive across it and no sheet forms through the ligand.
def implicit_grid(pts, nrm, h=0.6, sigma=1.0, reach=2.5, pad=2.0):
    """Signed distance to the oriented point cloud on a grid:
    f(x) = sum_i w_i n_i.(x - p_i) / sum_i w_i, w_i = exp(-|x-p_i|^2 / 2 sigma^2),
    stamped from each point onto grid nodes within `reach`. Nodes no point
    reaches are invalid (NaN), so the mesh stops where the samples stop."""
    lo = pts.min(0) - pad
    shape = np.ceil((pts.max(0) + pad - lo) / h).astype(int) + 1
    r = int(np.ceil(reach / h))
    off = np.stack(np.meshgrid(*[np.arange(-r, r + 1)] * 3, indexing="ij"), -1).reshape(-1, 3)
    off = off[(off ** 2).sum(1) * h * h <= reach * reach]
    base = np.round((pts - lo) / h).astype(int)
    num = np.zeros(int(np.prod(shape)))
    den = np.zeros_like(num)
    for s in range(0, len(pts), 512):
        b = base[s:s + 512, None] + off[None]                     # (k, m, 3)
        ok = np.all((b >= 0) & (b < shape), -1)
        x = lo + b * h
        d = x - pts[s:s + 512, None]
        w = np.exp(-(d ** 2).sum(-1) / (2 * sigma ** 2)) * ok
        sd = (d * nrm[s:s + 512, None]).sum(-1)
        flat = np.ravel_multi_index(tuple(np.clip(b, 0, shape - 1).reshape(-1, 3).T), shape)
        num += np.bincount(flat, (w * sd).ravel(), len(num))
        den += np.bincount(flat, w.ravel(), len(den))
    f = np.full(len(num), np.nan)
    good = den > 1e-3
    f[good] = num[good] / den[good]
    return f.reshape(shape), lo, h

CORNERS = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0],
                    [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]])
EDGES = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7),
         (0, 4), (1, 5), (2, 6), (3, 7)]

def surface_nets(F, lo, h):
    """Naive surface nets on the zero level set of F (NaN = invalid).
    -> vertices (V,3), triangles (T,3)."""
    nx, ny, nz = F.shape
    c = np.stack([F[i:nx-1+i, j:ny-1+j, k:nz-1+k] for i, j, k in CORNERS], -1)  # cells x 8
    valid = np.isfinite(c).all(-1)
    neg = c < 0
    cross = valid & neg.any(-1) & (~neg).any(-1)
    idx = np.argwhere(cross)
    cc = c[cross]
    acc, cnt = np.zeros((len(idx), 3)), np.zeros(len(idx))
    for a, b in EDGES:
        fa, fb = cc[:, a], cc[:, b]
        m = (fa < 0) != (fb < 0)
        t = np.where(m, fa / np.where(m, fa - fb, 1), 0)
        pa, pb = CORNERS[a], CORNERS[b]
        acc += m[:, None] * (pa + t[:, None] * (pb - pa))
        cnt += m
    verts = lo + (idx + acc / cnt[:, None]) * h
    vid = -np.ones(cross.shape, int)
    vid[cross] = np.arange(len(idx))
    tris = []
    shape = np.array(vid.shape)
    for ax in range(3):
        sl0, sl1 = [slice(None)] * 3, [slice(None)] * 3
        sl0[ax] = slice(0, -1); sl1[ax] = slice(1, None)
        f0, f1 = F[tuple(sl0)], F[tuple(sl1)]
        e = np.isfinite(f0) & np.isfinite(f1) & ((f0 < 0) != (f1 < 0))
        p = np.argwhere(e)
        if not len(p):
            continue
        u, v = [a for a in range(3) if a != ax]
        q = []
        for du, dv in ((0, 0), (-1, 0), (-1, -1), (0, -1)):
            c = p.copy()
            c[:, u] += du
            c[:, v] += dv
            ok = np.all((c >= 0) & (c < shape), 1)
            ids = np.full(len(p), -1)
            ids[ok] = vid[tuple(c[ok].T)]
            q.append(ids)
        q = np.stack(q, 1)
        keep = (q >= 0).all(1)
        q, p = q[keep], p[keep]
        t = np.concatenate([q[:, [0, 1, 2]], q[:, [0, 2, 3]]])
        # f < 0 inside (normals point out), so where the lower node is inside
        # the outward direction is +ax: orient every face to agree
        out = np.where(f0[tuple(p.T)] < 0, 1.0, -1.0)
        out = np.concatenate([out, out])
        fn = np.cross(verts[t[:, 1]] - verts[t[:, 0]], verts[t[:, 2]] - verts[t[:, 0]])
        flip = np.sign(fn[:, ax]) != out
        t[flip] = t[flip][:, ::-1]
        tris.append(t)
    return verts, (np.concatenate(tris) if tris else np.zeros((0, 3), int))


def mesh_surface(pts, nrm, h=0.6, sigma=1.0, reach=2.5, max_gap=2.0):
    """Oriented points -> (vertices, triangles), trimmed to within max_gap A of
    a sample so the mesh stops where the samples do. 2 A, not tighter: at ~1
    point per A^2 the gaps between samples reach ~2 A, and a tighter cut
    punches holes (checked on a sampled sphere in the self-test)."""
    F, lo, h = implicit_grid(pts, nrm, h, sigma, reach)
    V, T = surface_nets(F, lo, h)
    if not len(T):
        return V, T
    keep = _min_dist(V, pts) <= max_gap
    return _subset(V, T, keep)


def _min_dist(a, b, chunk=1024):
    out = np.empty(len(a))
    for s in range(0, len(a), chunk):
        out[s:s + chunk] = np.sqrt(((a[s:s + chunk, None] - b[None]) ** 2).sum(-1).min(1))
    return out


def _subset(V, T, keep):
    """Keep the vertices in `keep` and the triangles made only of them."""
    T = T[keep[T].all(1)]
    new = -np.ones(len(V), int)
    new[keep] = np.arange(int(keep.sum()))
    return V[keep], new[T]


def smooth_onto(V, pts, vals, s=1.2, chunk=1024):
    """Gaussian-weighted average of point values at each vertex: (V, C)."""
    out = np.zeros((len(V), vals.shape[1]))
    for a in range(0, len(V), chunk):
        d2 = ((V[a:a + chunk, None] - pts[None]) ** 2).sum(-1)
        w = np.exp(-d2 / (2 * s * s)) * (d2 <= (3 * s) ** 2)
        out[a:a + chunk] = (w @ vals) / np.maximum(w.sum(1, keepdims=True), 1e-12)
    return out


def html_viewer(meta, x, path, crop=18.0, mesh=True, smooth=1.2, grid=0.6):
    """One protein -> a self-contained plotly page with a channel dropdown.

    mesh: a triangulated surface (mesh_surface) with every channel smoothed onto
    its vertices (Gaussian, `smooth` A); otherwise the raw sampled points.
    Falls back to points when the mesh comes out empty."""
    p, subj, ref, lab = x
    d = _min_dist(p["xyz"], p["lig_xyz"])
    # mesh a margin beyond the crop so the cut edge is clean, then trim
    keep = d <= (crop + 3.0 if mesh else crop) if crop else np.ones(len(d), bool)
    chans = {}
    for group, src in (("learned", subj), ("reference", ref)):
        for n, v in src.items():
            chans[n] = (group, v[keep])
    chans["pocket_label"] = ("labels", p["pocket_label"][keep].astype(np.float32))
    chans["site"] = ("labels", p["site"][keep].astype(np.float32))
    if lab is not None:
        for c, n in enumerate(PLIP_CLASSES):
            chans[f"plip_{n}"] = ("labels", lab[keep, c].astype(np.float32))
    pts = p["xyz"][keep].astype(np.float64)
    V = T = None
    if mesh and "normals" in p and len(pts) >= 50:
        V, T = mesh_surface(pts, p["normals"][keep].astype(np.float64), h=grid)
        if crop and len(T):
            V, T = _subset(V, T, _min_dist(V, p["lig_xyz"]) <= crop)
        if not len(T):
            V = T = None
    names = list(chans)
    M = np.nan_to_num(np.stack([chans[n][1] for n in names], 1).astype(np.float64))
    if V is not None:
        M = smooth_onto(V, pts, M, s=smooth)
        geom = V
    else:
        sel = (d[keep] <= crop) if crop else np.ones(len(pts), bool)
        M, geom = M[sel], pts[sel]
    data = {}
    for k, n in enumerate(names):
        g, v = chans[n][0], M[:, k]
        if g == "labels":
            lo, hi, scale = 0.0, 1.0, "binary"
        elif n.startswith("curv_"):
            hi = float(np.percentile(np.abs(v), 98)) or 1.0
            lo, scale = -hi, "signed"
        else:
            lo, hi = (float(q) for q in np.percentile(v, [2, 98]))
            hi, scale = (hi if hi > lo else lo + 1.0), "seq"
        data[n] = {"g": g, "v": np.round(v, 4).tolist(), "lo": lo, "hi": hi, "s": scale}
    xyz = np.round(geom, 2)
    tri = (T if V is not None else np.zeros((0, 3), int)).tolist()
    bx, by, bz = _bonds(p["lig_xyz"])
    first = "type_0" if "type_0" in data else next(iter(data))
    what = (f"surface mesh, {len(xyz)} vertices, features smoothed over {smooth:g} A"
            if V is not None else f"{len(xyz)} sampled points")
    title = (f"{meta['tag']} - {p['system_id']} ({p['group']}, {p['split']}); "
             f"{what}, " + (f"within {crop:g} A of the ligand" if crop else "whole surface"))
    opts = "".join(f'<option value="{html.escape(n)}"{" selected" if n == first else ""}>'
                   f"{html.escape(c['g'])}: {html.escape(n)}</option>" for n, c in data.items())
    page = f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(p['system_id'])}</title>
<script src="{PLOTLY}"></script>
<style>body{{font-family:system-ui,sans-serif;margin:0 16px;background:#fff;color:#222}}
#plot{{width:100%;height:82vh}} select{{font-size:14px;padding:4px;max-width:100%}}</style>
</head><body><h3 style="font-weight:500">{html.escape(title)}</h3>
<label>channel <select id="ch">{opts}</select></label>
<div id="plot"></div>
<script>
const D = {json.dumps(data)};
const X = {json.dumps(xyz[:, 0].tolist())}, Y = {json.dumps(xyz[:, 1].tolist())},
      Z = {json.dumps(xyz[:, 2].tolist())};
const TRI = {json.dumps(tri)};
const L = {json.dumps(np.round(p['lig_xyz'], 2).tolist())};
const SC = {{signed: 'RdBu', seq: 'Viridis', binary: [[0, '#d9d9d9'], [1, '#d62728']]}};
function trace(n) {{
  const c = D[n];
  const bar = {{title: n, thickness: 14}};
  if (TRI.length) return {{type: 'mesh3d', x: X, y: Y, z: Z, name: n,
    i: TRI.map(t => t[0]), j: TRI.map(t => t[1]), k: TRI.map(t => t[2]),
    intensity: c.v, intensitymode: 'vertex', colorscale: SC[c.s], cmin: c.lo,
    cmax: c.hi, reversescale: c.s === 'signed', colorbar: bar, flatshading: false,
    lighting: {{ambient: 0.55, diffuse: 0.7, specular: 0.05, roughness: 0.9}},
    hovertemplate: n + ': %{{intensity:.3f}}<extra></extra>'}};
  return {{type: 'scatter3d', mode: 'markers', x: X, y: Y, z: Z, name: n,
    marker: {{size: 3, color: c.v, colorscale: SC[c.s], cmin: c.lo, cmax: c.hi,
      reversescale: c.s === 'signed', colorbar: bar}},
    hovertemplate: n + ': %{{marker.color:.3f}}<extra></extra>'}};
}}
const lig = {{type: 'scatter3d', mode: 'markers', name: 'ligand',
  x: L.map(a => a[0]), y: L.map(a => a[1]), z: L.map(a => a[2]),
  marker: {{size: 5, color: '#111'}}, hoverinfo: 'skip'}};
const bonds = {{type: 'scatter3d', mode: 'lines', name: 'bonds', x: {json.dumps(bx)},
  y: {json.dumps(by)}, z: {json.dumps(bz)}, line: {{color: '#111', width: 5}},
  hoverinfo: 'skip'}};
const layout = {{margin: {{l: 0, r: 0, t: 0, b: 0}}, showlegend: false, uirevision: 'keep',
  scene: {{aspectmode: 'data', xaxis: {{visible: false}}, yaxis: {{visible: false}},
           zaxis: {{visible: false}}}}}};
Plotly.newPlot('plot', [trace('{first}'), lig, bonds], layout);
document.getElementById('ch').onchange = e =>
  Plotly.react('plot', [trace(e.target.value), lig, bonds], layout);
</script></body></html>"""
    Path(path).write_text(page)


def pymol_files(meta, x, folder, names, crop=18.0):
    """points_<channel>.pdb (B-factor = channel), ligand.pdb and load.pml."""
    p, subj, ref, lab = x
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    d = np.sqrt(((p["xyz"][:, None] - p["lig_xyz"][None]) ** 2).sum(-1)).min(1)
    keep = d <= crop if crop else np.ones(len(d), bool)
    allc = {**subj, **ref}
    if lab is not None:
        allc.update({f"plip_{n}": lab[:, c].astype(np.float32)
                     for c, n in enumerate(PLIP_CLASSES)})

    def atoms(xyz, b, res, el):
        return "".join(f"HETATM{i + 1:5d}  {el:<3s} {res} X{1:4d}    "
                       f"{c[0]:8.3f}{c[1]:8.3f}{c[2]:8.3f}  1.00{float(v):6.2f}          {el:>2s}\n"
                       for i, (c, v) in enumerate(zip(xyz, b)))
    written = []
    for n in names:
        if n not in allc:
            continue
        v = np.clip(np.nan_to_num(allc[n][keep]), -99.99, 999.99)
        (folder / f"points_{n}.pdb").write_text(atoms(p["xyz"][keep], v, "PTS", "C") + "END\n")
        written.append(n)
    (folder / "ligand.pdb").write_text(
        atoms(p["lig_xyz"], np.zeros(len(p["lig_xyz"])), "LIG", "C") + "END\n")
    pml = ["load ligand.pdb", "show sticks, ligand", "color black, ligand"]
    for n in written:
        pml += [f"load points_{n}.pdb", f"hide everything, points_{n}",
                f"show spheres, points_{n}", f"set sphere_scale, 0.25, points_{n}",
                f"spectrum b, blue_white_red, points_{n}", f"disable points_{n}"]
    if written:
        pml.append(f"enable points_{written[0]}")
    (folder / "load.pml").write_text("\n".join(pml + ["zoom ligand, 12", ""]))
    return written


def report(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    plip_kw = {"lig_radius": a.plip_lig_radius} if a.plip_lig_radius else {}
    texts, sids = [], []
    for path in a.dumps:
        meta, prots = unpack(path)
        sids.append([p["system_id"] for p in prots])
        res = analyse(meta, prots, a.plip, plip_kw)
        txt = format_report(meta, res)
        texts.append(txt)
        print(txt + "\n")
        tag = str(meta["tag"])
        if res["types"] is not None:
            np.savetxt(out / f"{tag}_type_corr.tsv", res["types"]["corr"], fmt="%.3f",
                       delimiter="\t")
        hdir = out / f"{tag}_html"
        hdir.mkdir(exist_ok=True)
        for x in res["per"][:a.n_html]:
            html_viewer(meta, x, hdir / f"{x[0]['system_id']}.html", crop=a.crop,
                        mesh=a.viewer == "mesh", smooth=a.smooth, grid=a.grid)
            pymol_files(meta, x, out / f"{tag}_pymol" / x[0]["system_id"],
                        a.pymol_channels.split(","), crop=a.crop)
        print(f"  {min(a.n_html, len(res['per']))} viewers in {hdir}/, PyMOL in "
              f"{out / (tag + '_pymol')}/\n")
    if len(sids) > 1 and any(s != sids[0] for s in sids[1:]):
        print("[warn] the dumps hold different proteins; dump the later checkpoints "
              "with --rows_from the first so their tables compare like with like")
    (out / "report.txt").write_text("\n\n".join(texts) + "\n")
    print(f"wrote {out / 'report.txt'}")


# ---------------------------------------------------------------------------
def selftest():
    import tempfile

    fails = []

    def check(name, ok, detail=""):
        print(f"  {'ok  ' if ok else 'FAIL'} {name}" + (f"   {detail}" if detail else ""))
        if not ok:
            fails.append(name)

    rng = np.random.default_rng(0)
    check("auroc: perfect, inverted, ties, one class",
          auroc([1, 2, 3, 4], [0, 0, 1, 1]) == 1.0 and auroc([4, 3, 2, 1], [0, 0, 1, 1]) == 0.0
          and auroc([1, 1, 1, 1], [0, 1, 0, 1]) == 0.5 and np.isnan(auroc([1, 2], [1, 1])))

    from chemotype import feature_names as chemo_names
    from plip_labels import KINDS

    names = feature_names()
    T = 8
    prots = []
    for k in range(3):
        n = 1500
        xyz = rng.normal(size=(n, 3)) * 10
        lig = rng.normal(size=(12, 3)) + np.array([8.0, 0, 0])
        dl = np.sqrt(((xyz[:, None] - lig[None]) ** 2).sum(-1)).min(1)
        pocket = dl < 4.0
        site = dl < 6.0
        atoms = xyz[rng.choice(n, 600)] + rng.normal(size=(600, 3)) * 0.8
        atype = rng.choice([0, 2, 3, 4], 600, p=[0.6, 0.2, 0.15, 0.05])
        feats = rng.normal(size=(n, len(names))) * 0.1
        feats[:, 6] = np.where(pocket, -0.5, 0.2) + rng.normal(size=n) * 0.05   # curv_H5
        es = element_shares(xyz.astype(np.float32), atoms.astype(np.float32), atype)
        feats[:, 10] = es[:, 1] + rng.normal(size=n) * 0.01                     # atomnet_0 ~ O
        donors = xyz[site][:6]                       # PLIP protein-side donor atoms
        dd = np.sqrt(((xyz[:, None] - donors[None]) ** 2).sum(-1)).min(1)
        types = rng.dirichlet(np.ones(T + 1), n)[:, :T]
        types[:, 2] = np.where(dd < 3.0, 0.9, 0.02)                            # type 2 = donor
        prots.append({"xyz": xyz, "normals": rng.normal(size=(n, 3)), "feats": feats,
                      "pre_atomnet": rng.normal(size=(n, 6)),
                      "chemo_ref": rng.normal(size=(n, 12)), "point_emb": rng.normal(size=(n, 16)),
                      "pocket_logit": np.where(pocket, 2.0, -2.0), "types": types,
                      "sal": np.full(n, 0.5), "pocket_label": pocket, "site": site,
                      "lig_xyz": lig, "atom_xyz": atoms, "atom_type": atype,
                      "atom_groups": np.zeros((600, 6), bool), "system_id": f"sys{k}",
                      "receptor_key": f"rec{k}", "group": f"up:P{k}", "split": "val",
                      "row": k, "fallback": False,
                      "_plip": {"kind": np.full(6, KINDS.index("hbond_pdon"), np.int8),
                                "prot_xyz": donors.astype(np.float32),
                                "lig_xyz": donors.astype(np.float32) + 2.9}})
    meta = {"ckpt": "planted.pt", "step": 5000, "tag": "planted",
            "feat_names": np.asarray(names), "chemo_ref_names": np.asarray(chemo_names()),
            "pocket_select": "site", "cfg_json": "{}"}
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "plip").mkdir()
        for p in prots:
            np.savez(td / "plip" / f"{p['system_id']}.npz", **p.pop("_plip"))
        np.savez_compressed(td / "d.npz", **pack(prots, meta))
        m2, back = unpack(td / "d.npz")
        check("pack / unpack round trip (float16 channels within tolerance)",
              len(back) == 3 and m2["tag"] == "planted"
              and np.allclose(back[1]["feats"], prots[1]["feats"], atol=2e-3)
              and (back[2]["site"] == prots[2]["site"]).all()
              and back[0]["system_id"] == "sys0" and len(back[0]["lig_xyz"]) == 12)
        res = analyse(m2, back, td / "plip")
        nm = {r[0]: r[1] for r in res["naming"]}
        check("naming: atomnet_0 is the oxygen share", nm["atomnet_0"].startswith("elem_O +"),
              nm["atomnet_0"])
        pk = {r[0]: r[1] for r in res["pocket"]}
        check("pocket: the planted curvature is LOW on pocket points (AUROC ~0)",
              pk["curv_H5"] < 0.05 and pk["pocket_prob"] > 0.95,
              f"curv_H5 {pk['curv_H5']:.3f}, pocket_prob {pk['pocket_prob']:.3f}")
        pl = {r[0]: r[1:] for r in res["plip"]}
        check("plip: type_2 finds the donor contacts, type_5 does not",
              pl["type_2"][0] > 0.95 and abs(pl["type_5"][0] - 0.5) < 0.15,
              f"type_2 donor {pl['type_2'][0]:.3f}, type_5 donor {pl['type_5'][0]:.3f}")
        t = res["types"]
        check("types: usage sums to 1, 8 x 8 correlation, effective count in range",
              abs(t["usage"].sum() - 1) < 1e-6 and t["corr"].shape == (8, 8)
              and 1 <= t["effective"] <= 8, f"effective {t['effective']:.2f}")
        txt = format_report(m2, res)
        check("report text has every section", all(s in txt for s in
              ("site types", "naming", "pocket:", "plip:")))

        class A:
            dumps, plip, out, n_html, crop = [td / "d.npz"], td / "plip", td / "rep", 2, 18.0
            plip_lig_radius, pymol_channels = None, "type_2,salience,atomnet_0,plip_donor"
            viewer, smooth, grid = "points", 1.2, 0.6
        report(A)
        pages = sorted((td / "rep" / "planted_html").glob("*.html"))
        page = pages[0].read_text() if pages else ""
        check("two HTML viewers, each carrying plotly, the ligand and every channel",
              len(pages) == 2 and PLOTLY in page and '"type_2"' in page
              and '"plip_donor"' in page and '"curv_H1"' in page and "ligand" in page)
        pm = td / "rep" / "planted_pymol" / "sys0"
        check("PyMOL: one PDB per requested channel + ligand + load.pml",
              all((pm / f"points_{c}.pdb").exists() for c in
                  ("type_2", "salience", "atomnet_0", "plip_donor"))
              and (pm / "load.pml").exists() and (pm / "ligand.pdb").exists())
        line = (pm / "points_type_2.pdb").read_text().splitlines()[0]
        check("PDB lines are fixed-width (coordinates in columns 31-54)",
              len(line) >= 66 and float(line[30:38]) == round(float(line[30:38]), 3),
              repr(line[:66]))
        check("report.txt written", (td / "rep" / "report.txt").exists())

        # mesh: a sampled sphere has a known surface, orientation and channel
        R = 8.0
        u = rng.normal(size=(int(4 * np.pi * R * R), 3))
        u /= np.linalg.norm(u, axis=1, keepdims=True)
        V, Tm = mesh_surface(u * R, u)
        E = {tuple(sorted(e)) for a_, b_, c_ in Tm for e in ((a_, b_), (b_, c_), (c_, a_))}
        fn = np.cross(V[Tm[:, 1]] - V[Tm[:, 0]], V[Tm[:, 2]] - V[Tm[:, 0]])
        r = np.linalg.norm(V, axis=1)
        check("mesh: closed sphere (Euler 2), on the samples, faces outward",
              len(V) - len(E) + len(Tm) == 2 and abs(r.mean() - R) < 0.2 and r.std() < 0.1
              and ((fn * V[Tm].mean(1)).sum(1) > 0).mean() > 0.99,
              f"{len(V)} vertices, radius {r.mean():.2f} +- {r.std():.2f}")
        step = (u[:, 2:3] > 0).astype(np.float64)       # 1 on the upper half
        sm = smooth_onto(V, u * R, step, s=1.2)[:, 0]
        far = np.abs(V[:, 2]) > 3.0
        check("smooth_onto: a step channel is exact away from the step, graded across it",
              (np.abs(sm[far] - (V[far, 2] > 0)) < 0.01).all()
              and 0.2 < sm[np.abs(V[:, 2]) < 0.3].mean() < 0.8,
              f"max err away {np.abs(sm[far] - (V[far, 2] > 0)).max():.3f}")
        half = u[u[:, 2] > 0.2]
        Vh, Th = mesh_surface(half * R, half)
        rh = np.linalg.norm(Vh, axis=1)
        check("open patch: the mesh stops within 2 A of the last sample and puts no "
              "cap across the opening",
              len(Th) and Vh[:, 2].min() > 0.2 * R - 2.0 and np.abs(rh - R).max() < 0.5,
              f"apron {0.2 * R - Vh[:, 2].min():.2f} A, max |r - R| {np.abs(rh - R).max():.2f}")
        sph = dict(prots[0], xyz=u * R, normals=u,
                   lig_xyz=np.zeros((3, 3)) + [[0, 0, 5], [0, 1, 5], [1, 0, 5]])
        n = len(u)
        for k in ("feats", "pre_atomnet", "chemo_ref", "point_emb", "types"):
            sph[k] = rng.normal(size=(n, prots[0][k].shape[1]))
        sph.update(pocket_logit=np.zeros(n), sal=np.full(n, 0.5),
                   pocket_label=np.zeros(n, bool), site=np.zeros(n, bool))
        subj, ref, _ = channels(m2, sph)
        html_viewer(m2, (sph, subj, ref, None), td / "mesh.html", crop=0)
        page = (td / "mesh.html").read_text()
        check("mesh viewer: a mesh3d trace with triangles and per-vertex intensity",
              "mesh3d" in page and "surface mesh" in page and "const TRI = [[" in page)

    print("\nPASS" if not fails else f"\nFAIL: {fails}")
    raise SystemExit(1 if fails else 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("dump", help="container, GPU")
    from_train = True
    try:
        from train import add_data_args, add_region_args
    except Exception:                    # CPU venv: dump is unavailable there
        from_train = False
    if from_train:
        add_data_args(p)
        add_region_args(p, defaults=False)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True, help=".npz to write")
    p.add_argument("--tag", default="", help="name in the report (default: run directory)")
    p.add_argument("--n", type=int, default=64, help="val proteins, group-balanced")
    p.add_argument("--groups", default="",
                   help="comma-separated target groups to add (UniProt or up:/pli: ids)")
    p.add_argument("--rows_from", default="", help="an earlier dump: use its proteins")
    p.add_argument("--no_pretrained", action="store_true",
                   help="skip the published-weights AtomNet comparison")
    p = sub.add_parser("report", help="numpy")
    p.add_argument("--dumps", nargs="+", required=True)
    p.add_argument("--plip", default="data/plip", help="plip_labels.py run output")
    p.add_argument("--plip_lig_radius", type=float, default=None,
                   help="only label surface on the side facing the ligand partner")
    p.add_argument("--out", default="out/viz/report")
    p.add_argument("--n_html", type=int, default=12, help="proteins with a viewer")
    p.add_argument("--viewer", choices=("mesh", "points"), default="mesh",
                   help="mesh: triangulated surface, features smoothed onto it")
    p.add_argument("--smooth", type=float, default=1.2,
                   help="mesh viewer: Gaussian smoothing of features onto vertices (A)")
    p.add_argument("--grid", type=float, default=0.6, help="mesh viewer: grid spacing (A)")
    p.add_argument("--crop", type=float, default=18.0,
                   help="viewer / PyMOL: points within this many A of the ligand (0 = all)")
    p.add_argument("--pymol_channels",
                   default="salience,type_0,type_1,type_2,type_3,type_4,type_5,type_6,"
                           "type_7,atomnet_0,atomnet_1,atomnet_2,atomnet_3,atomnet_4,"
                           "atomnet_5,plip_donor,plip_acceptor,plip_hydrophobic")
    sub.add_parser("selftest")
    a = ap.parse_args()
    if a.cmd == "dump" and not from_train:
        raise SystemExit("dump needs train.py's loaders (torch): run it in the container")
    {"dump": dump, "report": report, "selftest": lambda a: selftest()}[a.cmd](a)


if __name__ == "__main__":
    main()
