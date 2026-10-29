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


def html_viewer(meta, x, path, crop=18.0):
    """One protein -> a self-contained plotly page with a channel dropdown."""
    p, subj, ref, lab = x
    d = np.sqrt(((p["xyz"][:, None] - p["lig_xyz"][None]) ** 2).sum(-1)).min(1)
    keep = d <= crop if crop else np.ones(len(d), bool)
    chans = {}
    for group, src in (("learned", subj), ("reference", ref)):
        for n, v in src.items():
            chans[n] = (group, v[keep])
    chans["pocket_label"] = ("labels", p["pocket_label"][keep].astype(np.float32))
    chans["site"] = ("labels", p["site"][keep].astype(np.float32))
    if lab is not None:
        for c, n in enumerate(PLIP_CLASSES):
            chans[f"plip_{n}"] = ("labels", lab[keep, c].astype(np.float32))
    data = {}
    for n, (g, v) in chans.items():
        v = np.nan_to_num(v.astype(np.float64))
        if g == "labels":
            lo, hi, scale = 0.0, 1.0, "binary"
        elif n.startswith("curv_"):
            hi = float(np.percentile(np.abs(v), 98)) or 1.0
            lo, scale = -hi, "signed"
        else:
            lo, hi = (float(q) for q in np.percentile(v, [2, 98]))
            hi, scale = (hi if hi > lo else lo + 1.0), "seq"
        data[n] = {"g": g, "v": np.round(v, 4).tolist(), "lo": lo, "hi": hi, "s": scale}
    xyz = np.round(p["xyz"][keep], 2)
    bx, by, bz = _bonds(p["lig_xyz"])
    first = "type_0" if "type_0" in data else next(iter(data))
    title = (f"{meta['tag']} - {p['system_id']} ({p['group']}, {p['split']}); "
             f"{int(keep.sum())} points within {crop:g} A of the ligand")
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
const L = {json.dumps(np.round(p['lig_xyz'], 2).tolist())};
const SC = {{signed: 'RdBu', seq: 'Viridis', binary: [[0, '#d9d9d9'], [1, '#d62728']]}};
function trace(n) {{
  const c = D[n];
  return {{type: 'scatter3d', mode: 'markers', x: X, y: Y, z: Z, name: n,
    marker: {{size: 3, color: c.v, colorscale: SC[c.s], cmin: c.lo, cmax: c.hi,
      reversescale: c.s === 'signed', colorbar: {{title: n, thickness: 14}}}},
    hovertemplate: n + ': %{{marker.color:.3f}}<extra></extra>'}};
}}
const lig = {{type: 'scatter3d', mode: 'markers', name: 'ligand',
  x: L.map(a => a[0]), y: L.map(a => a[1]), z: L.map(a => a[2]),
  marker: {{size: 5, color: '#111'}}, hoverinfo: 'skip'}};
const bonds = {{type: 'scatter3d', mode: 'lines', name: 'bonds', x: {json.dumps(bx)},
  y: {json.dumps(by)}, z: {json.dumps(bz)}, line: {{color: '#111', width: 5}},
  hoverinfo: 'skip'}};
const layout = {{margin: {{l: 0, r: 0, t: 0, b: 0}}, showlegend: false,
  scene: {{aspectmode: 'data', xaxis: {{visible: false}}, yaxis: {{visible: false}},
           zaxis: {{visible: false}}}}}};
Plotly.newPlot('plot', [trace('{first}'), lig, bonds], layout);
document.getElementById('ch').onchange = e => {{
  const t = trace(e.target.value);
  Plotly.restyle('plot', {{marker: [t.marker], name: [t.name],
    hovertemplate: [t.hovertemplate]}}, [0]);
}};
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
            html_viewer(meta, x, hdir / f"{x[0]['system_id']}.html", crop=a.crop)
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
