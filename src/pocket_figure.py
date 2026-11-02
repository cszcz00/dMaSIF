"""Figure: how the pocket logit becomes anchors (the v8 path), on a real protein.

Three panels, one protein, from any checkpoint with --region_source predicted:

    1  the pocket logit on the surface (sigmoid, per point), ligand drawn
    2  the selector: top 5% of the smoothed map = candidates, then up to
       k_pos balls of radius R (ball_R8: greedy seeds, mean-shifted centres),
       numbered in rank order; the one matching the ligand's pocket marked
    3  each ball -> pair pooling -> BatchNorm -> MLP -> a unit 128-d anchor,
       drawn as arrows on a sphere with the ligand's vector; the score is the
       max cosine over anchors. The sphere is a 3D PROJECTION (SVD) of the
       128-d directions, so angles there are approximate; the cosines printed
       beside it are the exact 128-d values.

Two stages:

    dump     container, GPU, ~minutes. A checkpoint + N val proteins -> one
             .npz with, per protein: surface points / normals, pocket logit,
             the selector's candidate mask (the SAME smoothing and top
             fraction as anchors.ball_regions), the anchor regions, their
             128-d anchors, the ligand's 128-d vector, region labels, the
             fallback flag, contact label, in-pocket PLIP weight, ligand atoms.
    figure   numpy only (CPU venv is fine). One protein of a dump -> an SVG
             with the two surface views as embedded PNG renders (a numpy
             point-splat renderer over channel_viz's surface mesh) and
             everything else as vectors. Prints a table of the dump's
             proteins first, so you can pick one with --system_id; the
             default is the best clean success (no fallback, the top-scoring
             anchor is the matching one).
    selftest numpy: a planted protein through figure, end to end.

Example:
    sbatch src/slurm/pocket_figure.sbatch            # dump + figure, v8 best.pt
    python src/pocket_figure.py figure --dump out/figures/pocket_v8.npz \\
        --out docs/figures/pocket_selection.svg --system_id <id>

Depends on: numpy, channel_viz (mesh); torch + train.py's loaders (dump only).
"""

import argparse
import base64
import json
import struct
import sys
import zlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

PER_PROT = ("xyz", "normals", "pocket_logit", "cand", "contact", "plip_pocket",
            "lig_xyz", "region_ptr", "region_idx", "region_label", "anchors",
            "anchor_cos", "lig_vec", "fallback_idx", "fallback_vec")


# ---------------------------------------------------------------------------
# dump (container, GPU)
# ---------------------------------------------------------------------------
def dump(a):
    import torch

    import anchors as anchors_mod
    from channel_viz import pick_rows
    from train import (POCKET_DEFAULTS, anchor_block, build_batch, encode_ligands,
                       load_model_from_ckpt, load_rows, pocket_source,
                       use_neighbor_device)
    from train_data import LabelCache, SurfaceStore, fallback_region

    use_neighbor_device(a, a.device)
    model, ckpt, saved = load_model_from_ckpt(a.ckpt, a.device, overrides=a)
    cfg = model.encoder.cfg
    if cfg.region_source != "predicted":
        print(f"[warn] {a.ckpt} has region_source {cfg.region_source}: the regions "
              "are not the pocket logit's, so panel 2 does not show the selector")
    a.n_decoys = 0
    table, ligands = load_rows(a)
    for k, v in POCKET_DEFAULTS.items():
        if getattr(a, k, None) is None:
            setattr(a, k, getattr(saved, k, None) or v)
    store = SurfaceStore(a.surface_cache, ram=a.surface_ram, device=a.device,
                         keep_hydrogens=saved.keep_hydrogens,
                         pockets=pocket_source(a, cfg),
                         atom_feats=getattr(cfg, "atom_feats", False))
    plip = a.plip or getattr(saved, "plip", "")
    plip = plip if plip and Path(plip).is_dir() else None
    labels = LabelCache(table, cutoff=a.cutoff, ram=a.label_ram, plip=plip,
                        plip_soft=getattr(saved, "plip_soft", 0.0),
                        plip_union=False, plip_pocket=plip is not None)
    tr = np.asarray(ckpt["train_rows"])
    ligands.set_scaler(table.fp_row[tr[tr < len(table)]])
    rows = pick_rows(table, ckpt, saved, a)
    print(f"{a.ckpt}: step {ckpt['step']}; {len(rows)} proteins; regions "
          f"{cfg.region_source}/{cfg.region_method} R={cfg.region_ball_radius} "
          f"top {cfg.region_top_fraction} k_pos {cfg.k_pos}")

    out, gen = {}, torch.Generator().manual_seed(a.seed)
    model.eval()
    i_prot = 0
    with torch.no_grad():
        for s in range(0, len(rows), a.eval_batch):
            idx = rows[s:s + a.eval_batch]
            P, labs = build_batch(model, store, labels, table, idx, a.device, 0)
            blk = anchor_block(model, P, labs, a, generator=gen)
            o = blk["out"]
            lig = encode_ligands(model, ligands, table.fp_row[idx], a.device)
            K = o["anchor_valid"].shape[1]
            for b, i in enumerate(idx):
                sel = (o["batch"] == b).nonzero().squeeze(1)
                xyz = o["xyz"][sel].float().cpu()
                prob = torch.sigmoid(o["pocket_logit"][sel].float()).cpu()
                sm = anchors_mod.smoothed(xyz, prob, cfg.smooth_radius)
                cand = anchors_mod.candidates(sm, cfg.region_threshold,
                                              cfg.region_top_fraction)
                regs, lab_r, vec, cos = [], [], [], []
                lv = lig[b]
                for k, r in enumerate(o["anchor_regions"][b]):
                    if not bool(o["anchor_is_pos"][b, k] & o["anchor_valid"][b, k]):
                        continue
                    regs.append((r - sel[0]).cpu().numpy().astype(np.int32))
                    lab_r.append(int(blk["label"][b, k]))
                    v = blk["anchors"][b, k]
                    vec.append(v.float().cpu().numpy())
                    cos.append(float((v * lv).sum()))
                lb = labs[b]
                fb_idx, fb_vec = np.zeros(0, np.int32), np.zeros(0, np.float32)
                if bool(blk["fallback"][b]):
                    pk = lb["pocket"].to(o["xyz"].device)
                    fb_idx = fallback_region(o["xyz"][sel], pk, cfg.region_ball_radius
                                             ).cpu().numpy().astype(np.int32)
                    fb_vec = blk["anchors"][b, K].float().cpu().numpy()
                pp = lb.get("plip_pocket")
                row = table.rows[int(i)]
                d = {"xyz": xyz.numpy(), "normals": o["normals"][sel].float().cpu().numpy(),
                     "pocket_logit": o["pocket_logit"][sel].float().cpu().numpy(),
                     "cand": cand.numpy(), "contact": lb["pocket"].cpu().numpy(),
                     "plip_pocket": (np.zeros(len(sel), np.float16) if pp is None
                                     else np.asarray(pp, np.float16)),
                     "lig_xyz": np.asarray(lb["lig"], np.float32),
                     "region_ptr": np.cumsum([0] + [len(r) for r in regs]).astype(np.int64),
                     "region_idx": (np.concatenate(regs) if regs else np.zeros(0, np.int32)),
                     "region_label": np.asarray(lab_r, np.int8),
                     "anchors": np.asarray(vec, np.float32).reshape(len(vec), -1),
                     "anchor_cos": np.asarray(cos, np.float32),
                     "lig_vec": lv.float().cpu().numpy(),
                     "fallback_idx": fb_idx, "fallback_vec": fb_vec}
                for k, v in d.items():
                    out[f"p{i_prot}_{k}"] = v
                out[f"p{i_prot}_system_id"] = np.asarray(row["system_id"])
                out[f"p{i_prot}_group"] = np.asarray(row["group"])
                i_prot += 1
            print(f"    {s + len(idx)}/{len(rows)}", end="\r", flush=True)
    meta = {"ckpt": str(a.ckpt), "step": int(ckpt["step"]), "n": i_prot,
            "k_pos": cfg.k_pos, "ball_radius": cfg.region_ball_radius,
            "top_fraction": cfg.region_top_fraction, "smooth_radius": cfg.smooth_radius,
            "region_method": cfg.region_method, "region_source": cfg.region_source,
            "pool_mode": cfg.pool_mode, "pair_types": cfg.pair_types,
            "pair_bins": cfg.pair_bins, "out_dim": cfg.out_dim,
            "tau": 1.0 / float(model.scale().detach()),
            "pocket_target": getattr(saved, "pocket_target", "contact")}
    out["meta_json"] = np.asarray(json.dumps(meta))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, **out)
    print(f"\nwrote {a.out}: {i_prot} proteins")


def load_dump(path):
    d = np.load(path, allow_pickle=False)
    meta = json.loads(str(d["meta_json"]))
    prots = []
    for i in range(meta["n"]):
        p = {k: d[f"p{i}_{k}"] for k in PER_PROT}
        p["system_id"] = str(d[f"p{i}_system_id"])
        p["group"] = str(d[f"p{i}_group"])
        ptr = p.pop("region_ptr")
        idx = p.pop("region_idx")
        p["regions"] = [idx[ptr[j]:ptr[j + 1]] for j in range(len(ptr) - 1)]
        prots.append(p)
    return meta, prots


# ---------------------------------------------------------------------------
# rendering (numpy)
# ---------------------------------------------------------------------------
REGION_COL = ["#2f6fdf", "#e07b1a", "#2a9d6f", "#9a6fd8", "#d64f6f", "#3fa7c4",
              "#b59a2a", "#7d8590"]
INK, MUTED, FAINT, RULE = "#1f2328", "#59636e", "#8c959f", "#d0d7de"
BASE_RGB = np.array([0.86, 0.88, 0.91])
# sequential map for pocket probability: light grey -> orange -> dark red
PROB_STOPS = [(0.0, (0.90, 0.91, 0.93)), (0.35, (0.95, 0.80, 0.62)),
              (0.7, (0.90, 0.45, 0.16)), (1.0, (0.58, 0.14, 0.05))]


def hex_rgb(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)])


def prob_rgb(p):
    p = np.clip(p, 0, 1)
    xs = np.array([s[0] for s in PROB_STOPS])
    cs = np.array([s[1] for s in PROB_STOPS])
    return np.stack([np.interp(p, xs, cs[:, c]) for c in range(3)], -1)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def vertex_normals(V, T):
    n = np.zeros_like(V)
    f = np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]])
    for k in range(3):
        np.add.at(n, T[:, k], f)
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)


def camera(centre, toward):
    """Orthonormal (x, y, z) with z pointing from the protein toward the viewer."""
    z = toward / np.linalg.norm(toward)
    up = np.array([0.0, 0.0, 1.0]) if abs(z[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.stack([x, y, z])


class View:
    """Orthographic projection of a scene into an S x S image."""

    def __init__(self, pts, R, size, margin=0.06):
        self.c = pts.mean(0)
        self.R = R
        q = (pts - self.c) @ R.T
        half = np.abs(q[:, :2]).max() * (1 + margin)
        self.size, self.scale = size, size / (2 * half)

    def proj(self, p):
        q = (np.atleast_2d(p) - self.c) @ self.R.T
        x = self.size / 2 + q[:, 0] * self.scale
        y = self.size / 2 - q[:, 1] * self.scale       # image y down
        return x, y, q[:, 2]


def splat(view, V, N, rgb, radius_px):
    """Z-buffered disc splats with Lambert + a little specular -> (RGBA uint8
    image, depth map with -inf where empty)."""
    S = view.size
    x, y, z = view.proj(V)
    nz = N @ view.R[2]
    shade = 0.38 + 0.62 * np.clip(nz, 0, 1) + 0.12 * np.clip(nz, 0, 1) ** 24
    col = np.clip(rgb * shade[:, None], 0, 1)
    r = int(np.ceil(radius_px))
    off = [(dx, dy) for dx in range(-r, r + 1) for dy in range(-r, r + 1)
           if dx * dx + dy * dy <= radius_px * radius_px]
    ix, iy = np.round(x).astype(int), np.round(y).astype(int)
    F, D, I = [], [], []
    vid = np.arange(len(V))
    for dx, dy in off:
        px, py = ix + dx, iy + dy
        ok = (px >= 0) & (px < S) & (py >= 0) & (py < S) & (nz > -0.2)
        F.append(py[ok] * S + px[ok])
        D.append(z[ok])
        I.append(vid[ok])
    F, D, I = np.concatenate(F), np.concatenate(D), np.concatenate(I)
    order = np.lexsort((D, F))                 # by pixel, then depth ascending
    F, D, I = F[order], D[order], I[order]
    last = np.r_[F[1:] != F[:-1], True]        # nearest (largest z) per pixel
    F, D, I = F[last], D[last], I[last]
    img = np.zeros((S * S, 4))
    img[F, :3] = col[I]
    img[F, 3] = 1
    depth = np.full(S * S, -np.inf)
    depth[F] = D
    return (img.reshape(S, S, 4) * 255).round().astype(np.uint8), depth.reshape(S, S)


def png_bytes(rgba):
    h, w = rgba.shape[:2]
    raw = b"".join(b"\x00" + rgba[r].tobytes() for r in range(h))

    def chunk(t, d):
        return (struct.pack(">I", len(d)) + t + d
                + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


def occluded(depth, x, y, z, tol):
    S = depth.shape[0]
    i, j = int(np.clip(round(y), 0, S - 1)), int(np.clip(round(x), 0, S - 1))
    return np.isfinite(depth[i, j]) and depth[i, j] > z + tol


def choose(prots):
    """Default protein: no fallback, best anchor is a match; then the largest
    cosine margin of that anchor over the next."""
    best, key = 0, None
    for i, p in enumerate(prots):
        if not len(p["anchor_cos"]):
            continue
        j = int(np.argmax(p["anchor_cos"]))
        ok = (not len(p["fallback_idx"])) and p["region_label"][j] == 1
        srt = np.sort(p["anchor_cos"])[::-1]
        margin = srt[0] - (srt[1] if len(srt) > 1 else -1)
        k = (ok, margin)
        if key is None or k > key:
            best, key = i, k
    return best


def protein_table(prots):
    print(f"{'#':>3} {'system_id':40s} {'points':>7} {'anchors':>7} {'fb':>3} "
          f"{'best':>5} {'label':>5} {'cos':>6}")
    for i, p in enumerate(prots):
        c = p["anchor_cos"]
        j = int(np.argmax(c)) if len(c) else -1
        print(f"{i:3d} {p['system_id'][:40]:40s} {len(p['xyz']):7d} {len(c):7d} "
              f"{'yes' if len(p['fallback_idx']) else '':>3} {j + 1:5d} "
              f"{int(p['region_label'][j]) if j >= 0 else 0:5d} "
              f"{(c[j] if j >= 0 else np.nan):6.3f}")


# ---------------------------------------------------------------------------
# figure (numpy)
# ---------------------------------------------------------------------------
def figure(a, meta=None, prots=None):
    from channel_viz import _min_dist, mesh_surface, smooth_onto

    if prots is None:
        meta, prots = load_dump(a.dump)
    protein_table(prots)
    if a.system_id:
        hit = [i for i, p in enumerate(prots) if p["system_id"] == a.system_id]
        if not hit:
            raise SystemExit(f"--system_id {a.system_id} is not in the dump")
        pi = hit[0]
    else:
        pi = a.index if a.index is not None else choose(prots)
    p = prots[pi]
    print(f"\nfigure: #{pi} {p['system_id']}")

    xyz, nrm = p["xyz"].astype(np.float64), p["normals"].astype(np.float64)
    nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-9)
    prob = sigmoid(p["pocket_logit"].astype(np.float64))
    V, T = mesh_surface(xyz, nrm, h=a.grid)
    if not len(T):
        raise SystemExit("empty mesh; try a larger --grid")
    VN = vertex_normals(V, T)
    # nearest sample point per vertex, for categorical channels
    near = np.empty(len(V), int)
    for s in range(0, len(V), 2048):
        d2 = ((V[s:s + 2048, None] - xyz[None]) ** 2).sum(-1)
        near[s:s + 2048] = d2.argmin(1)
    vprob = smooth_onto(V, xyz, prob[:, None], s=1.2)[:, 0]

    n_reg = len(p["regions"])
    reg_of = -np.ones(len(xyz), int)               # best-ranked region per point
    for k in range(n_reg - 1, -1, -1):
        reg_of[p["regions"][k]] = k
    # a ball's badge sits on its SURFACE point nearest the ball's centroid:
    # the centroid of a curved patch lies under the surface, and a depth test
    # on it would call every ball hidden
    centres = []
    for r in p["regions"]:
        c = xyz[r].mean(0)
        centres.append(xyz[r][((xyz[r] - c) ** 2).sum(1).argmin()])
    lig_c = p["lig_xyz"].mean(0)
    R = camera(xyz.mean(0), lig_c - xyz.mean(0))
    S = a.px
    view = View(np.vstack([V, p["lig_xyz"]]), R, S)
    rad = max(1.2, 0.85 * a.grid * view.scale)

    img1, dep1 = splat(view, V, VN, prob_rgb(vprob), rad)
    rgb2 = np.tile(BASE_RGB, (len(V), 1))
    cand_v = p["cand"][near].astype(bool)
    rgb2[cand_v] = np.array([0.98, 0.86, 0.70])
    rv = reg_of[near]
    for k in range(n_reg):
        m = rv == k
        c = hex_rgb(REGION_COL[k % len(REGION_COL)])
        rgb2[m] = 0.35 * rgb2[m] + 0.65 * c
    img2, dep2 = splat(view, V, VN, rgb2, rad)

    # ------------------------------------------------------------ SVG
    out = []
    w = out.append

    def esc(s):
        return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    def text(x, y, s, size=13, col=INK, anchor="start", weight="normal", extra=""):
        w(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{col}" '
          f'text-anchor="{anchor}" font-weight="{weight}" {extra}>{esc(s)}</text>')

    def circle(x, y, r, fill, stroke="none", sw=1, extra=""):
        w(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="{fill}" stroke="{stroke}" '
          f'stroke-width="{sw}" {extra}/>')

    def line(x1, y1, x2, y2, col=FAINT, sw=1.5, extra=""):
        w(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{col}" '
          f'stroke-width="{sw}" {extra}/>')

    def image(x, y, rgba, size):
        b64 = base64.b64encode(png_bytes(rgba)).decode("ascii")
        w(f'<image x="{x}" y="{y}" width="{size}" height="{size}" '
          f'href="data:image/png;base64,{b64}"/>')

    Wd, Hd, D = 1800, 770, 500           # canvas, displayed panel size
    k_disp = D / S
    w(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {Wd} {Hd}" width="{Wd}" '
      f'height="{Hd}" font-family="Helvetica, Arial, sans-serif">')
    w('<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
      f'markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" '
      f'fill="{FAINT}"/></marker></defs>')
    w(f'<rect width="{Wd}" height="{Hd}" fill="#ffffff"/>')
    text(40, 50, "From pocket logit to anchors: where the protein is read, and what each "
         "read becomes", 24, INK, weight="bold")
    text(40, 78, f"{p['system_id']}  ·  target {p['group']}  ·  checkpoint step "
         f"{meta['step']:,}  ·  ligand drawn for reference only (the selector never sees it)",
         14, MUTED)

    X1, X2, X3, Y0 = 40, 600, 1160, 130
    # panel 1
    text(X1, Y0 - 12, "1 · Pocket logit on every surface point", 17, INK, weight="bold")
    image(X1, Y0, img1, D)

    def ligand(x0, y0):
        lx, ly, _ = view.proj(p["lig_xyz"])
        for xx, yy in zip(lx, ly):
            circle(x0 + xx * k_disp, y0 + yy * k_disp, max(1.8, 0.33 * view.scale * k_disp),
                   "#1f2328", "#ffffff", 0.6, 'opacity="0.75"')
    ligand(X1, Y0)
    # colour bar
    cb_y = Y0 + D + 22
    for i in range(100):
        r, g, b = (prob_rgb(np.array([i / 99]))[0] * 255).astype(int)
        w(f'<rect x="{X1 + 120 + i * 2.4:.1f}" y="{cb_y}" width="2.6" height="10" '
          f'fill="rgb({r},{g},{b})"/>')
    text(X1, cb_y + 9, "sigmoid(logit)", 12, MUTED)
    text(X1 + 120, cb_y + 26, "0", 11, MUTED)
    text(X1 + 360, cb_y + 26, "1", 11, MUTED, "end")
    tgt = ("trained toward: 4 Å contact = 0.5, PLIP-interacting = 1, elsewhere 0"
           if meta.get("pocket_target") == "contact_plip"
           else "trained toward: 4 Å ligand contact = 1, elsewhere 0")
    text(X1, cb_y + 48, tgt, 12.5, MUTED)
    text(X1, cb_y + 66, "● ligand heavy atoms (crystal pose)", 12.5, MUTED)

    # panel 2
    tf = meta.get("top_fraction")
    text(X2, Y0 - 12, f"2 · Top {100 * tf:.0f}% → up to {meta['k_pos']} balls of "
         f"{meta['ball_radius']:.0f} Å" if tf else
         f"2 · Candidates → up to {meta['k_pos']} balls of {meta['ball_radius']:.0f} Å",
         17, INK, weight="bold")
    image(X2, Y0, img2, D)
    ligand(X2, Y0)
    match = [k for k in range(n_reg) if p["region_label"][k] == 1]
    for k, c in enumerate(centres):
        cx, cy, cz = view.proj(c)
        x, y = X2 + cx[0] * k_disp, Y0 + cy[0] * k_disp
        hid = occluded(dep2, cx[0], cy[0], cz[0], tol=1.5)
        col = REGION_COL[k % len(REGION_COL)]
        if hid:
            circle(x, y, 11, "#ffffff", col, 2, 'stroke-dasharray="3 2" opacity="0.9"')
            text(x, y + 4.5, k + 1, 12, col, "middle", "bold")
        else:
            circle(x, y, 11, col, "#ffffff", 2)
            text(x, y + 4.5, k + 1, 12, "#ffffff", "middle", "bold")
        if k in match:
            circle(x, y, 16, "none", INK, 2)
    ly = Y0 + D + 31
    w(f'<rect x="{X2}" y="{ly - 9}" width="14" height="10" fill="#fadcb3"/>')
    text(X2 + 20, ly, f"candidates (smoothed over {meta['smooth_radius']:.0f} Å)", 12.5, MUTED)
    circle(X2 + 7, ly + 18, 7, REGION_COL[0], "#ffffff", 1.5)
    text(X2 + 20, ly + 22, "ball centre, numbered in rank order; dashed = on the far side",
         12.5, MUTED)
    circle(X2 + 7, ly + 40, 9, "none", INK, 1.8)
    text(X2 + 20, ly + 44, "overlaps the ligand's pocket (cover ≥ 0.5, precision ≥ 0.3)",
         12.5, MUTED)
    if len(p["fallback_idx"]):
        text(X2, ly + 64, "no ball matched the pocket: training used a fallback ball there",
             12.5, "#d64f6f")

    # arrows between panels
    w(f'<line x1="{X1 + D + 8}" y1="{Y0 + D / 2}" x2="{X2 - 10}" y2="{Y0 + D / 2}" '
      f'stroke="{FAINT}" stroke-width="2" marker-end="url(#ah)"/>')
    w(f'<line x1="{X2 + D + 8}" y1="{Y0 + D / 2}" x2="{X3 - 10}" y2="{Y0 + D / 2}" '
      f'stroke="{FAINT}" stroke-width="2" marker-end="url(#ah)"/>')

    # panel 3: anchors as directions
    nb = meta["pair_types"] * (meta["pair_types"] + 1) // 2 * meta["pair_bins"]
    text(X3, Y0 - 12, f"3 · Each ball → one {meta['out_dim']}-d anchor", 17, INK,
         weight="bold")
    text(X3, Y0 + 8, f"pair pooling ({meta['pair_types']} site types² × "
         f"{meta['pair_bins']} distance bins = {nb})" if meta["pool_mode"] == "pairs"
         else f"{meta['pool_mode']} pooling", 12.5, MUTED)
    text(X3, Y0 + 25, "→ BatchNorm → MLP → unit vector", 12.5, MUTED)
    A = p["anchors"].astype(np.float64)
    L = p["lig_vec"].astype(np.float64)
    M = np.vstack([A, L[None]])
    M /= np.maximum(np.linalg.norm(M, axis=1, keepdims=True), 1e-9)
    _, _, Vt = np.linalg.svd(M, full_matrices=False)
    Q = M @ Vt[:3].T
    Q /= np.maximum(np.linalg.norm(Q, axis=1, keepdims=True), 1e-9)
    # rotate so the ligand points up-right and slightly toward the viewer
    tgt_dir = np.array([0.45, 0.75, 0.48])
    tgt_dir /= np.linalg.norm(tgt_dir)
    v = np.cross(Q[-1], tgt_dir)
    s, c = np.linalg.norm(v), float(Q[-1] @ tgt_dir)
    if s > 1e-9:
        K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        Rot = np.eye(3) + K + K @ K * ((1 - c) / s ** 2)
        Q = Q @ Rot.T
    cx, cy, rs = X3 + 165, Y0 + 210, 140
    circle(cx, cy, rs, "#f6f8fa", RULE, 1.2)
    w(f'<ellipse cx="{cx}" cy="{cy}" rx="{rs}" ry="{rs * 0.28:.1f}" fill="none" '
      f'stroke="{RULE}" stroke-dasharray="4 3"/>')
    order = np.argsort(Q[:, 2])                    # back to front
    for i in order:
        q = Q[i]
        x, y = cx + rs * q[0], cy - rs * q[1]
        back = q[2] < 0
        if i == len(Q) - 1:
            col, sw, lab = INK, 3.2, "ligand"
        else:
            col, sw, lab = REGION_COL[i % len(REGION_COL)], 2.2, str(i + 1)
        op = 0.45 if back else 1.0
        line(cx, cy, x, y, col, sw, f'opacity="{op}" stroke-linecap="round"')
        circle(x, y, 4.5 if i == len(Q) - 1 else 3.5, col, "#ffffff", 1, f'opacity="{op}"')
        dx, dy = q[0], -q[1]
        nrm2 = max(np.hypot(dx, dy), 1e-6)
        text(x + 12 * dx / nrm2, y + 12 * dy / nrm2 + 4, lab, 12.5 if lab == "ligand" else 12,
             col, "middle", "bold", f'opacity="{op}"')
    text(cx, cy + rs + 22, f"directions projected from {meta['out_dim']}-d to 3-d (SVD); "
         "angles approximate", 11.5, FAINT, "middle")

    # cosine bars (exact)
    cosv = p["anchor_cos"]
    best = int(np.argmax(cosv)) if len(cosv) else -1
    by = Y0 + 420
    text(X3, by - 14, "cos(anchor, ligand), exact", 13, INK, weight="bold")
    bx, bw = X3 + 52, 300
    lo = min(0.0, float(cosv.min())) if len(cosv) else 0.0
    hi = max(1e-6, float(cosv.max())) if len(cosv) else 1.0
    zero = bx + bw * (0 - lo) / (hi - lo)
    for k, v_ in enumerate(cosv):
        y = by + k * 22
        col = REGION_COL[k % len(REGION_COL)]
        x0, x1 = sorted([zero, bx + bw * (v_ - lo) / (hi - lo)])
        w(f'<rect x="{x0:.1f}" y="{y}" width="{max(x1 - x0, 1):.1f}" height="14" rx="2" '
          f'fill="{col}" opacity="{1.0 if k == best else 0.55}"/>')
        text(X3, y + 11.5, f"ball {k + 1}", 12, MUTED)
        lbl = f"{v_:+.2f}".replace("-0.00", "0.00").replace("+0.00", "0.00")
        text(max(x1, zero) + 6, y + 11.5, lbl + ("  ← max" if k == best else ""),
             12, INK if k == best else MUTED, weight="bold" if k == best else "normal")
    line(zero, by - 3, zero, by + 22 * len(cosv) - 6, RULE, 1)
    sy = by + 22 * max(len(cosv), 1) + 22
    text(X3, sy, "score(protein, ligand) = max over balls of cos / τ", 13.5, INK,
         weight="bold")
    text(X3, sy + 19, f"τ = {meta['tau']:.3f}; trained by InfoNCE against the batch's "
         "other proteins and ligands", 12, MUTED)

    w("</svg>")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text("\n".join(out) + "\n")
    print(f"wrote {a.out}  ({Path(a.out).stat().st_size / 1e6:.1f} MB; "
          f"{len(V):,} mesh vertices, {n_reg} balls, best ball {best + 1}, "
          f"label {int(p['region_label'][best]) if best >= 0 else 'n/a'})")
    return a.out


# ---------------------------------------------------------------------------
def planted(seed=0):
    """A bumpy sphere with a dent, a ligand in the dent, logit high there,
    8 greedy balls on the top 5%, anchors with ball 1 nearest the ligand."""
    rng = np.random.default_rng(seed)
    n = 9000
    u = rng.normal(size=(n, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    dent_dir = np.array([0.3, 0.2, 0.93]) / np.linalg.norm([0.3, 0.2, 0.93])
    cosang = u @ dent_dir
    r = 18 + 1.2 * np.sin(5 * u[:, 0]) * np.cos(4 * u[:, 1]) - 5 * np.exp(-(1 - cosang) * 12)
    xyz = u * r[:, None]
    nrm = u.copy()
    lig = dent_dir * 15.5 + rng.normal(scale=1.6, size=(22, 3))
    d = np.sqrt(((xyz[:, None] - lig[None]) ** 2).sum(-1)).min(1)
    contact = d < 4.0
    logit = 4 * np.exp(-(1 - cosang) * 6) - 3 + 2.2 * np.exp(-(1 - u @ np.array([-1, 0, 0])) * 5) \
        + rng.normal(scale=0.3, size=n)
    prob = sigmoid(logit)
    cand = prob >= np.quantile(prob, 0.95)
    avail, regions = cand.copy(), []
    while avail.any() and len(regions) < 8:
        idx = np.nonzero(avail)[0]
        seed_i = idx[np.argmax(prob[idx])]
        c = xyz[seed_i]
        d2 = ((xyz - c) ** 2).sum(1)
        avail &= d2 >= 64
        avail[seed_i] = False
        reg = np.nonzero(d2 < 64)[0]
        if len(reg) >= 20:
            regions.append(reg.astype(np.int32))
    lab = []
    for reg in regions:
        cover = contact[reg].sum() / max(contact.sum(), 1)
        prec = contact[reg].mean()
        lab.append(1 if (cover >= 0.5 and prec >= 0.3) else
                   (-1 if (cover >= 0.1 or prec >= 0.1) else 0))
    D = 128
    L = rng.normal(size=D)
    L /= np.linalg.norm(L)
    A = rng.normal(size=(len(regions), D))
    A[0] += 6 * L
    A /= np.linalg.norm(A, axis=1, keepdims=True)
    p = {"xyz": xyz.astype(np.float32), "normals": nrm.astype(np.float32),
         "pocket_logit": logit.astype(np.float32), "cand": cand, "contact": contact,
         "plip_pocket": np.zeros(n, np.float16), "lig_xyz": lig.astype(np.float32),
         "regions": regions, "region_label": np.asarray(lab, np.int8),
         "anchors": A.astype(np.float32), "anchor_cos": (A @ L).astype(np.float32),
         "lig_vec": L.astype(np.float32), "fallback_idx": np.zeros(0, np.int32),
         "fallback_vec": np.zeros(0, np.float32), "system_id": "planted__1__1.A__1.B",
         "group": "up:PLANTED"}
    meta = {"ckpt": "planted", "step": 1234, "n": 1, "k_pos": 8, "ball_radius": 8.0,
            "top_fraction": 0.05, "smooth_radius": 2.0, "region_method": "ball",
            "region_source": "predicted", "pool_mode": "pairs", "pair_types": 8,
            "pair_bins": 8, "out_dim": D, "tau": 0.05, "pocket_target": "contact_plip"}
    return meta, [p]


def selftest(a):
    import tempfile

    fails = []

    def check(name, ok, detail=""):
        print(f"  {'ok  ' if ok else 'FAIL'} {name}" + (f"   {detail}" if detail else ""))
        if not ok:
            fails.append(name)

    # PNG writer round-trip
    img = np.zeros((7, 5, 4), np.uint8)
    img[2, 3] = (10, 20, 30, 255)
    b = png_bytes(img)
    raw = zlib.decompress(b[b.index(b"IDAT") + 4:b.index(b"IEND") - 8])
    check("png: signature and IDAT decode to the pixels",
          b[:8] == b"\x89PNG\r\n\x1a\n" and len(raw) == 7 * (1 + 5 * 4)
          and raw[2 * 21 + 1 + 3 * 4: 2 * 21 + 1 + 3 * 4 + 4] == bytes([10, 20, 30, 255]))
    # z-buffer: of two overlapping splats the nearer one wins
    R = np.eye(3)
    V = np.array([[0, 0, 0.0], [0, 0, 5.0], [10, 10, 0]])
    N = np.array([[0, 0, 1.0]] * 3)
    view = View(V, R, 64)
    rgb = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]], float)
    im, dep = splat(view, V, N, rgb, 2)
    x, y, _ = view.proj(V[1])
    px = im[int(round(y[0])), int(round(x[0]))]
    check("splat: nearer point wins the pixel", px[2] > 0 and px[0] == 0, str(px))
    check("occluded: a point behind the surface is hidden, one in front is not",
          occluded(dep, x[0], y[0], 0.0, 1.0) and not occluded(dep, x[0], y[0], 6.0, 1.0))
    meta, prots = planted()
    check("planted: ball 1 matches the ligand's pocket", prots[0]["region_label"][0] == 1,
          str(prots[0]["region_label"].tolist()))
    with tempfile.TemporaryDirectory() as td:
        ns = argparse.Namespace(dump=None, system_id="", index=None, grid=0.8, px=640,
                                out=str(Path(td) / "f.svg"))
        figure(ns, meta, prots)
        svg = Path(ns.out).read_text()
        check("figure: an SVG with two embedded renders and every ball numbered",
              svg.count("data:image/png;base64,") == 2
              and all(f">{k + 1}</text>" in svg for k in range(len(prots[0]["regions"]))))
        check("figure: the max cosine is marked on the best ball", "← max" in svg)
        if a.keep:
            Path(a.keep).write_text(svg)
            print(f"  kept {a.keep}")
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
    except Exception:
        from_train = False
    if from_train:
        add_data_args(p)
        add_region_args(p, defaults=False)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--plip", default="", help="PLIP dir (default: the checkpoint's)")
    p.add_argument("--n", type=int, default=12, help="val proteins, group-balanced")
    p.add_argument("--groups", default="", help="comma-separated UniProt ids to add")
    p.add_argument("--rows_from", default="", help="(channel_viz) reuse a dump's proteins")
    p = sub.add_parser("figure", help="numpy")
    p.add_argument("--dump", required=True)
    p.add_argument("--out", default="docs/figures/pocket_selection.svg")
    p.add_argument("--system_id", default="")
    p.add_argument("--index", type=int, default=None)
    p.add_argument("--grid", type=float, default=0.7, help="mesh spacing, A")
    p.add_argument("--px", type=int, default=1000, help="render size per panel, pixels")
    p = sub.add_parser("selftest")
    p.add_argument("--keep", default="", help="also write the planted figure here")
    a = ap.parse_args()
    if a.cmd == "dump" and not from_train:
        raise SystemExit("dump needs train.py's loaders (torch): run it in the container")
    {"dump": dump, "figure": figure, "selftest": selftest}[a.cmd](a)


if __name__ == "__main__":
    main()
