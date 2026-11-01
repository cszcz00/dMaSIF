"""Docking vs. our scoring: one simulation per pair, or two encoders and a dot product.

Same pocket and molecules as the intro slide (intro_fig.py): one items.tsv
row's pocket, its crystal binder and the first decoys of the same random draw.

  left   docking: the pocket fans out to every ligand, one docking run per
         pocket-ligand pair - cost grows with pockets x ligands, and a new
         pocket repeats the whole library.
  right  ours: a surface encoder turns the pocket into one vector, a ligand
         encoder turns each ligand into one, both in the same space; a ligand
         is scored by its angle to the pocket (cos theta, a dot product).
         Ligand vectors are computed once and reused for every pocket.

The 3D space is an ILLUSTRATION, not model output: the real vectors have
128 dimensions, and the arrows are placed to show the idea (the binder near
the pocket, decoys away from it). The figure says so.

Example (row 2 of big_targets is 4J7G, as on the intro slide):
    python plots/docking_vs_vectors.py --subset plots/subsets/big_targets/big_targets.tsv \\
        --items out/runs/dataset_v2/items.tsv --row 2 --out plots/output/docking_vs_vectors

Depends on: numpy, pandas, matplotlib, rdkit, biopython; intro_fig,
pipeline_fig, contrastive_fig, structview (this folder).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))

POCKET = "#d1495b"      # the pocket's arrow, as the pocket tint elsewhere
BINDER = "#0b0b0b"      # the crystal binder: ink, like the ligand drawings
DECOY = "#9aa0a8"
INK_2 = "#52514e"


def pocket_thumb(ax, row, root=".", half=None):
    """The empty pocket, as in intro_fig panel B: mesh seen down the opening,
    contact patch tinted."""
    from intro_fig import pocket_mesh, pocket_view, draw_mesh
    from pipeline_fig import Camera, frame
    from structview import load_ligand, load_receptor, load_surface

    rec, lig, surf = load_receptor(row, root), load_ligand(row, root), load_surface(row, root)
    cam = Camera(rec["xyz"], lig["xyz"], pocket_view(surf, lig))
    L = cam(lig["xyz"])
    half = half or max(9.0, np.ptp(L[:, :2], axis=0).max() / 2 + 5.0)
    frame(ax, L.mean(0), half)
    V, T = pocket_mesh(surf, lig, half * 1.5)
    draw_mesh(ax, V, T, cam, lig)


def ligand_tile(fig, rect, smiles, binder=False, px=260):
    from contrastive_fig import ligand_image

    g = fig.add_axes(rect)
    g.imshow(ligand_image(smiles, px=px, bond_px=16))
    g.set_xticks([]), g.set_yticks([])
    for sp in g.spines.values():
        sp.set_edgecolor(BINDER if binder else "#d0d4da")
        sp.set_linewidth(2.0 if binder else 0.8)
    return g


def arrow(fig, a, b, text=None, rad=0.0, color="#555"):
    from matplotlib.patches import FancyArrowPatch

    fig.add_artist(FancyArrowPatch(a, b, transform=fig.transFigure, arrowstyle="-|>",
                                   mutation_scale=14, color=color, lw=1.3,
                                   connectionstyle=f"arc3,rad={rad}"))
    if text:
        fig.text((a[0] + b[0]) / 2, (a[1] + b[1]) / 2 + 0.018, text, ha="center",
                 va="bottom", fontsize=9.5, color=INK_2)


def pill(fig, x, y, text, fc="#eef0f3", ec="#c9ced6", color="#333", size=8.5):
    fig.text(x, y, text, ha="center", va="center", fontsize=size, color=color,
             bbox=dict(boxstyle="round,pad=0.3", fc=fc, ec=ec, lw=0.8))


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def rotate_toward(p, d, deg):
    """p rotated by deg degrees in the plane of p and d."""
    p, d = unit(p), unit(d)
    q = unit(d - (d @ p) * p)
    t = np.radians(deg)
    return np.cos(t) * p + np.sin(t) * q


def embedding_space(ax, n_decoys, elev=18, azim=28):
    """Illustrative 3D space: pocket arrow, binder close to it, decoys away,
    the angle between pocket and binder marked. Directions are set in the
    camera's terms (pocket up, binder toward screen-right) so the labels
    land apart whatever the view."""
    az = np.radians(azim)
    right = np.array([-np.sin(az), np.cos(az), 0.0])         # screen-right at this azim
    p = unit([0.12, -0.1, 1.0])
    vecs = [("binder", rotate_toward(p, right, 22), BINDER)]
    dirs = [-right + [0, 0, -0.2], right + [0, 0, -0.6], [0.6, 0.4, -1], -right + [0, 0, -1]]
    angles = [75, 85, 120, 100]
    for k in range(n_decoys):
        vecs.append(("decoy", rotate_toward(p, dirs[k % 4], angles[k % 4]), DECOY))

    # a faint unit sphere for orientation
    u, v = np.mgrid[0:2 * np.pi:28j, 0:np.pi:14j]
    ax.plot_wireframe(np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v),
                      color="#e3e6ea", lw=0.4, rstride=1, cstride=1)
    ax.quiver(0, 0, 0, *p, color=POCKET, lw=3, arrow_length_ratio=0.12)
    ax.text(*(p * 1.12), "pocket  ", color=POCKET, fontsize=11, weight="bold", ha="right")
    for name, w, c in vecs:
        ax.quiver(0, 0, 0, *w, color=c, lw=2.2 if name == "binder" else 1.6,
                  arrow_length_ratio=0.1)
        ax.text(*(w * 1.12), ("  " if name == "binder" else "") + name, color=c,
                fontsize=10, weight="bold" if name == "binder" else "normal",
                ha="left" if name == "binder" else "center")
    # the angle the score reads
    b = vecs[0][1]
    arc = np.array([unit((1 - t) * p + t * b) * 0.45 for t in np.linspace(0, 1, 20)])
    ax.plot(*arc.T, color="#333", lw=1.2)
    m = arc[10] * 1.25
    ax.text(*m, "θ", fontsize=12, color="#333")
    ax.set_xlim(-1, 1), ax.set_ylim(-1, 1), ax.set_zlim(-1, 1)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()


def draw(row, items, out, root=".", n_lig=4, seed=0):
    import matplotlib.pyplot as plt

    from intro_fig import decoy_rows

    dec, n_total = decoy_rows(items, row, n_lig - 1, seed)
    smiles = [row.smiles] + list(dec.smiles)
    fig = plt.figure(figsize=(15.5, 7.2))

    # ---- left: docking -----------------------------------------------------
    fig.text(0.0, 0.985, "Docking: one simulation per pocket–ligand pair", fontsize=14,
             weight="bold", va="top")
    fig.text(0.0, 0.935, "each pair is placed and scored in 3D, from scratch",
             fontsize=10, color=INK_2, va="top")
    pocket_thumb(fig.add_axes([0.135, 0.56, 0.17, 0.32]), row, root)
    slots = n_lig + 1
    w, gap, y0, h = 0.072, 0.012, 0.17, 0.15
    x0 = 0.22 - (slots * w + (slots - 1) * gap) / 2
    top = (0.22, 0.56)
    for k in range(slots):
        x = x0 + k * (w + gap)
        tip = (x + w / 2, y0 + h + 0.005)
        mid = (top[0] + 0.55 * (tip[0] - top[0]), top[1] + 0.55 * (tip[1] - top[1]))
        if k < n_lig:
            ligand_tile(fig, [x, y0, w, h], smiles[k], binder=(k == 0))
            arrow(fig, top, tip)
            pill(fig, *mid, "dock")
            pill(fig, x + w / 2, y0 - 0.035, "score", fc="white")
        else:
            fig.text(x + w / 2, y0 + h / 2, f"…\n× {n_total:,}", ha="center", va="center",
                     fontsize=11, color=INK_2)
    fig.text(0.0, 0.04, f"1 pocket × {n_total:,} ligands = {n_total:,} docking runs;  "
             "a new pocket repeats them all  →  cost ∝ pockets × ligands",
             fontsize=10.5, color="#222")

    # ---- right: ours -------------------------------------------------------
    X = 0.5
    fig.text(X, 0.985, "Ours: encode each side once, score by angle", fontsize=14,
             weight="bold", va="top")
    fig.text(X, 0.935, "proteins and ligands become vectors in one shared space",
             fontsize=10, color=INK_2, va="top")
    pocket_thumb(fig.add_axes([X, 0.58, 0.12, 0.26]), row, root)
    gw, gh = 0.055, 0.12
    for k in range(n_lig):
        r_, c_ = divmod(k, 2)
        ligand_tile(fig, [X + c_ * (gw + 0.008), 0.30 - r_ * (gh + 0.012), gw, gh],
                    smiles[k], binder=(k == 0), px=220)
    ax3 = fig.add_axes([0.67, 0.07, 0.33, 0.86], projection="3d")
    embedding_space(ax3, n_lig - 1)
    arrow(fig, (X + 0.125, 0.71), (0.72, 0.66), "surface encoder", rad=-0.15, color=POCKET)
    arrow(fig, (X + 2 * gw + 0.015, 0.25), (0.72, 0.40), "ligand encoder", rad=0.15)
    fig.text(0.82, 0.16, "score = cos θ  (a dot product)", ha="center", fontsize=11,
             color="#222")
    fig.text(0.82, 0.12, "illustration: the real space has 128 dimensions",
             ha="center", fontsize=8.5, color=INK_2, style="italic")
    fig.text(X, 0.04, f"1 + {n_total:,} encoder passes; ligand vectors reused for every "
             "pocket  →  cost ∝ pockets + ligands", fontsize=10.5, color="#222")
    fig.add_artist(plt.Line2D([0.465, 0.465], [0.03, 0.97], transform=fig.transFigure,
                              color="#e0e2e6", lw=1))

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(out.with_suffix(f".{ext}"), dpi=200, bbox_inches="tight",
                    facecolor="white")
    print(f"wrote {out.with_suffix('.png')} and .svg")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--subset", required=True)
    ap.add_argument("--items", required=True, help="the full items.tsv (decoys, count)")
    ap.add_argument("--row", default="2", help="row number or system_id (2 = 4J7G)")
    ap.add_argument("--out", default="plots/output/docking_vs_vectors")
    ap.add_argument("--n", type=int, default=4, help="ligands shown (binder + decoys)")
    ap.add_argument("--seed", type=int, default=0, help="the decoy draw; 0 = intro slide's")
    a = ap.parse_args()

    from structview import load_row

    row = load_row(a.subset, int(a.row) if a.row.isdigit() else a.row)
    draw(row, a.items, a.out, n_lig=a.n, seed=a.seed)


if __name__ == "__main__":
    main()
