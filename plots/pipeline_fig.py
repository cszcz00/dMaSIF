"""From a crystal complex to a ligand graph and a set of pocket surface points.

One real items.tsv row, five panels left to right:

  1  the complex: receptor atoms with the ligand in place (whole protein; the
     dashed box is the window panels 2 and 4 zoom into, sized to the ligand)
  2  fpocket's alpha spheres; the pocket(s) at the ligand's site in orange,
     the centres of every other pocket fpocket found as blue dots
  3  the split: the receptor as 3D atoms, the ligand as a 2D graph - the two
     inputs the two encoders see
  4  dMaSIF's sampled surface over the site pocket's alpha spheres; the points
     inside them (the site region) in red
  5  only those points, each with its outward normal: the oriented points that
     are encoded and pooled into the pocket vector

Panel 4's region is the training code's own: external_pockets.site_region
with train.py's defaults (site pockets: >= 25% of a pocket's sphere centres,
or >= 50% of the ligand's heavy atoms, within 3 A of the other; region = the
surface points inside those pockets' spheres). So the red points are exactly
what `--pocket_select site` pools for this row.

All 3D panels share one camera, looking from outside the protein straight into
the pocket (protein centroid -> ligand centroid), orthographic, with sizes in
true Angstroms: atoms 3.4 A wide, surface samples 1 A, alpha spheres at their
real radii.

Inputs, per row (plots/fetch_subset.py fetches all of them):
    structures/<receptor_key>.cif, systems/<system_id>/ligand_files/*.sdf,
    data/surfaces/<input_id>.npz, data/pockets/fpocket/<input_id>.npz

Example:
    python plots/pipeline_fig.py --subset plots/subsets/big_targets/big_targets.tsv \\
        --row 0 --out plots/output/pipeline

Depends on: numpy, pandas, matplotlib, rdkit, biopython; structview and
contrastive_fig (this folder), external_pockets (src/, numpy only).
"""

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))

GREY = "#b9bec6"           # receptor atoms
POCKET_COLOR = "#d1495b"   # the site region (as in contrastive_fig / structview)
SITE_SPHERE = "#e0a030"    # fpocket's site pocket
OTHER_SPHERE = "#4f78b0"   # centres of fpocket's other pockets
SURF_GREY = "#c9ced6"
ATOM_A, POINT_A = 3.4, 1.0


def load_fpocket(row, root="."):
    from external_pockets import load_pockets

    return load_pockets(Path(root) / "data" / "pockets" / "fpocket" / f"{row.input_id}.npz")


class Camera:
    """Orthographic view with the pocket facing the viewer; x right, y up,
    depth toward the viewer."""

    def __init__(self, rec_xyz, lig_xyz):
        from contrastive_fig import view_basis

        self.B = view_basis(rec_xyz, lig_xyz)
        self.o = rec_xyz.mean(0)

    def __call__(self, xyz):
        return (np.asarray(xyz) - self.o) @ self.B.T

    def tilted(self, deg):
        """The same camera rotated about its x axis: the pocket seen at an
        angle, so normals that point at the viewer show their length."""
        t = np.radians(deg)
        R = np.array([[1, 0, 0], [0, np.cos(t), -np.sin(t)], [0, np.sin(t), np.cos(t)]])
        cam = Camera.__new__(Camera)
        cam.B, cam.o = R @ self.B, self.o
        return cam


def frame(ax, centre, half):
    ax.set_xlim(centre[0] - half, centre[0] + half)
    ax.set_ylim(centre[1] - half, centre[1] + half)
    ax.set_aspect("equal")
    ax.axis("off")


def pt_per_a(ax):
    w = ax.get_window_extent().width * 72 / ax.figure.dpi
    return w / np.ptp(ax.get_xlim())


def discs(ax, p, color, diam_a, shade=True, alpha=1.0, edge=0.25, z=1):
    """Projected spheres, back to front, darker toward the back."""
    from matplotlib.colors import to_rgba_array

    if not len(p):
        return
    order = np.argsort(p[:, 2])
    col = to_rgba_array(color, alpha)
    col = np.broadcast_to(col, (len(p), 4)).copy() if len(col) == 1 else col.copy()
    if shade:
        t = (p[:, 2] - p[:, 2].min()) / (np.ptp(p[:, 2]) + 1e-9)
        k = (0.55 + 0.45 * t)[:, None]
        col[:, :3] = col[:, :3] * k + 0.95 * (1 - k)
    s = (diam_a * pt_per_a(ax)) ** 2
    ax.scatter(p[order, 0], p[order, 1], s=s, c=col[order],
               edgecolors=(0, 0, 0, edge), linewidths=0.3, zorder=z)


def ligand_3d(ax, cam, lig, z=5, ball_a=0.55, bond_a=0.3):
    """Black ball-and-stick, drawn over the protein (it sits in the pocket
    opening, which faces the viewer). ball_a / bond_a: sizes in A."""
    p = cam(lig["xyz"])
    k = pt_per_a(ax)
    lw = float(np.clip(bond_a * k, 0.6, 2.2))
    for i, j in lig["bonds"]:
        ax.plot(p[[i, j], 0], p[[i, j], 1], color="black", lw=lw, zorder=z,
                solid_capstyle="round")
    ax.scatter(p[:, 0], p[:, 1], s=max(ball_a * k, 1.5) ** 2, c="black", zorder=z + 1)


def spheres(ax, cam, xyz, r, color, face=0.07, edge=0.45, z=3):
    """Alpha spheres at their true radii, as faint bubbles: a pocket has
    dozens to hundreds, so opaque discs merge into one block, while a faint
    face with a coloured outline shows the cluster's shape and density."""
    from matplotlib.collections import PatchCollection
    from matplotlib.colors import to_rgba
    from matplotlib.patches import Circle

    if not len(xyz):
        return
    p = cam(xyz)
    o = np.argsort(p[:, 2])
    pc = PatchCollection([Circle(p[i, :2], r[i]) for i in o],
                         facecolor=to_rgba(color, face), edgecolor=to_rgba(color, edge),
                         linewidth=0.5, zorder=z)
    ax.add_collection(pc)


def centres(ax, cam, xyz, color, diam_a=0.8, z=4):
    """Alpha-sphere centres as small dots: where fpocket's void is."""
    if not len(xyz):
        return
    p = cam(xyz)
    ax.scatter(p[:, 0], p[:, 1], s=max(diam_a * pt_per_a(ax), 1.0) ** 2, c=color,
               linewidths=0, zorder=z)


def site_info(surf, pk, lig):
    """(site pocket ids, site region indices into surf['xyz']) as training sees them."""
    from external_pockets import site_pockets, site_region

    ids = site_pockets(pk, lig["xyz"])
    region, _ = site_region(surf["xyz"], pk, lig["xyz"])
    return ids, (np.zeros(0, np.int64) if region is None else region)


def draw(row, out, root=".", window=10.0, normal_a=1.6, tilt=35.0, max_normals=250,
         title=None):
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, Rectangle

    from contrastive_fig import ligand_image
    from structview import load_ligand, load_receptor, load_surface

    rec, lig = load_receptor(row, root), load_ligand(row, root)
    surf, pk = load_surface(row, root), load_fpocket(row, root)
    ids, region = site_info(surf, pk, lig)
    cam = Camera(rec["xyz"], lig["xyz"])
    P, L, S = cam(rec["xyz"]), cam(lig["xyz"]), cam(surf["xyz"])
    c = L.mean(0)
    print(f"{row.system_id}: {len(rec['xyz'])} receptor atoms, {len(lig['xyz'])} ligand "
          f"atoms, {len(surf['xyz'])} surface points; fpocket {len(pk['pocket_score'])} "
          f"pockets, site pocket(s) {ids.tolist()} -> {len(region)} site points")
    if not len(region):
        print("  fpocket found no pocket at this ligand: panels 4-5 will be empty "
              "(training falls back to a ball here); try another row")

    # the zoom fits the ligand: at least `window`, else its extent + 6 A
    win = max(window, np.ptp(L[:, :2], axis=0).max() / 2 + 6.0)
    # window panels: drop what sits in front of the pocket opening, so the
    # zoomed views look into the site rather than at a wall
    front = lambda q: q[:, 2] <= c[2] + 4.0
    near = lambda q: (np.abs(q[:, :2] - c[:2]) <= win * 1.15).all(1)

    W, H, gap = 3.2, 3.2, 0.55
    fig = plt.figure(figsize=(5 * W + 4 * gap, H + 0.9))
    axes = []
    for k in range(5):
        x0 = (k * (W + gap)) / fig.get_figwidth()
        axes.append(fig.add_axes([x0, 0.04, W / fig.get_figwidth(), H / fig.get_figheight()]))
    a1, a2, a3, a4, a5 = axes

    # 1 complex, whole protein
    half = np.ptp(P[:, :2], axis=0).max() / 2 * 1.05
    frame(a1, P[:, :2].min(0) + np.ptp(P[:, :2], axis=0) / 2, half)
    discs(a1, P, GREY, ATOM_A)
    ligand_3d(a1, cam, lig)
    a1.add_patch(Rectangle(c[:2] - win, 2 * win, 2 * win, fill=False,
                           ls="--", lw=1, ec="#555", zorder=10))

    # 2 alpha spheres at the site window
    frame(a2, c, win)
    m = near(P) & front(P)
    discs(a2, P[m], GREY, ATOM_A, alpha=0.5, edge=0.1)
    site = np.isin(pk["sphere_pocket"], ids)
    other = ~site & near(cam(pk["sphere_xyz"]))
    # other pockets: centres only, so they mark fpocket's other voids without
    # drowning the site
    centres(a2, cam, pk["sphere_xyz"][other], OTHER_SPHERE, 0.9, z=3)
    spheres(a2, cam, pk["sphere_xyz"][site], pk["sphere_r"][site], SITE_SPHERE, z=4)
    centres(a2, cam, pk["sphere_xyz"][site], "#b5791a", 0.7, z=5)
    ligand_3d(a2, cam, lig, z=6)

    # 3 split: receptor atoms (top) + ligand graph (bottom)
    a3.axis("off")
    pos = a3.get_position()
    top = fig.add_axes([pos.x0, pos.y0 + pos.height * 0.5, pos.width, pos.height * 0.5])
    bot = fig.add_axes([pos.x0, pos.y0, pos.width, pos.height * 0.44])
    frame(top, P[:, :2].min(0) + np.ptp(P[:, :2], axis=0) / 2, half)
    discs(top, P, GREY, ATOM_A)
    # one ligand here, so scale it to fill the box (-1: no fixed bond length)
    bot.imshow(ligand_image(row.smiles, px=600, bond_px=-1))
    bot.axis("off")
    a3.text(0.5, 0.47, "+", ha="center", va="center", fontsize=18, color="#555",
            transform=a3.transAxes)

    # 4 surface samples over the site pocket
    frame(a4, c, win)
    ms = near(S) & front(S)
    in_site = np.zeros(len(S), bool)
    in_site[region] = True
    spheres(a4, cam, pk["sphere_xyz"][site], pk["sphere_r"][site], SITE_SPHERE,
            face=0.05, edge=0.3, z=2)
    discs(a4, S[ms & ~in_site], SURF_GREY, POINT_A, edge=0.15, z=3)
    discs(a4, S[ms & in_site], POCKET_COLOR, POINT_A, edge=0.15, z=4)

    # 5 the site points alone, with their normals, seen at an angle (straight
    # on, the normals point at the viewer and vanish) and zoomed to the region
    cam5 = cam.tilted(tilt)
    R, Nr = cam5(surf["xyz"][region]), surf["normals"][region] @ cam5.B.T
    if len(R):
        tips = np.vstack([R, R + normal_a * Nr])[:, :2]
        frame(a5, tips.min(0) + np.ptp(tips, axis=0) / 2, np.ptp(tips, axis=0).max() / 2 * 1.15)
    else:
        frame(a5, c, win)
    if len(R):
        # every point drawn, normals on an even subsample: hundreds of
        # crossing sticks read as hatching, not as a surface's orientation
        tip = R + normal_a * Nr
        pick = np.random.default_rng(0).permutation(len(R))[:max_normals]
        for i in pick[np.argsort(R[pick, 2])]:
            a5.plot([R[i, 0], tip[i, 0]], [R[i, 1], tip[i, 1]], color="#7a2230",
                    lw=0.6, alpha=0.75, zorder=5, solid_capstyle="round")
        # smaller than panel 4's 1 A: this panel is zoomed in, and the
        # normals are the point of it
        discs(a5, R, POCKET_COLOR, 0.6 * POINT_A, edge=0.15, z=4)

    titles = ["1  Protein–ligand complex", "2  fpocket alpha spheres",
              "3  Receptor + ligand graph", "4  dMaSIF surface samples",
              "5  Pocket surface points"]
    subs = ["receptor atoms, ligand in place", "site pocket orange, other pockets' centres blue",
            "the two encoders' inputs", f"site region red ({len(region)} points)",
            f"position + normal (tilted {tilt:g}°) → encoded, pooled"]
    for ax, t, s in zip(axes, titles, subs):
        p = ax.get_position()
        fig.text(p.x0, p.y1 + 0.13, t, fontsize=12, weight="bold", va="bottom")
        fig.text(p.x0, p.y1 + 0.06, s, fontsize=9, color="#555", va="bottom")
    for k in range(4):
        p0, p1 = axes[k].get_position(), axes[k + 1].get_position()
        y = p0.y0 + p0.height / 2
        fig.add_artist(FancyArrowPatch((p0.x1 + 0.004, y), (p1.x0 - 0.004, y),
                                       transform=fig.transFigure, arrowstyle="-|>",
                                       mutation_scale=16, color="#555", lw=1.4))
    if title:
        fig.suptitle(title, x=0.0, ha="left", y=1.12, fontsize=15, weight="bold")

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(out.with_suffix(f".{ext}"), dpi=200, bbox_inches="tight",
                    facecolor="white")
    print(f"wrote {out.with_suffix('.png')} and .svg")
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--subset", required=True, help="a subset (or items) TSV")
    ap.add_argument("--row", default="0", help="row number, or a system_id")
    ap.add_argument("--out", default="plots/output/pipeline")
    ap.add_argument("--window", type=float, default=10.0,
                    help="minimum half-width in A of the zoomed panels (grows to fit the ligand)")
    ap.add_argument("--max_normals", type=int, default=250,
                    help="panel 5 draws normals on at most this many points")
    ap.add_argument("--tilt", type=float, default=35.0,
                    help="degrees panel 5's camera is tilted, to show the normals")
    ap.add_argument("--title", default="")
    a = ap.parse_args()

    from structview import load_row

    row = load_row(a.subset, int(a.row) if a.row.isdigit() else a.row)
    draw(row, a.out, window=a.window, tilt=a.tilt, max_normals=a.max_normals,
         title=a.title or None)


if __name__ == "__main__":
    main()
