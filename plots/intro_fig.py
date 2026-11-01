"""The problem slide: what binding looks like, and why finding binders is hard.

Two panels from one real items.tsv row:

  A  the binding site: dMaSIF's surface around the pocket as a shaded mesh,
     seen down the pocket's opening, the crystal ligand in it, and PLIP's
     protein-ligand contacts as dashed lines to the surface over each contact (H-bonds, salt bridges, pi-stacking, halogen bonds, hydrophobic
     contacts), the H-bonding / charged / stacking residues labelled.
     "Shape and chemistry complement each other."
  B  needle in a haystack: the same pocket, empty, then "?" and a grid of
     ligands - the crystal binder and random molecules from the dataset, the
     binder outlined - captioned with how many distinct ligands the dataset
     holds. Decoys are drawn uniformly from the dataset's ligands, excluding
     every ligand of the same target group (so none is a known binder of this
     protein) and anything under 6 heavy atoms (ions, solvents).

The tinted pocket (--pocket): `contact`, the surface within 4 A of a ligand
heavy atom (the pocket label); `fallback`, training's fallback region
(train_data.fallback_region): a ball of --ball_radius (8 A, train.py's
default) at the centroid of the contact patch, marked +; or `site`, the
fpocket pocket(s) at the ligand (external_pockets.site_region), which is what
`--pocket_select site` pools whenever fpocket found one. Every run prints all
three regions' sizes and how many PLIP contacts each covers (most of the
contact's surface cap - points within 3 A of its protein atom, 4 A of a ring /
charge centre, plip_labels' convention - lies in the region).

Picking the row (--row auto, the default): among subset rows whose files are
present, a ligand of 15-40 heavy atoms with the most kinds of specific contact
(H-bond, salt bridge, pi, halogen), then the most contacts. The candidates are
printed, so a different --row is one look away.

Inputs per row (plots/fetch_subset.py fetches them): structures/<receptor_key>.cif,
the ligand SDF, data/surfaces/<input_id>.npz, data/plip/<system_id>.npz; and
--items, the full items.tsv the subset came from (for decoys and the count).

Example:
    python plots/intro_fig.py --subset plots/subsets/big_targets/big_targets.tsv \\
        --items out/runs/dataset_v2/items.tsv --out plots/output/intro

Depends on: numpy, pandas, matplotlib, rdkit, biopython; structview,
pipeline_fig, contrastive_fig (this folder); channel_viz, plip_labels (src/,
numpy only at import).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "src"))

SURFACE = "#d5dbe3"
POCKET_TINT = "#eab3b9"     # the pocket region, either definition below
BALL_R = 8.0                # train.py --ball_radius default: the fallback ball
BINDER = "#d1495b"          # outline of the true binder in panel B
# PLIP kinds -> what the slide calls them (water kinds never occur: no waters)
CONTACTS = {
    "H-bond": (("hbond_pdon", "hbond_ldon"), "#2f6db5", (0, (3, 2))),
    "salt bridge": (("saltbridge_ppos", "saltbridge_pneg"), "#e0a030", (0, (3, 2))),
    "π-stacking / π-cation": (("pistack", "pication_pring", "pication_pcharge"),
                              "#3f9e6a", (0, (3, 2))),
    "halogen bond": (("halogen",), "#7b4fa0", (0, (3, 2))),
    "hydrophobic": (("hydrophobic",), "#8a8f96", (0, (1, 2))),
}
SPECIFIC = ("H-bond", "salt bridge", "π-stacking / π-cation", "halogen bond")


def load_plip(row, root="."):
    from plip_labels import KINDS

    with np.load(Path(root) / "data" / "plip" / f"{row.system_id}.npz") as d:
        kind = [KINDS[k] for k in d["kind"].astype(int)]
        return {"kind": np.asarray(kind), "prot_xyz": d["prot_xyz"], "lig_xyz": d["lig_xyz"]}


def contact_label(kind):
    for name, (kinds, _, _) in CONTACTS.items():
        if kind in kinds:
            return name
    return None


def n_heavy(smiles):
    from rdkit import Chem

    m = Chem.MolFromSmiles(str(smiles))
    return m.GetNumHeavyAtoms() if m is not None else 0


def has_files(row, root="."):
    r = Path(root)
    need = [r / "structures" / f"{row.receptor_key}.cif",
            r / "data" / "surfaces" / f"{row.input_id}.npz",
            r / "data" / "plip" / f"{row.system_id}.npz"]
    need += [r / p for p in str(row.sdf_paths).split(",")]
    return all(p.exists() for p in need)


def auto_row(sub, root="."):
    rows = []
    for i, r in enumerate(sub.itertuples()):
        if not has_files(r, root):
            continue
        labels = [contact_label(k) for k in load_plip(r, root)["kind"]]
        kinds = len({x for x in labels if x in SPECIFIC})
        ha = n_heavy(r.smiles)
        rows.append((15 <= ha <= 40, kinds, len(labels), i, r.system_id, r.group, ha))
    if not rows:
        raise SystemExit("no row of the subset has all its files; run fetch_subset's rsync")
    rows.sort(reverse=True)
    print("candidates (drug-sized, contact kinds, contacts):")
    for ok, k, n, i, sid, g, ha in rows[:8]:
        print(f"  row {i:3d} {sid:32s} {g:16s} {ha:3d} heavy atoms, {k} kinds, {n} contacts"
              + ("" if ok else "  (outside 15-40 heavy atoms)"))
    return rows[0][3]


# ---------------------------------------------------------------------------
# panel A: the shaded pocket surface
# ---------------------------------------------------------------------------
def pocket_mesh(surf, lig, radius):
    """Mesh only the surface near the ligand: faster, and nothing else is shown."""
    from structview import mesh

    c = lig["xyz"].mean(0)
    keep = np.linalg.norm(surf["xyz"] - c, axis=1) <= radius + 6.0
    return mesh({"xyz": surf["xyz"][keep], "normals": surf["normals"][keep]})


def nearest_index(q, xyz, chunk=4096):
    """Index of the nearest xyz point for each q (chunked, numpy only)."""
    out = np.empty(len(q), np.int64)
    for s in range(0, len(q), chunk):
        d = ((q[s:s + chunk, None, :] - xyz[None]) ** 2).sum(-1)
        out[s:s + chunk] = d.argmin(1)
    return out


def region_masks(surf, lig, radius=BALL_R, pk=None):
    """The candidate pocket regions as boolean masks over the surface points.

    contact   surface within 4 A of a ligand heavy atom: the pocket LABEL.
    fallback  train_data.fallback_region: a ball of `radius` at the centroid
              of the labelled points - what training uses when no other
              region covers the site; the same shape as the ball_R8 selector.
    site      external_pockets.site_region with train.py's defaults: the
              fpocket pocket(s) at the ligand, which `--pocket_select site`
              (v3c, v5) pools for every row where fpocket found one.
    -> ({name: mask}, fallback centre)
    """
    from structview import POCKET_A, min_dist

    xyz = surf["xyz"]
    label = min_dist(xyz, lig["xyz"]) <= POCKET_A
    if not label.any():
        raise SystemExit("no surface point within 4 A of the ligand")
    centre = xyz[label].mean(0)
    masks = {"contact": label,
             "fallback": ((xyz - centre) ** 2).sum(-1) < radius * radius}
    if pk is not None:
        from external_pockets import site_region

        idx, _ = site_region(xyz, pk, lig["xyz"])
        site = np.zeros(len(xyz), bool)
        if idx is not None:
            site[idx] = True
        masks["site"] = site
    return masks, centre


def coverage(surf, plip, masks, atom_radius=3.0, centre_radius=4.0):
    """PLIP contacts under each region -> {name: (covered, with a cap)}, and
    how many contacts have no surface within reach.

    A contact's CAP is the surface over its protein side, by plip_labels'
    site_labels convention: points within 3 A of the protein atom (4 A of a
    ring / charge-group centre; the surface sits ~1.5-2 A out from atom
    centres). A region covers the contact when it holds most of that cap.
    """
    from plip_labels import CENTRE_KINDS, KINDS

    xyz = surf["xyz"]
    centre = CENTRE_KINDS[[KINDS.index(k) for k in plip["kind"]]]
    out = {k: 0 for k in masks}
    capped = 0
    for p_, c_ in zip(plip["prot_xyz"].astype(np.float32), centre):
        r = centre_radius if c_ else atom_radius
        cap = ((xyz - p_) ** 2).sum(-1) <= r * r
        if not cap.any():
            continue
        capped += 1
        for k, m in masks.items():
            out[k] += m[cap].mean() > 0.5
    return {k: (int(v), capped) for k, v in out.items()}, len(plip["kind"]) - capped


def as_test(surf, mask):
    """A surface-point mask -> test(points) for mesh faces: a face belongs to
    the region when its nearest surface sample does."""
    return lambda q: mask[nearest_index(q.astype(np.float32), surf["xyz"])]


def draw_mesh(ax, V, T, cam, lig, front_cut=None, tint=True, z=1, region=None):
    """Painter's-algorithm render of a triangle mesh in the camera's view:
    Lambert shading from a light over the viewer's left shoulder, the 4 A
    contact patch tinted (or `region`, a points -> mask test from
    region_masks + as_test). front_cut (A in front of the ligand) drops faces
    nearer the viewer than that; off by default, because cutting the pocket's
    rim leaves holes, and the camera already looks straight into the pocket."""
    from matplotlib.collections import PolyCollection
    from matplotlib.colors import to_rgb

    from structview import POCKET_A, min_dist

    p = cam(V)
    cz = cam(lig["xyz"]).mean(0)[2]
    f = p[T]                                                   # (F, 3, 3)
    keep = (np.ones(len(f), bool) if front_cut is None
            else f[:, :, 2].mean(1) <= cz + front_cut)
    f, Tk = f[keep], T[keep]
    n = np.cross(f[:, 1] - f[:, 0], f[:, 2] - f[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    light = np.array([-0.45, 0.55, 1.0])
    light /= np.linalg.norm(light)
    lam = np.abs(n @ light)                                    # both faces lit: no black backs
    shade = 0.42 + 0.58 * lam
    base = np.tile(np.asarray(to_rgb(SURFACE)), (len(f), 1))
    if tint:
        centres = V[Tk].mean(1)
        near = (region(centres) if region is not None else
                min_dist(centres.astype(np.float32), lig["xyz"]) <= POCKET_A)
        base[near] = to_rgb(POCKET_TINT)
    col = np.clip(base * shade[:, None] + 0.08 * (1 - shade[:, None]), 0, 1)
    order = np.argsort(f[:, :, 2].mean(1))
    ax.add_collection(PolyCollection(f[order][:, :, :2], facecolors=col[order],
                                     edgecolors=col[order], linewidths=0.15, zorder=z))


def residue_name(rec, xyz):
    """'Asp189' for the receptor atom nearest this point."""
    i = int(np.argmin(((rec["xyz"] - xyz) ** 2).sum(1)))
    _, res, _ = rec["label"][i].split(" ", 2)
    name, num = res[:3], res[3:]
    return name.capitalize() + num


def surface_ends(surf, plip, atom_radius=3.0, centre_radius=4.0):
    """Where to draw each contact's protein end: on the surface, not at the
    buried atom. The point of the contact's surface cap (coverage()'s
    convention) nearest its ligand partner - the patch of surface the ligand
    actually faces - or the atom itself when no surface is within reach."""
    from plip_labels import CENTRE_KINDS, KINDS

    xyz = surf["xyz"]
    centre = CENTRE_KINDS[[KINDS.index(k) for k in plip["kind"]]]
    out = plip["prot_xyz"].astype(np.float32).copy()
    for i, (p_, l_, c_) in enumerate(zip(plip["prot_xyz"], plip["lig_xyz"], centre)):
        r = centre_radius if c_ else atom_radius
        cap = np.flatnonzero(((xyz - p_) ** 2).sum(-1) <= r * r)
        if len(cap):
            out[i] = xyz[cap[((xyz[cap] - l_) ** 2).sum(-1).argmin()]]
    return out


def pocket_view(surf, lig):
    """Look down the pocket's opening: the mean outward normal of the 4 A
    contact patch. None (use the default camera) when the patch wraps round
    the ligand so far that its normals cancel."""
    from structview import POCKET_A, min_dist

    m = min_dist(surf["xyz"], lig["xyz"]) <= POCKET_A
    v = surf["normals"][m].mean(0)
    return v if np.linalg.norm(v) > 0.2 else None


def panel_a(ax, rec, lig, surf, plip, cam, half, max_labels=6, region=None, centre=None):
    from pipeline_fig import frame, ligand_3d

    c = cam(lig["xyz"]).mean(0)
    frame(ax, c, half)
    V, T = pocket_mesh(surf, lig, half * 1.5)
    draw_mesh(ax, V, T, cam, lig, region=region)
    if centre is not None:                                     # the fallback ball's centre
        cc = cam(centre[None])[0]
        ax.scatter([cc[0]], [cc[1]], marker="+", s=120, c="#7a2230", lw=2, zorder=9)
    ligand_3d(ax, cam, lig, z=6, ball_a=0.32, bond_a=0.22)

    ends = surface_ends(surf, plip)
    labels, used = [], set()
    for name, (kinds, color, ls) in CONTACTS.items():
        m = np.isin(plip["kind"], kinds)
        for pp, lp in zip(cam(ends[m]), cam(plip["lig_xyz"][m])):
            ax.plot([lp[0], pp[0]], [lp[1], pp[1]], color=color, lw=1.6, ls=ls, zorder=7)
            ax.scatter([pp[0]], [pp[1]], s=22, c=color, zorder=8, edgecolors="white",
                       linewidths=0.6)
        if name in SPECIFIC:
            for xyz, pp in zip(plip["prot_xyz"][m], cam(ends[m])):
                labels.append((name, residue_name(rec, xyz), pp))
    for name, res, pp in labels:
        if res in used or len(used) >= max_labels:
            continue
        used.add(res)
        d = pp[:2] - c[:2]
        d = d / (np.linalg.norm(d) + 1e-9)
        ax.annotate(res, pp[:2], xytext=pp[:2] + 2.2 * d, fontsize=9, weight="bold",
                    color="#222", ha="center", va="center", zorder=9,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))
    return {name for name, (kinds, _, _) in CONTACTS.items() if np.isin(plip["kind"], kinds).any()}


# ---------------------------------------------------------------------------
# panel B: one pocket, many molecules
# ---------------------------------------------------------------------------
def decoy_rows(items, row, n, seed=0, min_heavy=6):
    """n random distinct ligands of the dataset (items.tsv rows), none from this
    target group, none under min_heavy heavy atoms -> (DataFrame, n distinct)."""
    df = pd.read_csv(items, sep="\t", dtype=str)
    banned = set(df.loc[df.group == row.group, "fp_row"])
    lig = df.drop_duplicates("fp_row")
    n_total = len(lig)
    pool = lig[~lig.fp_row.isin(banned)].sample(frac=1, random_state=seed)
    keep = []
    for i, s in zip(pool.index, pool.smiles):
        if n_heavy(s) >= min_heavy:
            keep.append(i)
        if len(keep) == n:
            break
    return pool.loc[keep], n_total


def decoys(items, row, n, seed=0, min_heavy=6):
    """decoy_rows' SMILES, in the same order."""
    rows, n_total = decoy_rows(items, row, n, seed, min_heavy)
    return list(rows.smiles), n_total


def draw(row, items, out, root=".", half=None, grid=(4, 4), seed=0, title=None,
         pocket="contact", ball_radius=BALL_R, view="pocket"):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import FancyArrowPatch, Rectangle

    from contrastive_fig import ligand_image
    from pipeline_fig import Camera, frame
    from structview import load_ligand, load_receptor, load_surface

    rec, lig, surf, plip = (load_receptor(row, root), load_ligand(row, root),
                            load_surface(row, root), load_plip(row, root))
    # down the pocket's opening (normal 'view'), else protein centroid -> ligand
    cam = Camera(rec["xyz"], lig["xyz"], pocket_view(surf, lig) if view == "pocket" else None)
    L = cam(lig["xyz"])
    half = half or max(9.0, np.ptp(L[:, :2], axis=0).max() / 2 + 5.0)
    pk_path = Path(root) / "data" / "pockets" / "fpocket" / f"{row.input_id}.npz"
    pk = None
    if pk_path.exists():
        from pipeline_fig import load_fpocket
        pk = load_fpocket(row, root)
    masks, ball_c = region_masks(surf, lig, ball_radius, pk)
    span = np.sqrt(((lig["xyz"][:, None] - lig["xyz"][None]) ** 2).sum(-1)).max()
    cov, n_buried = coverage(surf, plip, masks)
    print(f"ligand spans {span:.1f} A; PLIP contacts whose surface cap lies mostly in "
          f"each region ({n_buried} with no surface within reach, left out):")
    for k, m in masks.items():
        a_, b_ = cov[k]
        print(f"  {k:9s} {int(m.sum()):5d} points  covers {a_}/{b_} contacts"
              + (f"  (ball R={ball_radius:g} A)" if k == "fallback" else ""))
    if pocket not in masks:
        raise SystemExit(f"--pocket {pocket}: no fpocket file at {pk_path}")
    if not masks[pocket].any():
        raise SystemExit(f"--pocket {pocket}: empty for this row (fpocket found no pocket "
                         "at the ligand; training falls back to the ball here)")
    region = as_test(surf, masks[pocket])
    centre = ball_c if pocket == "fallback" else None
    rows_, cols = grid
    dec, n_total = decoys(items, row, rows_ * cols - 1, seed)
    print(f"{row.system_id} ({row.group}): {len(plip['kind'])} PLIP contacts; "
          f"{n_total} distinct ligands in {items}")

    fig = plt.figure(figsize=(15.5, 6.6))
    a = fig.add_axes([0.0, 0.06, 0.40, 0.80])
    shown = panel_a(a, rec, lig, surf, plip, cam, half, region=region, centre=centre)
    handles = [Line2D([], [], color=CONTACTS[k][1], ls=CONTACTS[k][2], lw=2, label=k)
               for k in CONTACTS if k in shown]
    a.legend(handles=handles, loc="lower left", fontsize=9, frameon=True, framealpha=0.9,
             bbox_to_anchor=(0.0, -0.06))

    # B: the empty pocket, "?", the grid
    p = fig.add_axes([0.445, 0.30, 0.15, 0.36])
    frame(p, L.mean(0), half)
    V, T = pocket_mesh(surf, lig, half * 1.5)
    draw_mesh(p, V, T, cam, lig, tint=True, region=region)
    fig.add_artist(FancyArrowPatch((0.605, 0.48), (0.655, 0.48), transform=fig.transFigure,
                                   arrowstyle="-|>", mutation_scale=22, color="#555", lw=1.6))
    fig.text(0.63, 0.53, "?", fontsize=30, weight="bold", ha="center", color="#333")

    smiles = dec[:]
    true_at = np.random.default_rng(seed).integers(0, len(smiles) + 1)
    smiles.insert(true_at, row.smiles)
    x0, y0, w, h = 0.665, 0.06, 0.335, 0.80
    cw, ch = w / cols, h / rows_
    for k, s in enumerate(smiles):
        r_, c_ = divmod(k, cols)
        g = fig.add_axes([x0 + c_ * cw + 0.004, y0 + (rows_ - 1 - r_) * ch + 0.006,
                          cw - 0.008, ch - 0.012])
        g.imshow(ligand_image(s, px=300, bond_px=20))
        g.set_xticks([]), g.set_yticks([])
        binder = k == true_at
        for sp in g.spines.values():
            sp.set_edgecolor(BINDER if binder else "#d0d4da")
            sp.set_linewidth(2.6 if binder else 0.8)
        if binder:
            g.text(0.03, 0.04, "crystal binder", transform=g.transAxes, fontsize=8,
                   color=BINDER, weight="bold")

    fig.text(0.0, 0.97, "A  Binding: shape and chemistry complement each other",
             fontsize=14, weight="bold", va="top")
    fig.text(0.0, 0.915, f"{row.receptor_key.split('__')[0].upper()} · {row.group} · "
             "dMaSIF surface, crystal ligand, PLIP contacts; "
             + {"fallback": f"pocket = {ball_radius:g} Å ball at the contact patch (+)",
                "site": "pocket = fpocket's pocket at the ligand",
                "contact": "pocket ≤ 4 Å tinted"}[pocket],
             fontsize=9.5, color="#555", va="top")
    fig.text(0.445, 0.97, "B  Finding binders: a needle in a haystack",
             fontsize=14, weight="bold", va="top")
    fig.text(0.445, 0.915, f"1 pocket · {n_total:,} distinct ligands in the dataset · "
             f"{len(smiles) - 1} random ones shown with the crystal binder",
             fontsize=9.5, color="#555", va="top")
    if title:
        fig.suptitle(title, x=0.0, ha="left", y=1.06, fontsize=16, weight="bold")

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(out.with_suffix(f".{ext}"), dpi=200, bbox_inches="tight",
                    facecolor="white")
    print(f"wrote {out.with_suffix('.png')} and .svg")
    return fig


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--subset", required=True, help="a subset TSV (fetched)")
    ap.add_argument("--items", required=True, help="the full items.tsv it came from")
    ap.add_argument("--row", default="auto", help="'auto', a row number, or a system_id")
    ap.add_argument("--out", default="plots/output/intro")
    ap.add_argument("--half", type=float, default=0.0,
                    help="half-width in A of panel A (default: fits the ligand)")
    ap.add_argument("--grid", type=int, nargs=2, default=(4, 4), metavar=("ROWS", "COLS"))
    ap.add_argument("--seed", type=int, default=0, help="which random decoys")
    ap.add_argument("--pocket", default="contact", choices=("contact", "fallback", "site"),
                    help="tint the 4 A contact patch, training's fallback ball, or the "
                         "fpocket site region --pocket_select site pools")
    ap.add_argument("--view", default="pocket", choices=("pocket", "centroid"),
                    help="camera down the pocket opening (mean contact-patch normal), or "
                         "along protein centroid -> ligand")
    ap.add_argument("--ball_radius", type=float, default=BALL_R,
                    help="fallback ball radius in A (train.py --ball_radius)")
    ap.add_argument("--title", default="")
    a = ap.parse_args()

    sub = pd.read_csv(a.subset, sep="\t", dtype=str)
    if a.row == "auto":
        i = auto_row(sub)
    else:
        i = int(a.row) if a.row.isdigit() else int(np.flatnonzero(sub.system_id == a.row)[0])
    draw(sub.iloc[i], a.items, a.out, half=a.half or None, grid=tuple(a.grid),
         seed=a.seed, title=a.title or None, pocket=a.pocket, ball_radius=a.ball_radius,
         view=a.view)


if __name__ == "__main__":
    main()
