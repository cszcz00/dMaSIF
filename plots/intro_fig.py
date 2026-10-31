"""The problem slide: what binding looks like, and why finding binders is hard.

Two panels from one real items.tsv row:

  A  the binding site: dMaSIF's surface around the pocket as a shaded mesh,
     the crystal ligand in it, and PLIP's protein-ligand contacts as dashed
     lines (H-bonds, salt bridges, pi-stacking, halogen bonds, hydrophobic
     contacts), the H-bonding / charged / stacking residues labelled.
     "Shape and chemistry complement each other."
  B  needle in a haystack: the same pocket, empty, then "?" and a grid of
     ligands - the crystal binder and random molecules from the dataset, the
     binder outlined - captioned with how many distinct ligands the dataset
     holds. Decoys are drawn uniformly from the dataset's ligands, excluding
     every ligand of the same target group (so none is a known binder of this
     protein) and anything under 6 heavy atoms (ions, solvents).

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
POCKET_TINT = "#eab3b9"     # surface within 4 A of the ligand
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


def draw_mesh(ax, V, T, cam, lig, front_cut=None, tint=True, z=1):
    """Painter's-algorithm render of a triangle mesh in the camera's view:
    Lambert shading from a light over the viewer's left shoulder, the 4 A
    contact patch tinted. front_cut (A in front of the ligand) drops faces
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
        near = min_dist(V[Tk].mean(1).astype(np.float32), lig["xyz"]) <= POCKET_A
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


def panel_a(ax, rec, lig, surf, plip, cam, half, max_labels=6):
    from pipeline_fig import frame, ligand_3d

    c = cam(lig["xyz"]).mean(0)
    frame(ax, c, half)
    V, T = pocket_mesh(surf, lig, half * 1.5)
    draw_mesh(ax, V, T, cam, lig)
    ligand_3d(ax, cam, lig, z=6, ball_a=0.32, bond_a=0.22)

    labels, used = [], set()
    for name, (kinds, color, ls) in CONTACTS.items():
        m = np.isin(plip["kind"], kinds)
        for pp, lp in zip(cam(plip["prot_xyz"][m]), cam(plip["lig_xyz"][m])):
            ax.plot([lp[0], pp[0]], [lp[1], pp[1]], color=color, lw=1.6, ls=ls, zorder=7)
            ax.scatter([pp[0]], [pp[1]], s=22, c=color, zorder=8, edgecolors="white",
                       linewidths=0.6)
        if name in SPECIFIC:
            for xyz, pp in zip(plip["prot_xyz"][m], cam(plip["prot_xyz"][m])):
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
def decoys(items, row, n, seed=0, min_heavy=6):
    """n random SMILES from the dataset's distinct ligands, none from this
    target group, none under min_heavy heavy atoms."""
    df = pd.read_csv(items, sep="\t", dtype=str)
    banned = set(df.loc[df.group == row.group, "fp_row"])
    lig = df.drop_duplicates("fp_row")
    n_total = len(lig)
    pool = lig[~lig.fp_row.isin(banned)].sample(frac=1, random_state=seed)
    out = []
    for s in pool.smiles:
        if n_heavy(s) >= min_heavy:
            out.append(s)
        if len(out) == n:
            break
    return out, n_total


def draw(row, items, out, root=".", half=None, grid=(4, 4), seed=0, title=None):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import FancyArrowPatch, Rectangle

    from contrastive_fig import ligand_image
    from pipeline_fig import Camera, frame
    from structview import load_ligand, load_receptor, load_surface

    rec, lig, surf, plip = (load_receptor(row, root), load_ligand(row, root),
                            load_surface(row, root), load_plip(row, root))
    cam = Camera(rec["xyz"], lig["xyz"])
    L = cam(lig["xyz"])
    half = half or max(9.0, np.ptp(L[:, :2], axis=0).max() / 2 + 5.0)
    rows_, cols = grid
    dec, n_total = decoys(items, row, rows_ * cols - 1, seed)
    print(f"{row.system_id} ({row.group}): {len(plip['kind'])} PLIP contacts; "
          f"{n_total} distinct ligands in {items}")

    fig = plt.figure(figsize=(15.5, 6.6))
    a = fig.add_axes([0.0, 0.06, 0.40, 0.80])
    shown = panel_a(a, rec, lig, surf, plip, cam, half)
    handles = [Line2D([], [], color=CONTACTS[k][1], ls=CONTACTS[k][2], lw=2, label=k)
               for k in CONTACTS if k in shown]
    a.legend(handles=handles, loc="lower left", fontsize=9, frameon=True, framealpha=0.9,
             bbox_to_anchor=(0.0, -0.06))

    # B: the empty pocket, "?", the grid
    p = fig.add_axes([0.445, 0.30, 0.15, 0.36])
    frame(p, L.mean(0), half)
    V, T = pocket_mesh(surf, lig, half * 1.5)
    draw_mesh(p, V, T, cam, lig, tint=True)
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
             "dMaSIF surface, crystal ligand, PLIP contacts; pocket ≤ 4 Å tinted",
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
    ap.add_argument("--title", default="")
    a = ap.parse_args()

    sub = pd.read_csv(a.subset, sep="\t", dtype=str)
    if a.row == "auto":
        i = auto_row(sub)
    else:
        i = int(a.row) if a.row.isdigit() else int(np.flatnonzero(sub.system_id == a.row)[0])
    draw(sub.iloc[i], a.items, a.out, half=a.half or None, grid=tuple(a.grid),
         seed=a.seed, title=a.title or None)


if __name__ == "__main__":
    main()
