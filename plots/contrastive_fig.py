"""A contrastive-learning figure built from real proteins, ligands and vectors.

The CLIP-style picture: a batch of proteins down the left, each drawn atom by
atom with its pocket highlighted, encoded to a vector; their ligands across the
top, drawn in 2D, encoded to vectors; and between them the protein x ligand
similarity matrix whose diagonal (each protein with its own ligand) InfoNCE
pulls up and whose off-diagonal it pushes down.

Every number in it is the model's. Protein vectors come from a
`retrieval_report.py dump` (the GPU half of src/slurm/report.sbatch), which
embeds 400 train + 400 val receptors with a checkpoint, plus every ligand with
the same checkpoint. The dump cannot be pointed at chosen proteins, so the
figure's proteins are PICKED FROM THE DUMP's rows:

    pick   dump + items.tsv + ligand_feats.npz -> a subset TSV (fetch_subset
           format, plus dump_split / dump_index columns) and its .files list.
           Proteins from distinct target groups with distinct ligands.
           --prefer_correct keeps rows whose own ligand the model ranks highly
           among all ligands, and a batch whose diagonal wins each row - an
           illustrative choice, and the figure is no evidence of accuracy.
    (rsync the .files list, exactly as for fetch_subset.py)
    draw   subset TSV + dump -> the figure (.png and .svg).

The protein vector drawn is the anchor (pocket-region vector) that scores
highest against its own ligand; the matrix entry for protein i and ligand j is
the model's score, the max over protein i's usable anchors of anchor . ligand,
as in training. With --softmax each row becomes InfoNCE's probabilities at the
temperature given (the checkpoint learns its own; ~0.06-0.07 for C).

Example (v3c / C on dataset_v1, the dump report.sbatch makes by default):
    # cluster, GPU, ~2 min
    SELECTS=site sbatch src/slurm/report.sbatch
    # laptop
    rsync -avR "$H/./out/reports/v3c_site_pairs_bn_site.npz" .
    python plots/contrastive_fig.py pick --dump out/reports/v3c_site_pairs_bn_site.npz \\
        --items out/runs/v1/items.tsv --lig_feats out/runs/v1/ligand_feats.npz \\
        --name contrastive --n 4 --prefer_correct
    rsync -avr --files-from=plots/subsets/contrastive/contrastive.files \\
        alpha.empire-ai.org:/mnt/home/cjs2301/dmasif/ .
    python plots/contrastive_fig.py draw --subset plots/subsets/contrastive/contrastive.tsv \\
        --dump out/reports/v3c_site_pairs_bn_site.npz --out plots/output/contrastive

Depends on: numpy, pandas, matplotlib, rdkit, biopython; structview,
fetch_subset (this folder).
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

POCKET_COLOR = "#d1495b"
CMAP = "RdBu_r"


# ---------------------------------------------------------------------------
# pick
# ---------------------------------------------------------------------------
def load_dump(path):
    with np.load(path, allow_pickle=False) as d:
        return {k: d[k] for k in d.files}


def table_rows(items, lig_feats):
    """items.tsv as train.load_rows sees it: rows whose ligand failed
    featurisation dropped, so a dump's row indices are positions in this."""
    df = pd.read_csv(items, sep="\t", dtype=str)
    with np.load(lig_feats, allow_pickle=False) as f:
        failed = f["failed"] if "failed" in f.files else np.zeros(len(f["fp"]), bool)
    fp = df.fp_row.astype(int).to_numpy()
    return df[~failed[fp]].reset_index(drop=True)


def scores(anchors, usable, lig):
    """(R, K, D) x (L, D) -> (R, L): max over usable anchors, as in training."""
    s = np.einsum("rkd,ld->rkl", anchors, lig)
    return np.where(usable[:, :, None], s, -np.inf).max(1)


def own_rank(dump, split, table):
    """Each dumped row's true-ligand rank among all ligands (1 = best)."""
    an, us, rows = dump[f"{split}_anchors"], dump[f"{split}_usable"], dump[f"{split}_rows"]
    fp = table.fp_row.astype(int).to_numpy()[rows]
    ranks = np.empty(len(rows), np.int64)
    for s in range(0, len(rows), 64):
        sc = scores(an[s:s + 64], us[s:s + 64], dump["lig_vec"])
        own = sc[np.arange(len(sc)), fp[s:s + 64]]
        ranks[s:s + 64] = 1 + (sc > own[:, None]).sum(1)
    return ranks


def pick(a):
    from fetch_subset import row_files

    dump = load_dump(a.dump)
    table = table_rows(a.items, a.lig_feats)
    split = a.split
    rows = dump[f"{split}_rows"]
    ok = dump[f"{split}_usable"].any(1)
    rank = own_rank(dump, split, table)
    cand = np.flatnonzero(ok & (rank <= a.max_rank if a.prefer_correct else ok))
    print(f"{a.dump}: {len(rows)} {split} receptors, {ok.sum()} with a usable anchor; "
          f"own-ligand rank median {int(np.median(rank[ok]))} of {len(dump['lig_vec'])}; "
          f"{len(cand)} candidates")
    if len(cand) < a.n:
        raise SystemExit("too few candidates; raise --max_rank or drop --prefer_correct")

    rng = np.random.default_rng(a.seed)
    fp_all = table.fp_row.astype(int).to_numpy()
    for attempt in range(500):
        order = rng.permutation(cand)
        chosen, groups, ligs = [], set(), set()
        for i in order:
            r = table.iloc[rows[i]]
            if r.group in groups or r.fp_row in ligs:
                continue
            chosen.append(i)
            groups.add(r.group)
            ligs.add(r.fp_row)
            if len(chosen) == a.n:
                break
        if len(chosen) < a.n:
            raise SystemExit("not enough distinct target groups / ligands")
        idx = np.asarray(chosen)
        S = scores(dump[f"{split}_anchors"][idx], dump[f"{split}_usable"][idx],
                   dump["lig_vec"][fp_all[rows[idx]]])
        if not a.prefer_correct or (S.argmax(1) == np.arange(a.n)).all():
            break
    else:
        raise SystemExit("no batch with a winning diagonal in 500 tries; change --seed")

    sel = table.iloc[rows[idx]].copy()
    sel["dump_split"], sel["dump_index"] = split, idx
    sel["own_rank"] = rank[idx]
    out = Path(f"plots/subsets/{a.name}")
    out.mkdir(parents=True, exist_ok=True)
    sel.to_csv(out / f"{a.name}.tsv", sep="\t", index=False)
    files = list(dict.fromkeys(f for r in sel.itertuples() for f in row_files(r)))
    (out / f"{a.name}.files").write_text("\n".join(files) + "\n")
    print(f"\nbatch ({'diagonal wins every row' if a.prefer_correct else 'as drawn'}):")
    for r in sel.itertuples():
        print(f"  {r.system_id:32s} {r.group:16s} own-ligand rank {r.own_rank:5d}  "
              f"{str(r.smiles)[:50]}")
    print(f"\nwrote {out / (a.name + '.tsv')}\nfetch with:\n  rsync -avr --files-from="
          f"{out / (a.name + '.files')} alpha.empire-ai.org:/mnt/home/cjs2301/dmasif/ .")


# ---------------------------------------------------------------------------
# draw
# ---------------------------------------------------------------------------
def view_basis(rec_xyz, lig_xyz):
    """Orthonormal (right, up, toward-viewer) with the pocket facing the viewer."""
    z = lig_xyz.mean(0) - rec_xyz.mean(0)
    z /= np.linalg.norm(z) + 1e-9
    helper = np.array([0.0, 0.0, 1.0]) if abs(z[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    x = np.cross(helper, z)
    x /= np.linalg.norm(x)
    return np.stack([x, np.cross(z, x), z])


def draw_protein(ax, rec, lig, pocket_a=4.5):
    """Atom-wise depiction: heavy atoms as shaded discs, painter's order, the
    pocket (atoms within pocket_a of the ligand) in the accent colour, the rest
    by element, faded with depth. The ligand itself is not drawn - it is on the
    other side of the figure."""
    from matplotlib.colors import to_rgb

    from structview import ELEMENT_COLORS, min_dist

    B = view_basis(rec["xyz"], lig["xyz"])
    p = (rec["xyz"] - rec["xyz"].mean(0)) @ B.T
    depth = p[:, 2]
    t = (depth - depth.min()) / (np.ptp(depth) + 1e-9)            # 0 back, 1 front
    pocket = min_dist(rec["xyz"], lig["xyz"]) <= pocket_a
    base = np.array([to_rgb(ELEMENT_COLORS.get(e, "#909090")) for e in rec["element"]])
    base[rec["element"] == "C"] = to_rgb("#c8ccd2")               # pale carbons
    # elements muted toward grey, so the pocket is the only saturated colour
    # (pocket red would otherwise read as oxygen)
    base = 0.18 * base + 0.82 * np.array(to_rgb("#b9bec6"))
    base[pocket] = to_rgb(POCKET_COLOR)
    shade = 0.45 + 0.55 * t[:, None]                   # back atoms fade toward the background
    col = base * shade + (1 - shade) * 0.92
    order = np.argsort(depth)
    # marker area in pt^2 for a ~3.4 A wide atom at this panel's real scale
    half = np.ptp(p[:, :2], axis=0).max() / 2 * 1.05
    c = p[:, :2].min(0) + np.ptp(p[:, :2], axis=0) / 2
    ax.set_xlim(c[0] - half, c[0] + half)
    ax.set_ylim(c[1] - half, c[1] + half)
    fig = ax.figure
    pt_per_a = ax.get_window_extent().width * 72 / fig.dpi / (2 * half)
    size = (3.4 * pt_per_a) ** 2 * (0.8 + 0.2 * t)
    ax.scatter(p[order, 0], p[order, 1], s=size[order], c=col[order],
               edgecolors=(0, 0, 0, 0.25), linewidths=0.3)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")


def ligand_image(smiles, px=420, bond_px=34):
    """2D depiction, black on white (matches structview's black ligand), at a
    fixed bond length so ligand sizes compare across the row; a molecule too
    big for the box is shrunk to fit."""
    import io

    import matplotlib.image as mpimg
    from rdkit import Chem
    from rdkit.Chem.Draw import rdMolDraw2D

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
    d = rdMolDraw2D.MolDraw2DCairo(px, px)
    o = d.drawOptions()
    o.useBWAtomPalette()
    o.bondLineWidth = 2
    o.padding = 0.08
    o.fixedBondLength = bond_px
    o.clearBackground = True
    d.DrawMolecule(mol)
    d.FinishDrawing()
    return mpimg.imread(io.BytesIO(d.GetDrawingText()), format="png")


def batch_vectors(sub, dump):
    """-> (protein vectors (B, D), ligand vectors (B, D), score matrix (B, B))."""
    split = sub.dump_split.iloc[0]
    idx = sub.dump_index.astype(int).to_numpy()
    an, us = dump[f"{split}_anchors"][idx], dump[f"{split}_usable"][idx]
    lig = dump["lig_vec"][sub.fp_row.astype(int).to_numpy()]
    S = scores(an, us, lig)
    own = np.einsum("bkd,bd->bk", an, lig)
    own = np.where(us, own, -np.inf).argmax(1)
    return an[np.arange(len(idx)), own], lig, S


def draw(a):
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from matplotlib.patches import FancyArrowPatch, Rectangle

    from structview import load_ligand, load_receptor

    sub = pd.read_csv(a.subset, sep="\t", dtype=str)
    dump = load_dump(a.dump)
    P, L, S = batch_vectors(sub, dump)
    n, D = P.shape
    if a.dims:
        P, L = P[:, :a.dims], L[:, :a.dims]
    if a.softmax:
        Z = S / a.softmax
        M = np.exp(Z - Z.max(1, keepdims=True))
        M /= M.sum(1, keepdims=True)
        mlabel, vmin, vmax, cmap = f"softmax over ligands (τ = {a.softmax})", 0, 1, "Reds"
    else:
        M, mlabel, cmap = S, "similarity (protein · ligand)", "Reds"
        vmin, vmax = min(0, S.min()), S.max()
    vlim = np.abs(np.concatenate([P, L])).max()

    # columns: protein | encoder arrow | protein vector | n matrix columns
    # rows:    ligand drawing | encoder arrow | ligand vector | n matrix rows
    fig = plt.figure(figsize=(2.0 + 2.2 * n, 2.3 + 2.2 * n))
    gs = GridSpec(3 + n, 3 + n, figure=fig,
                  width_ratios=[1.15, 0.45, 0.35] + [1] * n,
                  height_ratios=[1.15, 0.45, 0.35] + [1] * n, wspace=0.08, hspace=0.08)

    for i, r in enumerate(sub.itertuples()):
        rec, lig = load_receptor(r), load_ligand(r)
        draw_protein(fig.add_subplot(gs[3 + i, 0]), rec, lig)
        ax = fig.add_subplot(gs[3 + i, 2])                         # protein vector, a row
        ax.imshow(P[i][:, None], aspect="auto", cmap=CMAP,
                  vmin=-vlim, vmax=vlim)
        ax.set_xticks([]), ax.set_yticks([])
        ax = fig.add_subplot(gs[0, 3 + i])                         # ligand drawing
        ax.imshow(ligand_image(r.smiles))
        ax.axis("off")
        ax = fig.add_subplot(gs[2, 3 + i])                         # ligand vector, a column
        ax.imshow(L[i][None, :], aspect="auto", cmap=CMAP, vmin=-vlim, vmax=vlim)
        ax.set_xticks([]), ax.set_yticks([])
        for k, txt in ((gs[3 + i, 1], "→"), (gs[1, 3 + i], "↓")):
            ax = fig.add_subplot(k)
            ax.text(0.5, 0.5, txt, ha="center", va="center", fontsize=22, color="#555")
            ax.axis("off")

    ax = fig.add_subplot(gs[3:, 3:])                               # the matrix
    ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=12,
                    color="white" if (M[i, j] - vmin) > 0.6 * (vmax - vmin) else "#222")
        ax.add_patch(Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, ec="black", lw=2.5))
    ax.set_xticks([]), ax.set_yticks([])

    # captions
    first = gs[0, 0].get_position(fig)
    fig.text(first.x0, first.y1, a.title, fontsize=16, weight="bold", va="top")
    bot = gs[3 + n - 1, 0].get_position(fig)
    fig.text(bot.x0 + bot.width / 2, bot.y0 - 0.012, "red: pocket atoms\n(≤ 4.5 Å of the ligand)",
             color=POCKET_COLOR, fontsize=9, ha="center", va="top")
    fig.text(gs[3, 1].get_position(fig).x0, gs[2, 1].get_position(fig).y0,
             "surface\nencoder", fontsize=10, color="#555", va="bottom")
    lv = gs[1, 3].get_position(fig)
    fig.text(lv.x0 - 0.01, lv.y0 + lv.height / 2, "ligand encoder", fontsize=10,
             color="#555", ha="right", va="center")
    mat = gs[3 + n - 1, 3 + n - 1].get_position(fig)
    fig.text(mat.x1, mat.y0 - 0.012,
             f"{mlabel}; boxed = true pairs (pulled together), others pushed apart.  "
             f"Vectors: {P.shape[1]} of {D} dims.", fontsize=9, ha="right", va="top",
             color="#444")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(out.with_suffix(f".{ext}"), dpi=200, bbox_inches="tight",
                    facecolor="white")
    print(f"wrote {out.with_suffix('.png')} and .svg")
    print("score matrix (rows proteins, columns ligands):")
    print(np.array2string(S, precision=3))


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("pick", help="choose the batch from a dump's rows")
    p.add_argument("--dump", required=True)
    p.add_argument("--items", required=True, help="the items.tsv the dump was made from")
    p.add_argument("--lig_feats", required=True)
    p.add_argument("--name", default="contrastive")
    p.add_argument("--n", type=int, default=4)
    p.add_argument("--split", default="train", choices=("train", "val"))
    p.add_argument("--prefer_correct", action="store_true")
    p.add_argument("--max_rank", type=int, default=20,
                   help="with --prefer_correct: own ligand in the model's top this many")
    p.add_argument("--seed", type=int, default=0)

    p = sub.add_parser("draw", help="render the figure")
    p.add_argument("--subset", required=True, help="the TSV `pick` wrote")
    p.add_argument("--dump", required=True)
    p.add_argument("--out", default="plots/output/contrastive")
    p.add_argument("--softmax", type=float, default=0.0,
                   help="show InfoNCE row probabilities at this temperature (e.g. 0.07)")
    p.add_argument("--dims", type=int, default=0, help="show only the first N dims")
    p.add_argument("--title", default="Contrastive training: proteins and ligands in one space")

    a = ap.parse_args()
    pick(a) if a.cmd == "pick" else draw(a)


if __name__ == "__main__":
    main()
