"""Generate docs/figures/pipeline.svg: the whole model on one slide (the v8 configuration).

    python3 docs/figures/make_pipeline_svg.py docs/figures/pipeline.svg

Every number is the v8 configuration in code (src/slurm/train_v8.sbatch and
the defaults it leaves alone): surface_encoder.EncoderConfig (10 curvatures,
6 AtomNet, 16 DirAtomNet channels over atom_features' 13 values; conv 3
layers x 12 A x 16 dims; ball_R8 from the top 5%, k_pos 8), pair_pool.PairPool
(8 types, 8 bins, 3-16 A), ligand_features (47 atom / 6 bond features),
ligand_encoder (GINE 4 x 128, mean + max pool), train.py (batch 32, InfoNCE,
false-negative mask at Tanimoto 0.8), plip_loss.pocket_target, lig_sim.py.
"""
import math
import sys

INK, MUTED, FAINT, RULE, BG, PANEL = "#1f2328", "#59636e", "#8c959f", "#d0d7de", "#ffffff", "#f6f8fa"
PROT, PROT_L = "#2f6fdf", "#dbe7fd"
LIG, LIG_L = "#2a9d6f", "#d5efe3"
LOSS, LOSS_L = "#e07b1a", "#fde9d6"
HOT = ["#e6e8eb", "#f4d2b0", "#f0a060", "#e0601a", "#9c2a0e"]
TYPE_COL = ["#2f6fdf", "#e07b1a", "#2a9d6f", "#9a6fd8", "#d64f6f", "#3fa7c4", "#b59a2a", "#7d8590"]
W, H = 1600, 870

out = []
w = out.append


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def text(x, y, s, size=13, col=INK, anchor="start", weight="normal", style="normal"):
    w(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{col}" text-anchor="{anchor}" '
      f'font-weight="{weight}" font-style="{style}">{esc(s)}</text>')


def rect(x, y, wd, ht, fill, stroke="none", rx=0, sw=1, extra=""):
    w(f'<rect x="{x:.1f}" y="{y:.1f}" width="{wd:.1f}" height="{ht:.1f}" rx="{rx}" '
      f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}" {extra}/>')


def circle(x, y, r, fill, stroke="none", sw=1, extra=""):
    w(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="{fill}" stroke="{stroke}" '
      f'stroke-width="{sw}" {extra}/>')


def line(x1, y1, x2, y2, col=INK, sw=1.2, extra=""):
    w(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{col}" '
      f'stroke-width="{sw}" {extra}/>')


def path(d, col, sw=1.5, fill="none", extra=""):
    w(f'<path d="{d}" stroke="{col}" stroke-width="{sw}" fill="{fill}" {extra}/>')


def arrow(x1, y1, x2, y2, col=FAINT, sw=2):
    line(x1, y1, x2, y2, col, sw, 'marker-end="url(#ah)"')


def tag(x, y, label, col):
    wd = 7.2 * len(label) + 12
    rect(x, y - 12, wd, 17, "#ffffff", col, 8, 1.2)
    text(x + wd / 2, y + 1, label, 11, col, "middle", "bold")


# A fixed, smooth "surface" curve every protein glyph reuses
def surf_pts(x0, y0, wd, n=26):
    pts = []
    for i in range(n):
        t = i / (n - 1)
        x = x0 + t * wd
        y = y0 - 14 * math.sin(math.pi * t) + 16 * math.exp(-((t - 0.55) / 0.12) ** 2)
        pts.append((x, y))
    return pts


def box(x, y, wd, ht, accent, num, title, lines, src):
    rect(x, y, wd, ht, "#ffffff", RULE, 10)
    rect(x, y, wd, 4, accent, rx=2)
    text(x + 12, y + 132, f"{num}  {title}", 14, INK, weight="bold")
    for k, s in enumerate(lines):
        text(x + 12, y + 151 + 16 * k, s, 12, MUTED)
    if src:
        tag(x + 12, y + ht - 12, src, FAINT if src == "dMaSIF" else accent)


w(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
  'font-family="Helvetica, Arial, sans-serif">')
w('<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
  f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{FAINT}"/></marker>'
  '<pattern id="hatch" width="5" height="5" patternUnits="userSpaceOnUse" '
  'patternTransform="rotate(45)"><line x1="0" y1="0" x2="0" y2="5" stroke="#b8bcc4" '
  'stroke-width="2"/></pattern></defs>')
rect(0, 0, W, H, BG)

text(40, 50, "The model: a protein becomes a few pocket vectors, a ligand becomes one, "
     "and binding is a dot product", 24, INK, weight="bold")
text(40, 78, "Trained end to end, contrastively, on dataset_v2. At screening time each side is "
     "encoded once; no per-pair docking or simulation.", 14, MUTED)

# ================================================================ protein row
text(40, 118, "PROTEIN", 13, PROT, weight="bold")
BX, BY, BW, BH, GAP = 40, 128, 196, 232, 21
xs = [BX + i * (BW + GAP) for i in range(7)]

# 1 receptor atoms
x = xs[0]
box(x, BY, BW, BH, PROT, "1", "Receptor atoms", ["heavy atoms of the", "PLINDER receptor;", "the ligand is removed"], "")
for i, (dx, dy, r) in enumerate([(40, 60, 9), (62, 48, 8), (84, 62, 9), (106, 50, 8), (128, 64, 9),
                                  (150, 52, 8), (58, 80, 8), (96, 84, 9), (136, 86, 8), (78, 100, 7),
                                  (118, 104, 7), (160, 76, 7)]):
    circle(x + dx, BY + dy, r, ["#9aa4b1", "#d64f6f", "#3f7fe0", "#9aa4b1", "#b59a2a"][i % 5],
           "#ffffff", 1.2)

# 2 surface
x = xs[1]
box(x, BY, BW, BH, PROT, "2", "Surface points", ["dMaSIF samples the", "a smooth molecular surface:", "~1 point / Å², normals"], "dMaSIF")
pts = surf_pts(x + 18, BY + 80, BW - 36)
for k, (px, py) in enumerate(pts):
    if k % 2 == 0 and 0 < k < len(pts) - 1:
        qx, qy = pts[k + 1]
        dx, dy = qx - px, qy - py
        n = math.hypot(dx, dy)
        line(px, py, px + 11 * dy / n, py - 11 * dx / n, FAINT, 1)
    circle(px, py, 3, PROT)

# 3 input features
x = xs[2]
box(x, BY, BW, BH, PROT, "3", "32 input channels", ["10 curvatures + 6 AtomNet", "+ 16 learned atom typing:", "13 atom features + polar dir."], "ours + dMaSIF")
vals = [0.5, 0.8, 0.35, 0.6, 0.9, 0.3, 0.7, 0.45, 0.55, 0.4,
        0.65, 0.3, 0.8, 0.5, 0.35, 0.7,
        0.4, 0.9, 0.3, 0.6, 0.75, 0.35, 0.5, 0.85, 0.4, 0.6, 0.3, 0.7, 0.55, 0.45, 0.8, 0.35]
for k, v in enumerate(vals):
    col = "#b8bcc4" if k < 10 else ("#8fb2ef" if k < 16 else PROT)
    rect(x + 16 + k * 5.2, BY + 104 - 70 * v, 4.2, 70 * v, col)
line(x + 14, BY + 105, x + 16 + 32 * 5.2, BY + 105, MUTED, 1)
text(x + 16, BY + 22, "curv.", 10.5, FAINT)
text(x + 70, BY + 22, "AtomNet", 10.5, FAINT)
text(x + 118, BY + 22, "new", 10.5, PROT, weight="bold")

# 4 convolution
x = xs[3]
box(x, BY, BW, BH, PROT, "4", "Surface convolution", ["dMaSIFConv, 3 layers,", "12 Å radius → a 16-d", "embedding per point"], "dMaSIF")
pts = surf_pts(x + 18, BY + 84, BW - 36)
cx, cy = pts[12]
circle(cx, cy - 6, 34, PROT_L, PROT, 1, 'opacity="0.7" stroke-dasharray="4 3"')
for k, (px, py) in enumerate(pts):
    circle(px, py, 3, PROT if abs(k - 12) <= 5 else "#b8bcc4")
for k in range(7, 18):
    if k != 12:
        px, py = pts[k]
        line(px, py, cx, cy, PROT, 0.7, 'opacity="0.5"')
circle(cx, cy, 5, INK)
text(cx, BY + 36, "12 Å", 11, PROT, "middle")

# 5 pocket logit
x = xs[4]
box(x, BY, BW, BH, PROT, "5", "Pocket logit", ["one score per point;", "trained toward contact 0.5,", "PLIP-interacting 1, else 0"], "ours")
pts = surf_pts(x + 18, BY + 90, BW - 36)
for k, (px, py) in enumerate(pts):
    h = math.exp(-((k - 14.3) / 3.2) ** 2)
    circle(px, py, 4.2, HOT[min(4, int(h * 4.99))])
for i, (dx, dy) in enumerate([(0, -18), (9, -26), (-9, -26), (0, -34)]):
    circle(pts[14][0] + dx, pts[14][1] + dy - 2, 3.5, INK, "#ffffff", 0.8, 'opacity="0.55"')
text(x + 14, BY + 24, "ligand: for training labels only", 10.5, FAINT)

# 6 balls
x = xs[5]
box(x, BY, BW, BH, PROT, "6", "8 pocket regions", ["top 5% of the smoothed", "logit → greedy balls of 8 Å", "(no gradient through choice)"], "ours")
pts = surf_pts(x + 18, BY + 92, BW - 36)
for k, (px, py) in enumerate(pts):
    circle(px, py, 3, "#b8bcc4")
for k, (i, col) in enumerate([(14, TYPE_COL[0]), (4, TYPE_COL[1]), (22, TYPE_COL[2])]):
    px, py = pts[i]
    circle(px, py - 4, 24, col, col, 1.2, 'fill-opacity="0.18"')
    circle(px, py - 30, 9, col, "#ffffff", 1.5)
    text(px, py - 26, str(k + 1), 11, "#ffffff", "middle", "bold")

# 7 pair pooling -> anchors
x = xs[6]
box(x, BY, BW, BH, PROT, "7", "Pair pooling → anchors", ["per ball: which site types", "sit at what distance (288)", "→ BatchNorm → MLP → 128-d"], "ours")
g0x, g0y, c = x + 20, BY + 26, 9
for i in range(8):
    for j in range(8):
        if j >= i:
            v = (math.sin(i * 1.7 + j * 2.3) + 1) / 2
            rect(g0x + j * c, g0y + i * c, c - 1, c - 1, PROT, extra=f'fill-opacity="{0.15 + 0.75 * v:.2f}"')
text(g0x + 4 * c, g0y + 8 * c + 14, "type × type × 8 bins", 10.5, FAINT, "middle")
ox, oy = x + 150, BY + 70
for k, ang in enumerate([60, 20, -25]):
    a = math.radians(ang)
    line(ox, oy, ox + 34 * math.cos(a), oy - 34 * math.sin(a), TYPE_COL[k], 2.4,
         'stroke-linecap="round"')
circle(ox, oy, 2.5, INK)
text(ox + 2, oy + 26, "unit vectors", 10.5, FAINT, "middle")

for i in range(6):
    arrow(xs[i] + BW + 2, BY + 70, xs[i + 1] - 3, BY + 70)

# ================================================================ ligand row
LY = 610
text(40, LY - 10, "LIGAND", 13, LIG, weight="bold")
lx = [BX + i * (BW + GAP) for i in range(4)]
x = lx[0]
box(x, LY, BW, BH, LIG, "1", "2D molecule", ["SMILES from PLINDER;", "no crystal pose, so nothing", "about the answer leaks in"], "")
text(x + BW / 2, LY + 66, "c1ccc(cc1)C(=O)N…", 15, LIG, "middle")

x = lx[1]
box(x, LY, BW, BH, LIG, "2", "Heavy-atom graph", ["47 atom features (element,", "charge, aromaticity, …),", "6 bond features (RDKit)"], "ours")
hx, hy, r = x + 80, LY + 64, 26
hexp = [(hx + r * math.cos(math.radians(60 * k + 30)), hy + r * math.sin(math.radians(60 * k + 30))) for k in range(6)]
for k in range(6):
    line(*hexp[k], *hexp[(k + 1) % 6], LIG, 2)
tail = [hexp[0], (hexp[0][0] + 26, hexp[0][1] - 10), (hexp[0][0] + 50, hexp[0][1] + 2)]
line(*tail[0], *tail[1], LIG, 2)
line(*tail[1], *tail[2], LIG, 2)
line(tail[1][0], tail[1][1], tail[1][0] + 4, tail[1][1] - 24, LIG, 2)
for p in hexp + tail[1:] + [(tail[1][0] + 4, tail[1][1] - 24)]:
    circle(*p, 4.5, "#ffffff", LIG, 2)

x = lx[2]
box(x, LY, BW, BH, LIG, "3", "Graph network", ["GINE, 4 layers × 128,", "messages along bonds;", "mean + max over atoms"], "ours")
nodes = [(x + 40, LY + 50), (x + 80, LY + 32), (x + 120, LY + 52), (x + 100, LY + 92),
         (x + 56, LY + 92), (x + 155, LY + 80)]
edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (2, 5)]
for a, b in edges:
    line(*nodes[a], *nodes[b], LIG_L, 5)
for a, b in [(0, 1), (2, 1), (4, 0), (5, 2)]:
    (x1, y1), (x2, y2) = nodes[a], nodes[b]
    arrow(x1 + 0.25 * (x2 - x1), y1 + 0.25 * (y2 - y1), x1 + 0.7 * (x2 - x1), y1 + 0.7 * (y2 - y1), LIG, 1.6)
for p in nodes:
    circle(*p, 7, LIG, "#ffffff", 1.5)

x = lx[3]
box(x, LY, BW, BH, LIG, "4", "Ligand vector", ["MLP → one unit vector", "in the same 128-d space", "as the anchors"], "ours")
ox, oy = x + 70, LY + 90
line(ox, oy, ox + 60, oy - 50, INK, 3, 'stroke-linecap="round"')
circle(ox, oy, 2.5, INK)
text(ox + 66, oy - 52, "ℓ", 18, INK, weight="bold")

for i in range(3):
    arrow(lx[i] + BW + 2, LY + 70, lx[i + 1] - 3, LY + 70)

# ================================================================ score + loss
SX, SY, SW, SH = 940, 430, 620, 412
rect(SX, SY, SW, SH, PANEL, RULE, 12)
rect(SX, SY, SW, 4, LOSS, rx=2)
text(SX + 20, SY + 34, "Score and training", 17, INK, weight="bold")
text(SX + 20, SY + 60, "score(protein, ligand) = max over its 8 anchors of cos(aₖ, ℓ) / τ", 14,
     INK, weight="bold")
text(SX + 20, SY + 79, "a protein binds a ligand if ANY of its pockets matches; τ is learned", 12.5, MUTED)

# batch matrix
mx, my, cs, n = SX + 24, SY + 112, 24, 8
masked = {(1, 4), (4, 1), (2, 6), (6, 2), (5, 7), (7, 5)}
for i in range(n):
    for j in range(n):
        if i == j:
            rect(mx + j * cs, my + i * cs, cs - 2, cs - 2, LOSS, rx=3)
        elif (i, j) in masked:
            rect(mx + j * cs, my + i * cs, cs - 2, cs - 2, "url(#hatch)", RULE, 3)
        else:
            rect(mx + j * cs, my + i * cs, cs - 2, cs - 2, LOSS_L, rx=3)
text(mx + n * cs / 2, my - 10, "32 ligands in the batch", 11.5, MUTED, "middle")
w(f'<text x="{mx - 10}" y="{my + n * cs / 2}" font-size="11.5" fill="{MUTED}" text-anchor="middle" '
  f'transform="rotate(-90 {mx - 10} {my + n * cs / 2})">32 proteins</text>')

tx = mx + n * cs + 26
lines = [
    ("InfoNCE over the batch", INK, "bold", 14),
    ("each true pair (diagonal) must outscore every", MUTED, "normal", 12.5),
    ("other protein and every other ligand", MUTED, "normal", 12.5),
    ("", MUTED, "normal", 6),
    ("False negatives masked (hatched)", INK, "bold", 13),
    ("same target, same ligand, or Tanimoto ≥ 0.8", MUTED, "normal", 12.5),
    ("One row per target per batch", INK, "bold", 13),
    ("so popular targets cannot dominate", MUTED, "normal", 12.5),
]
yy = my + 6
for s, col, wt, sz in lines:
    if s:
        text(tx, yy, s, sz, col, weight=wt)
    yy += sz + 7

line(SX + 20, SY + 330, SX + SW - 20, SY + 330, RULE)
text(SX + 20, SY + 354, "+ auxiliary losses", 13, INK, weight="bold")
text(SX + 20, SY + 375, "pocket BCE on the logit (box 5): keeps the regions on real pockets", 12.5, MUTED)
text(SX + 20, SY + 394, "Tanimoto pull, weight 0.05: ligands at Tanimoto T ≥ 0.5 asked for cos ≥ T",
     12.5, MUTED)

# connectors into the score box
path(f"M {xs[6] + BW / 2} {BY + BH + 2} L {xs[6] + BW / 2} {SY - 10}", FAINT, 2,
     extra='marker-end="url(#ah)"')
text(xs[6] + BW / 2 + 8, (BY + BH + SY) / 2 + 4, "8 anchors a₁…a₈", 12.5, PROT, weight="bold")
arrow(lx[3] + BW + 2, LY + 70, SX - 4, LY + 70)
text(lx[3] + BW + 12, LY + 60, "ℓ", 14, LIG, weight="bold")

# inference note
rect(40, 400, 860, 170, "#ffffff", RULE, 10, 1, 'stroke-dasharray="5 4"')
text(60, 430, "At screening time", 15, INK, weight="bold")
text(60, 454, "encode every protein once and every ligand once and store the vectors; a screen is then", 13, MUTED)
text(60, 474, "one matrix of dot products between them, instead of one docking or simulation run per", 13, MUTED)
text(60, 494, "protein-ligand pair. The protein side needs no ligand and no known binding site:", 13, MUTED)
text(60, 514, "the pocket logit chooses where to look (boxes 5-6).", 13, MUTED)
tag(60, 548, "dMaSIF", FAINT)
text(124, 552, "published method, weights fine-tuned here", 12, MUTED)
tag(420, 548, "ours", PROT)
text(468, 552, "built in this project", 12, MUTED)

w("</svg>")
p = sys.argv[1] if len(sys.argv) > 1 else "docs/figures/pipeline.svg"
with open(p, "w") as fh:
    fh.write("\n".join(out) + "\n")
print(f"wrote {p}")
