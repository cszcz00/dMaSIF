"""Generate docs/figures/pair_pool.svg, a schematic of src/pair_pool.py at its defaults.

    python3 docs/figures/make_pair_pool_svg.py docs/figures/pair_pool.svg
"""
import math
import random
import sys

R_MIN, R_MAX, B, T, EXCL = 3.0, 16.0, 8, 8, 2.0
CENTRES = [R_MIN + k * (R_MAX - R_MIN) / (B - 1) for k in range(B)]
WIDTH = (R_MAX - R_MIN) / (B - 1)


def K(d):
    return [0.0 if d < EXCL else math.exp(-0.5 * ((d - c) / WIDTH) ** 2) for c in CENTRES]


INK, MUTED, RULE, BG, PANEL = "#1f2328", "#59636e", "#d0d7de", "#ffffff", "#f6f8fa"
C1, C2, C3, GREY = "#2f6fdf", "#e07b1a", "#2a9d6f", "#b8bcc4"
TYPE_COL = [C1, C2, C3, "#9a6fd8", "#d64f6f", "#3fa7c4", "#b59a2a", "#7d8590"]

out = []
w = out.append


def text(x, y, s, size=13, col=INK, anchor="start", weight="normal", style="normal"):
    s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    w(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{col}" text-anchor="{anchor}" '
      f'font-weight="{weight}" font-style="{style}">{s}</text>')


def rect(x, y, wd, ht, fill, stroke="none", rx=0, extra=""):
    w(f'<rect x="{x:.1f}" y="{y:.1f}" width="{wd:.1f}" height="{ht:.1f}" rx="{rx}" '
      f'fill="{fill}" stroke="{stroke}" {extra}/>')


def line(x1, y1, x2, y2, col=INK, sw=1.2, extra=""):
    w(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{col}" '
      f'stroke-width="{sw}" {extra}/>')


def arrow(x1, y1, x2, y2, col=MUTED):
    line(x1, y1, x2, y2, col, 1.4, 'marker-end="url(#ah)"')


def bars(x, y, wd, ht, vals, cols, gap=2, base=True):
    n = len(vals)
    bw = (wd - gap * (n - 1)) / n
    for i, v in enumerate(vals):
        h = max(v, 0) * ht
        c = cols[i] if isinstance(cols, list) else cols
        rect(x + i * (bw + gap), y + ht - h, bw, h, c)
    if base:
        line(x - 2, y + ht, x + wd + 2, y + ht, MUTED, 1)


def cluster(cx, cy, n, spread, col, rng, rmin=2.6, rmax=4.2):
    for _ in range(n):
        a, r = rng.uniform(0, 2 * math.pi), spread * math.sqrt(rng.random())
        w(f'<circle cx="{cx + r * math.cos(a):.1f}" cy="{cy + r * math.sin(a):.1f}" '
          f'r="{rng.uniform(rmin, rmax):.1f}" fill="{col}" fill-opacity="0.9"/>')


W, H = 1280, 800
w(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
  f'font-family="-apple-system, Segoe UI, Helvetica, Arial, sans-serif">')
w('<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
  f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{MUTED}"/></marker>'
  '<pattern id="hatch" width="5" height="5" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">'
  f'<line x1="0" y1="0" x2="0" y2="5" stroke="{GREY}" stroke-width="2"/></pattern></defs>')
rect(0, 0, W, H, BG)

text(24, 36, "Pair pooling: one pocket region → one 312-dim vector", 21, weight="600")
text(24, 58, "src/pair_pool.py, --pool_mode pairs, defaults: T = 8 learned site types, "
     "B = 8 distance bins 3–16 Å, E = 16", 13, MUTED)

# ---------------------------------------------------------------- panels
PY, PH = 78, 430
panels = [(20, 290), (326, 290), (632, 330), (978, 282)]
titles = ["① One region, n surface points", "② Two small heads per point",
          "③ Type × type × distance histogram", "④ Concatenate"]
for (x, wd), t in zip(panels, titles):
    rect(x, PY, wd, PH, PANEL, RULE, 8)
    text(x + 14, PY + 26, t, 15, weight="600")
for (x1, w1), (x2, _) in zip(panels, panels[1:]):
    arrow(x1 + w1 + 2, PY + PH / 2, x2 - 3, PY + PH / 2)

# ① region
rng = random.Random(3)
x0 = panels[0][0]
cx, cy = x0 + 145, PY + 150
w(f'<path d="M{cx-115},{cy} C{cx-120},{cy-95} {cx+90},{cy-120} {cx+118},{cy-20} '
  f'C{cx+135},{cy+70} {cx-20},{cy+120} {cx-80},{cy+80} C{cx-110},{cy+60} {cx-112},{cy+30} '
  f'{cx-115},{cy} z" fill="#ffffff" stroke="{RULE}" stroke-width="1.5"/>')
for _ in range(170):
    while True:
        px, py = rng.uniform(-110, 115), rng.uniform(-105, 105)
        if (px / 112) ** 2 + (py / 92) ** 2 < 0.9:
            break
    w(f'<circle cx="{cx+px:.1f}" cy="{cy+py:.1f}" r="1.6" fill="{GREY}"/>')
cluster(cx - 55, cy - 30, 14, 9, C1, rng)
cluster(cx + 45, cy - 40, 14, 9, C2, rng)
cluster(cx + 20, cy + 45, 22, 16, C3, rng, 2.2, 3.2)
cluster(cx - 50, cy + 40, 10, 8, C2, rng, 2.2, 3.4)
ly = PY + 290
for i, (c, lab) in enumerate([(C1, "type 1"), (C2, "type 2"), (C3, "type 3")]):
    w(f'<circle cx="{x0 + 22 + i * 82}" cy="{ly}" r="4" fill="{c}"/>')
    text(x0 + 31 + i * 82, ly + 4, lab, 12)
text(x0 + 14, ly + 22, "colour = argmax learned type (unnamed),", 12, MUTED)
text(x0 + 14, ly + 38, "dot size = salience s_i; grey = not a site", 12, MUTED)
text(x0 + 14, ly + 66, "per point i:", 12, weight="600")
text(x0 + 14, ly + 84, "z_i = [ h_i ‖ input channels ]", 12)
text(x0 + 14, ly + 100, "  conv embedding + curvature, AtomNet,", 11, MUTED)
text(x0 + 14, ly + 114, "  chemo (drop with --no_pair_feats)", 11, MUTED)

# ② heads
x0 = panels[1][0]
rect(x0 + 16, PY + 50, 64, 30, "#ffffff", RULE, 5)
text(x0 + 48, PY + 70, "z_i", 14, anchor="middle", weight="600")
rect(x0 + 120, PY + 50, 152, 30, "#ffffff", RULE, 5)
text(x0 + 196, PY + 70, "type_head  z→32→T+1", 12, anchor="middle")
arrow(x0 + 80, PY + 65, x0 + 118, PY + 65)
rect(x0 + 120, PY + 92, 152, 30, "#ffffff", RULE, 5)
text(x0 + 196, PY + 112, "sal_head  z→32→1", 12, anchor="middle")
w(f'<path d="M{x0+48},{PY+80} L{x0+48},{PY+107} L{x0+117},{PY+107}" fill="none" '
  f'stroke="{MUTED}" stroke-width="1.4" marker-end="url(#ah)"/>')
text(x0 + 16, PY + 150, "type_head → softmax, drop the ∅ slot:", 12, MUTED)
soft = [0.06, 0.48, 0.07, 0.03, 0.05, 0.04, 0.06, 0.05, 0.16]
bx, by, bwd, bht = x0 + 22, PY + 160, 238, 60
gap = 4
bw = (bwd - gap * 8) / 9
for i, v in enumerate(soft):
    h = v / 0.5 * bht
    fill = "url(#hatch)" if i == 8 else TYPE_COL[i]
    rect(bx + i * (bw + gap), by + bht - h, bw, h, fill, GREY if i == 8 else "none")
    text(bx + i * (bw + gap) + bw / 2, by + bht + 14, str(i + 1) if i < 8 else "∅", 10,
         MUTED, "middle")
line(bx - 2, by + bht, bx + bwd + 2, by + bht, MUTED, 1)
_xb = bx + 8 * (bw + gap)
line(_xb - 2, by + bht - 24, _xb + bw + 2, by + bht + 2, "#c62828", 1.5)
line(_xb - 2, by + bht + 2, _xb + bw + 2, by + bht - 24, "#c62828", 1.5)
text(x0 + 16, PY + 260, "→ a_i ∈ ℝ^T   soft site type", 13)

text(x0 + 16, PY + 296, "s_i = σ(sal_head(z_i)) · w_i", 13)
text(x0 + 16, PY + 312, "w_i = the encoder's pool weight", 11, MUTED)
rect(x0 + 16, PY + 340, 258, 58, "#ffffff", C1, 6)
text(x0 + 145, PY + 365, "m_i = s_i · a_i   (T,)", 15, anchor="middle", weight="600")
text(x0 + 145, PY + 385, "site-type mass of point i", 12, MUTED, "middle")

# ③ histogram
x0, pw = panels[2]
rng = random.Random(11)
sx, sy = x0 + 90, PY + 70
cluster(sx, sy, 10, 8, C1, rng)
cluster(sx + 120, sy - 5, 10, 8, C2, rng)
line(sx + 8, sy, sx + 110, sy - 4, INK, 1.2, 'stroke-dasharray="4 3"')
text(sx + 60, sy - 12, "d_ij ≈ 7 Å", 12, anchor="middle")
text(sx + 60, sy + 28, "i ∈ type 1,  j ∈ type 2", 12, MUTED, "middle")

# gaussian bins plot
gx, gy, gw, gh = x0 + 30, PY + 130, pw - 52, 72
px_per_A = gw / 17.0
rect(gx, gy, EXCL * px_per_A, gh, "#eceff2")
for k, c in enumerate(CENTRES):
    pts = []
    for s in range(0, 171):
        d = s / 10
        v = 0 if d < EXCL else math.exp(-0.5 * ((d - c) / WIDTH) ** 2)
        pts.append(f"{gx + d * px_per_A:.1f},{gy + gh - v * gh:.1f}")
    w(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{MUTED}" stroke-width="1" '
      f'stroke-opacity="0.75"/>')
d7 = 7.0
line(gx + d7 * px_per_A, gy - 4, gx + d7 * px_per_A, gy + gh, "#c62828", 1.4)
line(gx, gy + gh, gx + gw, gy + gh, INK, 1)
for t in (0, 2, 3, 5, 10, 16):
    line(gx + t * px_per_A, gy + gh, gx + t * px_per_A, gy + gh + 4, INK, 1)
    text(gx + t * px_per_A, gy + gh + 16, str(t), 10, MUTED, "middle")
text(gx + gw, gy + gh + 16, "Å", 10, MUTED, "end")
text(gx + 2, gy - 6, "K_b(d): 8 Gaussian bins", 11, MUTED)
text(gx + 3, gy + gh - 4, "<2 Å", 9, MUTED)
text(x0 + 14, gy + gh + 34, "pairs under 2 Å dropped: one atom's own point cluster", 11, MUTED)

text(x0 + pw / 2, PY + 270, "H[t₁,t₂,b] = Σ_ij m_i[t₁] m_j[t₂] K_b(d_ij)", 14, anchor="middle",
     weight="600")

# upper-triangle grid
ux, uy, cs = x0 + 22, PY + 292, 15
for r in range(T):
    for c in range(T):
        if c < r:
            continue
        hi = (r, c) == (0, 1)
        rect(ux + c * cs, uy + r * cs, cs - 1.5, cs - 1.5, C1 if hi else "#ffffff",
             C1 if hi else RULE, 1.5)
text(ux, uy + T * cs + 16, "36 type pairs (t₁ ≤ t₂)", 11, MUTED)
hx, hy = ux + T * cs + 40, uy + 12
line(ux + 2 * cs, uy + cs / 2, hx - 6, hy + 30, C1, 1, 'stroke-dasharray="3 3"')
bars(hx, hy, 120, 66, K(7.0), C1)
text(hx, hy - 8, "cell (1, 2): 8 bins", 11, MUTED)
text(hx + 60, hy + 82, "3 Å  →  16 Å", 10, MUTED, "middle")

# ④ output vector
x0, pw = panels[3]
vx, vy, vw, vh = x0 + 22, PY + 50, 40, 360
segs = [(16, C3, "mean", "Σ w_i h_i / Σ w_i", "E = 16  what it's made of"),
        (8, C2, "mass", "log1p(Σ_i m_i / 10)", "T = 8  how much of each type"),
        (288, C1, "pairs", "log1p(H[t₁≤t₂] / 100)", "36 × 8 = 288  which type at what distance")]
y = vy
for n, col, name, f, note in segs:
    h = vh * n / 312
    rect(vx, y, vw, h, col, "#ffffff", 0, 'stroke-width="0.8"')
    y += h
rect(vx, vy, vw, vh, "none", INK, 0, 'stroke-width="1"')
lx = vx + vw + 14
for (n, col, name, f, note), ty, ay in zip(segs, (PY + 62, PY + 136, PY + 230),
                                          (vy + 3, vy + 13, vy + 120)):
    line(vx + vw + 1, ay, lx - 3, ty - 4, col, 1)
    text(lx, ty, name, 14, col, weight="600")
    text(lx, ty + 17, f, 12)
    words, cur, lines = note.split("  "), "", []
    text(lx, ty + 33, words[0], 11, MUTED)
    text(lx, ty + 47, words[1], 11, MUTED)
text(lx, PY + 340, "= 312 dims", 14, weight="600")
text(lx, PY + 362, "→ anchor_norm", 12, MUTED)
text(lx, PY + 378, "→ projection MLP", 12, MUTED)
text(lx, PY + 394, "→ 128-d anchor", 12, MUTED)

# ---------------------------------------------------------------- bottom row
BY = PY + PH + 22
rect(20, BY, 860, H - BY - 20, PANEL, RULE, 8)
text(34, BY + 26, "Why: same composition, different arrangement", 15, weight="600")
text(34, BY + 44, "mean and mass cannot tell these two pockets apart; pairs can", 12, MUTED)
for k, (d, lab) in enumerate([(5.0, "A: sites 5 Å apart"), (12.0, "B: sites 12 Å apart")]):
    ox = 40 + k * 420
    rng = random.Random(20 + k)
    sc = 9.0
    cluster(ox + 40, BY + 110, 9, 7, C1, rng)
    cluster(ox + 40 + d * sc, BY + 110, 9, 7, C2, rng)
    line(ox + 48, BY + 110, ox + 32 + d * sc, BY + 110, INK, 1, 'stroke-dasharray="3 3"')
    text(ox + 40 + d * sc / 2, BY + 98, f"{d:.0f} Å", 11, anchor="middle")
    text(ox, BY + 72, lab, 13, weight="600")
    text(ox, BY + 150, "mean, mass:", 12, MUTED)
    text(ox + 76, BY + 150, "identical", 12, weight="600")
    text(ox, BY + 176, "pairs (1, 2):", 12, MUTED)
    bars(ox + 82, BY + 162, 150, 64, K(d), C1)
    text(ox + 82, BY + 242, "3 Å", 10, MUTED)
    text(ox + 232, BY + 242, "16 Å", 10, MUTED, "end")
line(450, BY + 64, 450, H - 34, RULE, 1)

rect(896, BY, 364, H - BY - 20, PANEL, RULE, 8)
text(910, BY + 26, "Properties (checked by the self-test)", 15, weight="600")
props = [("rigid-motion invariant", "only distances inside the region"),
         ("permutation invariant", "sums over points"),
         ("differentiable", "through a_i, s_i, mean; not the bins"),
         ("counts, not fractions", "two ≠ five donors; log1p for scale"),
         ("memory", "n×n×B per region, checkpointed;"),
         ("", ">4096 points thinned, H × (n/kept)²")]
yy = BY + 52
for a, b in props:
    if a:
        text(910, yy, a, 12, weight="600")
    text(1052, yy, b, 11, MUTED)
    yy += 22 if a or True else 0

w("</svg>")
open(sys.argv[1], "w").write("\n".join(out) + "\n")
