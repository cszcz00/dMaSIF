"""Generate docs/figures/pair_pool_slide.svg: pair pooling in two boxes, for a talk.

    python3 docs/figures/make_pair_pool_slide_svg.py docs/figures/pair_pool_slide.svg

The slide-readable companion to pair_pool.svg (make_pair_pool_svg.py), which
stays as the detailed reference: same method, two boxes, no formulas.

The inset histogram is computed with pair_pool.PairPool's own arithmetic,
reimplemented in numpy (the module itself needs torch), on the cartoon pocket
that the figure draws:

    K_b(d) = exp(-0.5 ((d - c_b) / w)^2),  zeroed below `exclude`
    H[t1, t2, b] = sum_ij m_i[t1] m_j[t2] K_b(d_ij)

at the defaults in surface_encoder.EncoderConfig: T = 8 types, B = 8 bins,
centres linspace(3, 16, 8) A, width = the spacing, exclude 2 A.
"""
import math
import sys

import numpy as np

INK, MUTED, FAINT, RULE, BG, PANEL = "#1f2328", "#59636e", "#8c959f", "#d0d7de", "#ffffff", "#f6f8fa"
ACC = "#2f6fdf"
T1, T2, T3 = "#2f6fdf", "#e07b1a", "#2a9d6f"
W, H = 1600, 660

# ---------------------------------------------------------------- real arithmetic
T, B, R_MIN, R_MAX, EXCL = 8, 8, 3.0, 16.0, 2.0
CENTRES = np.linspace(R_MIN, R_MAX, B)
WIDTH = (R_MAX - R_MIN) / (B - 1)


def site(cx, cy, n, spread, rng):
    """One chemical group = a cluster of ~n surface points ~1 A apart."""
    a = rng.uniform(0, 2 * math.pi, n)
    r = spread * np.sqrt(rng.random(n))
    return np.stack([cx + r * np.cos(a), cy + r * np.sin(a)], 1)


# the cartoon pocket, in angstroms: two polar sites ~7 A apart, one patch below
rng = np.random.default_rng(4)
XY = np.concatenate([site(-5.0, 1.0, 10, 1.7, rng),
                     site(2.0, 2.0, 10, 1.7, rng),
                     site(-1.0, -6.0, 8, 1.9, rng)])
TY = np.concatenate([np.zeros(10, int), np.ones(10, int), np.full(8, 2)])
D_PAIR = float(np.hypot(7.0, 1.0))                       # the drawn distance, ~7.1 A

d = np.sqrt(((XY[:, None] - XY[None]) ** 2).sum(-1))
k = np.exp(-0.5 * ((d[..., None] - CENTRES) / WIDTH) ** 2) * (d >= EXCL)[..., None]
HIST = np.einsum("i,ijb,j->b", (TY == 0).astype(float), k, (TY == 1).astype(float))
HIT_BIN = int(np.argmin(np.abs(CENTRES - D_PAIR)))        # the bin that pair lands in

out = []
w = out.append


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def text(x, y, s, size=13, col=INK, anchor="start", weight="normal"):
    w(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{col}" text-anchor="{anchor}" '
      f'font-weight="{weight}">{esc(s)}</text>')


def rect(x, y, wd, ht, fill, stroke="none", rx=0, sw=1, extra=""):
    w(f'<rect x="{x:.1f}" y="{y:.1f}" width="{wd:.1f}" height="{ht:.1f}" rx="{rx}" '
      f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}" {extra}/>')


def circle(x, y, r, fill, stroke="none", sw=1, extra=""):
    w(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r:.1f}" fill="{fill}" stroke="{stroke}" '
      f'stroke-width="{sw}" {extra}/>')


def line(x1, y1, x2, y2, col=INK, sw=1.2, extra=""):
    w(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{col}" '
      f'stroke-width="{sw}" {extra}/>')


w(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
  'font-family="Helvetica, Arial, sans-serif">')
w('<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" '
  f'markerHeight="8" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{FAINT}"/>'
  '</marker></defs>')
rect(0, 0, W, H, BG)

text(40, 50, "Pair pooling: one pocket region → one vector", 27, INK, weight="bold")
text(40, 80, "A mean over the region keeps what it is made of and loses where the pieces sit. "
     "This keeps which kinds of site lie at which distances from each other.", 15, MUTED)

BY, BH = 112, 500
B1X, B2X, BW = 40, 820, 740
for x in (B1X, B2X):
    rect(x, BY, BW, BH, "#ffffff", RULE, 10)
    rect(x, BY, BW, 4, ACC, rx=2)
line(B1X + BW + 14, BY + BH / 2, B2X - 14, BY + BH / 2, FAINT, 2.4,
     'marker-end="url(#ah)"')

# ================================================================ box 1
x = B1X
text(x + 26, BY + 44, "1   Type every point, measure every pair", 19, INK, weight="bold")
text(x + 26, BY + 70, "two small heads give each surface point a soft mix over 8 LEARNED", 13.5, MUTED)
text(x + 26, BY + 89, "site types and a salience weight — not hand-assigned chemistry", 13.5, MUTED)

# the pocket cartoon
pcx, pcy, SC = x + 215, BY + 225, 12.0
c = XY.mean(0)
w(f'<ellipse cx="{pcx:.1f}" cy="{pcy - 8:.1f}" rx="128" ry="118" fill="{PANEL}" '
  f'stroke="{RULE}" stroke-dasharray="5 4"/>')
text(pcx, BY + 128, "one pocket region", 12.5, FAINT, "middle")


def px_of(p):
    return pcx + (p[0] - c[0]) * SC, pcy - (p[1] - c[1]) * SC


for p, t in zip(XY, TY):
    qx, qy = px_of(p)
    circle(qx, qy, 5.2, [T1, T2, T3][t], "#ffffff", 1.2)
# the measured pair
a1x, a1y = px_of(XY[:10].mean(0))
a2x, a2y = px_of(XY[10:20].mean(0))
for qx, qy in ((a1x, a1y), (a2x, a2y)):
    circle(qx, qy, 25, "none", INK, 1.3, 'stroke-dasharray="3 3" opacity="0.55"')
line(a1x + 25, a1y, a2x - 25, a2y, INK, 1.6)
mx_, my_ = (a1x + a2x) / 2, (a1y + a2y) / 2
rect(mx_ - 34, my_ - 32, 68, 21, "#ffffff", INK, 10, 1.2)
text(mx_, my_ - 17, f"d ≈ {D_PAIR:.0f} Å", 13.5, INK, "middle", "bold")

# legend
lx, ly = x + 390, BY + 190
for kk, lab in enumerate(("type 1", "type 2", "type 3")):
    circle(lx + 8, ly + kk * 28, 6.5, [T1, T2, T3][kk], "#ffffff", 1.2)
    text(lx + 26, ly + kk * 28 + 5, lab, 13.5, MUTED)
text(lx, ly + 104, "…8 types in all", 13, FAINT)
text(lx, ly + 124, "one chemical group ≈ a", 13, FAINT)
text(lx, ly + 142, "cluster of points ~1 Å apart", 13, FAINT)

# distance bins
gx, gy, gw, gh = x + 70, BY + 368, 600, 62
for b in range(B):
    pts = []
    for i in range(81):
        dd = (R_MIN - 2) + i * ((R_MAX + 2) - (R_MIN - 2)) / 80
        v = math.exp(-0.5 * ((dd - CENTRES[b]) / WIDTH) ** 2) * (dd >= EXCL)
        pts.append(f"{gx + i * gw / 80:.1f},{gy + gh - v * gh:.1f}")
    hit = b == HIT_BIN
    w(f'<polyline points="{" ".join(pts)}" fill="none" stroke="{ACC if hit else RULE}" '
      f'stroke-width="{2.6 if hit else 1.2}"/>')
line(gx, gy + gh, gx + gw, gy + gh, MUTED, 1.2)
dm = gx + (D_PAIR - (R_MIN - 2)) / ((R_MAX + 2) - (R_MIN - 2)) * gw
line(dm, gy - 6, dm, gy + gh, "#d64f6f", 1.6, 'stroke-dasharray="3 3"')
text(gx, gy + gh + 17, "3 Å", 12, FAINT)
text(gx + gw, gy + gh + 17, "16 Å", 12, FAINT, "end")
text(gx - 4, gy - 16, "distance is soft: 8 overlapping bins, 3–16 Å", 13.5, MUTED)
text(x + 26, BY + BH - 22, "pairs under 2 Å are dropped — that is one group with itself",
     12.5, FAINT)

# ================================================================ box 2
x = B2X
text(x + 26, BY + 44, "2   Stack the counts into one vector", 19, INK, weight="bold")
text(x + 26, BY + 70, "one distance histogram for every pair of types, kept as COUNTS", 13.5, MUTED)
text(x + 26, BY + 89, "(two donors and five are different pockets), then log-scaled", 13.5, MUTED)

g0x, g0y, cell = x + 40, BY + 128, 25
for i in range(T):
    for j in range(i, T):
        v = 0.14 + 0.82 * abs(math.sin(i * 1.9 + j * 2.6))
        rect(g0x + j * cell, g0y + i * cell, cell - 3, cell - 3, ACC, rx=2.5,
             extra=f'fill-opacity="{v:.2f}"')
text(g0x, g0y + T * cell + 24, "36 type pairs  (8 × 8, upper triangle)", 13.5, MUTED)

ix, iy, iw, ih = x + 430, BY + 170, 210, 74
rect(ix - 20, iy - 34, iw + 40, ih + 76, PANEL, RULE, 8)
hmax = HIST.max()
bw = iw / B
for i, v in enumerate(HIST):
    hh = ih * v / hmax
    rect(ix + i * bw + 1.5, iy + ih - hh, bw - 3, hh, ACC, rx=2)
line(ix, iy + ih, ix + iw, iy + ih, MUTED, 1.2)
text(ix, iy + ih + 18, "3 Å", 12, FAINT)
text(ix + iw, iy + ih + 18, "16 Å", 12, FAINT, "end")
text(ix + iw / 2, iy - 14, "8 bins in each cell", 12.5, FAINT, "middle")
line(g0x + 6 * cell, g0y + 2 * cell, ix - 24, iy + 20, FAINT, 1.2, 'stroke-dasharray="3 3"')

vx, vy, vw, vh = x + 40, BY + 372, 660, 36
rect(vx, vy, vw * 288 / 312, vh, ACC, rx=5)
rect(vx + vw * 288 / 312, vy, vw * 8 / 312, vh, T2)
rect(vx + vw * 296 / 312, vy, vw * 16 / 312, vh, T3, rx=5)
text(vx + vw * 144 / 312, vy + 24, "288 pair counts", 15, "#ffffff", "middle", "bold")
circle(vx + 8, vy + vh + 28, 6, T2, "#ffffff", 1.2)
text(vx + 22, vy + vh + 33, "8 type masses", 13, MUTED)
circle(vx + 190, vy + vh + 28, 6, T3, "#ffffff", 1.2)
text(vx + 204, vy + vh + 33, "16-d mean", 13, MUTED)
text(vx + 350, vy + vh + 33, "=  312 numbers", 13.5, INK, weight="bold")
text(vx, BY + BH - 20, "→ BatchNorm → MLP → one 128-d anchor", 16, INK, weight="bold")

w("</svg>")
p = sys.argv[1] if len(sys.argv) > 1 else "docs/figures/pair_pool_slide.svg"
with open(p, "w") as fh:
    fh.write("\n".join(out) + "\n")
print(f"wrote {p}")
print(f"  cartoon pair d = {D_PAIR:.2f} A -> bin {HIT_BIN} (centre {CENTRES[HIT_BIN]:.2f} A)")
print(f"  type1 x type2 histogram: {np.round(HIST, 1)}")
