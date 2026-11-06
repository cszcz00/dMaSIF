"""Generate the two pictographs for the dataset slide.

    picto_druglike.svg   why the raw PDB ligands are slimmed to drug-like ones:
                         a size window. Fragments fall out below it, polymers
                         (peptides, glycans) above it, and what is left is the
                         200-900 Da band dataset_v2 keeps.
    picto_clusters.svg   why "data" is not "usable": interactions group into
                         similarity clusters, and a cluster is held out WHOLE
                         or the held-out rows leak into training.

Pictographs, not charts - the molecules are schematic ball-and-stick, and the
cluster dots are placed from a fixed seed, so neither carries data beyond the
band edges and the labels. The numbers both of them stand on are in
make_dataset_fig.py's docstring.

Why these two. docs/RESULTS.md, "Dataset funnel, first run" - the ligand-class
and mass filters are the big cut (341,945 -> 107,544 rows before any structure
quality filter runs), and the validation carve by target group is what makes a
training run smaller than the dataset it is built from.

Style matches make_dataset_fig.py: serif to sit in the deck, accent #2a5599
(passes the lightness band, chroma floor and 3:1 on white), grey for what is
excluded or held out.

    python3 docs/figures/make_slide_pictos.py docs/figures
"""
import math
import random
import sys
from pathlib import Path

FONT = "Palatino, 'Palatino Linotype', 'Book Antiqua', Georgia, serif"
INK, MUTED, FAINT = "#1a1a1a", "#5a5a56", "#86857f"
ACCENT, ACCENT_BG = "#2a5599", "#eaeff7"
GREY, GREY_BG = "#b4b3ac", "#f2f1ec"
RULE = "#dcdbd4"


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def text(out, x, y, s, size=15, fill=INK, anchor="middle", weight="normal",
         spacing=None):
    extra = f' letter-spacing="{spacing}"' if spacing else ""
    out.append(f'<text x="{x:.1f}" y="{y:.1f}" font-family="{FONT}" '
               f'font-size="{size}" fill="{fill}" text-anchor="{anchor}" '
               f'font-weight="{weight}"{extra}>{esc(s)}</text>')


def svg(w, h, body):
    return "\n".join([
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" '
        f'width="{w}" height="{h}">',
        f'<rect width="{w}" height="{h}" fill="#ffffff"/>', *body, "</svg>"])


# ---------------------------------------------------------------------------
# molecule glyphs: schematic ball-and-stick
# ---------------------------------------------------------------------------
def bond(out, a, b, color, w=2.2):
    out.append(f'<line x1="{a[0]:.1f}" y1="{a[1]:.1f}" x2="{b[0]:.1f}" '
               f'y2="{b[1]:.1f}" stroke="{color}" stroke-width="{w}" '
               f'stroke-linecap="round"/>')


def atom(out, p, color, r=5.0):
    out.append(f'<circle cx="{p[0]:.1f}" cy="{p[1]:.1f}" r="{r}" fill="{color}"/>')


def ring(cx, cy, r=15.0, start=-90):
    return [(cx + r * math.cos(math.radians(start + 60 * k)),
             cy + r * math.sin(math.radians(start + 60 * k))) for k in range(6)]


def draw_graph(out, pts, bonds, color, r=5.0):
    for i, j in bonds:
        bond(out, pts[i], pts[j], color)
    for p in pts:
        atom(out, p, color, r)


def fragment(out, cx, cy, color):
    """A handful of atoms: below the mass window."""
    pts = ring(cx, cy, 13)
    draw_graph(out, pts, [(k, (k + 1) % 6) for k in range(6)], color, 4.5)


def druglike(out, cx, cy, color):
    """A ring, a linker and a small second ring: the band that is kept."""
    a = ring(cx - 22, cy, 15)
    b = ring(cx + 26, cy + 6, 13)
    draw_graph(out, a, [(k, (k + 1) % 6) for k in range(6)], color)
    draw_graph(out, b, [(k, (k + 1) % 6) for k in range(6)], color)
    bond(out, a[2], b[5], color)       # lower-right to upper-left: clears both rings
    tail = [(cx - 30, cy - 26), (cx - 16, cy - 34)]
    bond(out, a[0], tail[0], color)
    bond(out, tail[0], tail[1], color)
    for p in tail:
        atom(out, p, color)


def polymer(out, cx, cy, color):
    """Repeating units in a chain: peptides and glycans, above the window."""
    xs = [cx - 56, cx - 19, cx + 18, cx + 55]
    prev = None
    for i, x in enumerate(xs):
        y = cy + (6 if i % 2 else -6)
        pts = ring(x, y, 11)
        draw_graph(out, pts, [(k, (k + 1) % 6) for k in range(6)], color, 4.0)
        if prev is not None:
            bond(out, prev[2], pts[5], color, 2.0)
        prev = pts


# ---------------------------------------------------------------------------
# picto 1: the drug-like window
# ---------------------------------------------------------------------------
def druglike_fig():
    W, H = 560, 300
    out = []
    cols = [(95, "Fragments", "under 200 Da", fragment, False),
            (280, "Drug-like", "200 – 900 Da", druglike, True),
            (465, "Polymers", "peptides, glycans", polymer, False)]
    cy = 120

    # The kept column sits on a tinted panel; the other two are plain.
    for cx, title, sub, draw, keep in cols:
        if keep:
            out.append(f'<rect x="{cx - 88}" y="42" width="176" height="158" '
                       f'rx="8" fill="{ACCENT_BG}"/>')
    for cx, title, sub, draw, keep in cols:
        draw(out, cx, cy, ACCENT if keep else GREY)
        text(out, cx, 170, title, 17, INK if keep else MUTED,
             weight="bold" if keep else "normal")
        text(out, cx, 191, sub, 13.5, FAINT)
        text(out, cx, 232, "keep" if keep else "drop", 14,
             ACCENT if keep else FAINT, weight="bold" if keep else "normal")

    # A size axis underneath, so the three columns read as one continuum.
    y = 258
    out.append(f'<line x1="60" y1="{y}" x2="{W - 74}" y2="{y}" '
               f'stroke="{RULE}" stroke-width="1.4"/>')
    out.append(f'<path d="M{W - 74} {y - 5} L{W - 60} {y} L{W - 74} {y + 5} Z" '
               f'fill="{RULE}"/>')
    text(out, 60, y + 24, "ligand size", 13.5, FAINT, anchor="start")
    return W, H, svg(W, H, out)


# ---------------------------------------------------------------------------
# picto 2: interaction clusters
# ---------------------------------------------------------------------------
def clusters_fig():
    W, H = 560, 300
    out = []
    rng = random.Random(7)
    split = 392                                  # the dashed divider

    def blob(cx, cy, n, rx, ry, color, bg):
        out.append(f'<ellipse cx="{cx}" cy="{cy}" rx="{rx}" ry="{ry}" '
                   f'fill="{bg}" stroke="{RULE}" stroke-width="1"/>')
        # Dart-throwing with a minimum separation: plain rejection sampling
        # clumps the dots and they read as one blot rather than n interactions.
        placed, tries = [], 0
        while len(placed) < n and tries < 4000:
            tries += 1
            dx, dy = rng.uniform(-1, 1), rng.uniform(-1, 1)
            if dx * dx + dy * dy > 0.58:
                continue
            p = (cx + dx * rx, cy + dy * ry)
            if all((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 > 15.5 ** 2
                   for q in placed):
                placed.append(p)
        for p in placed:
            atom(out, p, color, 4.6)

    text(out, 60, 46, "TRAINS ON", 13, FAINT, anchor="start", weight="bold",
         spacing="1.6")
    text(out, W - 60, 46, "HELD OUT", 13, FAINT, anchor="end", weight="bold",
         spacing="1.6")
    out.append(f'<line x1="60" y1="60" x2="{W - 60}" y2="60" '
               f'stroke="{RULE}" stroke-width="1"/>')

    # train side: several clusters of differing size
    for cx, cy, n, rx, ry in ((122, 118, 7, 46, 33), (254, 108, 5, 38, 28),
                              (330, 178, 4, 32, 25), (150, 206, 6, 44, 30),
                              (248, 178, 3, 26, 21)):
        blob(cx, cy, n, rx, ry, ACCENT, ACCENT_BG)
    # held-out side: whole clusters, never a slice of one
    for cx, cy, n, rx, ry in ((466, 125, 6, 42, 31), (470, 207, 4, 34, 25)):
        blob(cx, cy, n, rx, ry, GREY, GREY_BG)

    out.append(f'<line x1="{split}" y1="74" x2="{split}" y2="252" '
               f'stroke="{FAINT}" stroke-width="1.6" stroke-dasharray="6 5"/>')
    text(out, 60, 282, "one dot = one interaction · one blob = one cluster",
         13.5, FAINT, anchor="start")
    text(out, W - 60, 282, "a cluster is never split", 13.5, MUTED, anchor="end")
    return W, H, svg(W, H, out)


if __name__ == "__main__":
    outdir = Path(sys.argv[1] if len(sys.argv) > 1 else "docs/figures")
    outdir.mkdir(parents=True, exist_ok=True)
    for name, fn in (("picto_druglike", druglike_fig), ("picto_clusters", clusters_fig)):
        w, h, s = fn()
        (outdir / f"{name}.svg").write_text(s)
        print(f"wrote {outdir / name}.svg  ({w} x {h})")
