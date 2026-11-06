"""Generate docs/figures/dataset_selection.svg, the dataset slide's figure.

What dataset_v2 keeps of PLINDER, as two tiles: INTERACTIONS and DISTINCT
TRAINING TARGETS. They are on different scales, so they are two tiles rather
than one chart with two axes, and the point is that they disagree - v2 holds
a tenth of PLINDER's interactions but covers a third of its targets. Breadth,
not depth, because v3c+ plateaued by MEMORISATION at ~67 visits per target
group (STATUS.md) and only more distinct targets moves that.

Form: a hero number per tile over a meter (the fill is v2's share of the
PLINDER track). Accent #2a5599 passes the lightness band, chroma floor and
3:1 contrast against white; both meters are direct-labelled.

The numbers are transcribed, not recomputed (the tables are HPC-side):
    PLINDER rows        docs/DATA.md - manifest.parquet, 341,945 system x
                        proper-ligand rows, all splits
    PLINDER targets     docs/RESULTS.md, "Dataset funnel, first run" - the
                        manifest holds 24,327 TRAIN target groups
    dataset_v2          docs/RESULTS.md, "dataset_v2 BUILT" - 35,407
                        interactions and items.tsv's 7,895 train target groups

Rows are all splits; targets are the train split.

    python3 docs/figures/make_dataset_fig.py docs/figures/dataset_selection.svg
"""
import sys

# (eyebrow, dataset_v2, PLINDER, noun)
TILES = [
    ("Interactions", 35_407, 341_945, "interactions"),
    ("Distinct training targets", 7_895, 24_327, "targets"),
]

# Serif, to sit in the deck rather than beside it. The reference design system
# is all-sans; the deck is the design system here, so this is a substitution.
FONT = "Palatino, 'Palatino Linotype', 'Book Antiqua', Georgia, serif"
INK, MUTED, FAINT = "#1a1a1a", "#5a5a56", "#86857f"
TRACK = "#ecebe6"                          # empty part of the meter
ACCENT = "#2a5599"                         # dataset_v2
RULE = "#dcdbd4"

W, H = 1240, 282
X0, TILE_DX, TRACK_W = 60, 620, 500
BAR_H = 26


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def text(out, x, y, s, size=15, fill=INK, anchor="start", weight="normal",
         spacing=None):
    extra = f' letter-spacing="{spacing}"' if spacing else ""
    out.append(f'<text x="{x:.1f}" y="{y:.1f}" font-family="{FONT}" '
               f'font-size="{size}" fill="{fill}" text-anchor="{anchor}" '
               f'font-weight="{weight}"{extra}>{esc(s)}</text>')


def build():
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
           f'width="{W}" height="{H}">',
           f'<rect width="{W}" height="{H}" fill="#ffffff"/>']
    text(out, X0, 46, "dataset_v2", 17, INK, weight="bold")
    out.append(f'<line x1="{X0}" y1="62" x2="{W - X0}" y2="62" '
               f'stroke="{RULE}" stroke-width="1"/>')

    for i, (eyebrow, n, top, noun) in enumerate(TILES):
        x = X0 + i * TILE_DX
        text(out, x, 102, eyebrow.upper(), 13, FAINT, weight="bold", spacing="1.6")
        text(out, x, 164, f"{n:,}", 46, INK, weight="bold")
        y = 190
        out.append(f'<rect x="{x}" y="{y}" width="{TRACK_W}" height="{BAR_H}" '
                   f'rx="4" fill="{TRACK}"/>')
        w = max(TRACK_W * n / top, 3.0)
        out.append(f'<rect x="{x}" y="{y}" width="{w:.1f}" height="{BAR_H}" '
                   f'rx="{min(4.0, w / 2):.1f}" fill="{ACCENT}"/>')
        text(out, x + w + 10, y + BAR_H / 2 + 5, f"{n / top:.0%}", 14, FAINT)
        text(out, x, y + BAR_H + 30,
             f"of PLINDER's {top:,} {noun}", 14.5, MUTED)
    out.append("</svg>")
    return "\n".join(out)


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "docs/figures/dataset_selection.svg"
    with open(path, "w") as fh:
        fh.write(build())
    print(f"wrote {path}")
