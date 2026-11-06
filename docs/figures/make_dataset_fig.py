"""Generate docs/figures/dataset_selection.svg, the dataset slide's figure.

Two meters, side by side: how much of PLINDER each dataset keeps, measured
first in INTERACTIONS and then in DISTINCT TRAINING TARGETS. They are on
different scales, so they are two panels rather than one chart with two axes.

The point of the figure is that the two disagree. dataset_v2 holds 1.7x
dataset_v1's interactions but 3.0x its targets, which is the whole reason it
was built: v3c+ plateaued by MEMORISATION at ~67 visits per target group
(STATUS.md), and only more distinct targets moves that. Visits per group fall
67 -> 20 at the 5,000 x 32 budget dataset_funnel.py reports.

Form: a meter per row (a fill against the PLINDER track), emphasis colouring -
dataset_v2 in the accent, everything else grey. Accent #2a5599 passes the
lightness band, chroma floor and 3:1 contrast against white; every bar is
direct-labelled, which is the relief the grey fills need.

The numbers are transcribed, not recomputed (the tables are HPC-side):
    PLINDER rows        docs/DATA.md - manifest.parquet, 341,945 system x
                        proper-ligand rows, all splits
    PLINDER targets     docs/RESULTS.md, "Dataset funnel, first run" - the
                        manifest holds 24,327 TRAIN target groups
    dataset_v1          docs/DATA.md (21,209 interactions) and the same
                        funnel section (2,605 train target groups)
    dataset_v2          docs/RESULTS.md, "dataset_v2 BUILT" - 35,407
                        interactions and items.tsv's 7,895 train target groups
    visits              dataset_funnel.py: steps x batch / groups at 5,000 x 32

Rows are all splits; targets are the train split. Both ratios quoted in the
footnote are taken within their own panel, so neither mixes the two scopes.

    python3 docs/figures/make_dataset_fig.py docs/figures/dataset_selection.svg
"""
import sys

# (label, interactions, train target groups)
ROWS = [
    ("PLINDER", 341_945, 24_327),
    ("dataset_v1", 21_209, 2_605),
    ("dataset_v2", 35_407, 7_895),
]
ACCENT_ROW = "dataset_v2"

# Serif, to sit in the deck rather than beside it. The reference design system
# is all-sans; the deck is the design system here, so this is a substitution.
FONT = "Palatino, 'Palatino Linotype', 'Book Antiqua', Georgia, serif"
INK, MUTED, FAINT = "#1a1a1a", "#5a5a56", "#86857f"
TRACK = "#ecebe6"                          # empty part of the meter
GREY = "#b4b3ac"                           # context fills
ACCENT = "#2a5599"                         # dataset_v2
RULE = "#dcdbd4"

W, H = 1420, 420
PANELS = [
    ("Interactions", 1, 60),               # (heading, index into ROWS tuple, x0)
    ("Distinct training targets", 2, 760),
]
LAB_W, TRACK_W, VAL_W = 115, 400, 85       # gutter, meter, value column
BAR_H, ROW_PITCH, Y0 = 30, 64, 118         # Y0 = top of the first bar


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def text(out, x, y, s, size=15, fill=INK, anchor="start", weight="normal",
         style="normal", spacing=None):
    extra = f' letter-spacing="{spacing}"' if spacing else ""
    out.append(f'<text x="{x:.1f}" y="{y:.1f}" font-family="{FONT}" '
               f'font-size="{size}" fill="{fill}" text-anchor="{anchor}" '
               f'font-weight="{weight}" font-style="{style}"{extra}>{esc(s)}</text>')


def meter(out, x, y, frac, color):
    """Track plus a proportional fill, both with 4px rounded ends -> fill width."""
    out.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{TRACK_W}" height="{BAR_H}" '
               f'rx="4" fill="{TRACK}"/>')
    w = max(TRACK_W * frac, 3.0)
    out.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{BAR_H}" '
               f'rx="{min(4.0, w / 2):.1f}" fill="{color}"/>')
    return w


def build():
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
           f'width="{W}" height="{H}">',
           f'<rect width="{W}" height="{H}" fill="#ffffff"/>']

    for heading, col, x0 in PANELS:
        top = ROWS[0][col]
        tx = x0 + LAB_W                                  # track left edge
        vx = tx + TRACK_W + VAL_W                        # value column, right-aligned
        text(out, x0, 72, heading.upper(), 13, FAINT, weight="bold", spacing="1.6")
        out.append(f'<line x1="{x0}" y1="88" x2="{vx}" y2="88" '
                   f'stroke="{RULE}" stroke-width="1"/>')

        for i, (label, *vals) in enumerate(ROWS):
            n = vals[col - 1]
            y = Y0 + i * ROW_PITCH
            mid = y + BAR_H / 2 + 5                      # text baseline in the bar
            accent = label == ACCENT_ROW
            w = meter(out, tx, y, n / top, ACCENT if accent else GREY)
            text(out, x0, mid, label, 15, INK if accent else MUTED,
                 weight="bold" if accent else "normal")
            text(out, vx, mid, f"{n:,}", 16, INK if accent else MUTED,
                 anchor="end", weight="bold" if accent else "normal")
            if i:        # the share, just past the fill; PLINDER is 100% by definition
                text(out, tx + w + 10, mid, f"{n / top:.0%}", 13, FAINT)

    # The punchline: the two panels disagree, and that is the whole point.
    yb = Y0 + (len(ROWS) - 1) * ROW_PITCH + BAR_H + 46
    out.append(f'<line x1="60" y1="{yb}" x2="{W - 60}" y2="{yb}" '
               f'stroke="{RULE}" stroke-width="1"/>')
    ra = ROWS[2][1] / ROWS[1][1]
    rb = ROWS[2][2] / ROWS[1][2]
    text(out, 60, yb + 38,
         f"dataset_v2 holds {ra:.1f}× dataset_v1's interactions "
         f"but {rb:.1f}× its distinct targets.", 19, INK)
    text(out, 60, yb + 66,
         "At a fixed training budget that cuts visits per target from 67 to 20 "
         "— the memorisation ceiling, not row count, was the lever.", 15, MUTED)
    out.append("</svg>")
    return "\n".join(out)


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "docs/figures/dataset_selection.svg"
    with open(path, "w") as fh:
        fh.write(build())
    print(f"wrote {path}")
