"""Generate docs/figures/dataset_selection.svg, the dataset slide's figure.

What dataset_v2 holds, and what ONE TRAINING RUN can actually use. Four
numbers, no bars: the held-out share is 7.0% of rows and 3.2% of groups, and
a bar at 93% full cannot show the difference between those two, so the
numbers carry it themselves.

The distinction is not bookkeeping. Validation is carved by TARGET GROUP -
UniProt, else PLINDER's 95% strong interaction component (probe_retrieval
.group_of) - so a cluster is held out WHOLE or it leaks. Holding out 251 of
7,895 target groups (3.2%) therefore costs 2,404 of 34,265 rows (7.0%): the
held-out clusters carry 9.6 rows each against the 4.3 of an average one,
because they are dataset_v1's validation targets, which were the data-rich
ones. Only 400 of those 2,404 rows are ever scored.

BOTH columns are the TRAIN SPLIT, deliberately. items.tsv's 7,895 is a
train-split group count, so pairing it with the whole dataset's 35,407
interactions (train 34,316 / val 356 / test 735) would put two scopes in one
table. Quote 35,407 as the dataset's size in prose if the slide needs it.

The numbers are transcribed, not recomputed (the tables are HPC-side), all
from docs/RESULTS.md, "dataset_v2 BUILT":
    34,265 /    out/runs/dataset_v2/items.tsv - the train split as prepared,
    7,895       after v1's --exclude and one unfeaturisable ligand, and its
                target groups
    31,861 /    what a run trains on under the fixed validation set
    7,644       out/valset_v1v2 (400 rows over 240 groups)

    python3 docs/figures/make_dataset_fig.py docs/figures/dataset_selection.svg
"""
import sys

# (row label, dataset_v2 train split, what a run trains on)
ROWS = [
    ("Interactions", 34_265, 31_861),
    ("Distinct targets", 7_895, 7_644),
]
COL_A, COL_B = "dataset_v2, train split", "a training run uses"

# Serif, to sit in the deck rather than beside it. The reference design system
# is all-sans; the deck is the design system here, so this is a substitution.
FONT = "Palatino, 'Palatino Linotype', 'Book Antiqua', Georgia, serif"
INK, MUTED, FAINT = "#1a1a1a", "#5a5a56", "#86857f"
ACCENT = "#2a5599"                         # what the run trains on
RULE = "#dcdbd4"

W, H = 720, 332
X0, XA, XARR, XB = 60, 440, 470, 660   # labels, col A right, arrow, col B right
Y_HEAD, Y0, ROW_PITCH = 60, 150, 72


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def text(out, x, y, s, size=15, fill=INK, anchor="start", weight="normal",
         spacing=None):
    extra = f' letter-spacing="{spacing}"' if spacing else ""
    out.append(f'<text x="{x:.1f}" y="{y:.1f}" font-family="{FONT}" '
               f'font-size="{size}" fill="{fill}" text-anchor="{anchor}" '
               f'font-weight="{weight}"{extra}>{esc(s)}</text>')


def arrow(out, x, y, w=34):
    """Drawn, not a glyph, so it does not depend on the font's arrow."""
    out.append(f'<path d="M{x} {y} H{x + w - 6}" stroke="{FAINT}" '
               f'stroke-width="1.6" fill="none"/>')
    out.append(f'<path d="M{x + w - 9} {y - 4.5} L{x + w} {y} '
               f'L{x + w - 9} {y + 4.5} Z" fill="{FAINT}"/>')


def build():
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
           f'width="{W}" height="{H}">',
           f'<rect width="{W}" height="{H}" fill="#ffffff"/>']
    text(out, XA, Y_HEAD, COL_A.upper(), 13, FAINT, anchor="end",
         weight="bold", spacing="1.6")
    text(out, XB, Y_HEAD, COL_B.upper(), 13, FAINT, anchor="end",
         weight="bold", spacing="1.6")
    out.append(f'<line x1="{X0}" y1="{Y_HEAD + 16}" x2="{XB}" y2="{Y_HEAD + 16}" '
               f'stroke="{RULE}" stroke-width="1"/>')

    for i, (label, have, use) in enumerate(ROWS):
        y = Y0 + i * ROW_PITCH
        text(out, X0, y, label, 18, MUTED)
        text(out, XA, y, f"{have:,}", 40, INK, anchor="end")
        arrow(out, XARR, y - 13)
        text(out, XB, y, f"{use:,}", 40, ACCENT, anchor="end", weight="bold")

    yb = Y0 + (len(ROWS) - 1) * ROW_PITCH + 44
    out.append(f'<line x1="{X0}" y1="{yb}" x2="{XB}" y2="{yb}" '
               f'stroke="{RULE}" stroke-width="1"/>')
    text(out, X0, yb + 32,
         "Validation is carved by interaction cluster, so a cluster is held "
         "out whole.", 15, MUTED)
    out.append("</svg>")
    return "\n".join(out)


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "docs/figures/dataset_selection.svg"
    with open(path, "w") as fh:
        fh.write(build())
    print(f"wrote {path}")
