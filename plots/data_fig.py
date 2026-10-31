"""Data selection, simply: PLINDER -> dataset_v1 in four bars, dataset_v2 for scale.

Rows left after each group of filters, in the order build_manifest.py applies
them, with the criteria in small type under each bar. dataset_v2 is a separate
bar in a second colour: a relaxed rebuild, not a further step of the funnel.

The numbers are transcribed, not recomputed (the manifest is HPC-side):
    PLINDER rows      docs/DATA.md (manifest: 341,945 system x proper-ligand
                      rows; ions, artifacts and unloadable ligands already out)
    stages 2-3        docs/RESULTS.md, "Dataset funnel, first run" - rows left
                      after `interactions` and after `no_altlocs`
    dataset_v1        docs/DATA.md; 2,605 train target groups from the same
                      funnel section
    dataset_v2        docs/RESULTS.md, "dataset_v2 BUILT" (35,407 rows) and
                      "Funnel, fourth run" (7,897 train target groups)

Example:
    python plots/data_fig.py --out plots/output/data_selection

Depends on: matplotlib.
"""

import argparse
from pathlib import Path

# (label, rows, criteria) - the funnel, in filter order
STAGES = [
    ("PLINDER", 341_945,
     "protein–ligand systems with a proper small-molecule ligand"),
    ("Drug-like ligand", 88_678,
     "not a cofactor, oligomer or covalent · 200–800 Da · 3–50 interactions"),
    ("Well-defined pocket", 29_933,
     "one proper ligand · ≤ 2.5 Å resolution · pocket complete, resolved, no crystal contacts"),
    ("dataset_v1", 21_209,
     "one ligand chain (one molecule per example) · deduplicated · 2,605 train targets"),
]
V2 = ("dataset_v2", 35_407,
      "relaxed: resolution, buried fragments, cofactors, small companions · ≤ 50 rows per ligand"
      " · 7,897 train targets")

BAR = "#2a78d6"        # dataviz reference palette, slot 1
BAR_V2 = "#eb6834"     # slot 2: a different dataset, not a further step
INK, INK_2 = "#0b0b0b", "#52514e"


def draw(out):
    import matplotlib.pyplot as plt

    rows = STAGES + [V2]
    fig = plt.figure(figsize=(11, 5.6))
    ax = fig.add_axes([0.02, 0.02, 0.96, 0.84])
    top = rows[0][1]
    step, gap = 1.3, 0.4                     # row pitch; extra space before v2
    ys = [-(i * step + (gap if i == len(STAGES) else 0)) for i in range(len(rows))]
    for (label, n, crit), y in zip(rows, ys):
        color = BAR_V2 if label == "dataset_v2" else BAR
        ax.barh(y, n, height=0.34, color=color, edgecolor="none")
        ax.text(n + top * 0.008, y, f"{n:,}", va="center", ha="left", fontsize=12,
                color=INK, weight="bold")
        ax.text(0, y + 0.3, label, va="bottom", ha="left", fontsize=12.5, color=INK,
                weight="bold")
        ax.text(0, y - 0.24, crit, va="top", ha="left", fontsize=9.5, color=INK_2)
    ax.axhline((ys[len(STAGES) - 1] + ys[-1]) / 2, color="#d9d8d4", lw=0.8)
    ax.set_xlim(0, top * 1.12)
    ax.set_ylim(ys[-1] - 0.6, 0.7)
    ax.axis("off")
    fig.text(0.02, 0.99, "From PLINDER to a training set", fontsize=15, weight="bold",
             color=INK, va="top")
    fig.text(0.02, 0.935, "rows = protein–ligand pairs left after each group of filters",
             fontsize=10, color=INK_2, va="top")

    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(out.with_suffix(f".{ext}"), dpi=200, bbox_inches="tight",
                    facecolor="white")
    print(f"wrote {out.with_suffix('.png')} and .svg")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="plots/output/data_selection")
    draw(ap.parse_args().out)


if __name__ == "__main__":
    main()
