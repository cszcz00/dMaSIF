"""Generate docs/figures/dataset_v2.svg: how dataset_v2 ("the dataset") is built from PLINDER.

    python3 docs/figures/make_dataset_v2_svg.py docs/figures/dataset_v2.svg

Every number is a recorded one (docs/RESULTS.md, docs/DATA.md):
- strict attrition: "Dataset funnel, first run" (2026-10-29), rows after each
  filter in build_manifest.FILTERS order, ending at dataset_v1's 21,209
- per-entry core and the expansion ladder: "Funnel, third run" (cumulative NEW
  train target groups per step) and "Funnel, fourth run" (final 7,897, after
  the all-ligand cap and the detergent exclusion)
- the built dataset: "dataset_v2 BUILT" (2026-10-30)
PLINDER's own row count before the manifest's hard filters is not recorded
in the repo, so the figure starts at the manifest.
"""
import sys

INK, MUTED, FAINT, RULE, BG, PANEL = "#1f2328", "#59636e", "#8c959f", "#d0d7de", "#ffffff", "#f6f8fa"
CORE, CORE_L = "#2f6fdf", "#c9dafb"
GROW, GROW_L = "#2a9d6f", "#c4e8d8"
CAP = "#e07b1a"
W, H = 1600, 900

out = []
w = out.append


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def text(x, y, s, size=13, col=INK, anchor="start", weight="normal"):
    w(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{col}" '
      f'text-anchor="{anchor}" font-weight="{weight}">{esc(s)}</text>')


def rich(x, y, parts, size=13, anchor="start"):
    """One line of mixed styles: parts = [(text, colour, weight), ...]."""
    spans = "".join(f'<tspan fill="{c}" font-weight="{wt}">{esc(t)}</tspan>' for t, c, wt in parts)
    w(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" text-anchor="{anchor}">{spans}</text>')


def rect(x, y, wd, ht, fill, stroke="none", rx=0, sw=1):
    w(f'<rect x="{x:.1f}" y="{y:.1f}" width="{wd:.1f}" height="{ht:.1f}" rx="{rx}" '
      f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')


def line(x1, y1, x2, y2, col=RULE, sw=1, dash=""):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    w(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{col}" '
      f'stroke-width="{sw}"{d}/>')


def arrow(x1, y1, x2, y2, col=FAINT, sw=2):
    w(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{col}" '
      f'stroke-width="{sw}" marker-end="url(#ah)"/>')


def n(v):
    return f"{v:,}"


w(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
  'font-family="Helvetica, Arial, sans-serif">')
w('<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
  f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{FAINT}"/></marker></defs>')
rect(0, 0, W, H, BG)

# ---------------------------------------------------------------- title
text(40, 52, "The dataset: PLINDER, filtered to a drug-like, pocket-consistent core, "
     "then widened for target diversity", 24, INK, weight="bold")
text(40, 80, "One row = one protein–ligand interaction from a PLINDER system. "
     "Target group = UniProt accession, the unit training samples and validation holds out.",
     14, MUTED)

# ================================================================ column 1
X1, C1W = 40, 450
rect(X1, 105, C1W, 76, PANEL, RULE, 8)
text(X1 + 16, 131, "1 · Source: PLINDER 2024-06 (v2)", 16, INK, weight="bold")
text(X1 + 16, 152, "annotated protein–ligand systems with interaction", 13, MUTED)
text(X1 + 16, 169, "clusters, UniProt, ligand classes, train / val / test split", 13, MUTED)

text(X1, 214, "2 · A strict, drug-like core", 16, INK, weight="bold")
text(X1, 233, "rows left after each group of filters", 13, MUTED)

funnel = [
    (341945, "usable small molecules (manifest)",
     ["proper ligand; no ions, no crystallisation artifacts;", "RDKit-loadable"]),
    (107544, "drug-like",
     ["not a cofactor, oligomer or covalent ligand;", "MW 200–800 Da"]),
    (88678, "reliable ligand",
     ["coordinates correct; 3–50 protein contacts"]),
    (22443, "consistent pocket",
     ["one ligand chain, 5–100 pocket residues, complete and", "resolved, ≤ 2.5 Å, < 25% crystal contacts, no altlocs"]),
    (21209, "deduplicated core",
     ["one row per (interaction cluster, ligand): 2,605 targets,", "~67 training visits per target → memorisation"]),
]
y = 252
cx, maxw = X1 + 70, C1W - 70 - 80
for i, (v, title, desc) in enumerate(funnel):
    bw = max(maxw * v / funnel[0][0], 3)
    last = i == len(funnel) - 1
    rect(cx, y, bw, 22, CORE if last else CORE_L, rx=3)
    text(cx + bw + 8, y + 16, n(v), 15, CORE if last else INK, weight="bold")
    text(X1, y + 16, f"{100 * v / funnel[0][0]:.0f}%" if i else "100%", 12, FAINT)
    text(cx, y + 42, title, 14, INK, weight="bold")
    for k, d in enumerate(desc):
        text(cx, y + 60 + 16 * k, d, 12.5, MUTED)
    y += 70 + 16 * len(desc)

arrow(X1 + C1W + 4, 470, X1 + C1W + 40, 470)
text(X1, 776, "Clean but narrow: 6% of the usable rows and only", 13, MUTED)
text(X1, 793, "2,605 targets, too few for retrieval to generalise.", 13, MUTED)

# ================================================================ column 2
X2, C2W = 540, 620
text(X2, 131, "3 · Relax only where it adds new targets", 16, INK, weight="bold")
text(X2, 152, "training target groups, each step added on top of the previous ones", 13, MUTED)
text(X2, 169, "(symmetry copies removed first: one row per PDB entry, ligand and UniProt)", 13, MUTED)

base = 2379
ladder = [
    # (cumulative new groups, title, description)
    (0, "core, symmetry copies removed", "about half of the core's training rows were symmetry copies"),
    (210, "+ pockets with alternate conformations", "one conformer is used anyway"),
    (1028, "+ lower resolution", "≤ 3.5 Å, or ≤ 4.5 Å if the pocket is complete and rigid"),
    (2251, "+ small buried ligands", "120–200 Da when ≥ 90% buried; MW now up to 900 Da"),
    (2467, "+ covalent ligands", "glycosylation (sugars on Asn) excluded"),
    (3663, "+ small oligomers", "≤ 60 heavy atoms: short peptides, small sugars"),
    (4898, "+ systems with small companions", "extra chains allowed if each ≤ 6 heavy atoms (ions, SO₄, glycerol)"),
    (6025, "+ cofactors as targets", "HEM, FAD, NAD(P), SAH, nucleotides, ..."),
]
top = 8398
bar_x, bar_max = X2 + 340, 200
scale = bar_max / top
y = 196
prev = base
for k, (cum, title, desc) in enumerate(ladder):
    tot = base + cum
    text(X2, y + 14, title, 14, INK, weight="bold")
    text(X2, y + 31, desc, 12, MUTED)
    rect(bar_x, y + 2, prev * scale, 18, GROW_L if k else CORE, rx=3)
    if k:
        rect(bar_x + prev * scale, y + 2, (tot - prev) * scale, 18, GROW, rx=3)
        rich(bar_x + tot * scale + 8, y + 16,
             [(f"+{n(tot - prev)}", GROW, "bold")], 13)
    else:
        text(bar_x + tot * scale + 8, y + 16, n(tot), 13, CORE, weight="bold")
    prev = tot
    y += 52

text(bar_x + top * scale + 8, y - 52 + 34, f"= {n(top)}", 13, INK, weight="bold")
line(X2, y - 6, X2 + C2W - 20, y - 6, RULE)
text(X2, y + 18, "Hub control", 14, CAP, weight="bold")
text(X2, y + 35, "≤ 50 training rows per ligand, spread one per target first;", 12, MUTED)
text(X2, y + 50, "detergents excluded (they passed PLINDER's artifact flag)", 12, MUTED)
rect(bar_x, y + 6, 7897 * scale, 18, GROW, rx=3)
rect(bar_x + 7897 * scale, y + 6, (top - 7897) * scale, 18, "none", CAP, 3, 1.2)
text(bar_x + top * scale + 8, y + 20, f"{n(7897)} targets", 13, INK, weight="bold")
text(X2, y + 82, "Uncapped, cofactors alone would add +3,813 targets but bring a few dominant", 12, FAINT)
text(X2, y + 97, "ligands (HEM 2,896 rows, GDP 1,009, ADP 976): the cap keeps the targets, not the hubs.",
     12, FAINT)
arrow(X2 + C2W + 2, 470, X2 + C2W + 34, 470)

# ================================================================ column 3
X3, C3W = 1200, 360
rect(X3, 105, C3W, 760, PANEL, RULE, 10)
text(X3 + 20, 140, "4 · The dataset", 18, INK, weight="bold")
text(X3 + 20, 160, "(dataset_v2, built 2026-10-30)", 12.5, MUTED)

stats = [
    ("35,407", "protein–ligand interactions", INK),
    ("34,808", "receptor structures", INK),
    ("20,555", "distinct ligands", INK),
    ("7,895", "training target groups", GROW),
]
y = 214
for big, lab, col in stats:
    text(X3 + 20, y, big, 34, col, weight="bold")
    text(X3 + 20, y + 22, lab, 14, MUTED)
    y += 74

line(X3 + 20, y - 18, X3 + C3W - 20, y - 18)
rich(X3 + 20, y + 6, [("3× ", GROW, "bold"), ("the core's 2,605 targets", INK, "normal")], 14)
rich(X3 + 20, y + 28, [("67 → 20 ", GROW, "bold"), ("training visits per target", INK, "normal")], 14)
rich(X3 + 20, y + 50, [("≤ 50 ", CAP, "bold"), ("rows for any one ligand", INK, "normal")], 14)

y += 86
line(X3 + 20, y - 18, X3 + C3W - 20, y - 18)
text(X3 + 20, y + 4, "Splits", 14, INK, weight="bold")
text(X3 + 20, y + 24, "PLINDER train 34,316 · val 356 · test 735", 13, MUTED)
text(X3 + 20, y + 44, "Model selection: a fixed validation set of", 13, MUTED)
text(X3 + 20, y + 62, "400 rows over 240 targets, UniProt-disjoint", 13, MUTED)
text(X3 + 20, y + 80, "from training (7,644 targets, 31,861 rows)", 13, MUTED)

y += 116
line(X3 + 20, y - 18, X3 + C3W - 20, y - 18)
text(X3 + 20, y + 4, "Per receptor, precomputed", 14, INK, weight="bold")
text(X3 + 20, y + 24, "dMaSIF surface + atom features,", 13, MUTED)
text(X3 + 20, y + 42, "fpocket pockets, PLIP interaction labels", 13, MUTED)
text(X3 + 20, y + 60, "(all training receptors but one)", 13, MUTED)

# footnote
text(40, 884, "Step gains in 3 are from the dataset funnel's ladder before the final per-ligand cap; "
     "the hub-control row is the built result. Sources: docs/RESULTS.md (funnel runs 1, 3, 4; "
     "dataset_v2 built).", 11.5, FAINT)

w("</svg>")
path = sys.argv[1] if len(sys.argv) > 1 else "docs/figures/dataset_v2.svg"
with open(path, "w") as fh:
    fh.write("\n".join(out) + "\n")
print(f"wrote {path}")
