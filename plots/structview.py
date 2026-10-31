"""Load one items.tsv row's receptor, ligand and surface, and draw them (plotly).

For notebooks. Everything here runs in the CPU venv: Biopython for the
receptor, RDKit for the ligand, numpy for the cached surface. extract.py's
loader is not reused because importing it pulls in torch and torch_geometric.

    row = load_row("plots/subsets/demo/demo.tsv", 0)
    rec = load_receptor(row)       # protein atoms of structures/<receptor_key>.cif
    lig = load_ligand(row)         # heavy atoms + bonds from the row's SDF
    surf = load_surface(row)       # data/surfaces/<input_id>.npz
    fig_atoms(rec, lig).show()
    fig_surface(surf, lig).show()

Coordinates are all in the deposited frame (nothing is centred when a surface
is cached), so the three overlay without any alignment.

Depends on: numpy, pandas, biopython, rdkit, plotly.
"""

from pathlib import Path

import numpy as np
import pandas as pd

# CPK-ish, readable on white
ELEMENT_COLORS = {"C": "#909090", "N": "#3050f8", "O": "#ff0d0d", "S": "#e6c228",
                  "P": "#ff8000", "SE": "#ffa100", "H": "#dddddd", "F": "#90e050",
                  "CL": "#1ff01f", "BR": "#a62929", "I": "#940094", "FE": "#e06633"}
LIGAND_CARBON = "#2fb36b"      # ligand carbons, for lig_color="element"
LIGAND_COLOR = "black"         # default: the whole ligand one colour
LIGAND_SIZE = 4                # ligand ball size (receptor atoms are 3, surface 2.5)
POCKET_A = 4.0                 # the repo's pocket label: surface within 4 A of the ligand


def load_row(table, i=0):
    """Row i (int) or the row with system_id i (str) of a subset / items TSV."""
    df = pd.read_csv(table, sep="\t", dtype=str)
    if isinstance(i, str):
        df = df[df.system_id == i]
        if df.empty:
            raise KeyError(i)
        return df.iloc[0]
    return df.iloc[i]


def load_receptor(row, root="."):
    """Protein atoms of the receptor cif -> dict of arrays.

    Amino acids only (plus selenomethionine), first model, like
    extract.load_protein_atoms: waters, ions and ligands are dropped.
    """
    from Bio.PDB import MMCIFParser

    path = Path(root) / "structures" / f"{row.receptor_key}.cif"
    s = MMCIFParser(QUIET=True).get_structure(row.receptor_key, str(path))
    xyz, el, label = [], [], []
    for chain in next(iter(s)):
        for res in chain:
            if res.id[0] != " " and res.get_resname() != "MSE":
                continue
            for atom in res:
                e = (atom.element or "").upper()
                if e == "H" or e == "D":
                    continue
                xyz.append(atom.coord)
                el.append(e)
                label.append(f"{chain.id} {res.get_resname()}{res.id[1]} {atom.get_id()}")
    return {"xyz": np.asarray(xyz, np.float32), "element": np.asarray(el),
            "label": np.asarray(label), "path": str(path)}


def load_ligand(row, root="."):
    """Heavy atoms and bonds of the row's ligand SDF(s) -> dict of arrays."""
    from rdkit import Chem

    xyz, el, bonds = [], [], []
    for p in str(row.sdf_paths).split(","):
        mol = Chem.MolFromMolFile(str(Path(root) / p), sanitize=False, removeHs=False)
        if mol is None:
            raise ValueError(f"RDKit could not read {p}")
        conf = mol.GetConformer()
        keep = {}
        for a in mol.GetAtoms():
            if a.GetSymbol().upper() in ("H", "D"):
                continue
            keep[a.GetIdx()] = len(xyz)
            pos = conf.GetAtomPosition(a.GetIdx())
            xyz.append([pos.x, pos.y, pos.z])
            el.append(a.GetSymbol().upper())
        for b in mol.GetBonds():
            i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            if i in keep and j in keep:
                bonds.append((keep[i], keep[j]))
    return {"xyz": np.asarray(xyz, np.float32), "element": np.asarray(el),
            "bonds": np.asarray(bonds, np.int64).reshape(-1, 2), "smiles": row.smiles}


def load_surface(row, root="."):
    """The cached dMaSIF surface: xyz (N, 3), normals (N, 3), and its atoms."""
    path = Path(root) / "data" / "surfaces" / f"{row.input_id}.npz"
    with np.load(path, allow_pickle=False) as d:
        return {k: d[k] for k in d.files} | {"path": str(path)}


def min_dist(points, ref):
    """Each point's distance to its nearest ref point (chunked, numpy only)."""
    out = np.empty(len(points), np.float32)
    for s in range(0, len(points), 20000):
        p = points[s:s + 20000]
        out[s:s + 20000] = np.sqrt(((p[:, None, :] - ref[None]) ** 2).sum(-1)).min(1)
    return out


def _ligand_traces(lig, color=LIGAND_COLOR, size=LIGAND_SIZE):
    """Ball-and-stick: one colour, or color="element" for green carbons + CPK."""
    import plotly.graph_objects as go

    xyz = lig["xyz"]
    ex, ey, ez = [], [], []
    for i, j in lig["bonds"]:
        ex += [xyz[i, 0], xyz[j, 0], None]
        ey += [xyz[i, 1], xyz[j, 1], None]
        ez += [xyz[i, 2], xyz[j, 2], None]
    if color == "element":
        colors, bond = [LIGAND_CARBON if e == "C" else ELEMENT_COLORS.get(e, "#ff69b4")
                        for e in lig["element"]], "#1d7a47"
    else:
        colors, bond = color, color
    return [go.Scatter3d(x=ex, y=ey, z=ez, mode="lines", name="ligand bonds",
                         line=dict(color=bond, width=5), hoverinfo="skip",
                         showlegend=False),
            go.Scatter3d(x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], mode="markers",
                         name="ligand", text=lig["element"],
                         marker=dict(size=size, color=colors,
                                     line=dict(color="black", width=0.5)))]


def _layout(fig, title):
    fig.update_layout(title=title, height=700, margin=dict(l=0, r=0, t=40, b=0),
                      scene=dict(aspectmode="data", xaxis_visible=False,
                                 yaxis_visible=False, zaxis_visible=False),
                      legend=dict(itemsizing="constant"))
    return fig


def crop(xyz, lig, radius):
    """Mask of points within radius A of any ligand atom (None = keep all)."""
    if radius is None:
        return np.ones(len(xyz), bool)
    return min_dist(xyz, lig["xyz"]) <= radius


def fig_atoms(rec, lig, radius=None, title=None, lig_color=LIGAND_COLOR,
              lig_size=LIGAND_SIZE):
    """Receptor heavy atoms by element, ligand as ball-and-stick.

    radius: show only receptor atoms within this many A of the ligand
    (e.g. 12 for the binding site); None shows the whole protein.
    lig_color: any plotly colour, or "element"; lig_size: ball size.
    """
    import plotly.graph_objects as go

    m = crop(rec["xyz"], lig, radius)
    xyz = rec["xyz"][m]
    colors = [ELEMENT_COLORS.get(e, "#ff69b4") for e in rec["element"][m]]
    prot = go.Scatter3d(x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], mode="markers",
                        name=f"receptor ({m.sum()} atoms)", text=rec["label"][m],
                        marker=dict(size=3, color=colors, opacity=0.6))
    fig = go.Figure([prot] + _ligand_traces(lig, lig_color, lig_size))
    return _layout(fig, title or f"receptor + ligand  {lig['smiles']}")


def fig_surface(surf, lig, radius=None, color="distance", title=None,
                lig_color=LIGAND_COLOR, lig_size=LIGAND_SIZE):
    """dMaSIF's sampled surface points with the ligand.

    color: "distance" (to the nearest ligand atom, capped at 12 A), or
           "pocket" (within POCKET_A of the ligand = the repo's pocket label).
    radius, lig_color, lig_size: as in fig_atoms.
    """
    import plotly.graph_objects as go

    m = crop(surf["xyz"], lig, radius)
    xyz = surf["xyz"][m]
    d = min_dist(xyz, lig["xyz"])
    if color == "pocket":
        pocket = d <= POCKET_A
        marker = dict(size=2.5, color=np.where(pocket, "#d1495b", "#9db4d0"))
        name = f"surface ({m.sum()} pts, {pocket.sum()} pocket)"
    else:
        # near = the pocket red of color="pocket", far = pale blue, so the
        # ligand stays visible against the points around it
        marker = dict(size=2.5, color=np.minimum(d, 12.0),
                      colorscale=[[0, "#d1495b"], [POCKET_A / 12, "#f2a65a"],
                                  [1, "#c9d6e8"]],
                      cmin=0, cmax=12, colorbar=dict(title="Å to ligand", x=0.92))
        name = f"surface ({m.sum()} points)"
    pts = go.Scatter3d(x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], mode="markers",
                       name=name, marker=marker, text=[f"{v:.1f} Å" for v in d],
                       hoverinfo="text")
    fig = go.Figure([pts] + _ligand_traces(lig, lig_color, lig_size))
    return _layout(fig, title or f"sampled surface + ligand  {lig['smiles']}")
