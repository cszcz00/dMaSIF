"""Extract dMaSIF forward pass data for batch of protein structures.

Input : Directory containing .pdb/.cif files, or .txt listing paths to 
        (optionally "path CHAINS", e.g. "structs/1abc.pdb AB").
        Biological-assembly files (.pdb1) hold symmetry copies as separate
        MODELs with duplicate chain IDs; pass --merge_models to use them all,
        relabelled "<model>.<chain>" (1.A, 2.A, ...). Without it only the first
        model is read, which for an oligomer means an incomplete surface.
Output: one .npz per protein saved to --out directory, from a results dictionary containing:

    xyz            (N, 3)  Coordinates of dMaSIF-sampled surface points (in same frame as the input structure).
    normals        (N, 3)  outward normals of surface points.
    input_feats    (N, 16) 10 curvature features + 6 learned chemical features (KNN-based) per surface point.
    emb1, emb2     (N, 16) Two dMaSIF-produced embeddings per surface point. 
    nearest_atom   (N,)    Nearest protein atom per sampled surface point. 
    atom_xyz       (M, 3)  Coordinates of structure (filtered) atoms
    atom_type      (M,)    {C,H,O,N,S,SE} -> {0..5}
    atom_chain, atom_resnum, atom_icode, atom_resname, atom_name  (M,) atom metadata

Example (inside the container):
    python extract.py --inputs pdbs/ --out feats/ --repo ../dMaSIF
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import dmasif_compat

ELE2NUM = {"C": 0, "H": 1, "O": 2, "N": 3, "S": 4, "SE": 5}

# Configuration of models/dMaSIF_search_3layer_12A_16dim, from
# benchmark_scripts/dMaSIF_search.sh + Arguments.py defaults + checkpoint shapes.
SEARCH_CONFIG = {
    "embedding_layer": "dMaSIF",
    "search": True,
    "site": False,
    "radius": 12.0,
    "n_layers": 3,
    "emb_dims": 16,
    "in_channels": 16,
    "atom_dims": 6,
    "resolution": 1.0,
    "distance": 1.05,
    "variance": 0.1,
    "sup_sampling": 20,
    "use_mesh": False,
    "no_chem": False,
    "no_geom": False,
    "dropout": 0.0,
}


# ----------------------------------------------------------------------------
# Structure parsing
# ----------------------------------------------------------------------------
def _iter_chains(structure, merge_models):
    """Yield (label, chain) over the structure.

    A deposited file is the asymmetric unit: one model, unique chain IDs, and
    `label` is just the chain ID. A biological-assembly file repeats the same
    chain ID across MODELs, so with merge_models the label carries the model
    index ("1.A", "2.A") to keep chains distinguishable in atom_chain. The
    labelling matches PLINDER's own chain convention.
    """
    models = list(structure)
    if not merge_models:
        for chain in models[0]:
            yield chain.id, chain
        return
    multi = len(models) > 1
    for mi, model in enumerate(models, start=1):
        for chain in model:
            yield (f"{mi}.{chain.id}" if multi else chain.id), chain


def load_protein_atoms(path, chains=None, keep_hydrogens=True, merge_models=False):
    """Parse protein atoms from a PDB/mmCIF file.

    Keeps only amino-acid residues (plus selenomethionine). Waters, ligands,
    ions and other HETATM records are dropped, so a protein-ligand complex file
    yields just the receptor. Reads the first model only unless merge_models is
    set; see _iter_chains for why that matters for assembly files.
    """
    from Bio.PDB import MMCIFParser, PDBParser

    path = Path(path)
    if path.suffix.lower() in (".cif", ".mmcif"):
        parser = MMCIFParser(QUIET=True)
    else:
        parser = PDBParser(QUIET=True)
    structure = parser.get_structure(path.stem, str(path))

    xyz, types = [], []
    chain_ids, resnums, icodes, resnames, names = [], [], [], [], []
    n_skipped_element = 0
    for label, chain in _iter_chains(structure, merge_models):
        if chains and label not in chains:
            continue
        for res in chain:
            hetflag = res.id[0]
            if hetflag != " " and res.get_resname() != "MSE":
                continue
            for atom in res:
                el = (atom.element or "").upper()
                if el == "D":
                    el = "H"
                if el not in ELE2NUM:
                    n_skipped_element += 1
                    continue
                if el == "H" and not keep_hydrogens:
                    continue
                xyz.append(atom.get_coord())
                types.append(ELE2NUM[el])
                chain_ids.append(label)
                resnums.append(res.id[1])
                icodes.append(res.id[2].strip())
                resnames.append(res.get_resname())
                names.append(atom.get_id())

    if not xyz:
        raise ValueError(f"No protein atoms found in {path} (chains={chains})")

    return {
        "atom_xyz": np.asarray(xyz, dtype=np.float32),
        "atom_type": np.asarray(types, dtype=np.int64),
        "atom_chain": np.asarray(chain_ids, dtype="U8"),
        "atom_resnum": np.asarray(resnums, dtype=np.int64),
        "atom_icode": np.asarray(icodes),
        "atom_resname": np.asarray(resnames),
        "atom_name": np.asarray(names),
        "n_skipped_element": n_skipped_element,
    }


def read_inputs(inputs):
    """Return a list of (path, chains-or-None)."""
    p = Path(inputs)
    if p.is_dir():
        files = sorted(
            f for f in p.iterdir() if f.suffix.lower() in (".pdb", ".ent", ".cif", ".mmcif")
        )
        return [(f, None) for f in files]
    items = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        items.append((Path(parts[0]), parse_chains(parts[1]) if len(parts) > 1 else None))
    return items


def parse_chains(spec):
    """"AB" -> {"A","B"}; "1.A" -> {"1.A"}; "1.A,2.A" -> {"1.A","2.A"}.

    Dotted labels (assembly files, PLINDER receptors) are never split per
    character; several of them must be comma separated.
    """
    if spec is None:
        return None
    if "," in spec or "." in spec:
        return {c.strip() for c in spec.split(",") if c.strip()}
    return set(spec)


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------
def load_model(repo_dir, ckpt_name, device):
    dmasif_compat.add_repo_to_path(repo_dir)
    from Arguments import parser
    from model import dMaSIF

    args = parser.parse_args(["--experiment_name", ckpt_name])
    for k, v in SEARCH_CONFIG.items():
        setattr(args, k, v)
    args.device = device

    net = dMaSIF(args)
    ckpt = torch.load(Path(repo_dir) / "models" / ckpt_name, map_location=device, weights_only=False)
    net.load_state_dict(ckpt["model_state_dict"], strict=True)
    net = net.to(device).eval()
    return net, args


def nearest_atoms(points, atoms, chunk=8192):
    out = torch.empty(points.shape[0], dtype=torch.long, device=points.device)
    for i in range(0, points.shape[0], chunk):
        out[i : i + chunk] = torch.cdist(points[i : i + chunk], atoms).argmin(dim=1)
    return out


@torch.no_grad()
def embed_batch(net, proteins, device):
    """
    Run dMaSIF on a list of parsed proteins in a single batch.
    Return list of results dictionaries, one per protein in batch. 
    results per protein identify 3D coordinates sampled surface points,
    computed features/embeddings for each of these points, and protein 
    atom nearest to each sampled surface point. 
    """
    atom_xyz = torch.cat([torch.from_numpy(p["atom_xyz"]) for p in proteins]).to(device)
    atom_type = torch.cat([torch.from_numpy(p["atom_type"]) for p in proteins])
    atomtypes = torch.nn.functional.one_hot(atom_type, num_classes=6).float().to(device)
    batch_atoms = torch.cat(
        [torch.full((len(p["atom_type"]),), i, dtype=torch.long) for i, p in enumerate(proteins)]
    ).to(device)

    P = {
        "atoms": atom_xyz,
        "atom_xyz": atom_xyz,
        "atomtypes": atomtypes,
        "batch_atoms": batch_atoms,
        "mesh_labels": None,
        "triangles": None,
    }
    net.preprocess_surface(P)  # sample surface points on batch proteins. Adds P["xyz"], P["normals"], P["batch"]
    net(P)  # dMaSIF forward pass on batch surface samples. Adds P["input_features"], P["embedding_1"], P["embedding_2"]

    results = []
    for i, p in enumerate(proteins):
        m = P["batch"] == i # mask for ith protein
        a = batch_atoms == i # Atom indices of ith protein in proteins
        xyz = P["xyz"][m] # coordinates of dMaSIF sampled surface points of ith protein
        results.append(
            {
                "xyz": xyz,
                "normals": P["normals"][m],
                "input_feats": P["input_features"][m], # Geometric and Chemical features (KNN-based) per sample point
                "emb1": P["embedding_1"][m], 
                "emb2": P["embedding_2"][m],
                "nearest_atom": nearest_atoms(xyz, atom_xyz[a]), # Nearest protein atom to each surface point
            }
        )
    return results


def batches_by_atoms(items, max_atoms, max_proteins):
    batch, n = [], 0
    for it in items:
        na = len(it[1]["atom_type"])
        if batch and (n + na > max_atoms or len(batch) >= max_proteins):
            yield batch
            batch, n = [], 0
        batch.append(it)
        n += na
    if batch:
        yield batch


def to_np(t):
    return t.detach().float().cpu().numpy()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", required=True, help="directory of structures, or a list file")
    ap.add_argument("--out", required=True)
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--ckpt", default="dMaSIF_search_3layer_12A_16dim")
    ap.add_argument("--no_hydrogens", action="store_true", help="drop H atoms even if present")
    ap.add_argument(
        "--merge_models",
        action="store_true",
        help="merge all MODELs (use for biological-assembly .pdb1 files)",
    )
    ap.add_argument("--max_atoms", type=int, default=150_000, help="atom budget per GPU batch")
    ap.add_argument("--max_proteins", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    net, args = load_model(a.repo, a.ckpt, a.device)
    print(f"Loaded {a.ckpt} (strict=True) on {torch.cuda.get_device_name(a.device)}")

    todo = []
    for path, chains in read_inputs(a.inputs):
        tag = path.stem + ("_" + "".join(sorted(chains)).replace(".", "") if chains else "")
        if (out / f"{tag}.npz").exists() and not a.overwrite:
            continue
        try:
            prot = load_protein_atoms(
                path, chains, keep_hydrogens=not a.no_hydrogens,
                merge_models=a.merge_models,
            )
        except Exception as e:  # keep going on bad files
            print(f"[skip] {path}: {e}")
            continue
        todo.append((tag, prot))
    print(f"{len(todo)} proteins to embed")

    meta = {"ckpt": a.ckpt, "config": SEARCH_CONFIG, "seed": a.seed}
    t0, n_done = time.time(), 0
    for batch in batches_by_atoms(todo, a.max_atoms, a.max_proteins):
        tags, prots = zip(*batch)
        try:
            results = embed_batch(net, list(prots), a.device)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"[oom] batch of {len(batch)}; retrying one at a time")
            results = [embed_batch(net, [p], a.device)[0] for p in prots]
        for tag, prot, r in zip(tags, prots, results):
            has_h = bool((prot["atom_type"] == ELE2NUM["H"]).any())
            np.savez_compressed(
                out / f"{tag}.npz",
                **{k: to_np(v) for k, v in r.items() if k != "nearest_atom"},
                nearest_atom=r["nearest_atom"].cpu().numpy(),
                **{k: v for k, v in prot.items() if k.startswith("atom_")},
                has_hydrogens=has_h,
                meta=json.dumps(meta),
            )
            n_done += 1
            print(
                f"{tag}: {len(prot['atom_type'])} atoms -> {r['xyz'].shape[0]} surface points"
                + ("" if has_h else "  (no H in input)")
                + (f"  ({prot['n_skipped_element']} atoms with other elements dropped)" if prot["n_skipped_element"] else "")
            )
    dt = time.time() - t0
    print(f"Done: {n_done} proteins in {dt:.1f}s ({dt / max(n_done, 1):.2f}s/protein)")


if __name__ == "__main__":
    main()