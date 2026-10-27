"""Apo vs holo: does the encoder find the same pocket, and give it the same anchor,
when the receptor is not in its ligand-bound shape?

For each pair from apo_pairs.py, one trained head (test_encoder.py overfit
--save_ckpt) sees up to four versions of the same protein:

    holo      PLINDER's receptor (ligand removed), as in every other test
    holo_rs   the same atoms rotated at random and resampled - the noise floor:
              dMaSIF samples its surface at random, so an apo-vs-holo
              difference means something only if it exceeds this one
    apo       the linked apo PDB chain (linked_structures/apo.zip)
    pred      the linked AlphaFold model (linked_structures/pred.zip), with
              residues below --plddt_min removed (disordered, not real surface)

PLINDER's linked files are NOT in the holo frame (one apo file serves many
holo systems), so each is placed here:
    1. residues matched to the holo chain by sequence alignment (numbering
       differs between entries, and AlphaFold uses UniProt numbering)
    2. superposed by Kabsch on matched C-alpha atoms of the holo pocket
       (residues with an atom within --fit_radius of the ligand); falls back to
       all matched residues when the pocket has too few
    3. trimmed to the matched residues (--trim_to_holo), so extra domains or
       tags of the linked construct cannot add pockets the holo never had
The encoder itself is invariant to rotation, so placement only matters for the
labels and for comparing anchor positions; vector cosines would not change.

Labels on every version: surface points within --cutoff A of the HOLO ligand.
A closed apo pocket therefore shows up as few or no pocket points.

Per version: pocket AUROC / AP, pocket points, and the site vector (embeddings
pooled over the labelled points, probability-weighted). Per version and
selector (SELECTORS): top-ranked region's centre distance to the ligand
centroid (dcc1), best over the top --k_pos (dccA), top region IoU with the
label (iou1), and - against holo - how far the top region moved (shift) and
the cosine between the two top-region vectors (cos1). Vectors are the pooled
point embeddings (E dims), before the untrained projection.

Writes <out>/apo_eval_versions.tsv and <out>/apo_eval_regions.tsv and prints
summaries per set (A_near ... D_closed, eval). Reference: the median cosine
between top-region vectors of DIFFERENT proteins' holo versions.

Example (container; see src/slurm/apo_test.sbatch):
    python src/apo_eval.py --ckpt out/encoder_test/head_pretrained_300.pt \\
        --pairs out/apo_test/pairs.tsv --links_root $PLINDER_DIR/linked_structures \\
        --out out/apo_test/
"""

import argparse
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# the contenders from region_sweep.py; region settings as anchors.pocket_regions takes them
SELECTORS = {
    "watershed_old": dict(method="watershed", top_fraction=0.05, smooth_radius=2.0,
                          link_radius=2.0, persistence=0.0),
    "watershed_p02": dict(method="watershed", top_fraction=0.05, smooth_radius=2.0,
                          link_radius=3.0, persistence=0.02),
    "components": dict(method="components", top_fraction=0.05, smooth_radius=2.0,
                       link_radius=3.0),
    "ball_R8": dict(method="ball", top_fraction=0.05, smooth_radius=2.0, ball_radius=8.0,
                    ball_shift=2),
    "nms_R8": dict(method="nms", top_fraction=0.05, smooth_radius=2.0, ball_radius=8.0,
                   nms_score="sum", nms_iou=0.3, nms_alpha=0.3),
    "compball_R8": dict(method="compball", top_fraction=0.05, smooth_radius=2.0,
                        link_radius=3.0, ball_radius=8.0, ball_shift=2),
}
VERSIONS = ("holo", "holo_rs", "apo", "pred")
SETS = ("A_near", "B_mild", "C_large", "D_closed", "eval")
AA = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
      "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
      "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V", "MSE": "M"}


# ----------------------------------------------------------------------------
# structures
# ----------------------------------------------------------------------------
def residue_table(p):
    """Residues of a load_protein_atoms() dict, in file order -> list of dicts:
    chain, aa (one letter), ca (xyz or None), atom slice, mean B-factor."""
    key = np.char.add(np.char.add(p["atom_chain"].astype(str), "|"),
                      np.char.add(p["atom_resnum"].astype(str), p["atom_icode"].astype(str)))
    start = np.r_[0, np.nonzero(key[1:] != key[:-1])[0] + 1]
    end = np.r_[start[1:], len(key)]
    out = []
    for s, e in zip(start, end):
        ca = np.nonzero(p["atom_name"][s:e] == "CA")[0]
        out.append({"chain": str(p["atom_chain"][s]),
                    "aa": AA.get(str(p["atom_resname"][s]).upper(), "X"),
                    "ca": p["atom_xyz"][s + ca[0]].astype(np.float64) if len(ca) else None,
                    "s": int(s), "e": int(e),
                    "b": float(p["atom_bfactor"][s:e].mean()) if "atom_bfactor" in p else 0.0})
    return out


def align_residues(seq_a, seq_b):
    """Global alignment, free end gaps -> (list of (i, j) matched positions,
    identity over matched positions)."""
    from Bio.Align import PairwiseAligner

    al = PairwiseAligner()
    al.mode = "global"
    al.match_score, al.mismatch_score = 2.0, -1.0
    al.open_gap_score, al.extend_gap_score = -5.0, -0.5
    al.end_gap_score = 0.0  # free end gaps (one name across Biopython versions)
    aln = al.align(seq_a, seq_b)[0]
    pairs = []
    for (a0, a1), (b0, b1) in zip(*aln.aligned):
        pairs += list(zip(range(a0, a1), range(b0, b1)))
    ident = float(np.mean([seq_a[i] == seq_b[j] for i, j in pairs])) if pairs else 0.0
    return pairs, ident


def kabsch(P, Q):
    """R, t minimising |P R^T + t - Q| over matched rows."""
    pc, qc = P.mean(0), Q.mean(0)
    U, _, Vt = np.linalg.svd((P - pc).T @ (Q - qc))
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R, qc - pc @ R.T


def random_rotation(rng):
    q = rng.normal(size=4)
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def place_linked(path, holo, holo_res, lig, a, is_pred):
    """Linked chain -> atoms in the holo frame, trimmed, plus placement stats."""
    import chemotype
    from extract import load_protein_atoms

    p = chemotype.annotate(load_protein_atoms(path, None, keep_hydrogens=False))
    res = residue_table(p)
    holo_seq = "".join(r["aa"] for r in holo_res)
    best = None
    for c in dict.fromkeys(r["chain"] for r in res):
        rc = [r for r in res if r["chain"] == c]
        pairs, ident = align_residues(holo_seq, "".join(r["aa"] for r in rc))
        if best is None or len(pairs) * ident > len(best[1]) * best[2]:
            best = (rc, pairs, ident)
    rc, pairs, ident = best
    if ident < a.min_ident:
        raise ValueError(f"sequence identity {ident:.2f} < {a.min_ident}")

    # holo pocket residues: any heavy atom within fit_radius of the ligand
    d = np.sqrt(((holo["atom_xyz"][:, None, :] - lig[None]) ** 2).sum(-1)).min(1)
    near = np.array([d[r["s"]:r["e"]].min() < a.fit_radius for r in holo_res])
    usable = [(i, j) for i, j in pairs if holo_res[i]["ca"] is not None and rc[j]["ca"] is not None]
    fit = [(i, j) for i, j in usable if near[i]]
    kind = "pocket"
    if len(fit) < a.min_fit:
        fit, kind = usable, "global"
    if len(fit) < 3:
        raise ValueError(f"only {len(fit)} matched C-alpha atoms to superpose")
    P = np.stack([rc[j]["ca"] for _, j in fit])
    Q = np.stack([holo_res[i]["ca"] for i, _ in fit])
    R, t = kabsch(P, Q)
    rmsd = float(np.sqrt((((P @ R.T + t) - Q) ** 2).sum(1).mean()))

    keep = set(j for _, j in pairs) if a.trim_to_holo else set(range(len(rc)))
    n_low = 0
    if is_pred:
        low = {j for j, r in enumerate(rc) if r["b"] < a.plddt_min}
        n_low = len(low & keep)
        keep -= low
    mask = np.zeros(len(p["atom_xyz"]), dtype=bool)
    for j in keep:
        mask[rc[j]["s"]:rc[j]["e"]] = True
    xyz = (p["atom_xyz"][mask].astype(np.float64) @ R.T + t).astype(np.float32)
    stats = {"ident": ident, "n_res": len(rc), "n_kept": len(keep), "n_low_plddt": n_low,
             "fit": kind, "n_fit": len(fit), "fit_rmsd": rmsd}
    # atom_groups must be subset by the SAME mask, or the typing silently
    # misaligns with the coordinates it describes.
    return {"atom_xyz": xyz, "atom_type": p["atom_type"][mask],
            "atom_groups": p["atom_groups"][mask]}, stats


# ----------------------------------------------------------------------------
# evaluation of one version
# ----------------------------------------------------------------------------
def evaluate_version(enc, it, to_holo, a, device):
    """Surface + head + regions for one version -> (version stats, per-selector stats).
    to_holo maps this version's coordinates back into the holo frame."""
    import torch

    from anchors import pocket_regions, pool_regions
    from test_encoder import attach_surfaces, auroc, average_precision, to_batch

    attach_surfaces(enc, [it], device, a.cutoff)
    with torch.no_grad():
        out = enc(to_batch([it], device))
    logit = out["pocket_logit"].float().cpu()
    prob, emb = torch.sigmoid(logit), out["point_emb"].float().cpu()
    xyz, pocket = it["xyz"].float(), it["pocket"]
    lig_c = torch.from_numpy(it["lig"]).float().mean(0)
    vst = {"n_points": len(xyz), "n_pocket": int(pocket.sum()), "min_dist": it["min_dist"],
           "auroc": auroc(logit, pocket), "ap": average_precision(logit, pocket),
           "site_vec": (pool_regions(emb, prob, [pocket.nonzero().squeeze(1)])[0]
                        if pocket.any() else None)}
    sel, cache = {}, {}
    for name, s in SELECTORS.items():
        regs = pocket_regions(xyz, prob, min_points=20, cache=cache, **s)[:a.k_pos]
        if not regs:
            sel[name] = {"n": 0}
            continue
        w = [prob[r] for r in regs]
        centre = torch.stack([(wi[:, None] * xyz[r]).sum(0) / wi.sum() for wi, r in zip(w, regs)])
        dcc = (centre - lig_c).norm(dim=1)
        n_pk = float(pocket.sum())
        inter = torch.tensor([float(pocket[r].sum()) for r in regs])
        size = torch.tensor([float(len(r)) for r in regs])
        iou = inter / (size + n_pk - inter) if n_pk else torch.zeros(len(regs))
        c_holo = centre[0].double().numpy() @ to_holo.T
        sel[name] = {"n": len(regs), "dcc1": float(dcc[0]), "dccA": float(dcc.min()),
                     "iou1": float(iou[0]), "iou": float(iou.max()),
                     "centre1": c_holo, "vec1": pool_regions(emb, prob, regs[:1])[0]}
    return vst, sel


def cosine(u, v):
    import torch

    if u is None or v is None:
        return float("nan")
    return float(torch.nn.functional.cosine_similarity(u[None], v[None])[0])


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, help="from test_encoder.py overfit --save_ckpt")
    ap.add_argument("--pairs", required=True, help="from apo_pairs.py")
    ap.add_argument("--links_root", required=True, help="PLINDER linked_structures/ directory")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--limit", type=int, default=0, help="first N pairs only (0 = all)")
    ap.add_argument("--cutoff", type=float, default=4.0, help="A; pocket label distance")
    ap.add_argument("--k_pos", type=int, default=8)
    ap.add_argument("--dcc", type=float, default=4.0, help="A; DCC success cutoff")
    ap.add_argument("--fit_radius", type=float, default=8.0,
                    help="A; holo residues this close to the ligand anchor the superposition")
    ap.add_argument("--min_fit", type=int, default=8,
                    help="fewer matched pocket C-alphas than this -> superpose on all matched")
    ap.add_argument("--min_ident", type=float, default=0.9,
                    help="drop linked chains below this sequence identity to holo")
    ap.add_argument("--plddt_min", type=float, default=70.0)
    ap.add_argument("--no_trim", dest="trim_to_holo", action="store_false",
                    help="keep linked residues that have no holo counterpart")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    import torch

    import chemotype
    from extract import load_protein_atoms
    from ids import parse_chains
    from test_encoder import load_checkpoint, read_sdf_heavy

    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    enc, ckpt = load_checkpoint(a.ckpt, a.device)
    seen = set(ckpt.get("train_ids", []))
    lines = Path(a.pairs).read_text().splitlines()
    header = lines[0].split("\t")
    pairs = [dict(zip(header, ln.split("\t"))) for ln in lines[1:]]
    if a.limit:
        pairs = pairs[:a.limit]
    zips = {k: zipfile.ZipFile(Path(a.links_root) / f"{k}.zip") for k in ("apo", "pred")}
    cache_dir = Path(a.out) / "linked"
    print(f"{a.ckpt}: head trained on {len(seen)} items; {len(pairs)} pairs; "
          f"selectors {', '.join(SELECTORS)}")

    def linked_file(kind, link_id):
        dest = cache_dir / kind / f"{link_id}.cif"
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_suffix(".part")
            tmp.write_bytes(zips[kind].read(f"{link_id}.cif"))
            tmp.rename(dest)
        return dest

    vrows, rrows, fails = [], [], []
    t0 = time.time()
    for n, pr in enumerate(pairs):
        sid = pr["system_id"]
        try:
            holo = chemotype.annotate(load_protein_atoms(
                pr["input_path"], parse_chains(pr["input_chains"]), keep_hydrogens=False))
            lig = np.concatenate([read_sdf_heavy(p) for p in pr["sdf_paths"].split(",")])
            holo_res = residue_table(holo)
        except Exception as e:  # noqa: BLE001 - report and move on
            fails.append((sid, "holo", repr(e)))
            continue
        R0 = random_rotation(rng)
        versions = {"holo": ({"atom_xyz": holo["atom_xyz"], "atom_type": holo["atom_type"],
                              "atom_groups": holo["atom_groups"]},
                             lig, np.eye(3), {}),
                    "holo_rs": ({"atom_xyz": (holo["atom_xyz"] @ R0.T).astype(np.float32),
                                 "atom_type": holo["atom_type"],
                                 "atom_groups": holo["atom_groups"]},
                                (lig @ R0.T).astype(np.float32), R0.T, {})}
        for kind in ("apo", "pred"):
            link_id = pr[f"{kind}_id"]
            if not link_id:
                continue
            try:
                atoms, st = place_linked(linked_file(kind, link_id), holo, holo_res, lig, a,
                                         kind == "pred")
                st["link"] = link_id
                versions[kind] = (atoms, lig, np.eye(3), st)
            except Exception as e:  # noqa: BLE001
                fails.append((sid, kind, repr(e)))

        res = {}
        for v, (atoms, lg, back, st) in versions.items():
            it = {"id": f"{sid}:{v}", "atom_xyz": atoms["atom_xyz"],
                  "atom_type": atoms["atom_type"],
                  "atom_groups": atoms["atom_groups"], "lig": lg}
            try:
                res[v] = evaluate_version(enc, it, back, a, a.device) + (st,)
            except Exception as e:  # noqa: BLE001
                fails.append((sid, v, repr(e)))
        if "holo" not in res:
            continue
        h_vst, h_sel, _ = res["holo"]
        base = {"system_id": sid, "sets": pr["sets"], "split": pr["split"],
                "seen": int(sid in seen), "apo_pocket_lddt": pr["apo_pocket_lddt"],
                "apo_clash": pr["apo_clash"], "pred_pocket_lddt": pr["pred_pocket_lddt"]}
        for v, (vst, sel, st) in res.items():
            vrows.append({**base, "version": v, "link": st.get("link", ""),
                          **{k: vst[k] for k in ("n_points", "n_pocket", "min_dist", "auroc", "ap")},
                          "site_cos": cosine(vst["site_vec"], h_vst["site_vec"]),
                          **{k: st.get(k, "") for k in ("ident", "n_res", "n_kept", "n_low_plddt",
                                                        "fit", "n_fit", "fit_rmsd")}})
            for name, s in sel.items():
                h = h_sel.get(name, {})
                both = s.get("n", 0) and h.get("n", 0)
                rrows.append({**base, "version": v, "selector": name, "n": s.get("n", 0),
                              **{k: s.get(k, float("nan")) for k in ("dcc1", "dccA", "iou1", "iou")},
                              "shift": (float(np.linalg.norm(s["centre1"] - h["centre1"]))
                                        if both else float("nan")),
                              "cos1": cosine(s["vec1"], h["vec1"]) if both else float("nan"),
                              "_vec1": s.get("vec1")})
        print(f"  {n + 1}/{len(pairs)} {sid} [{pr['sets']}] versions: {','.join(res)} "
              f"{time.time() - t0:.0f}s", flush=True)

    report(vrows, rrows, fails, a)


# ----------------------------------------------------------------------------
def report(vrows, rrows, fails, a):
    import torch

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    def write(path, rows):
        if not rows:
            return
        cols = [c for c in rows[0] if not c.startswith("_")]
        path.write_text("\t".join(cols) + "\n" + "".join(
            "\t".join(f"{r[c]:.4f}" if isinstance(r[c], float) else str(r[c]) for c in cols) + "\n"
            for r in rows))
        print(path)

    def med(x):
        x = np.array([v for v in x if v == v], dtype=float)  # drop nan
        return np.median(x) if len(x) else float("nan")

    def mean(x):
        x = np.array([v for v in x if v == v], dtype=float)
        return x.mean() if len(x) else float("nan")

    print(f"\n{len(fails)} failures" + (":" if fails else ""))
    for sid, v, e in fails[:30]:
        print(f"  {sid} {v}: {e[:160]}")

    # scale for cosines: top-region vectors of different proteins (holo)
    print("\nreference: median cosine between top-region vectors of DIFFERENT proteins (holo)")
    for name in SELECTORS:
        vecs = [r["_vec1"] for r in rrows if r["version"] == "holo" and r["selector"] == name
                and r["_vec1"] is not None]
        if len(vecs) > 1:
            V = torch.nn.functional.normalize(torch.stack(vecs), dim=1)
            c = (V @ V.T)[torch.triu(torch.ones(len(V), len(V), dtype=torch.bool), 1)]
            print(f"  {name:14s} {float(c.median()):.3f}")

    for st in SETS:
        vr = [r for r in vrows if st in r["sets"].split(",") and not r["seen"]]
        if not vr:
            continue
        print(f"\n===== {st}: {len({r['system_id'] for r in vr})} systems =====")
        print(f"{'version':8s} {'n':>4s} {'AUROC':>6s} {'AP':>6s} {'pocket':>7s} {'site_cos':>9s} "
              f"{'fit_rmsd':>9s} {'kept':>6s}")
        for v in VERSIONS:
            rows = [r for r in vr if r["version"] == v]
            if not rows:
                continue
            kept = [r["n_kept"] / r["n_res"] for r in rows if r["n_res"] not in ("", 0)]
            rmsd = [r["fit_rmsd"] for r in rows if r["fit_rmsd"] != ""]
            print(f"{v:8s} {len(rows):4d} {mean([r['auroc'] for r in rows]):6.3f} "
                  f"{mean([r['ap'] for r in rows]):6.3f} {med([r['n_pocket'] for r in rows]):7.0f} "
                  f"{med([r['site_cos'] for r in rows]):9.3f} {med(rmsd):9.2f} {med(kept):6.2f}")
        print(f"\n{'selector':14s} {'version':8s} {'dcc1':>6s} {'dccA':>6s} {'iou1':>6s} "
              f"{'shift':>6s} {'cos1':>6s}   (dcc = share within {a.dcc:g} A; shift, cos1 = "
              f"median vs holo)")
        for name in SELECTORS:
            for v in VERSIONS:
                rows = [r for r in rrows if st in r["sets"].split(",") and not r["seen"]
                        and r["selector"] == name and r["version"] == v]
                if not rows:
                    continue
                print(f"{name:14s} {v:8s} "
                      f"{mean([float(r['dcc1'] < a.dcc) for r in rows if r['n']]):6.2f} "
                      f"{mean([float(r['dccA'] < a.dcc) for r in rows if r['n']]):6.2f} "
                      f"{mean([r['iou1'] for r in rows]):6.2f} "
                      f"{med([r['shift'] for r in rows]) if v != 'holo' else float('nan'):6.1f} "
                      f"{med([r['cos1'] for r in rows]) if v != 'holo' else float('nan'):6.3f}")

    write(out / "apo_eval_versions.tsv", vrows)
    write(out / "apo_eval_regions.tsv", rrows)


if __name__ == "__main__":
    main()
