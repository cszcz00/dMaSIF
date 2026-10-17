"""Cheap test: which ligand encoder pairs with our surface anchors?

Full contrastive training needs the data pipeline, balanced sampling and hours
of GPU. This answers the one question that decides the ligand side first, on
FROZEN protein anchors, in minutes:

    1. dump    (GPU, container) run a trained pocket-head checkpoint over
               receptors it never trained on. Save each protein's pooled
               region embeddings - the anchors - with each region's overlap
               with the labelled pocket. One pass, then the GPU is done.
    2. probe   (CPU is enough) train ONLY the ligand encoder and the
               projection onto those frozen anchors, contrastively. Report
               retrieval metrics per arm (ligand_encoder.ARMS).

What it can and cannot tell us. The protein side is frozen and its projection
is random at the start, so these numbers do NOT predict end-to-end
performance; the pocket head here was trained on 32 receptors for 300 steps.
What they do give is a RANKING of ligand encoders against real surface
vectors, and a floor: an arm that cannot beat chance here will not be rescued
by training the protein side. The run also exercises the whole ligand data
path - featurisation, grouping, masking, the sampler, the loss, the metrics -
which is the part that would otherwise go unexercised until the first full
training job.

What is dumped is the POOLED (pre-projection) region embedding, not
anchor_vec. The projection has only ever seen pocket BCE, so its weights are
effectively random; freezing them would freeze noise. The probe trains a
projection of the same shape, so what it learns is what full training would
learn there.

Retrieval score, matching the training design: s(P, L) = max over P's pocket
anchors of cos(anchor, ligand) / tau, a multiple-instance score. The selector's
negative patches are never scored. `--oracle` additionally reports the score
restricted to anchors that genuinely match the labelled pocket, which bounds
what better anchor RANKING could buy.

Region labels, from each region's overlap with the labelled pocket:
    positive   cover >= --pos_cover and precision >= --pos_prec (the `hit` rule
               of test_encoder / region_sweep)
    ignore     any other region with cover or precision >= --ignore_overlap,
               near-duplicates of the pocket included: they are not negatives
    negative   everything else
With --other_ligands (a labels.py output directory), a region over ANOTHER
ligand's site on the same receptor is ignored too, rather than training as a
negative against a genuine pocket. That directory is read for its lig_xyz
COORDINATES, not receptor_union.npz's indices: this script samples a fresh
surface per protein, and labels.py's own warning applies - saved indices
address the point cloud they were computed on and nothing else.

False negatives are masked out of every denominator and every ranking: a pair
is masked when it shares a target group (system_pocket_UniProt, falling back to
the PLINDER 95% interaction component), is the same ligand, or has ECFP4
Tanimoto >= --tanimoto.

Example:
    # 1. CPU venv: pick receptors the checkpoint has not seen, and featurise
    python src/probe_retrieval.py prepare --index datasets/dataset_v1/index.parquet \\
        --exclude encoder_test/items.tsv --n 1500 --out probe/
    python src/ligand_features.py --index datasets/dataset_v1/index.parquet \\
        --out probe/
    # 2. container, GPU: freeze the anchors (see src/slurm/probe_retrieval.sbatch)
    python src/probe_retrieval.py dump --items probe/items.tsv \\
        --ckpt encoder_test/head_pretrained_300.pt --region_method ball \\
        --ball_radius 8 --ball_shift 2 --out probe/anchors.pt
    # 3. anywhere with torch: compare the arms
    python src/probe_retrieval.py probe --anchors probe/anchors.pt \\
        --ligands probe/ligand_feats.npz --arms fp,fp_linear,desc,fp_desc,gnn \\
        --out probe/arms.tsv
    # no data needed: planted-signal end-to-end check of loss + metrics
    python src/probe_retrieval.py selftest

Depends on: numpy; torch (dump, probe, selftest); pandas (prepare);
            ligand_encoder, ligand_features, surface_encoder, anchors, ids.
"""

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

ITEM_COLS = ["system_id", "receptor_key", "split", "input_path", "input_chains",
             "sdf_paths", "fp_row", "group", "smiles"]


# ---------------------------------------------------------------------------
# prepare (CPU venv: pandas)
# ---------------------------------------------------------------------------
def group_of(row):
    """Target group for balancing and false-negative masking.

    UniProt first: it is what "the same target" means. The PLINDER 95% strong
    interaction component is the fallback, and a system with neither is its own
    group, which is conservative - it is never masked against anything else.
    """
    up = row.get("system_pocket_UniProt")
    if isinstance(up, str) and up.strip():
        return f"up:{up.strip()}"
    comp = row.get("pli_unique_qcov__95__strong__component")
    if comp is not None and str(comp) not in ("", "nan", "None"):
        return f"pli:{comp}"
    return f"sys:{row['system_id']}"


def prepare(a):
    import pandas as pd

    from ids import ligand_chain, ligand_sdf_paths, receptor_key

    df = pd.read_parquet(a.index)
    if a.split:
        df = df[df["split"].isin(a.split.split(","))]
    n_all = len(df)

    excluded = set()
    for path in a.exclude or []:
        lines = Path(path).read_text().splitlines()
        header = lines[0].split("\t")
        if "system_id" not in header:
            raise SystemExit(f"{path}: no system_id column to exclude on")
        col = header.index("system_id")
        # by RECEPTOR: another ligand on a receptor the checkpoint trained on
        # is still a surface it has seen
        excluded |= {receptor_key(ln.split("\t")[col]) for ln in lines[1:] if ln.strip()}
    if excluded:
        before = len(df)
        df = df[~df["receptor_key"].isin(excluded)]
        print(f"excluded {len(excluded)} receptors from {len(a.exclude)} file(s): "
              f"{before} -> {len(df)} rows")

    # one interaction per receptor: each dumped surface appears once, so no
    # protein is over-represented in the retrieval metrics
    df = df.drop_duplicates("input_id").sample(frac=1, random_state=a.seed)
    rows, n_no_sdf = [], 0
    for r in df.itertuples():
        d = r._asdict()
        sdfs = ligand_sdf_paths(a.systems, r.system_id, ligand_chain(r.ligand_id))
        if not sdfs:
            n_no_sdf += 1
            continue
        rows.append({
            "system_id": r.system_id, "receptor_key": r.receptor_key, "split": r.split,
            "input_path": r.input_path, "input_chains": r.input_chains,
            "sdf_paths": ",".join(map(str, sdfs)), "fp_row": int(r.fp_row),
            "group": group_of(d),
            "smiles": str(d.get("ligand_rdkit_canonical_smiles", "")),
        })
        if len(rows) == a.n:
            break

    if not rows:
        raise SystemExit("no usable rows; are systems/ and the index in place?")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "items.tsv").write_text(
        "\t".join(ITEM_COLS) + "\n"
        + "".join("\t".join(str(r[c]) for c in ITEM_COLS) + "\n" for r in rows))

    groups = [r["group"] for r in rows]
    sizes = np.unique(groups, return_counts=True)[1]
    print(f"\n{out / 'items.tsv'}: {len(rows)} items from {n_all} rows "
          f"({n_no_sdf} skipped for a missing ligand SDF)")
    print(f"{len(sizes)} target groups; largest {sorted(sizes)[-5:]}, "
          f"median {int(np.median(sizes))}")
    print(f"{len({r['fp_row'] for r in rows})} distinct ligands")
    print("\nnext: src/ligand_features.py for ligand_feats.npz, then `dump` on the GPU")


# ---------------------------------------------------------------------------
# dump (GPU container)
# ---------------------------------------------------------------------------
def load_probe_items(path, n=None):
    """items.tsv -> item dicts with atoms and ligand coordinates loaded."""
    import chemotype
    from extract import load_protein_atoms
    from ids import parse_chains
    from test_encoder import read_sdf_heavy

    lines = Path(path).read_text().splitlines()
    header = lines[0].split("\t")
    body = lines[1:] if n is None else lines[1:1 + n]
    items, failed = [], []
    for ln in body:
        if not ln.strip():
            continue
        r = dict(zip(header, ln.split("\t")))
        try:
            prot = chemotype.annotate(load_protein_atoms(
                r["input_path"], parse_chains(r["input_chains"]), keep_hydrogens=False))
            lig = np.concatenate([read_sdf_heavy(p) for p in r["sdf_paths"].split(",")])
        except Exception as e:
            failed.append((r["system_id"], f"{type(e).__name__}: {e}"))
            continue
        items.append({"id": r["system_id"], "receptor_key": r.get("receptor_key", ""),
                      "split": r["split"], "group": r["group"],
                      "fp_row": int(r["fp_row"]), "smiles": r.get("smiles", ""),
                      "atom_xyz": prot["atom_xyz"], "atom_type": prot["atom_type"],
                      "atom_groups": prot["atom_groups"], "lig": lig})
    if failed:
        print(f"[skip] {len(failed)} items failed to load; first few:")
        for sid, msg in failed[:5]:
            print(f"    {sid}: {msg[:120]}")
    return items


REGION_OVERRIDES = {
    "region_method": "region_method", "top_fraction": "region_top_fraction",
    "ball_radius": "region_ball_radius", "ball_shift": "region_ball_shift",
    "ball_rank": "region_ball_rank", "nms_iou": "region_nms_iou",
    "nms_alpha": "region_nms_alpha", "nms_score": "region_nms_score",
    "persistence": "region_persistence", "k_pos": "k_pos", "k_neg": "k_neg",
    "min_region_points": "min_region_points",
}


def apply_overrides(cfg, a):
    """Override the checkpoint's REGION settings from the CLI.

    Legitimate because region selection has no learned parameters - it is the
    premise of region_sweep.py - and necessary because the checkpoint was saved
    with the old `watershed` default while the sweep's leader is `ball`.
    Anything that would change the WEIGHTS' meaning is not overridable here.
    """
    changed = {}
    for flag, field in REGION_OVERRIDES.items():
        val = getattr(a, flag, None)
        if val is None:
            continue
        if field == "region_top_fraction" and val <= 0:
            val = None
        if getattr(cfg, field) != val:
            changed[field] = (getattr(cfg, field), val)
            setattr(cfg, field, val)
    if changed:
        print("region settings overridden from the checkpoint:")
        for k, (old, new) in changed.items():
            print(f"    {k}: {old!r} -> {new!r}")
    return cfg


def index_other_ligands(labels_dir):
    """labels.py output directory -> {receptor_key: [(system_id, path), ...]}.

    Only the file names are read here; a receptor's ligand coordinates are
    loaded when an item needs them, which is a handful of small files rather
    than the whole directory.
    """
    out = defaultdict(list)
    for p in sorted(Path(labels_dir).glob("*.npz")):
        sid = p.stem
        if sid in ("receptor_union", "pocket_qc"):
            continue
        out["__".join(sid.split("__")[:3])].append((sid, p))
    return out


def other_site_mask(xyz, entries, own_id, cutoff):
    """Surface points within `cutoff` of a DIFFERENT ligand on this receptor.

    By coordinates, recomputed against the surface in hand. The indices in a
    labels.py file belong to the surface it was written against; this script
    samples its own, so only lig_xyz survives the trip.
    """
    import torch

    mask = torch.zeros(len(xyz), dtype=torch.bool)
    n = 0
    for sid, path in entries:
        if sid == own_id:
            continue
        with np.load(path, allow_pickle=False) as d:
            lig = d["lig_xyz"]
        if not len(lig):
            continue
        mask |= torch.cdist(xyz, torch.from_numpy(lig).float()).min(1).values < cutoff
        n += 1
    return mask, n


def region_labels(regions, pocket, other_sites, a):
    """-> (label, cover, prec) per region: 1 positive, 0 negative, -1 ignore."""
    import torch

    from anchors import region_overlap

    prec, cover = region_overlap(regions, pocket)
    label = torch.zeros(len(regions), dtype=torch.long)
    pos = (cover >= a.pos_cover) & (prec >= a.pos_prec)
    near = (cover >= a.ignore_overlap) | (prec >= a.ignore_overlap)
    label[near] = -1
    label[pos] = 1
    if other_sites is not None and other_sites.any():
        o_prec, o_cover = region_overlap(regions, other_sites)
        touches = ((o_cover >= a.ignore_overlap) | (o_prec >= a.ignore_overlap))
        label[touches & (label == 0)] = -1
    return label, cover, prec


def dump(a):
    import torch

    from test_encoder import load_checkpoint, to_batch

    dev = a.device
    enc, ckpt = load_checkpoint(a.ckpt, dev)
    apply_overrides(enc.cfg, a)
    trained_on = {k for k in ckpt.get("train_ids", [])}
    trained_rec = {"__".join(s.split("__")[:3]) for s in trained_on}
    print(f"\ncheckpoint {a.ckpt}: trained on {len(trained_on)} items "
          f"({len(trained_rec)} receptors)")

    others = None
    if a.other_ligands:
        others = index_other_ligands(a.other_ligands)
        print(f"{a.other_ligands}: labelled ligands for {len(others)} receptors")
    else:
        print("[warn] no --other_ligands: a region over a receptor's OTHER ligand "
              "site cannot be ignored, so a few genuine pockets train as negatives")

    items = load_probe_items(a.items, a.limit)
    seen = [it for it in items if it["receptor_key"] in trained_rec or it["id"] in trained_on]
    if seen:
        print(f"[warn] {len(seen)} items are receptors this checkpoint TRAINED on; "
              f"marking them seen=True so `probe` can drop them")
    print(f"{len(items)} items loaded; sampling surfaces one at a time")

    records, t0 = [], time.time()
    n_no_pocket = n_no_anchor = n_with_other = 0
    for i, it in enumerate(items, 1):
        P = to_batch([it], dev, with_surface=False)
        with torch.no_grad():
            xyz, normals, _ = enc.sample_surface(P["atom_xyz"], P["atomtypes"],
                                                 P["batch_atoms"])
        it["xyz"], it["normals"] = xyz.cpu(), normals.cpu()
        d = torch.cdist(it["xyz"], torch.from_numpy(it["lig"])).min(1).values
        pocket = d < a.cutoff
        if not pocket.any():
            n_no_pocket += 1
            continue

        with torch.no_grad():
            out = enc(to_batch([it], dev))
        base = int((out["batch"] == 0).nonzero()[0])
        regions = [(r - base).cpu() for r in out["anchor_regions"][0]]
        valid = out["anchor_valid"][0].cpu()
        is_pos = out["anchor_is_pos"][0].cpu()
        if not bool((valid & is_pos).any()):
            n_no_anchor += 1
            continue

        other = None
        if others is not None:
            entries = others.get(it["receptor_key"], [])
            if entries:
                other, n_other = other_site_mask(it["xyz"], entries, it["id"], a.cutoff)
                other &= ~pocket  # this ligand's own pocket is not an "other site"
                n_with_other += int(n_other > 0)
        label = torch.full((len(valid),), -1, dtype=torch.long)
        cover = torch.zeros(len(valid))
        prec = torch.zeros(len(valid))
        if regions:
            lab_r, cov_r, pre_r = region_labels(regions, pocket, other, a)
            label[:len(regions)] = lab_r
            cover[:len(regions)] = cov_r
            prec[:len(regions)] = pre_r

        records.append({
            "id": it["id"], "receptor_key": it["receptor_key"], "split": it["split"],
            "group": it["group"], "fp_row": it["fp_row"], "smiles": it["smiles"],
            "seen": it["receptor_key"] in trained_rec or it["id"] in trained_on,
            "pooled": out["anchor_pooled"][0].float().cpu(),
            "valid": valid, "is_pos": is_pos, "label": label,
            "cover": cover, "prec": prec,
            "size": out["anchor_size"][0].cpu(),
            "center": out["anchor_center"][0].float().cpu(),
            "n_points": int(len(it["xyz"])), "n_pocket": int(pocket.sum()),
            "lig_min_dist": float(d.min()),
        })
        if i % 50 == 0 or i == len(items):
            print(f"  {i}/{len(items)}  ({(time.time() - t0) / i:.2f}s/item)", flush=True)

    if not records:
        raise SystemExit("nothing dumped")
    hit = float(np.mean([bool((r["label"] == 1).any()) for r in records]))
    n_pos = float(np.mean([int((r["valid"] & r["is_pos"]).sum()) for r in records]))
    print(f"\n{len(records)} proteins dumped "
          f"({n_no_pocket} dropped: ligand not on the surface; "
          f"{n_no_anchor} dropped: no pocket anchor)")
    print(f"{n_pos:.1f} pocket anchors per protein; "
          f"{hit:.2f} of proteins have an anchor matching the labelled pocket "
          f"(cover >= {a.pos_cover}, prec >= {a.pos_prec})")
    print(f"{len({r['group'] for r in records})} target groups, "
          f"{len({r['fp_row'] for r in records})} distinct ligands")
    if others is not None:
        print(f"{n_with_other} proteins had another labelled ligand site, now ignored")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"records": records, "args": vars(a),
                "encoder_config": {k: v for k, v in vars(enc.cfg).items()},
                "ckpt_path": str(a.ckpt), "ckpt_train_ids": sorted(trained_on)}, a.out)
    print(f"\nwrote {a.out}")


# ---------------------------------------------------------------------------
# probe: losses, masks, metrics
# ---------------------------------------------------------------------------
def tanimoto(fp_a, fp_b):
    """Pairwise Tanimoto between two sets of 0/1 fingerprints -> (A, B)."""
    import torch

    a = fp_a.float()
    b = fp_b.float()
    inter = a @ b.T
    union = a.sum(1)[:, None] + b.sum(1)[None, :] - inter
    return inter / union.clamp_min(1.0)


def mask_pairs(group_q, group_c, row_q, row_c, fp_q, fp_c, thresh):
    """True where candidate c must not count as a negative for query q.

    Same target group, the same ligand, or a near-identical ligand. The
    diagonal is left False: a query's own answer is never masked.
    """
    import torch

    m = (group_q[:, None] == group_c[None, :]) | (row_q[:, None] == row_c[None, :])
    if thresh is not None and thresh < 1.0:
        m |= tanimoto(fp_q, fp_c) >= thresh
    n = min(len(group_q), len(group_c))
    if len(group_q) == len(group_c):
        m[torch.arange(n), torch.arange(n)] = False
    return m


def info_nce(scores, mask):
    """Symmetric InfoNCE over an aligned (B, B) score matrix.

    Row i and column i are the matching protein and ligand. Masked pairs are
    pushed to -inf so they leave the denominator entirely, rather than being
    trained against as negatives.
    """
    import torch
    import torch.nn.functional as F

    s = scores.masked_fill(mask, float("-inf"))
    target = torch.arange(len(s), device=s.device)
    return 0.5 * (F.cross_entropy(s, target) + F.cross_entropy(s.T, target))


def retrieval_metrics(scores, mask, ks=(1, 5, 10)):
    """Rank of each query's true partner among its unmasked candidates.

    scores (Q, C) with the answer to query q on the diagonal. Ties count as
    better, so the reported rank is the pessimistic one. `chance` is the exact
    expected MRR of a random ranking over the same candidate counts, which is
    the only honest floor when the candidate set is small.
    """
    import torch

    Q = len(scores)
    ar = torch.arange(Q)
    keep = ~mask
    keep[ar, ar] = True
    true = scores[ar, ar]
    better = ((scores > true[:, None]) & keep).sum(1)
    rank = (better + 1).float()
    n_cand = keep.sum(1).float()
    harmonic = torch.tensor([float(sum(1.0 / i for i in range(1, int(n) + 1)))
                             for n in n_cand])
    out = {"n": Q, "cands": float(n_cand.mean()), "mrr": float((1 / rank).mean()),
           "median_rank": float(rank.median()),
           "chance": float((harmonic / n_cand).mean())}
    for k in ks:
        out[f"top{k}"] = float((rank <= k).float().mean())
    return out


def split_by_group(groups, val_frac, seed, tol=0.25):
    """Group-disjoint train / val row indices.

    Disjoint by target: dataset_v1 has one UniProt with hundreds of rows, so a
    random row split would put the same target on both sides and the metric
    would read as memorisation.

    A group that would take the validation set past `target * (1 + tol)` rows
    is left in train instead of being accepted. Stopping only AFTER the target
    is reached is not enough - PDE10A's 747 rows, drawn first, would land a
    "30% validation" split at 87% - so the overshoot has to be refused per
    group, not detected afterwards. If every group is larger than that cap,
    validation is the single smallest group, which is the most this data allows.
    """
    groups = np.asarray(groups)
    uniq, counts = np.unique(groups, return_counts=True)
    rng = np.random.default_rng(seed)
    target = val_frac * len(groups)
    cap = target * (1 + tol)
    val_groups, n = set(), 0
    for i in rng.permutation(len(uniq)):
        if n >= target:
            break
        if n + counts[i] > cap:
            continue
        val_groups.add(uniq[i])
        n += int(counts[i])
    if not val_groups:
        i = int(np.argmin(counts))
        val_groups = {uniq[i]}
    in_val = np.isin(groups, list(val_groups))
    return np.nonzero(~in_val)[0], np.nonzero(in_val)[0]


def anchor_spread(vecs, max_n=1500, seed=0):
    """How much do different proteins' anchor vectors actually differ?

    `median_cos` is directly comparable to the reference apo_eval prints (0.993
    for ball_R8 on holo): for random 16-dim unit vectors it would be ~0, so a
    value near 1 means the space is a narrow cone and the projection has to
    recover a small residual rather than read off a spread-out code.

    `shared` is ||mean of the unit vectors||, the cleanest single statement of
    the problem: 1.0 means every protein points the same way, ~0 means the
    directions are isotropic. `eff_dims` is the participation ratio of the
    centred spectrum, (sum lambda)^2 / sum lambda^2 - the effective number of
    dimensions in use, out of emb_dims.
    """
    import torch
    import torch.nn.functional as F

    # The pairwise matrix is n^2 and torch.quantile has a size cap, so bound it.
    if len(vecs) > max_n:
        g = torch.Generator().manual_seed(seed)
        sel = torch.randperm(len(vecs), generator=g)[:max_n].to(vecs.device)
        vecs = vecs[sel]
    U = F.normalize(vecs.float(), dim=1)
    n = len(U)
    eye = torch.eye(n, dtype=torch.bool, device=U.device)
    off = (U @ U.T)[~eye]
    m = U.mean(0)
    Uc = F.normalize(U - m, dim=1)
    offc = (Uc @ Uc.T)[~eye]
    lam = torch.linalg.svdvals(U - m) ** 2
    return {"n": n, "median_cos": float(off.median()),
            "p90_cos": float(off.quantile(0.9)),
            "shared": float(m.norm()),
            "median_cos_centred": float(offc.median()),
            "eff_dims": float(lam.sum() ** 2 / (lam ** 2).sum().clamp_min(1e-12)),
            "dims": U.shape[1]}


class ProbeModel:
    """The two trainable pieces: the projection onto frozen anchors, and the
    ligand encoder. Kept as a plain holder so the optimiser and the state dict
    stay obvious."""

    def __init__(self, emb_dim, cfg, map_kind="mlp", tau=0.07, device="cpu"):
        import torch
        import torch.nn as nn

        from ligand_encoder import LigandEncoder

        D = cfg.out_dim
        self.project = (nn.Linear(emb_dim, D) if map_kind == "linear" else
                        nn.Sequential(nn.Linear(emb_dim, max(D, emb_dim)),
                                      nn.LeakyReLU(0.2),
                                      nn.Linear(max(D, emb_dim), D))).to(device)
        self.ligand = LigandEncoder(cfg).to(device)
        self.logit_scale = nn.Parameter(
            torch.log(torch.tensor(1.0 / tau, device=device)))
        self.device = device
        # Standardisation of the frozen anchors, fitted on TRAIN proteins only.
        # Identity until set_anchor_norm() says otherwise, so the default path
        # is unchanged.
        self.anchor_mean = torch.zeros(emb_dim, device=device)
        self.anchor_std = torch.ones(emb_dim, device=device)

    def set_anchor_norm(self, pooled, usable, mode):
        """Fit anchor standardisation on these proteins' usable anchors.

        Worth having because ~88% of a real pooled vector is one shared
        direction (apo_eval: different proteins sit at cosine 0.993), which a
        projection starting from random init has to learn to subtract. Centring
        hands it that for free. Fitted on the TRAIN split alone, so the val
        proteins contribute nothing to the statistics.
        """
        if mode == "none":
            return
        x = pooled[usable].float()
        if not len(x):
            return
        self.anchor_mean = x.mean(0)
        if mode == "whiten":
            self.anchor_std = x.std(0).clamp_min(1e-6)

    def parameters(self):
        return (list(self.project.parameters()) + list(self.ligand.parameters())
                + [self.logit_scale])

    def train(self, flag=True):
        self.project.train(flag)
        self.ligand.train(flag)

    def anchors(self, pooled, valid):
        """(B, K, E) pooled embeddings -> (B, K, D) normalised anchor vectors."""
        import torch.nn.functional as F

        x = (pooled - self.anchor_mean) / self.anchor_std
        return F.normalize(self.project(x), dim=-1) * valid[..., None]

    def scores(self, pooled, usable, lig_vec):
        """max over each protein's usable anchors of cos(anchor, ligand), scaled.

        -> (B_prot, B_lig). A protein with no usable anchor would make the max
        over an empty set; callers drop those rows first, and the -1e4 fill is
        the belt-and-braces version of that.
        """
        import torch

        a = self.anchors(pooled, usable)
        sim = torch.einsum("bkd,ld->bkl", a, lig_vec)
        sim = sim.masked_fill(~usable[..., None], -1e4)
        scale = self.logit_scale.exp().clamp(max=100.0)
        return sim.max(dim=1).values * scale

    def state_dict(self):
        return {"project": self.project.state_dict(),
                "ligand": self.ligand.state_dict(),
                "logit_scale": self.logit_scale.detach().cpu(),
                "anchor_mean": self.anchor_mean.detach().cpu(),
                "anchor_std": self.anchor_std.detach().cpu()}


def stack_records(records, device):
    """-> (pooled, valid, is_pos, label) tensors padded to a common K."""
    import torch

    K = max(len(r["valid"]) for r in records)
    E = records[0]["pooled"].shape[1]
    pooled = torch.zeros(len(records), K, E)
    valid = torch.zeros(len(records), K, dtype=torch.bool)
    is_pos = torch.zeros(len(records), K, dtype=torch.bool)
    label = torch.zeros(len(records), K, dtype=torch.long)
    for i, r in enumerate(records):
        k = len(r["valid"])
        pooled[i, :k] = r["pooled"]
        valid[i, :k] = r["valid"]
        is_pos[i, :k] = r["is_pos"]
        label[i, :k] = r["label"]
    return (pooled.to(device), valid.to(device), is_pos.to(device), label.to(device))


def sample_weights(groups, alpha):
    """Per-row sampling weight 1 / size^alpha, normalised.

    alpha 1 makes sampling uniform over TARGETS, which is the whole point:
    PDE10A has 747 rows in dataset_v1 and would otherwise dominate every batch.
    alpha 0 is plain row-uniform sampling, kept so the effect is measurable.

    This is the DEFINITION of alpha. draw_batch() implements the same
    distribution group-first, because it also needs one-row-per-group and
    cannot afford a rejection loop; see its docstring for the equivalence.
    """
    groups = np.asarray(groups)
    uniq, inv, counts = np.unique(groups, return_inverse=True, return_counts=True)
    w = 1.0 / np.power(counts[inv].astype(np.float64), alpha)
    return w / w.sum()


def group_index(groups):
    """Row positions bucketed by target group -> (member arrays, counts).

    Built once per split and reused every step. The obvious alternative, a
    boolean compare per group per step, is O(rows x groups) and would dominate
    the step at dataset_v1's scale.
    """
    groups = np.asarray(groups)
    _, inv, counts = np.unique(groups, return_inverse=True, return_counts=True)
    order = np.argsort(inv, kind="stable")
    return np.split(order, np.cumsum(counts)[:-1]), counts


def draw_batch(rng, members, counts, size, alpha=1.0):
    """Up to `size` rows, at most one per target group.

    Groups are chosen first, then one row within each. That is EQUIVALENT to
    the per-row weighting of sample_weights(): a row weight of count^-alpha
    sums to count^(1-alpha) over a group, so drawing groups with that
    probability and then uniformly inside the group gives the same per-row
    distribution - at O(size) instead of a rejection loop.

    Two rows of the same group would be masked against each other anyway, so a
    batch never spends two slots on a pair that carries no gradient.

    The group-first form also avoids a real hazard. numpy's weighted choice
    WITHOUT replacement is a rejection loop that must find every element it
    returns, so asking it for (nearly) the whole population with skewed weights
    can take unboundedly long - dataset_v1 pairs PDE10A's 747 rows against
    singletons, a 700x weight spread. When `size` covers every group there is
    nothing to sample, so permute instead.
    """
    n = len(members)
    k = min(size, n)
    if k >= n:
        chosen = rng.permutation(n)
    else:
        w = counts.astype(np.float64) ** (1.0 - alpha)
        chosen = rng.choice(n, size=k, replace=False, p=w / w.sum())
    out = np.empty(k, dtype=np.int64)
    for i, c in enumerate(chosen):
        m = members[c]
        out[i] = m[rng.integers(len(m))]
    return out


def encode_ligands(model, data, fp_rows, device):
    """Ligand vectors for these ligand rows.

    The `frozen` arm's input is a precomputed embedding that lives beside the
    featurised arrays rather than in the collated batch, so it is attached
    here - the single path both training and evaluation take, so neither can
    forget it.
    """
    import torch

    batch = data["ligands"].collate(fp_rows, device)
    if data["frozen_dim"]:
        batch["frozen"] = torch.as_tensor(data["frozen_emb"][fp_rows]).to(device)
    return model.ligand(batch)


def evaluate(model, data, rows, a, oracle=False):
    """Retrieval + site metrics over `rows`, all proteins against all ligands."""
    import torch

    pooled, valid, is_pos, label = data["tensors"]
    rows_t = torch.as_tensor(rows, device=pooled.device)
    usable = (valid & is_pos)[rows_t]
    if oracle:
        usable = (valid & (label == 1))[rows_t]
        keep = usable.any(1)
        if not bool(keep.all()):
            rows_t = rows_t[keep]
            usable = usable[keep]
            rows = np.asarray(rows)[keep.cpu().numpy()]
    model.train(False)
    with torch.no_grad():
        fp_rows = data["fp_row"][rows]
        lig = encode_ligands(model, data, fp_rows, pooled.device)
        scores = model.scores(pooled[rows_t], usable, lig)
        groups = torch.as_tensor(data["group_id"][rows], device=pooled.device)
        fps = torch.as_tensor(data["ligands"].fp[fp_rows].astype(np.float32),
                              device=pooled.device)
        frows = torch.as_tensor(fp_rows, device=pooled.device)
        mask = mask_pairs(groups, groups, frows, frows, fps, fps, a.tanimoto)
        met = retrieval_metrics(scores.T.cpu(), mask.cpu())          # ligand -> protein
        met_p = retrieval_metrics(scores.cpu(), mask.cpu())           # protein -> ligand
        met["p2l_mrr"] = met_p["mrr"]
        met["p2l_top1"] = met_p["top1"]

        # site: is the winning anchor one that matches the labelled pocket?
        a_vec = model.anchors(pooled[rows_t], usable)
        sim = torch.einsum("bkd,bd->bk", a_vec, lig)
        sim = sim.masked_fill(~usable, -1e4)
        best = sim.argmax(1)
        lab = label[rows_t].gather(1, best[:, None]).squeeze(1)
        has_pos = (valid & (label == 1))[rows_t].any(1)
        met["site"] = (float((lab[has_pos] == 1).float().mean())
                       if bool(has_pos.any()) else float("nan"))
        met["site_cov"] = float(has_pos.float().mean())
    model.train(True)
    return met


def train_arm(arm, data, train_rows, val_rows, a):
    """Train one ligand arm on the frozen anchors -> (metrics history, model)."""
    import torch

    from ligand_encoder import LigandConfig

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    pooled = data["tensors"][0]
    cfg = LigandConfig(arm=arm, out_dim=a.out_dim, hidden=a.hidden,
                       dropout=a.dropout, gnn_hidden=a.gnn_hidden,
                       gnn_layers=a.gnn_layers,
                       frozen_dim=data["frozen_dim"] if arm == "frozen" else 0)
    model = ProbeModel(pooled.shape[2], cfg, a.map, a.tau, a.device)
    _, valid0, is_pos0, _ = data["tensors"]
    tr = torch.as_tensor(train_rows, device=pooled.device)
    model.set_anchor_norm(pooled[tr], (valid0 & is_pos0)[tr], a.anchor_norm)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)

    members, counts = group_index(data["group"][train_rows])
    rng = np.random.default_rng(a.seed)
    _, valid, is_pos, _ = data["tensors"]
    history, losses = [], []
    t0 = time.time()
    for step in range(a.steps + 1):
        if step % a.eval_every == 0 or step == a.steps:
            met = evaluate(model, data, val_rows, a)
            met.update(step=step, loss=float(np.mean(losses[-a.eval_every:]))
                       if losses else float("nan"))
            history.append(met)
            print(f"  {arm:10s} {step:6d} {time.time() - t0:5.0f}s "
                  f"loss {met['loss']:7.4f}  "
                  f"val MRR {met['mrr']:.3f} (chance {met['chance']:.3f})  "
                  f"top1 {met['top1']:.3f}  site {met['site']:.3f}", flush=True)
        if step == a.steps:
            break

        idx = draw_batch(rng, members, counts, a.batch, a.alpha)
        rows = train_rows[idx]
        rows_t = torch.as_tensor(rows, device=pooled.device)
        usable = (valid & is_pos)[rows_t]
        keep = usable.any(1)
        if int(keep.sum()) < 2:
            continue
        rows, rows_t, usable = rows[keep.cpu().numpy()], rows_t[keep], usable[keep]

        fp_rows = data["fp_row"][rows]
        lig = encode_ligands(model, data, fp_rows, pooled.device)
        scores = model.scores(pooled[rows_t], usable, lig)
        g = torch.as_tensor(data["group_id"][rows], device=pooled.device)
        fps = torch.as_tensor(data["ligands"].fp[fp_rows].astype(np.float32),
                              device=pooled.device)
        fr = torch.as_tensor(fp_rows, device=pooled.device)
        loss = info_nce(scores, mask_pairs(g, g, fr, fr, fps, fps, a.tanimoto))
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip)
        opt.step()
        losses.append(float(loss.detach()))
    return history, model


def load_probe_data(a):
    """-> the dict every probe function reads: tensors, ligands, ids."""
    import torch

    from ligand_encoder import LigandData

    blob = torch.load(a.anchors, map_location="cpu", weights_only=False)
    records = blob["records"]
    n_seen = sum(1 for r in records if r.get("seen"))
    if n_seen and not a.keep_seen:
        records = [r for r in records if not r.get("seen")]
        print(f"dropped {n_seen} proteins the checkpoint trained on "
              f"(--keep_seen to keep them)")
    if len(records) < 10:
        raise SystemExit(f"only {len(records)} proteins; not enough to rank anything")

    ligands = LigandData(a.ligands)
    fp_row = np.asarray([r["fp_row"] for r in records])
    if fp_row.max() >= len(ligands):
        raise SystemExit(
            f"fp_row up to {fp_row.max()} but {a.ligands} holds {len(ligands)} "
            "ligands; the anchors and the features come from different datasets")
    bad = ligands.failed[fp_row]
    if bad.any():
        print(f"dropped {int(bad.sum())} proteins whose ligand could not be featurised")
        records = [r for r, b in zip(records, bad) if not b]
        fp_row = fp_row[~bad]

    group = np.asarray([r["group"] for r in records])
    _, group_id = np.unique(group, return_inverse=True)
    data = {"records": records, "ligands": ligands, "fp_row": fp_row,
            "group": group, "group_id": group_id.astype(np.int64),
            "tensors": stack_records(records, a.device),
            "frozen_dim": 0}
    if getattr(a, "lig_emb", None):
        emb = np.load(a.lig_emb)
        if len(emb) != len(ligands):
            raise SystemExit(f"{a.lig_emb}: {len(emb)} rows, expected {len(ligands)}")
        data["frozen_emb"] = emb.astype(np.float32)
        data["frozen_dim"] = emb.shape[1]
    return data


def probe(a):
    import torch

    data = load_probe_data(a)
    records = data["records"]
    train_rows, val_rows = split_by_group(data["group"], a.val_frac, a.split_seed)
    print(f"\n{len(records)} proteins, {len(set(data['group']))} groups, "
          f"{len(set(data['fp_row']))} distinct ligands")
    print(f"group-disjoint split: {len(train_rows)} train / {len(val_rows)} val "
          f"({len(set(data['group'][train_rows]))} / "
          f"{len(set(data['group'][val_rows]))} groups)")
    data["ligands"].set_scaler(data["fp_row"][train_rows])
    print("descriptor standardisation fitted on the TRAIN ligands only")

    # Are the frozen anchors discriminative at all? This decides whether the arm
    # table below says anything about ligand encoders, or only about the protein
    # side's collapse. Computed on the TOP-ranked pocket anchor per protein, the
    # same vector apo_eval's reference cosine uses.
    pooled_t, valid_t, is_pos_t, _ = data["tensors"]
    top = []
    for i in range(len(pooled_t)):
        k = (valid_t[i] & is_pos_t[i]).nonzero().flatten()
        if len(k):
            top.append(pooled_t[i, int(k[0])])
    if len(top) > 2:
        sp = anchor_spread(torch.stack(top))
        print(f"\nfrozen anchor spread over {sp['n']} proteins, {sp['dims']} dims "
              f"(top-ranked pocket anchor each):")
        print(f"  median cosine between DIFFERENT proteins   {sp['median_cos']:+.3f} "
              f"(p90 {sp['p90_cos']:+.3f})   apo_eval reference: +0.993")
        print(f"  shared direction ||mean of unit vectors||  {sp['shared']:.3f} "
              f"(1 = all identical, 0 = isotropic)")
        print(f"  after centring, median cosine              {sp['median_cos_centred']:+.3f}")
        print(f"  effective dimensions in use                {sp['eff_dims']:.1f} of {sp['dims']}")
        if sp["median_cos"] > 0.9:
            print("  [warn] the anchors are a narrow cone: a weak arm table here is")
            print("         the protein side, not the ligand encoders. Compare")
            print("         --anchor_norm center / whiten against none.")
    print(f"anchor standardisation: {a.anchor_norm}")

    arms = [s.strip() for s in a.arms.split(",") if s.strip()]
    if "frozen" in arms and not data["frozen_dim"]:
        raise SystemExit("arm 'frozen' needs --lig_emb with a precomputed embedding")
    rows_out, histories = [], {}
    for arm in arms:
        print(f"\n=== {arm} ===")
        t0 = time.time()
        history, model = train_arm(arm, data, train_rows, val_rows, a)
        best = max(history, key=lambda m: m["mrr"])
        final = history[-1]
        rec = {"arm": arm, "params": model.ligand.n_params(),
               "sec": time.time() - t0,
               "mrr0": history[0]["mrr"], "chance": final["chance"],
               **{k: final[k] for k in ("mrr", "top1", "top5", "top10",
                                        "median_rank", "p2l_mrr", "p2l_top1",
                                        "site", "site_cov", "cands")},
               "best_mrr": best["mrr"], "best_step": best["step"]}
        if a.oracle:
            o = evaluate(model, data, val_rows, a, oracle=True)
            rec["oracle_mrr"] = o["mrr"]
            rec["oracle_site"] = o["site"]
        rows_out.append(rec)
        histories[arm] = history
        if a.save_dir:
            d = Path(a.save_dir)
            d.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": model.state_dict(), "arm": arm,
                        "history": history}, d / f"probe_{arm}.pt")

    cols = ["arm", "params", "mrr0", "mrr", "best_mrr", "chance", "top1", "top5",
            "top10", "median_rank", "p2l_mrr", "site", "site_cov", "sec"]
    if a.oracle:
        cols.insert(cols.index("top1"), "oracle_mrr")
    print(f"\n{'arm':12s}" + "".join(f"{c:>11s}" for c in cols[1:]))
    for r in rows_out:
        print(f"{r['arm']:12s}" + "".join(
            f"{r[c]:11d}" if isinstance(r[c], int) else f"{r[c]:11.3f}"
            for c in cols[1:]))
    print("\nmrr0   before training (the untrained floor)   chance  random ranking"
          "\nmrr    ligand -> protein MRR on the val groups; p2l_mrr the other direction"
          "\nsite   of the proteins that HAVE a pocket-matching anchor (site_cov),"
          "\n       how often the ligand's best anchor is that one"
          + ("\noracle_mrr  MRR when only pocket-matching anchors are scored:"
             "\n       the headroom in anchor RANKING, not in the ligand encoder."
             "\n       Computed on the site_cov subset that HAS such an anchor, so it"
             "\n       is an easier question than mrr, not just a better score"
             if a.oracle else ""))

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        with open(a.out, "w") as f:
            f.write("\t".join(cols) + "\n")
            for r in rows_out:
                f.write("\t".join(f"{r[c]:.4f}" if isinstance(r[c], float) else str(r[c])
                                  for c in cols) + "\n")
        hist_path = Path(a.out).with_suffix(".history.json")
        hist_path.write_text(json.dumps(histories, indent=1) + "\n")
        print(f"\nwrote {a.out} and {hist_path}")


# ---------------------------------------------------------------------------
# selftest: planted signal, no data, no GPU
# ---------------------------------------------------------------------------
def make_fake(n_prot=160, n_lig=160, emb=16, k=16, latent=6, noise=0.35, seed=0):
    """Anchors + ligand features sharing a planted latent code.

    Each protein gets one TRUE pocket anchor built from its ligand's latent
    vector; its other anchors are noise. The ligand's fingerprint bits are
    drawn from the same latent. So a learnable map on each side can recover the
    pairing, and MRR must climb far above chance. If it does not, the loss, the
    masking, the sampler or the metrics are wrong - which is exactly what
    cannot be checked without the real dump.
    """
    import torch

    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n_lig, latent))
    W_fp = rng.normal(size=(latent, 2048))
    bits = (1 / (1 + np.exp(-(z @ W_fp)))) > rng.random((n_lig, 2048))
    W_a = rng.normal(size=(latent, emb))

    records = []
    for i in range(n_prot):
        lig = i % n_lig
        pooled = rng.normal(scale=noise, size=(k, emb))
        true_slot = int(rng.integers(0, k // 2))      # a pocket anchor
        pooled[true_slot] += z[lig] @ W_a
        valid = torch.ones(k, dtype=torch.bool)
        is_pos = torch.zeros(k, dtype=torch.bool)
        is_pos[: k // 2] = True                        # first half = pocket anchors
        label = torch.zeros(k, dtype=torch.long)
        label[true_slot] = 1
        records.append({
            "id": f"fake{i:04d}", "receptor_key": f"fake{i:04d}", "split": "train",
            "group": f"g{i // 2}",   # 2 proteins per group: exercises the masking
            "fp_row": lig, "smiles": "", "seen": False,
            "pooled": torch.from_numpy(pooled).float(), "valid": valid,
            "is_pos": is_pos, "label": label,
            "cover": torch.zeros(k), "prec": torch.zeros(k),
            "size": torch.full((k,), 100), "center": torch.zeros(k, 3),
            "n_points": 5000, "n_pocket": 200, "lig_min_dist": 2.0,
        })

    sizes = rng.integers(5, 25, n_lig)
    atom_ptr = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
    from ligand_features import ATOM_DIM, BOND_DIM, DESC_DIM
    idx, bfeat, bond_ptr = [], [], [0]
    for s in sizes:
        e = np.stack([np.arange(s - 1), np.arange(1, s)]).astype(np.int32)
        idx.append(e)
        bfeat.append(rng.integers(0, 2, (s - 1, BOND_DIM), dtype=np.uint8))
        bond_ptr.append(bond_ptr[-1] + s - 1)
    # Atom features carry the latent too, so the gnn arm has something to find.
    # W_at is drawn ONCE: a projection re-drawn per molecule would give every
    # molecule a different z -> features mapping, i.e. no learnable signal at
    # all, which is exactly what made the gnn arm sit at chance.
    W_at = rng.normal(size=(latent, ATOM_DIM))
    atom_feat = np.zeros((atom_ptr[-1], ATOM_DIM), dtype=np.uint8)
    for i in range(n_lig):
        a0, a1 = atom_ptr[i], atom_ptr[i + 1]
        p = 1 / (1 + np.exp(-(z[i] @ W_at)))
        atom_feat[a0:a1] = (p > rng.random((a1 - a0, ATOM_DIM))).astype(np.uint8)
    # Descriptors: every column a function of the latent plus a little noise,
    # as real physchem descriptors are heavily intercorrelated. The earlier
    # [latent, pure noise] layout left 27 of 33 columns irrelevant, which both
    # diluted the signal and handed the MLP noise dimensions to memorise
    # individual training molecules with.
    W_d = rng.normal(size=(latent, DESC_DIM - latent))
    ligands = {
        "fp": bits.astype(np.uint8),
        "desc": np.concatenate(
            [z, z @ W_d + 0.5 * rng.normal(size=(n_lig, DESC_DIM - latent))],
            axis=1).astype(np.float32),
        "atom_feat": atom_feat, "atom_ptr": atom_ptr,
        "bond_feat": np.concatenate(bfeat),
        "bond_index": np.concatenate(idx, axis=1),
        "bond_ptr": np.asarray(bond_ptr, dtype=np.int64),
        "failed": np.zeros(n_lig, dtype=bool),
        "smiles": np.asarray([f"fake{i}" for i in range(n_lig)]),
        "n_heavy": sizes.astype(np.int32),
    }
    return records, ligands


def selftest(a):
    import torch

    from ligand_encoder import LigandData

    tmp = Path(a.tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    records, ligands = make_fake(seed=a.seed)
    torch.save({"records": records, "args": {}, "encoder_config": {}},
               tmp / "anchors.pt")
    np.savez(tmp / "ligand_feats.npz", **ligands)
    print(f"planted {len(records)} proteins / {len(ligands['fp'])} ligands in {tmp}")

    a.anchors = str(tmp / "anchors.pt")
    a.ligands = str(tmp / "ligand_feats.npz")
    a.out = str(tmp / "arms.tsv")
    probe(a)

    rows = [ln.split("\t") for ln in Path(a.out).read_text().splitlines()[1:]]
    cols = Path(a.out).read_text().splitlines()[0].split("\t")
    failures = []
    for r in rows:
        d = dict(zip(cols, r))
        mrr, mrr0 = float(d["mrr"]), float(d["mrr0"])
        best, chance = float(d["best_mrr"]), float(d["chance"])
        # Judged on the PEAK, not the final step. The question this test asks is
        # whether each arm CAN learn the planted pairing - that is what exercises
        # the loss, the masking, the sampler and the metrics. Every arm then
        # overfits 112 synthetic rows and declines from its peak, which is
        # expected behaviour on a fixture this small and says nothing about the
        # wiring. `mrr` and `best_mrr` are both in the table above, so the
        # overfitting stays visible rather than being asserted away.
        if best < max(a.min_mrr, 2 * chance):
            failures.append(f"{d['arm']}: best val MRR {best:.3f} did not clear "
                            f"max({a.min_mrr}, 2 x chance {chance:.3f})")
        if best <= mrr0:
            failures.append(f"{d['arm']}: training never improved on the untrained "
                            f"floor ({mrr0:.3f} -> best {best:.3f})")
        if mrr < 0.6 * best:
            print(f"  [note] {d['arm']}: overfits, {best:.3f} peak -> {mrr:.3f} final")
    # LigandData must round-trip what the featuriser writes
    LigandData(tmp / "ligand_feats.npz")
    if failures:
        print("\nSELFTEST FAILED")
        for f in failures:
            print(f"  {f}")
        raise SystemExit(1)
    print("\nselftest passed: every arm learned the planted pairing")


# ---------------------------------------------------------------------------
def add_probe_args(p):
    p.add_argument("--arms", default="fp,fp_linear,desc,fp_desc,gnn",
                   help="comma-separated, from ligand_encoder.ARMS")
    p.add_argument("--lig_emb", default="", help="arm 'frozen': .npy of precomputed "
                   "ligand embeddings, one row per ligand")
    p.add_argument("--device", default="cpu")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--clip", type=float, default=5.0)
    p.add_argument("--eval_every", type=int, default=250)
    p.add_argument("--out_dim", type=int, default=128)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--gnn_hidden", type=int, default=128)
    p.add_argument("--gnn_layers", type=int, default=4)
    p.add_argument("--map", default="mlp", choices=["mlp", "linear"],
                   help="shape of the trainable projection on the frozen anchors")
    p.add_argument("--anchor_norm", default="none",
                   choices=["none", "center", "whiten"],
                   help="standardise the frozen anchors before the projection, "
                        "fitted on the train split. Default none, so a first run "
                        "is the baseline to compare against")
    p.add_argument("--tau", type=float, default=0.07)
    p.add_argument("--alpha", type=float, default=1.0,
                   help="target-balanced sampling exponent; 1 = uniform over groups")
    p.add_argument("--tanimoto", type=float, default=0.8,
                   help="mask pairs whose ligands are this similar; 1 disables")
    p.add_argument("--val_frac", type=float, default=0.3)
    p.add_argument("--split_seed", type=int, default=0)
    p.add_argument("--keep_seen", action="store_true",
                   help="keep proteins the checkpoint trained on (default: drop)")
    p.add_argument("--oracle", action="store_true",
                   help="also score with only pocket-matching anchors")
    p.add_argument("--save_dir", default="")
    p.add_argument("--seed", type=int, default=0)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prepare", help="CPU venv: pick receptors, write items.tsv")
    p.add_argument("--index", required=True)
    p.add_argument("--systems", default="systems")
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=1500)
    p.add_argument("--split", default="train",
                   help="comma-separated splits to draw from; '' for all")
    p.add_argument("--exclude", action="append",
                   help="items.tsv whose RECEPTORS to exclude; repeatable")
    p.add_argument("--seed", type=int, default=0)

    p = sub.add_parser("dump", help="GPU container: freeze anchors from a checkpoint")
    p.add_argument("--items", required=True)
    p.add_argument("--ckpt", required=True, help="from test_encoder.py --save_ckpt")
    p.add_argument("--out", required=True)
    p.add_argument("--other_ligands", default="", help="labels.py output directory; "
                   "a receptor's OTHER ligand sites are then ignored, not negatives")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--cutoff", type=float, default=4.0)
    p.add_argument("--limit", type=int, help="only the first N items")
    p.add_argument("--pos_cover", type=float, default=0.5)
    p.add_argument("--pos_prec", type=float, default=0.3)
    p.add_argument("--ignore_overlap", type=float, default=0.1)
    for flag, kind in [("region_method", str), ("ball_rank", str), ("nms_score", str)]:
        p.add_argument(f"--{flag}", type=kind, default=None)
    for flag in ("top_fraction", "ball_radius", "nms_iou", "nms_alpha", "persistence"):
        p.add_argument(f"--{flag}", type=float, default=None)
    for flag in ("ball_shift", "k_pos", "k_neg", "min_region_points"):
        p.add_argument(f"--{flag}", type=int, default=None)

    p = sub.add_parser("probe", help="train the ligand arms on frozen anchors")
    p.add_argument("--anchors", required=True)
    p.add_argument("--ligands", required=True, help="ligand_features.py .npz")
    p.add_argument("--out", default="")
    add_probe_args(p)

    p = sub.add_parser("selftest", help="planted signal; no data, no GPU")
    p.add_argument("--tmp", default="probe_selftest")
    p.add_argument("--min_mrr", type=float, default=0.25)
    add_probe_args(p)
    # Cheap on purpose: this is a wiring check, not a benchmark. fp's 2048-wide
    # first layer dominates on CPU, so the step count and batch are what make
    # the difference between 20 seconds and 15 minutes.
    p.set_defaults(steps=300, eval_every=50, batch=32,
                   arms="fp,fp_linear,desc,fp_desc,gnn", val_frac=0.3, oracle=True)

    a = ap.parse_args()
    {"prepare": prepare, "dump": dump, "probe": probe, "selftest": selftest}[a.cmd](a)


if __name__ == "__main__":
    main()
