"""Contrastive protein-ligand training over the surface encoder's region anchors.

What is being learned. A protein becomes K region anchors (surface_encoder.py
steps 1-6) and a ligand becomes one vector (ligand_encoder.py). The score of a
pair is the best match between the protein's PREDICTED pocket anchors and the
ligand:

    s(P, L) = max over P's pocket anchors a of cos(a, L) / tau

which is a multiple-instance score, settled because anchor recall is far above
top-1 for ball_R8 (hit 0.84 against top1 0.64). A batch of B pairs gives a
B x B matrix whose diagonal is the true pairs, and symmetric InfoNCE over it is
the loss. The selector's negative patches are never scored, here or at
inference.

    loss = InfoNCE(scores)
         + lam_pocket * per-point pocket BCE       keeps the head, and so the
                                                   regions, from drifting
         + lam_site   * within-protein anchor CE   optional: WHICH anchor, the
                                                   known weak spot

Three things make the batch honest rather than easy:

  false negatives leave the denominator. A pair sharing a target group
    (UniProt, falling back to PLINDER's 95% interaction component), the same
    ligand, or ECFP4 Tanimoto >= --tanimoto is masked to -inf rather than
    trained against. dataset_v1 has 747 PDE10A rows; without this the loss
    would mostly be punishing correct answers.
  target-balanced sampling. One row per target group per batch, groups drawn
    with probability count^(1-alpha), which is the group-first form of a
    count^-alpha row weight. alpha 1 is uniform over targets.
  group-disjoint validation. A random row split would put PDE10A on both sides
    and the metric would read as memorisation.

All four are probe_retrieval's functions, imported rather than re-derived, so
the frozen-anchor probe and this loop agree by construction on what a batch, a
split, a mask and a metric are.

The pocket term is not optional in spirit. Anchors come from the predicted
pocket map, and region selection has no gradient, so if the head degrades the
regions degrade and the contrastive loss loses the places it was scoring. When
no predicted region matches the labelled pocket, the step falls back to a
selector-shaped ball at the pocket centroid (train_data.fallback_region) so it
still has a positive; the RATE of that is reported every evaluation and is the
honest measure of how much the loop is leaning on labels it will not have at
inference.

External pockets. `--region_source external --pockets DIR` takes the pocket
regions from an outside pocket finder (fpocket, via external_pockets.py)
instead of the head's map: its top k_pos pockets are the anchors, pooled as
plain means, and the head only matters through its BCE. `external_map` feeds
the finder's per-point map to the usual selector instead. The default,
`predicted`, is the original loop exactly. src/slurm/train_fpocket.sbatch.
`--pocket_select site` makes the one pocket region of each row the fpocket
pocket at that row's ligand (known site), and `--pool_mode pairs` pools it as
site types x distance bins (pair_pool.py): src/slurm/train_v3.sbatch.
`--n_decoys N` (site only) adds up to N of the receptor's OTHER fpocket
pockets as decoy anchors: never scored against other proteins' ligands, but
trained against this row's ligand by the lam_site term, which is the choice a
blind (topk) screen has to make. Same protein, so no decoy can be a homologue
of the positive (site_decoys).

Temperature. The learned logit scale is capped at 1 / --tau_min (default
0.01, CLIP's cap). A low temperature puts the gradient on the top-scoring
negatives, which with a bank of thousands are the likeliest false negatives;
the current value is in every evaluation line and in history.json.

Stages:
    prepare    CPU venv. index.parquet -> out/runs/<name>/items.tsv.
    cache      container, GPU. Prefill the surface cache. Optional - training
               fills it lazily - but shardable across array tasks, and the
               first epoch is otherwise dominated by surface sampling.
    train      container, GPU. The loop, with checkpoints, resume and a
               walltime budget.
    eval       container, GPU. Validation metrics from a checkpoint, nothing
               trained.
    selftest   no data, no GPU, no KeOps. A stub for the dMaSIF parts only;
               region selection, labels, the fallback, pooling, the loss, the
               sampler, the metrics and checkpoint round-tripping are the real
               code. See selftest() for what each assertion protects.

Example:
    # CPU venv
    python src/train.py prepare --index data/datasets/dataset_v1/index.parquet \\
        --out out/runs/v1 --split train
    python src/ligand_features.py --index data/datasets/dataset_v1/index.parquet \\
        --out out/runs/v1
    # container (see src/slurm/train.sbatch)
    python src/train.py train --items out/runs/v1/items.tsv \\
        --lig_feats out/runs/v1/ligand_feats.npz --out out/runs/v1 \\
        --head_ckpt out/encoder_test/head_pretrained_300.pt \\
        --region_method ball --ball_radius 8 --ball_shift 2 \\
        --surface_cache data/surfaces/ --steps 20000 --max_hours 11

Depends on: numpy, torch; train_data, probe_retrieval, surface_encoder,
            ligand_encoder, anchors, test_encoder, ids; external_pockets
            (only with an external --region_source).
"""

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


# ---------------------------------------------------------------------------
# the model: the two encoders and the temperature
# ---------------------------------------------------------------------------
def build_model(a, device):
    import torch
    import torch.nn as nn

    from ligand_encoder import LigandConfig, LigandEncoder
    from surface_encoder import EncoderConfig, SurfaceEncoder

    torch.manual_seed(a.seed)
    enc_cfg = EncoderConfig(
        chemo=a.chemo, k_pos=a.k_pos, k_neg=a.k_neg,
        # getattr / `or`: checkpoints from before external pockets have no
        # such arg, and eval's region flags default to None
        region_source=getattr(a, "region_source", None) or "predicted",
        region_method=a.region_method, region_top_fraction=(
            None if a.top_fraction <= 0 else a.top_fraction),
        region_threshold=a.threshold, region_ball_radius=a.ball_radius,
        region_ball_shift=a.ball_shift, region_ball_rank=a.ball_rank,
        region_persistence=a.persistence, region_nms_iou=a.nms_iou,
        region_nms_alpha=a.nms_alpha, region_nms_score=a.nms_score,
        pool_mode=a.pool_mode, n_shells=a.n_shells, anchor_norm=a.anchor_norm,
        # getattr: checkpoints from before pair pooling have no such args
        pair_types=getattr(a, "pair_types", 8), pair_bins=getattr(a, "pair_bins", 8),
        pair_rmin=getattr(a, "pair_rmin", 3.0), pair_rmax=getattr(a, "pair_rmax", 16.0),
        pair_exclude=getattr(a, "pair_exclude", 2.0),
        pair_feats=getattr(a, "pair_feats", True),
        # getattr: checkpoints saved before the descriptor branch existed have
        # no such args, and load_model_from_ckpt rebuilds from saved args
        desc_dims=getattr(a, "desc_dims", 0), desc_layers=getattr(a, "desc_layers", 3),
        out_dim=a.out_dim, dropout=a.enc_dropout,
        resolution=a.resolution, sup_sampling=a.sup_sampling)
    lig_cfg = LigandConfig(arm=a.arm, out_dim=a.out_dim, hidden=a.hidden,
                           dropout=a.lig_dropout, gnn_hidden=a.gnn_hidden,
                           gnn_layers=a.gnn_layers)

    # getattr: checkpoints from before --tau_min existed were capped at 100
    tau_min = getattr(a, "tau_min", 0.01)

    class RetrievalModel(nn.Module):
        """Both encoders plus the learnable temperature, as one module so a
        checkpoint, an optimiser and a .to(device) each take one object."""

        def __init__(self):
            super().__init__()
            self.encoder = SurfaceEncoder(enc_cfg)
            self.ligand = LigandEncoder(lig_cfg)
            self.logit_scale = nn.Parameter(torch.log(torch.tensor(1.0 / a.tau)))

        def scale(self):
            # Clamped like CLIP's: an unbounded temperature can run away early,
            # when every pair looks alike and sharpening is the cheapest way to
            # reduce the loss. The cap is 1 / --tau_min.
            return self.logit_scale.exp().clamp(max=1.0 / tau_min)

        def scores(self, anchors, usable, lig_vec):
            """(B, K, D) anchors x (L, D) ligands -> (B, L), max over anchors."""
            sim = torch.einsum("bkd,ld->bkl", anchors, lig_vec)
            sim = sim.masked_fill(~usable[..., None], -1e4)
            return sim.max(dim=1).values * self.scale()

    model = RetrievalModel().to(device)
    if a.tau < tau_min:
        print(f"[warn] --tau {a.tau} is below --tau_min {tau_min}: the "
              f"temperature starts at the floor")
    print(f"encoder: {enc_cfg.in_channels} input channels -> {enc_cfg.emb_dims} dims "
          + (f"+ {enc_cfg.desc_dims} descriptor ({enc_cfg.desc_layers} layers) "
             if enc_cfg.desc_dims else "")
          + f"-> {enc_cfg.pooled_dims} pooled ({enc_cfg.pool_mode}) -> {a.out_dim}; "
          f"anchor_norm {enc_cfg.anchor_norm}")
    print(f"surface sampler: resolution {enc_cfg.resolution}, sup_sampling "
          f"{enc_cfg.sup_sampling}"
          + ("" if enc_cfg.sup_sampling == 20 and enc_cfg.resolution == 1.0 else
             "   <- NOT the dMaSIF default (1.0 / 20). The surfaces this writes "
             "to the cache are coarser than a real run's, so keep them in a "
             "cache directory of their own."))
    n_enc = sum(p.numel() for p in model.encoder.parameters())
    print(f"parameters: encoder {n_enc:,}, ligand ({a.arm}) "
          f"{model.ligand.n_params():,}")
    if a.pretrained:
        print(f"loaded {len(model.encoder.load_pretrained())} pretrained dMaSIF tensors")
    if a.head_ckpt:
        load_head(model.encoder, a.head_ckpt)
    return model, enc_cfg, lig_cfg


def pocket_source(a, cfg):
    """external_pockets.PocketSource for an external region_source, else None.

    Region SHAPE settings come from the encoder config (ball radius, minimum
    region size), so an external ball is the same object as a ball_R8 region
    and the fallback anchor. Mapping only the top k_pos pockets is enough for
    `external` (select_regions takes no more); `external_map` maps them all
    unless --pocket_max caps it.
    """
    if cfg.region_source == "predicted":
        if getattr(a, "pockets", None):
            print(f"[warn] --pockets {a.pockets} ignored: --region_source is "
                  "predicted, so the regions come from the pocket head")
        return None
    if not getattr(a, "pockets", None):
        raise SystemExit(f"--region_source {cfg.region_source} needs --pockets, a "
                         "directory from `external_pockets.py run`")
    from external_pockets import PocketSource

    select = getattr(a, "pocket_select", None) or "topk"
    if select == "site" and cfg.region_source != "external":
        raise SystemExit("--pocket_select site picks THE pocket region per row, "
                         "which needs --region_source external")
    k = a.pocket_max or (cfg.k_pos if cfg.region_source == "external" else None)
    return PocketSource(a.pockets, k=k, rank=a.pocket_rank, margin=a.pocket_margin,
                        shape=a.pocket_shape, ball_radius=cfg.region_ball_radius,
                        min_points=cfg.min_region_points, select=select,
                        site_cutoff=a.site_cutoff, site_frac=a.site_frac,
                        site_lig_frac=a.site_lig_frac)


def load_head(enc, path):
    """Start from a trained pocket head (test_encoder.py --save_ckpt).

    Shape-compatible tensors only, reported by name count. The projection is
    deliberately allowed to mismatch: pool_mode or anchor_norm changes its
    input width, and in any case it has only ever seen pocket BCE, so its
    weights are effectively random and there is nothing there to keep. The
    POINT of this is the head and the conv, which reach AUROC 0.937 and
    therefore hand the contrastive loss usable regions from step 0 instead of
    noise.
    """
    import torch

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    # Zero-pad the three input-side tensors when this encoder has more input
    # channels (--chemo against a 16-channel head), exactly as load_pretrained
    # does, so the head's function is preserved instead of those tensors being
    # skipped and left random under a pretrained conv. A no-op otherwise.
    state = ckpt["state_dict"]
    if hasattr(enc, "_inflate_inputs"):
        try:
            state = enc._inflate_inputs(state)
        except RuntimeError as e:    # narrower than the head: left to the filter
            print(f"  not inflating: {e}")
    have = enc.state_dict()
    take = {k: v for k, v in state.items()
            if k in have and tuple(have[k].shape) == tuple(v.shape)}
    skipped = sorted(set(state) - set(take))
    enc.load_state_dict(take, strict=False)
    print(f"head checkpoint {path}: loaded {len(take)}/{len(state)} tensors"
          + (f"; shape-incompatible, left at init: {len(skipped)} "
             f"({', '.join(skipped[:4])}{', ...' if len(skipped) > 4 else ''})"
             if skipped else ""))
    n_train = len(ckpt.get("train_ids", []))
    if n_train:
        print(f"  that head trained on {n_train} items; "
              f"they are NOT excluded here - see --exclude in prepare")
    return take


# ---------------------------------------------------------------------------
# one batch of proteins: surfaces, labels, anchors
# ---------------------------------------------------------------------------
def site_decoys(pockets, site, lab, n, ignore_overlap=0.1):
    """The receptor's other external pockets, as this row's decoys -> list.

    `pockets` are the receptor's mapped pockets (PocketSource.map: its top
    --pocket_max, else k_pos, by --pocket_rank), best first; `site` is this
    row's site region list (empty when fpocket found nothing there). A pocket
    is dropped when region_labels would not call it a clean negative against
    the 4 A label and the sibling ligands' sites (cover or prec >= 0.1 with
    either), or when it overlaps the site region itself the same way - the
    site is a union of fpocket pockets, so its own members are in `pockets`
    too. The first n survivors, in rank order: the pockets a blind screen
    would rank highest are the hardest decoys.

    Not a homology question at all: a decoy is on the SAME protein, so the
    only way it binds this ligand is a second site no structure shows.
    """
    import torch

    from anchors import region_overlap
    from train_data import region_labels

    cand = [torch.as_tensor(r).long() for r in pockets]
    if not n or not cand:
        return []
    lab_c, _, _ = region_labels(cand, lab["pocket"], lab["other"],
                                ignore_overlap=ignore_overlap)
    keep = lab_c == 0
    if site:
        s = torch.zeros(len(lab["pocket"]), dtype=torch.bool)
        for r in site:
            s[torch.as_tensor(r).long()] = True
        prec, cover = region_overlap(cand, s)
        keep &= (prec < ignore_overlap) & (cover < ignore_overlap)
    return [cand[i] for i in keep.nonzero().flatten().tolist()][:n]


def build_batch(model, store, labels, table, rows, device, n_decoys=0):
    """-> (P, labs) for these row indices. Surfaces come from the store.

    n_decoys (--pocket_select site only): append up to this many of the
    receptor's other pockets after the site (site_decoys). Each lab then
    carries n_site (0 or 1) and n_decoy, which is how anchor_block tells the
    site slot from the decoy slots.
    """
    from test_encoder import to_batch

    import torch

    src = getattr(store, "pockets", None)
    site = src is not None and src.select == "site"
    items, labs = [], []
    for i in rows:
        entry = store.get(model.encoder, table.rows[i])
        lab = labels.get(int(i), entry)
        if site:
            # --pocket_select site: THE pocket region is the fpocket pocket at
            # this row's own ligand. Per row, not per surface (two ligands on
            # one receptor are two sites), so a shallow copy: the RAM entry is
            # shared by every row of the receptor. So is the label record.
            entry = dict(entry)
            found = [torch.from_numpy(r) for r in src.site(
                entry["xyz"].numpy(), entry.get("ext_raw"), lab["lig"])]
            decoys = site_decoys(entry["ext_regions"], found, lab, n_decoys)
            entry["ext_regions"] = found + decoys
            lab = dict(lab, n_site=len(found), n_decoy=len(decoys))
        labs.append(lab)
        items.append(entry)
    return to_batch(items, device), labs


def anchor_block(model, P, labs, a, generator=None):
    """Forward the proteins and assemble every anchor the loss can use.

    Returns a dict with, padded to a common slot count:
        anchors  (B, K', D)   normalised, the usable slots only
        usable   (B, K')      predicted pocket anchors, which is what a score
                              may be taken over - at training and at inference
        label    (B, K')      1 positive / 0 negative / -1 ignore per region
        decoy    (B, K')      site decoys (build_batch n_decoys): label 0,
                              never usable, so only lam_site sees them
        fallback (B,)         True where the positive is the fallback ball
        out                   the raw encoder output, for the pocket loss
    """
    import torch

    from surface_encoder import pool_region_vectors
    from train_data import fallback_region, region_labels

    cfg = model.encoder.cfg
    # project=False: the fallback anchor below has to go through the same
    # normalisation as the rest, in one call, so the tail is run here.
    out = model.encoder(P, generator=generator, project=False)
    pooled, valid = out["anchor_pooled"], out["anchor_valid"]
    is_pos = out["anchor_is_pos"]
    B, K = valid.shape
    # The weights the encoder pooled its own regions with - the pocket
    # probability, or ones under an external region_source - so the fallback
    # anchor is pooled the same way. .get for the selftest's stub encoder.
    prob = out.get("pool_weight")
    if prob is None:
        prob = torch.sigmoid(out["pocket_logit"])

    label = torch.zeros(B, K, dtype=torch.long, device=valid.device)
    decoy = torch.zeros(B, K, dtype=torch.bool, device=valid.device)
    extra_vec, extra_at = [], []
    site = (cfg.region_source == "external"
            and (getattr(a, "pocket_select", None) or "topk") == "site")
    for b, lb in enumerate(labs):
        sel = (out["batch"] == b).nonzero().squeeze(1)
        base = int(sel[0])
        regions = [(r - base).cpu() for r in out["anchor_regions"][b]]
        lab, _, _ = region_labels(regions, lb["pocket"], lb["other"],
                                  a.pos_cover, a.pos_prec, a.ignore_overlap)
        if len(lab):
            label[b, :len(lab)] = lab.to(valid.device)
        # A negative patch is never scored, so its label cannot be read as a
        # trainable negative; mark it ignore.
        label[b, ~is_pos[b]] = -1
        # .get: rows built without the site select carry neither count, and
        # a site row from before decoys always had its site in slot 0
        n_site = lb.get("n_site", 1)
        if site and n_site and bool(is_pos[b, 0] & valid[b, 0]):
            # --pocket_select site: the one pocket region IS the ligand's site
            # by construction, whatever its overlap with the 4 A label - a
            # merged fpocket site can be larger than the label and fail the
            # precision rule, and would then wrongly trigger the fallback
            label[b, 0] = 1
        n_dec = lb.get("n_decoy", 0) if site else 0
        if n_dec:
            # The decoys follow the site (none when fpocket missed it, and
            # then slot 0 is a decoy, not the site). site_decoys kept only
            # clean negatives; a different --pos_cover / --ignore_overlap can
            # still relabel one, and a decoy must never count as a positive.
            d = slice(n_site, n_site + n_dec)
            decoy[b, d] = is_pos[b, d] & valid[b, d]
            label[b] = torch.where(decoy[b] & (label[b] == 1), -1, label[b])

        pos_here = bool(((label[b] == 1) & valid[b] & is_pos[b] & ~decoy[b]).any())
        if not pos_here and a.fallback and bool(lb["pocket"].any()):
            # The label masks live on the CPU with the cached surface; this
            # indexes the encoder's own xyz, which is on the device, and a
            # boolean mask has to be on the tensor's device to index it.
            pk = lb["pocket"].to(out["xyz"].device)
            reg = fallback_region(out["xyz"][sel], pk, cfg.region_ball_radius)
            if len(reg) >= cfg.min_region_points:
                # pool_emb, not point_emb: the fallback anchor must be pooled
                # from the same features as every selector anchor, descriptor
                # branch included. .get for the selftest's stub encoder.
                feats = out.get("input_feats")
                extra_vec.append(pool_region_vectors(
                    cfg, getattr(model.encoder, "pair_pool", None),
                    out.get("pool_emb", out["point_emb"])[sel],
                    None if feats is None else feats[sel], prob[sel],
                    [reg.to(sel.device)], out["xyz"][sel])[0])
                extra_at.append(b)

    fallback = torch.zeros(B, dtype=torch.bool, device=valid.device)
    if extra_vec:
        at = torch.as_tensor(extra_at, device=valid.device)
        fallback[at] = True
        # One more slot, assembled as a fresh tensor and concatenated, rather
        # than written into the encoder's own output in place.
        slot = pooled.new_zeros(B, 1, pooled.shape[2])
        slot[at, 0] = torch.stack(extra_vec)
        pooled = torch.cat([pooled, slot], dim=1)
        col = lambda v, dt: torch.full((B, 1), v, dtype=dt, device=valid.device)
        v1, p1, l1 = col(False, torch.bool), col(False, torch.bool), col(-1, torch.long)
        v1[at, 0] = p1[at, 0] = True
        l1[at, 0] = 1
        valid = torch.cat([valid, v1], dim=1)
        is_pos = torch.cat([is_pos, p1], dim=1)
        label = torch.cat([label, l1], dim=1)
        decoy = torch.cat([decoy, torch.zeros_like(v1)], dim=1)
    # `valid`, not `usable`: the normalisation sees every real anchor, pocket
    # and negative alike, in training and in evaluation, so its statistics mean
    # the same thing in both.
    anchors = model.encoder.project_anchors(pooled, valid)
    return {"anchors": anchors, "usable": valid & is_pos & ~decoy, "label": label,
            "decoy": decoy, "valid": valid, "is_pos": is_pos, "fallback": fallback,
            "out": out, "pooled": pooled}


def pair_mask(table, rows, ligands, device, tanimoto):
    """False-negative mask for an all-vs-all block of these rows."""
    import torch

    from probe_retrieval import mask_pairs

    fp_rows = table.fp_row[rows]
    g = torch.as_tensor(table.group_id[rows], device=device)
    fr = torch.as_tensor(fp_rows, device=device)
    fps = torch.as_tensor(ligands.fp[fp_rows].astype(np.float32), device=device)
    return mask_pairs(g, g, fr, fr, fps, fps, tanimoto)


def cross_mask(table, rows_q, rows_c, ligands, device, tanimoto):
    """False-negative mask between two DIFFERENT row sets -> (Q, C) bool.

    The same rules as pair_mask (same target group, same ligand, Tanimoto >=
    `tanimoto`), but with no diagonal to protect. Not mask_pairs: that clears
    the diagonal whenever Q == C, which between two unrelated row sets would
    un-mask a real same-group pair whenever the sizes happen to match.
    """
    import torch

    from probe_retrieval import tanimoto as tanimoto_matrix

    gq = torch.as_tensor(table.group_id[rows_q], device=device)
    gc = torch.as_tensor(table.group_id[rows_c], device=device)
    fq = torch.as_tensor(table.fp_row[rows_q], device=device)
    fc = torch.as_tensor(table.fp_row[rows_c], device=device)
    m = (gq[:, None] == gc[None, :]) | (fq[:, None] == fc[None, :])
    if tanimoto is not None and tanimoto < 1.0:
        fpq = torch.as_tensor(ligands.fp[table.fp_row[rows_q]].astype(np.float32),
                              device=device)
        fpc = torch.as_tensor(ligands.fp[table.fp_row[rows_c]].astype(np.float32),
                              device=device)
        m |= tanimoto_matrix(fpq, fpc) >= tanimoto
    return m


def nce_extended(s, mask, s_bank=None, m_bank=None, s_lig=None, m_lig=None):
    """Symmetric InfoNCE with extra candidates on either side.

    s (B, B)          batch proteins x batch ligands, true pairs on the diagonal
    s_bank (Q, B)     memory-bank proteins x batch ligands: more PROTEIN
                      candidates for each ligand query (ligand -> protein, the
                      direction validation measures)
    s_lig (B, N)      batch proteins x extra ligands: more LIGAND candidates
                      for each protein query
    Masked entries leave the denominator, exactly as in info_nce. With neither
    extra this IS info_nce: the same two cross-entropies over the same
    entries.
    """
    import torch
    import torch.nn.functional as F

    t = torch.arange(len(s), device=s.device)
    p2l, mp = s, mask
    if s_lig is not None:
        p2l, mp = torch.cat([s, s_lig], 1), torch.cat([mask, m_lig], 1)
    l2p, ml = s.T, mask.T
    if s_bank is not None:
        l2p, ml = torch.cat([s.T, s_bank.T], 1), torch.cat([mask.T, m_bank.T], 1)
    return 0.5 * (F.cross_entropy(p2l.masked_fill(mp, float("-inf")), t)
                  + F.cross_entropy(l2p.masked_fill(ml, float("-inf")), t))


class ProteinBank:
    """The last `size` training proteins' pooled anchors, as extra negatives.

    Proteins are the expensive side - a surface, a conv and region selection
    each - and the batch IS the negative set, so batch 32 gave every ligand 31
    wrong proteins to beat. This keeps the POOLED vectors (before the
    normalisation and projection) of recent batches, detached, and re-projects
    them with the CURRENT projection at every step. So a bank entry is stale
    only in the conv, which moves slowly; the projection, which the
    contrastive loss moves fastest, is always current. MoCo's queue, minus the
    momentum encoder: the staleness here is a few dozen steps of a 30k-parameter
    conv.

    No gradient flows into the bank. It is not checkpointed: a resumed run
    starts it empty and refills it in size / batch steps.
    """

    def __init__(self, size, k, dim, device):
        import torch

        self.size = size
        self.pooled = torch.zeros(size, k, dim, device=device)
        self.valid = torch.zeros(size, k, dtype=torch.bool, device=device)
        self.usable = torch.zeros(size, k, dtype=torch.bool, device=device)
        self.rows = np.full(size, -1, dtype=np.int64)
        self.ptr = self.n = 0

    def push(self, pooled, valid, usable, rows):
        import torch

        m, k = len(rows), pooled.shape[1]
        idx = torch.as_tensor((self.ptr + np.arange(m)) % self.size,
                              device=self.pooled.device)
        # Zero the whole slot first: this batch may have one slot fewer than
        # the bank (no fallback anchor this step), and the old tail must not
        # survive as a phantom anchor.
        self.pooled[idx] = 0
        self.valid[idx] = False
        self.usable[idx] = False
        self.pooled[idx, :k] = pooled.detach()
        self.valid[idx, :k] = valid
        self.usable[idx, :k] = usable
        self.rows[idx.cpu().numpy()] = rows
        self.ptr = (self.ptr + m) % self.size
        self.n = min(self.n + m, self.size)

    def anchors(self, model):
        """-> (anchors (n, k, D), usable (n, k), rows (n,)), or None if empty.

        The normalisation runs in eval mode here: a BatchNorm in training mode
        would update its running statistics from stale vectors, and the batch
        statistics of the bank are not the batch's.
        """
        import torch

        if not self.n:
            return None
        enc = model.encoder
        was = enc.pool_norm.training
        enc.pool_norm.eval()
        try:
            with torch.no_grad():
                anchors = enc.project_anchors(self.pooled[:self.n],
                                              self.valid[:self.n])
        finally:
            enc.pool_norm.train(was)
        return anchors, self.usable[:self.n], self.rows[:self.n]


def lr_at(a, step):
    """Learning rate at `step`: linear warmup, then constant or cosine.

    A function of the step alone, so a resumed run lands on exactly the rate an
    uninterrupted one would have had - nothing to checkpoint. The cosine runs
    over --steps, the run's TOTAL target, not over one job of a chain.
    """
    import math

    if a.warmup and step < a.warmup:
        return a.lr * (step + 1) / a.warmup
    if a.lr_schedule == "const":
        return a.lr
    prog = min(1.0, (step - a.warmup) / max(1, a.steps - a.warmup))
    return a.lr * (a.min_lr_frac + (1 - a.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * prog)))


def site_loss(scores_per_anchor, label):
    """Within-protein: is the ligand's best anchor the one over its own pocket?

    -log( sum over positive anchors exp(s) / sum over positive + negative
    exp(s) ), with ignores left out entirely. Multi-positive because ball_R8
    makes several near-duplicate regions of one pocket and any of them is a
    correct answer.

    Separate from InfoNCE because InfoNCE only ever sees the max over anchors,
    so it can be satisfied by a protein whose best-matching anchor is in the
    wrong place. This term is the one that says WHERE, and top1 0.64 against
    hit 0.86 is the gap it addresses.

    The negatives here are the selector's OTHER predicted pocket regions, not
    its negative patches: anchor_block marks those ignore, because nothing
    scores them at inference and a loss over them would be free marks. Under
    --pocket_select site the negatives are the site decoys (--n_decoys), the
    receptor's other fpocket pockets, which a blind topk screen does score.
    """
    import torch

    pos = label == 1
    neg = label == 0
    rows = pos.any(1) & (pos | neg).any(1)
    if not bool(rows.any()):
        return scores_per_anchor.sum() * 0.0
    s = scores_per_anchor[rows]
    p, n = pos[rows], neg[rows]
    s_all = s.masked_fill(~(p | n), float("-inf"))
    s_pos = s.masked_fill(~p, float("-inf"))
    return -(torch.logsumexp(s_pos, 1) - torch.logsumexp(s_all, 1)).mean()


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------
def collect(model, store, labels, table, rows, a, device):
    """Run the proteins once, no grad -> stacked anchors and per-point metrics.

    Chunked, and the anchors are kept rather than the surfaces, so a validation
    set of a few thousand proteins costs K x out_dim floats each.
    """
    import torch

    from test_encoder import auroc, average_precision

    model.eval()
    anchors, usable, label, pooled = [], [], [], []
    aucs, aps, n_fallback, kept = [], [], 0, []
    # Its own generator, re-seeded per call: negative-region seeding is the one
    # random draw inside selection, and taking it from the global RNG would
    # make the training stream depend on how often validation ran.
    gen = torch.Generator().manual_seed(a.seed)
    with torch.no_grad():
        for s in range(0, len(rows), a.eval_batch):
            chunk = rows[s:s + a.eval_batch]
            P, labs = build_batch(model, store, labels, table, chunk, device,
                                  getattr(a, "n_decoys", 0) or 0)
            blk = anchor_block(model, P, labs, a, generator=gen)
            for b, lb in enumerate(labs):
                m = blk["out"]["batch"] == b
                lg = blk["out"]["pocket_logit"][m].cpu()
                aucs.append(auroc(lg, lb["pocket"]))
                aps.append(average_precision(lg, lb["pocket"]))
            n_fallback += int(blk["fallback"].sum())
            anchors.append(blk["anchors"].cpu())
            usable.append(blk["usable"].cpu())
            label.append(blk["label"].cpu())
            pooled.append(blk["pooled"].cpu())
            kept.append(np.asarray(chunk))
    model.train()

    # Chunks can differ in slot count: a chunk where some protein needed the
    # fallback anchor has one more. Pad to the widest, with -1 for labels so a
    # padded slot reads as "ignore" rather than as a negative.
    K = max(t.shape[1] for t in anchors)

    def pad(t, fill=0):
        if t.shape[1] == K:
            return t
        tail = torch.full((t.shape[0], K - t.shape[1], *t.shape[2:]), fill,
                          dtype=t.dtype)
        return torch.cat([t, tail], dim=1)

    return {"rows": np.concatenate(kept),
            "anchors": torch.cat([pad(t) for t in anchors]),
            "usable": torch.cat([pad(t, False) for t in usable]),
            "pooled": torch.cat([pad(t) for t in pooled]),
            "label": torch.cat([pad(t, -1) for t in label]),
            "auroc": float(np.nanmean(aucs)), "ap": float(np.nanmean(aps)),
            "fallback_rate": n_fallback / max(len(rows), 1)}


def site_rate(sim, label, usable, site_mode):
    """The `site` metric -> (rate, coverage): is the ligand's best anchor over
    its own pocket? sim (B, K) ligand x own anchors, label as anchor_block's.

    Selector / topk: the argmax over the scored (usable) anchors, over the
    proteins with a positive anchor anywhere. Site select: only the site is
    usable, so that would read 1.0 by construction; instead the argmax over
    the site AND its decoys (every label >= 0 slot), over proteins that have
    both - how often the ligand would pick its own pocket out of the
    receptor's others, which is what a blind screen asks. NaN without decoys.
    """
    import torch

    has_pos = (label == 1).any(1)
    if site_mode:
        cand = label >= 0
        rows = has_pos & (label == 0).any(1)
    else:
        cand, rows = usable, has_pos
    best = sim.masked_fill(~cand, -1e4).argmax(1)
    lb = label.gather(1, best[:, None]).squeeze(1)
    rate = (float((lb[rows] == 1).float().mean()) if bool(rows.any())
            else float("nan"))
    return rate, float(rows.float().mean())


def evaluate(model, store, labels, table, rows, ligands, a, device):
    """Retrieval + pocket + anchor-spread metrics over `rows`."""
    import torch

    from probe_retrieval import anchor_spread, retrieval_metrics

    got = collect(model, store, labels, table, rows, a, device)
    rows = got["rows"]
    anchors = got["anchors"].to(device)
    usable = got["usable"].to(device)
    keep = usable.any(1)
    if int(keep.sum()) < 2:
        # Fewer than two scorable proteins: there is no ranking to measure, and
        # every retrieval key has to exist anyway or the caller's table breaks.
        nan = float("nan")
        met = {k: nan for k in ("mrr", "chance", "top1", "top5", "top10",
                                "median_rank", "cands", "p2l_mrr", "p2l_top1",
                                "site", "site_cov", "cos_diff", "shared",
                                "eff_dims")}
        met.update(n=int(keep.sum()), auroc=got["auroc"], ap=got["ap"],
                   fallback_rate=got["fallback_rate"],
                   tau=1.0 / float(model.scale().detach()))
        return met
    kn = keep.cpu().numpy()
    rows, anchors, usable = rows[kn], anchors[keep], usable[keep]
    # collect() hands everything back on the CPU; `keep` was built from the
    # device copy, and a CUDA mask cannot index a CPU tensor.
    label = got["label"][keep.cpu()]

    model.eval()
    with torch.no_grad():
        lig = encode_ligands(model, ligands, table.fp_row[rows], device)
        scores = model.scores(anchors, usable, lig)
        mask = pair_mask(table, rows, ligands, device, a.tanimoto)
        met = retrieval_metrics(scores.T.cpu(), mask.cpu())      # ligand -> protein
        p2l = retrieval_metrics(scores.cpu(), mask.cpu())        # protein -> ligand
        met["p2l_mrr"], met["p2l_top1"] = p2l["mrr"], p2l["top1"]

        met["site"], met["site_cov"] = site_rate(
            torch.einsum("bkd,bd->bk", anchors, lig), label.to(device), usable,
            (getattr(a, "pocket_select", None) or "topk") == "site")

        # The 0.993 readout: are different proteins' anchors still one cone?
        # On the TOP-ranked pocket anchor, which is the vector apo_eval's
        # reference cosine uses, so the two numbers are comparable.
        pooled_k = got["pooled"][keep.cpu()]
        u_cpu = usable.cpu()
        top = [pooled_k[i, int(u_cpu[i].nonzero()[0])]
               for i in range(len(pooled_k)) if bool(u_cpu[i].any())]
        if len(top) > 2:
            sp = anchor_spread(torch.stack(top))
            met["cos_diff"] = sp["median_cos"]
            met["shared"] = sp["shared"]
            met["eff_dims"] = sp["eff_dims"]
    model.train()
    # The temperature the scores were taken at: 1 / the clamped logit scale,
    # so a run sitting at --tau_min reads exactly tau_min.
    met.update(auroc=got["auroc"], ap=got["ap"],
               fallback_rate=got["fallback_rate"], tau=1.0 / float(model.scale().detach()))
    return met


def encode_ligands(model, ligands, fp_rows, device):
    return model.ligand(ligands.collate(fp_rows, device))


# ---------------------------------------------------------------------------
# checkpoints
# ---------------------------------------------------------------------------
def save_ckpt(path, model, opt, step, split, a, enc_cfg, lig_cfg, history, counts):
    import torch

    from ligand_encoder import config_dict as lig_dict
    from surface_encoder import config_dict as enc_dict

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = {"step": step, "state_dict": model.state_dict(),
            "optimizer": opt.state_dict(),
            "encoder_config": enc_dict(enc_cfg), "ligand_config": lig_dict(lig_cfg),
            "args": vars(a), "history": history, "pos_neg": counts,
            "train_rows": split[0].tolist(), "val_rows": split[1].tolist(),
            "rng": {"torch": torch.get_rng_state(),
                    "numpy": np.random.get_state()}}
    if torch.cuda.is_available():
        blob["rng"]["cuda"] = torch.cuda.get_rng_state_all()
    tmp = path.with_suffix(".pt.tmp")
    torch.save(blob, tmp)
    tmp.replace(path)
    return path


def load_model_from_ckpt(path, device, overrides=None):
    """Rebuild encoder + ligand encoder exactly as a checkpoint had them."""
    import torch

    from probe_retrieval import apply_overrides

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    saved = argparse.Namespace(**ckpt["args"])
    # The checkpoint's own weights are about to replace them, so do not spend a
    # load on the pretrained or head starting points.
    saved.pretrained, saved.head_ckpt = False, ""
    model, enc_cfg, lig_cfg = build_model(saved, device)
    model.load_state_dict(ckpt["state_dict"])
    if overrides is not None:
        apply_overrides(model.encoder.cfg, overrides)
    return model, ckpt, saved


# ---------------------------------------------------------------------------
# rows: the table both stages see, and the validation subsample
# ---------------------------------------------------------------------------
def use_neighbor_device(a, device):
    """--gpu_neighbors: build region selection's neighbour lists on the GPU."""
    import anchors

    if getattr(a, "gpu_neighbors", False) and str(device).startswith("cuda"):
        anchors.NEIGHBOR_DEVICE = device


def load_rows(a):
    """-> (table, ligands), rows whose ligand failed featurisation dropped.

    One function for train and eval: row indices in a checkpoint are positions
    in THIS filtered table, so eval has to drop exactly the rows train dropped
    or every saved index points one row off after the first failure.
    """
    from ligand_encoder import LigandData
    from train_data import ItemTable, read_items

    table = ItemTable(read_items(a.items, a.limit))
    print(f"{a.items}: {table.summary()}")
    ligands = LigandData(a.lig_feats)
    if table.fp_row.max() >= len(ligands):
        raise SystemExit(
            f"fp_row up to {table.fp_row.max()} but {a.lig_feats} holds "
            f"{len(ligands)} ligands; items and features are different datasets")
    bad = ligands.failed[table.fp_row]
    if bad.any():
        print(f"dropping {int(bad.sum())} rows whose ligand could not be featurised")
        table = ItemTable([r for r, b in zip(table.rows, bad) if not b])
    return table, ligands


def subsample_val(rows, groups, n, seed, balance=False):
    """`n` of these validation rows.

    Without `balance`, uniform over ROWS - the original draw, kept bit-for-bit
    so out/runs/v1 reproduces. Its flaw: the val pool is a few big targets and many
    small ones, so a row-uniform 400 from out/runs/v1's ~2,070-row, ~283-group pool
    covered only 112 groups and the metric was dominated by a handful of
    targets, while training samples uniformly over TARGETS.

    With `balance`, round-robin over groups in random order: every group gets
    one row before any group gets a second. That matches the training sampler
    (alpha 1) and spreads the metric over as many targets as the cap allows.
    """
    rows = np.asarray(rows)
    rng = np.random.default_rng(seed)
    if not balance:
        return np.sort(rng.choice(rows, n, replace=False))
    g = np.asarray(groups)[rows]
    buckets = [rng.permutation(rows[g == u]) for u in rng.permutation(np.unique(g))]
    out, depth = [], 0
    while len(out) < n:
        out += [b[depth] for b in buckets if depth < len(b)][:n - len(out)]
        depth += 1
    return np.sort(np.asarray(out))


def freeze_encoder(model):
    """Freeze the surface encoder up to the pooled vector; train the rest.

    The control for "does training the protein side buy anything": the conv,
    the input features and the pocket head stay exactly as the head checkpoint
    left them (so the regions are fixed too), and only the projection, the
    optional pool_norm, the ligand encoder and the temperature learn. That is
    the frozen-anchor probe's setting (ProbeModel: a projection over frozen
    anchors plus a ligand encoder), run inside this loop on this split, so its
    MRR is directly comparable to an unfrozen run's.
    """
    n = 0
    for name, p in model.encoder.named_parameters():
        # pair_pool is pooling, not encoder: fresh weights that would stay
        # random if frozen, so it trains with the projection
        if not name.startswith(("project.", "pool_norm.", "pair_pool.")):
            p.requires_grad_(False)
            n += p.numel()
    live = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"--freeze_encoder: {n:,} encoder parameters frozen (conv, inputs, "
          f"pocket head); {live:,} still train (projection, ligand, temperature)")


# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------
def train(a):
    import torch

    from probe_retrieval import draw_batch, group_index, info_nce, split_by_group
    from plip_loss import plip_losses, site_points
    from surface_encoder import pocket_loss
    from train_data import LabelCache, SurfaceStore

    import anchors as anchors_mod

    device = a.device
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    use_neighbor_device(a, device)
    table, ligands = load_rows(a)

    model, enc_cfg, lig_cfg = build_model(a, device)
    if a.n_decoys:
        if enc_cfg.region_source != "external" or a.pocket_select != "site":
            raise SystemExit("--n_decoys needs --region_source external "
                             "--pocket_select site: a decoy is a pocket that is "
                             "NOT the row's site, so there has to be a site")
        if a.n_decoys + 1 > enc_cfg.k_pos:
            raise SystemExit(f"--n_decoys {a.n_decoys} + the site exceeds --k_pos "
                             f"{enc_cfg.k_pos}, the pocket slots per protein")
        if not a.lam_site:
            print("[warn] --n_decoys with --lam_site 0: the decoys are pooled "
                  "and reported (site) but nothing trains on them")
    plip_on = bool(getattr(a, "lam_plip_sal", 0) or getattr(a, "lam_plip_type", 0))
    if plip_on:
        # PLIP supervision of the pair-pooling heads (plip_loss.py): needs the
        # heads, and the row's SITE region to put the labels on.
        if model.encoder.pair_pool is None:
            raise SystemExit("--lam_plip_* supervises pair pooling's heads: "
                             "needs --pool_mode pairs")
        if enc_cfg.region_source != "external" or a.pocket_select != "site":
            raise SystemExit("--lam_plip_* puts PLIP labels on the row's site "
                             "region: needs --region_source external --pocket_select site")
        if not a.plip or not Path(a.plip).is_dir():
            raise SystemExit(f"--lam_plip_* needs --plip DIR (plip_labels.py run "
                             f"output); got {a.plip!r}")
        if enc_cfg.pair_types < 6:
            raise SystemExit(f"--pair_types {enc_cfg.pair_types}: PLIP supervision ties "
                             "types 0-5 to its 6 classes, so it needs >= 6")
    if a.freeze_encoder:
        freeze_encoder(model)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=a.lr, weight_decay=a.weight_decay)
    # Two stores over ONE disk cache. Validation touches every val protein and
    # would otherwise evict the whole training window on every evaluation, and
    # the training window back out on the next step - the surfaces are the
    # expensive thing, so the two sets get their own residency.
    store = SurfaceStore(a.surface_cache, ram=a.surface_ram, device=device,
                         keep_hydrogens=a.keep_hydrogens,
                         pockets=pocket_source(a, enc_cfg))
    val_store = SurfaceStore(a.surface_cache, ram=a.val_ram, device=device,
                             keep_hydrogens=a.keep_hydrogens,
                             pockets=pocket_source(a, enc_cfg))
    labels = LabelCache(table, cutoff=a.cutoff, ram=a.label_ram,
                        plip=a.plip if plip_on else None)

    # split, sampler, state
    if getattr(a, "val_set", "") and a.val_split:
        raise SystemExit("--val_set and --val_split are alternatives")
    if getattr(a, "val_set", ""):
        # One fixed validation set shared across datasets (src/valset.py):
        # validation = its rows, training = every row outside its held-out
        # target groups. Not subsampled: the set IS the comparison.
        from valset import apply as apply_val_set
        train_rows, val_rows, missing = apply_val_set(table.rows, a.val_set)
        if missing:
            raise SystemExit(f"--val_set {a.val_set}: {len(missing)} of its rows are "
                             f"not in {a.items} (e.g. {missing[:3]}); make the set "
                             "from this items.tsv (valset.py make --items_b)")
        print(f"validation = the fixed set {a.val_set} ({len(val_rows)} rows); "
              "training excludes its held-out target groups")
    elif a.val_split:
        is_val = np.asarray([r["split"] in a.val_split.split(",") for r in table.rows])
        train_rows, val_rows = np.nonzero(~is_val)[0], np.nonzero(is_val)[0]
        if len(val_rows) < 2:
            raise SystemExit(
                f"--val_split {a.val_split} matches {len(val_rows)} of "
                f"{len(table)} rows. `prepare --split train` keeps only the "
                "train split, so there is nothing to validate on; prepare with "
                "--split train,val or drop --val_split for a group-disjoint carve")
        print(f"validation = the dataset's own split(s) {a.val_split}; NOT "
              "group-disjoint from train unless the dataset made it so")
    else:
        train_rows, val_rows = split_by_group(table.group, a.val_frac, a.split_seed)
        print("validation carved out group-disjointly from the prepared rows")
    if a.val_max and len(val_rows) > a.val_max and not getattr(a, "val_set", ""):
        val_rows = subsample_val(val_rows, table.group, a.val_max, a.split_seed,
                                 a.val_balance)
        print(f"validation subsampled to {a.val_max} rows (--val_max"
              + (", group-balanced: --val_balance)" if a.val_balance else
                 ", uniform over rows)"))
    print(f"{len(train_rows)} train / {len(val_rows)} val rows "
          f"({len(set(table.group[train_rows]))} / "
          f"{len(set(table.group[val_rows]))} target groups)")
    ligands.set_scaler(table.fp_row[train_rows])
    print("ligand descriptor standardisation fitted on the TRAIN rows only")

    members, counts = group_index(table.group[train_rows])
    rng = np.random.default_rng(a.seed)
    # Extra ligand negatives come from a stream of their own, so switching
    # them on does not change which BATCHES a run sees.
    rng_x = np.random.default_rng([a.seed, 1])
    gen = torch.Generator().manual_seed(a.seed)   # negative-region seeding
    history, step0 = [], 0
    pos_neg = [0.0, 0.0]

    resume = a.resume or (str(out / "last.pt") if a.auto_resume
                          and (out / "last.pt").exists() else "")
    if resume:
        ck = torch.load(resume, map_location="cpu", weights_only=False)
        if ck["args"].get("freeze_encoder", False) != a.freeze_encoder:
            raise SystemExit(
                f"{resume} was trained with freeze_encoder="
                f"{ck['args'].get('freeze_encoder', False)}; resuming it with "
                f"{a.freeze_encoder} would change which parameters train (and "
                "the optimiser state would not load). Use the same flag or a "
                "different --out")
        model.load_state_dict(ck["state_dict"])
        opt.load_state_dict(ck["optimizer"])
        step0 = ck["step"]
        history = ck.get("history", [])
        pos_neg = list(ck.get("pos_neg", pos_neg))
        train_rows = np.asarray(ck["train_rows"])
        val_rows = np.asarray(ck["val_rows"])
        torch.set_rng_state(ck["rng"]["torch"])
        np.random.set_state(ck["rng"]["numpy"])
        members, counts = group_index(table.group[train_rows])
        # The sampler stream is NOT restored: a numpy Generator's state is not
        # in the checkpoint, so it is reseeded from (seed, step) instead. That
        # keeps a resumed run reproducible from its own checkpoint, which is
        # what matters, but a resumed run is not step-for-step identical to an
        # uninterrupted one. Said plainly rather than silently approximated.
        rng = np.random.default_rng([a.seed, step0])
        rng_x = np.random.default_rng([a.seed, step0, 1])
        # The checkpoint's split replaced the one computed above, so the
        # descriptor scaler has to be refitted on ITS train rows or the ligand
        # branch would see inputs standardised against a different split.
        ligands.set_scaler(table.fp_row[train_rows])
        print(f"resumed {resume} at step {step0} with its own split "
              f"({len(train_rows)} train / {len(val_rows)} val); sampler "
              f"reseeded from (seed, step) = ({a.seed}, {step0})")

    best = max((h.get("mrr", float("nan")) for h in history), default=float("nan"))
    if not np.isfinite(best):
        best = -1.0
    bank = None
    if a.bank:
        if a.bank < a.batch:
            raise SystemExit(f"--bank {a.bank} is smaller than --batch {a.batch}")
        bank = ProteinBank(a.bank, enc_cfg.k_pos + enc_cfg.k_neg + 1,
                           enc_cfg.pooled_dims, device)
    t_start = time.time()
    losses, nces, pocks = [], [], []
    plip_sals, plip_types, plip_pos = [], [], []
    # Per-phase wall clock since the last evaluation line, summed over steps.
    # Synchronised at each boundary, so GPU work lands in the phase that
    # launched it rather than in whichever one next waits on the device.
    cuda = str(device).startswith("cuda") and torch.cuda.is_available()

    def clock():
        if cuda:
            torch.cuda.synchronize()
        return time.perf_counter()
    T = dict.fromkeys(("load", "fwd", "select", "nbrs", "lig", "bwd", "steps"), 0.0)
    mean = lambda v: float(np.mean(v[-a.eval_every:])) if v else float("nan")
    print(f"\nsteps {step0} -> {a.steps}, batch {a.batch}, alpha {a.alpha}, "
          f"lam_pocket {a.lam_pocket}, lam_site {a.lam_site}, tau {a.tau}")
    if enc_cfg.region_source == "external":
        print(f"pocket regions: EXTERNAL, not the pocket head - "
              f"{store.pockets.describe()}; k_pos {enc_cfg.k_pos} + k_neg "
              f"{enc_cfg.k_neg}. A val protein with no external pocket has no "
              "anchor and drops out of the retrieval metrics")
    else:
        print(f"region selector: {enc_cfg.region_method} "
              f"R={enc_cfg.region_ball_radius} shift={enc_cfg.region_ball_shift}, "
              f"k_pos {enc_cfg.k_pos} + k_neg {enc_cfg.k_neg}"
              + ("" if enc_cfg.region_source == "predicted" else
                 f", on the EXTERNAL map ({store.pockets.describe()}), "
                 f"threshold {enc_cfg.region_threshold}"))
    print(f"negatives per step: {a.batch - 1} in-batch proteins"
          + (f" + a bank of {a.bank}" if a.bank else "")
          + f"; {a.batch - 1} in-batch ligands"
          + (f" + {a.extra_ligands} extra" if a.extra_ligands else "")
          + ("  (nce is over these, so it is not comparable across settings)"
             if a.bank or a.extra_ligands else ""))
    print(f"lr {a.lr} {a.lr_schedule}" + (f", warmup {a.warmup}" if a.warmup else "")
          + (f", floor {a.min_lr_frac} x" if a.lr_schedule == "cosine" else "")
          + f"; neighbour lists on {anchors_mod.NEIGHBOR_DEVICE or 'the CPU'}")
    print("fb = fraction of val proteins whose positive anchor had to be the "
          "fallback ball\n")
    if a.n_decoys:
        print(f"site decoys: up to {a.n_decoys} of each receptor's other pockets "
              f"(top {a.pocket_max or enc_cfg.k_pos} by {a.pocket_rank}, clean of "
              f"the site, its label and sibling sites), trained against the "
              f"row's ligand by lam_site {a.lam_site}; never scored across "
              f"proteins. site = how often the ligand scores its own pocket "
              f"above every decoy on the receptor")
    print(f"temperature: learned from {a.tau}, floor --tau_min {a.tau_min} "
          f"(column tau)")
    hdr = ("step", "sec", "loss", "nce", "pock", "mrr", "chance", "top1",
           "top10", "site", "AUROC", "AP", "fb", "cosdiff", "tau")
    print("".join(f"{h:>8s}" for h in hdr), flush=True)

    for step in range(step0, a.steps + 1):
        due = (step % a.eval_every == 0) or step == a.steps
        if due:
            met = evaluate(model, val_store, labels, table, val_rows, ligands,
                            a, device)
            met.update(step=step, sec=time.time() - t_start, loss=mean(losses),
                       nce=mean(nces), pock=mean(pocks))
            history.append(met)
            vals = (step, met["sec"], met["loss"], met["nce"], met["pock"],
                    met["mrr"], met["chance"], met["top1"], met["top10"],
                    met["site"], met["auroc"], met["ap"], met["fallback_rate"],
                    met.get("cos_diff", float("nan")), met["tau"])
            # Seconds as an integer: at 8.3f, 1000 s and up is 8+ characters
            # and runs into the step column ("22912.301" was step 2, 2912 s).
            print(f"{step:8d} {met['sec']:7.0f}" + "".join(
                f"{v:8.3f}" if isinstance(v, float) else f"{v:8d}"
                for v in vals[2:]), flush=True)
            if plip_on:
                met.update(plip_sal=mean(plip_sals), plip_type=mean(plip_types),
                           plip_pos=mean(plip_pos))
                print(f"         plip  salience BCE {met['plip_sal']:.3f}  type CE "
                      f"{met['plip_type']:.3f}  (contacted share of site points "
                      f"{met['plip_pos']:.3f}; type CE at chance = "
                      f"{np.log(enc_cfg.pair_types + 1):.3f})", flush=True)
            print(f"         train {store.report()}", flush=True)
            print(f"         val   {val_store.report()}", flush=True)
            if T["steps"]:
                n = T["steps"]
                rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
                gpu = (f", GPU peak {torch.cuda.max_memory_allocated() / 1e9:.1f} GB"
                       if cuda else "")
                print(f"         s/step: load {T['load'] / n:.2f}  fwd {T['fwd'] / n:.2f} "
                      f"(select+pool {T['select'] / n:.2f}, of it nbr lists "
                      f"{T['nbrs'] / n:.2f})  lig+loss {T['lig'] / n:.2f}  "
                      f"bwd {T['bwd'] / n:.2f}  | lr {lr_at(a, step):.2e} | "
                      f"RSS peak {rss:.1f} GB{gpu}", flush=True)
                T = dict.fromkeys(T, 0.0)
            (out / "history.json").write_text(json.dumps(history, indent=1) + "\n")
            if np.isfinite(met["mrr"]) and met["mrr"] > best:
                best = met["mrr"]
                save_ckpt(out / "best.pt", model, opt, step, (train_rows, val_rows),
                          a, enc_cfg, lig_cfg, history, pos_neg)
        if step and (step % a.save_every == 0 or step == a.steps):
            save_ckpt(out / "last.pt", model, opt, step, (train_rows, val_rows),
                      a, enc_cfg, lig_cfg, history, pos_neg)
        if step == a.steps:
            break
        if a.max_hours and (time.time() - t_start) / 3600 > a.max_hours:
            print(f"\n--max_hours {a.max_hours} reached at step {step}; saving and stopping")
            save_ckpt(out / "last.pt", model, opt, step, (train_rows, val_rows),
                      a, enc_cfg, lig_cfg, history, pos_neg)
            break

        for g in opt.param_groups:
            g["lr"] = lr_at(a, step)
        t0 = clock()
        rows = train_rows[draw_batch(rng, members, counts, a.batch, a.alpha)]
        P, labs = build_batch(model, store, labels, table, rows, device, a.n_decoys)
        t1 = clock()
        nbr0 = anchors_mod.NEIGHBOR_SECONDS[0]
        blk = anchor_block(model, P, labs, a, generator=gen)
        t2 = clock()
        T["load"] += t1 - t0
        T["fwd"] += t2 - t1
        T["select"] += model.encoder.last_select_sec
        T["nbrs"] += anchors_mod.NEIGHBOR_SECONDS[0] - nbr0
        keep = blk["usable"].any(1)
        if int(keep.sum()) < 2:
            continue                       # nothing to contrast this step
        kn = keep.cpu().numpy()
        rows_k = rows[kn]
        anchors_k, usable_k = blk["anchors"][keep], blk["usable"][keep]

        extra = (train_rows[draw_batch(rng_x, members, counts, a.extra_ligands, a.alpha)]
                 if a.extra_ligands else rows_k[:0])
        # One ligand forward for batch and extras together, so both go through
        # the same normalisation statistics.
        lig_all = encode_ligands(model, ligands,
                                 table.fp_row[np.concatenate([rows_k, extra])], device)
        lig, lig_x = lig_all[:len(rows_k)], lig_all[len(rows_k):]
        scores = model.scores(anchors_k, usable_k, lig)
        mask = pair_mask(table, rows_k, ligands, device, a.tanimoto)
        banked = bank.anchors(model) if bank is not None else None
        if banked is None and not len(extra):
            nce = info_nce(scores, mask)
        else:
            s_bank = m_bank = s_lig = m_lig = None
            if banked is not None:
                s_bank = model.scores(banked[0], banked[1], lig)
                m_bank = cross_mask(table, banked[2], rows_k, ligands, device,
                                    a.tanimoto)
            if len(extra):
                s_lig = model.scores(anchors_k, usable_k, lig_x)
                m_lig = cross_mask(table, rows_k, extra, ligands, device, a.tanimoto)
            nce = nce_extended(scores, mask, s_bank, m_bank, s_lig, m_lig)

        label = torch.cat([lb["pocket"] for lb in labs]).to(device).float()
        pos_neg[0] += float(label.sum())
        pos_neg[1] += float((1 - label).sum())
        pw = a.pos_weight or (pos_neg[1] / max(pos_neg[0], 1.0))
        pock = pocket_loss(blk["out"]["pocket_logit"], label, pw)

        loss = nce + a.lam_pocket * pock
        if plip_on:
            out_b = blk["out"]
            tls, sls, ys = [], [], []
            for b, lb in enumerate(labs):
                sp = site_points(out_b, b, lb, site_select=True)
                y = lb.get("plip_union")
                if sp is None or y is None:
                    continue
                glob, loc = sp
                f = out_b["input_feats"][glob] if enc_cfg.pair_feats else None
                tl, sl = model.encoder.pair_pool.heads(out_b["pool_emb"][glob], f)
                tls.append(tl)
                sls.append(sl)
                ys.append(y[loc.cpu().numpy()])
            if tls:
                p_sal, p_type, p_st = plip_losses(tls, sls, ys)
                loss = loss + a.lam_plip_sal * p_sal + a.lam_plip_type * p_type
                plip_sals.append(float(p_sal.detach()))
                plip_types.append(float(p_type.detach()))
                plip_pos.append(p_st["plip_pos"])
        if a.lam_site:
            per_anchor = (torch.einsum("bkd,bd->bk", blk["anchors"][keep], lig)
                          * model.scale())
            loss = loss + a.lam_site * site_loss(per_anchor, blk["label"][keep])
        t3 = clock()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip)
        opt.step()
        if bank is not None:
            # After the loss, so a batch is never its own bank negative.
            bank.push(blk["pooled"][keep], blk["valid"][keep], usable_k, rows_k)
        t4 = clock()
        T["lig"] += t3 - t2
        T["bwd"] += t4 - t3
        T["steps"] += 1
        losses.append(float(loss.detach()))
        nces.append(float(nce.detach()))
        pocks.append(float(pock.detach()))

    print(f"\ntrain {store.report()}\nval   {val_store.report()}")
    if labels.failed:
        print(f"[warn] {len(labels.failed)} sibling ligands failed to load, so a "
              f"few other-site masks are incomplete; first: "
              f"{list(labels.failed.items())[:2]}")
    print(f"best val MRR {best:.4f}; checkpoints in {out}/best.pt and {out}/last.pt")
    print("\nmrr / top1 / top10   ligand -> protein retrieval on the val groups"
          "\nchance               exact expected MRR of a random ranking"
          "\nsite                 of proteins WITH a pocket-matching anchor, how"
          "\n                     often the ligand's best anchor is that one; under"
          "\n                     --pocket_select site, how often it scores its own"
          "\n                     pocket above every decoy (NaN without --n_decoys)"
          "\nAUROC / AP           the per-point pocket head, unchanged in meaning"
          "\n                     from encoder_test, so it is comparable to 0.937 / 0.47"
          "\nfb                   fraction of val proteins needing the fallback ball;"
          "\n                     if this stays high the loop is leaning on labels"
          "\ncosdiff              median cosine between DIFFERENT proteins' top"
          "\n                     anchors. 0.993 before any contrastive training;"
          "\n                     this is the number the whole exercise has to move"
          "\ntau                  the learned temperature; at --tau_min it is pinned")


def evaluate_only(a):
    """Validation metrics from a checkpoint; nothing is trained."""
    from probe_retrieval import split_by_group
    from train_data import LabelCache, SurfaceStore

    use_neighbor_device(a, a.device)
    model, ckpt, saved = load_model_from_ckpt(a.ckpt, a.device, overrides=a)
    table, ligands = load_rows(a)
    # External-pocket settings: the CLI's where given, else the checkpoint's,
    # else the defaults - so a checkpoint trained on fpocket regions evaluates
    # on the same regions without repeating every flag.
    for k, v in POCKET_DEFAULTS.items():
        if getattr(a, k, None) is None:
            setattr(a, k, getattr(saved, k, None) or v)
    store = SurfaceStore(a.surface_cache, ram=a.surface_ram, device=a.device,
                         keep_hydrogens=saved.keep_hydrogens,
                         pockets=pocket_source(a, model.encoder.cfg))
    labels = LabelCache(table, cutoff=a.cutoff, ram=a.label_ram)
    if sum(bool(x) for x in (a.use_ckpt_split, a.recarve, getattr(a, "val_set", ""))) > 1:
        raise SystemExit("--use_ckpt_split, --recarve and --val_set are alternatives")
    if getattr(a, "val_set", ""):
        # The fixed cross-dataset set (src/valset.py): the same rows whatever
        # dataset the checkpoint trained on, which is what makes a v1 and a v2
        # checkpoint comparable. Not subsampled.
        from valset import apply as apply_val_set
        _, rows, missing = apply_val_set(table.rows, a.val_set)
        if missing:
            raise SystemExit(f"--val_set: {len(missing)} rows not in {a.items}")
        leak = np.intersect1d(rows, np.asarray(ckpt["train_rows"]))
        if len(leak):
            raise SystemExit(f"--val_set: {len(leak)} of its rows were TRAIN rows of "
                             "this checkpoint; is --items the table it trained on?")
        a.val_max = 0
        print(f"--val_set {a.val_set}: {len(rows)} rows")
    elif a.recarve:
        # The checkpoint stores only its SUBSAMPLED val rows; the full pool is
        # recomputed from the split it was carved with, which split_by_group
        # makes deterministic in (groups, val_frac, split_seed).
        if getattr(saved, "val_split", ""):
            raise SystemExit(f"{a.ckpt} validated on --val_split "
                             f"{saved.val_split}; --recarve rebuilds a carved split")
        _, rows = split_by_group(table.group, saved.val_frac, saved.split_seed)
        leak = np.intersect1d(rows, np.asarray(ckpt["train_rows"]))
        if len(leak):
            raise SystemExit(f"--recarve: {len(leak)} recarved val rows were "
                             "TRAIN rows of this checkpoint; different table?")
        print(f"--recarve: {len(rows)} val rows (val_frac {saved.val_frac}, "
              f"split_seed {saved.split_seed}), disjoint from the checkpoint's train")
    else:
        rows = (np.asarray(ckpt["val_rows"]) if a.use_ckpt_split
                else np.arange(len(table)))
    rows = rows[rows < len(table)]
    if a.val_max and len(rows) > a.val_max:
        rows = subsample_val(rows, table.group, a.val_max, saved.split_seed,
                             a.val_balance)
        print(f"subsampled to {a.val_max} rows"
              + (" (group-balanced)" if a.val_balance else ""))
    # The scaler must be the one training used, so it is refitted on the
    # checkpoint's OWN train rows. Row indices only mean anything against the
    # same items.tsv; a different table makes this - and --use_ckpt_split -
    # meaningless, which is why both are clipped loudly rather than silently.
    tr = np.asarray(ckpt["train_rows"])
    tr = tr[tr < len(table)]
    if len(tr) != len(ckpt["train_rows"]) or (a.use_ckpt_split
                                              and len(rows) != len(ckpt["val_rows"])):
        print(f"[warn] this items.tsv has {len(table)} rows but the checkpoint "
              f"indexed {max(ckpt['train_rows'] + ckpt['val_rows']) + 1}; row "
              "indices are being clipped, so this is a different table")
    ligands.set_scaler(table.fp_row[tr])
    print(f"{a.ckpt}: step {ckpt['step']}; evaluating {len(rows)} rows")
    met = evaluate(model, store, labels, table, rows, ligands, a, a.device)
    for k in sorted(met):
        v = met[k]
        print(f"  {k:16s} {v:.4f}" if isinstance(v, float) else f"  {k:16s} {v}")


# ---------------------------------------------------------------------------
# cache: prefill surfaces
# ---------------------------------------------------------------------------
def cache(a):
    """Sample and store every receptor's surface. Shardable across array tasks."""
    from train_data import ItemTable, SurfaceStore, read_items

    table = ItemTable(read_items(a.items, a.limit))
    model, _, _ = build_model(a, a.device)
    store = SurfaceStore(a.surface_cache, ram=1, device=a.device,
                         keep_hydrogens=a.keep_hydrogens)
    # One row per surface, in a fixed order, so shards are disjoint and a rerun
    # of shard i covers exactly the same receptors.
    firsts = sorted(table.siblings.items())
    if a.shard:
        i, n = (int(x) for x in a.shard.split("/"))
        firsts = firsts[i::n]
        print(f"shard {i}/{n}: {len(firsts)} surfaces")
    t0, built, failed = time.time(), 0, []
    for k, (input_id, rows) in enumerate(firsts, 1):
        row = table.rows[rows[0]]
        try:
            built += bool(store.ensure(model.encoder, row))
        except Exception as e:
            failed.append((input_id, f"{type(e).__name__}: {e}"))
        if k % 50 == 0 or k == len(firsts):
            print(f"  {k}/{len(firsts)}  {built} built  "
                  f"{(time.time() - t0) / k:.2f}s/surface", flush=True)
    print(f"\n{built} surfaces written to {a.surface_cache}, "
          f"{len(firsts) - built - len(failed)} already present, {len(failed)} failed")
    for input_id, msg in failed[:10]:
        print(f"    {input_id}: {msg[:140]}")


# ---------------------------------------------------------------------------
# selftest
# ---------------------------------------------------------------------------
def selftest(a):
    """Delegate to src/train_selftest.py, where the 990-line body now lives.

    Imported inside the function rather than at module level: train_selftest
    imports the pieces under test FROM this module, so a module-level import
    would be a cycle. By call time train.py is fully loaded.
    """
    from train_selftest import selftest as run

    return run(a)


# ---------------------------------------------------------------------------
POCKET_DEFAULTS = {"pockets": "", "pocket_rank": "score", "pocket_shape": "pocket",
                   "pocket_margin": 0.0, "pocket_max": 0, "pocket_select": "topk",
                   "site_cutoff": 3.0, "site_frac": 0.25, "site_lig_frac": 0.5,
                   "n_decoys": 0}


def add_region_args(p, defaults=True):
    """Selector flags. With defaults=False every one is None, which is what
    `eval` needs: probe_retrieval.apply_overrides changes only the fields it is
    given, so a flag with a default would silently re-select the regions of a
    checkpoint that was trained with something else."""
    d = (lambda v: v) if defaults else (lambda v: None)
    p.add_argument("--region_method", default=d("ball"),
                   help="watershed | components | ball | nms | compball; the "
                        "sweep's leader, and this default, is ball")
    p.add_argument("--top_fraction", type=float, default=d(0.05))
    p.add_argument("--threshold", type=float, default=d(0.5))
    p.add_argument("--ball_radius", type=float, default=d(8.0))
    p.add_argument("--ball_shift", type=int, default=d(2))
    p.add_argument("--ball_rank", default=d("seed"))
    p.add_argument("--persistence", type=float, default=d(0.02))
    p.add_argument("--nms_iou", type=float, default=d(0.3))
    p.add_argument("--nms_alpha", type=float, default=d(0.0))
    p.add_argument("--nms_score", default=d("sum"))
    p.add_argument("--k_pos", type=int, default=d(8))
    p.add_argument("--k_neg", type=int, default=d(8))
    # Where the pocket regions come from (EncoderConfig.region_source) and the
    # external pockets themselves (external_pockets.py).
    p.add_argument("--region_source", default=d("predicted"),
                   choices=("predicted", "external", "external_map"),
                   help="predicted: the pocket head's map through the selector "
                        "(the original). external: --pockets ARE the pocket "
                        "regions. external_map: their per-point map replaces "
                        "the head's in the selector, thresholded at --threshold")
    P = POCKET_DEFAULTS
    p.add_argument("--pockets", default=d(P["pockets"]),
                   help="directory of <input_id>.npz from external_pockets.py run")
    p.add_argument("--pocket_rank", default=d(P["pocket_rank"]),
                   choices=("score", "drug", "volume"),
                   help="order of the external pockets: fpocket's score (its own "
                        "ranking), druggability, or volume")
    p.add_argument("--pocket_shape", default=d(P["pocket_shape"]),
                   choices=("pocket", "ball"),
                   help="pocket: the surface points its alpha spheres cover; "
                        "ball: a --ball_radius ball at their centroid")
    p.add_argument("--pocket_margin", type=float, default=d(P["pocket_margin"]),
                   help="A added to each alpha sphere's radius when mapping it "
                        "onto the surface; external_pockets.py score sweeps it")
    p.add_argument("--pocket_select", default=d(P["pocket_select"]),
                   choices=("topk", "site"),
                   help="topk: the receptor's best k_pos pockets, scored by max "
                        "(blind, what a screen sees). site: per row, THE pocket "
                        "fpocket found at that row's ligand (known-site, "
                        "DrugCLIP-style); eval a site model blind with topk")
    p.add_argument("--site_cutoff", type=float, default=d(P["site_cutoff"]),
                   help="A between an alpha-sphere centre and a ligand atom")
    p.add_argument("--site_frac", type=float, default=d(P["site_frac"]),
                   help="a pocket is at the site if this fraction of its "
                        "alpha spheres is within --site_cutoff of the ligand ...")
    p.add_argument("--site_lig_frac", type=float, default=d(P["site_lig_frac"]),
                   help="... or this fraction of the ligand's atoms is within "
                        "--site_cutoff of its alpha spheres")
    p.add_argument("--pocket_max", type=int, default=d(P["pocket_max"]),
                   help="map at most this many pockets per protein (0 = k_pos "
                        "for external, all for external_map)")
    p.add_argument("--n_decoys", type=int, default=d(P["n_decoys"]),
                   help="--pocket_select site: also pool up to this many of the "
                        "receptor's other mapped pockets (by --pocket_rank) as "
                        "decoys, trained against the ligand by --lam_site and "
                        "reported as `site`; never scored across proteins")


def add_model_args(p):
    p.add_argument("--arm", default="gnn", help="ligand_encoder.ARMS")
    p.add_argument("--out_dim", type=int, default=128)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--gnn_hidden", type=int, default=128)
    p.add_argument("--gnn_layers", type=int, default=4)
    p.add_argument("--lig_dropout", type=float, default=0.0)
    p.add_argument("--enc_dropout", type=float, default=0.0)
    p.add_argument("--desc_dims", type=int, default=0,
                   help="width of a second, fresh conv used only for region "
                        "descriptors (0 = off); see EncoderConfig.desc_dims")
    p.add_argument("--desc_layers", type=int, default=3)
    p.add_argument("--chemo", action="store_true",
                   help="add chemotype.py's 12 pharmacophore channels")
    p.add_argument("--pool_mode", default="mean",
                   help="mean | mean_std | shells (anchors.pool_regions) | pairs "
                        "(pair_pool.PairPool: site types x distance bins)")
    p.add_argument("--n_shells", type=int, default=3)
    p.add_argument("--pair_types", type=int, default=8,
                   help="pool_mode pairs: learned soft site types")
    p.add_argument("--pair_bins", type=int, default=8,
                   help="pool_mode pairs: Gaussian distance bins")
    p.add_argument("--pair_rmin", type=float, default=3.0)
    p.add_argument("--pair_rmax", type=float, default=16.0)
    p.add_argument("--pair_exclude", type=float, default=2.0,
                   help="A; closer point pairs are one atom's own cluster")
    p.add_argument("--no_pair_feats", dest="pair_feats", action="store_false",
                   help="type head reads the conv embedding only, not also the "
                        "raw input channels")
    p.add_argument("--anchor_norm", default="none",
                   help="none | batch; 'batch' subtracts the shared direction "
                        "that puts different proteins at cosine 0.993")
    p.add_argument("--resolution", type=float, default=1.0,
                   help="dMaSIF surface sampler; leave at the default for any "
                        "run whose surfaces are meant to be reused")
    p.add_argument("--sup_sampling", type=int, default=20,
                   help="candidate points per atom in the sampler. Lowering it "
                        "makes a coarser surface FAR faster, which is only for "
                        "shape checks - never for a real run")
    p.add_argument("--pretrained", action="store_true",
                   help="start the dMaSIF parts from the published checkpoint")
    p.add_argument("--head_ckpt", default="",
                   help="start from a trained pocket head (test_encoder.py "
                        "--save_ckpt), so the regions are usable at step 0")


def add_data_args(p, require_ligands=True):
    p.add_argument("--items", required=True)
    p.add_argument("--lig_feats", required=require_ligands, default="",
                   help="ligand_features.py .npz")
    p.add_argument("--limit", type=int, help="only the first N items")
    p.add_argument("--surface_cache", default="",
                   help="directory of cached surfaces; empty = resample every "
                        "time a receptor leaves the RAM cache")
    p.add_argument("--surface_ram", type=int, default=512,
                   help="training surfaces resident in host RAM (~1.1 MB each)")
    p.add_argument("--val_ram", type=int, default=512,
                   help="validation surfaces resident in host RAM, in a store "
                        "of their own so the two sets do not evict each other; "
                        "set it to at least --val_max")
    p.add_argument("--label_ram", type=int, default=1024)
    p.add_argument("--cutoff", type=float, default=4.0,
                   help="A from a ligand heavy atom; the pocket label")
    p.add_argument("--keep_hydrogens", action="store_true")
    p.add_argument("--gpu_neighbors", action="store_true",
                   help="region selection's neighbour lists on the GPU instead "
                        "of an O(N^2) CPU cdist per surface load")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--eval_batch", type=int, default=8)
    p.add_argument("--pos_cover", type=float, default=0.5)
    p.add_argument("--pos_prec", type=float, default=0.3)
    p.add_argument("--ignore_overlap", type=float, default=0.1)
    p.add_argument("--tanimoto", type=float, default=0.8)
    # anchor_block needs this on every path, so it lives here and not only on
    # `train`; otherwise `eval` would fail on the attribute, not on the flag.
    p.add_argument("--no_fallback", dest="fallback", action="store_false",
                   help="drop a protein whose predicted regions miss the pocket "
                        "instead of forcing a selector-shaped ball there")
    p.add_argument("--seed", type=int, default=0)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prepare", help="CPU venv: index.parquet -> items.tsv")
    p.add_argument("--index", required=True)
    p.add_argument("--systems", default="systems")
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=0, help="0 = every usable row")
    p.add_argument("--split", default="train",
                   help="comma-separated splits to keep; empty = all")
    p.add_argument("--exclude", action="append",
                   help="items.tsv whose RECEPTORS to leave out")
    p.add_argument("--seed", type=int, default=0)

    p = sub.add_parser("cache", help="container: prefill the surface cache")
    add_data_args(p, require_ligands=False)
    add_model_args(p)
    add_region_args(p)
    p.add_argument("--shard", default="", help="i/n, for an array job")
    p.set_defaults(tau=0.07)

    p = sub.add_parser("train", help="container: the contrastive loop")
    add_data_args(p)
    add_model_args(p)
    add_region_args(p)
    p.add_argument("--out", required=True, help="run directory for checkpoints")
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch", type=int, default=8,
                   help="proteins per step; this IS the number of in-batch "
                        "negatives, so bigger is a better loss, not just faster")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lr_schedule", default="const", choices=("const", "cosine"),
                   help="cosine decays to --min_lr_frac x lr over --steps")
    p.add_argument("--warmup", type=int, default=0, help="linear warmup steps")
    p.add_argument("--min_lr_frac", type=float, default=0.05)
    p.add_argument("--bank", type=int, default=0,
                   help="memory bank of this many recent proteins' anchors as "
                        "extra ligand->protein negatives (0 = off)")
    p.add_argument("--extra_ligands", type=int, default=0,
                   help="this many extra training ligands per step as extra "
                        "protein->ligand negatives (0 = off)")
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--clip", type=float, default=5.0)
    p.add_argument("--tau", type=float, default=0.07)
    p.add_argument("--tau_min", type=float, default=0.01,
                   help="floor on the learned temperature (the logit scale is "
                        "capped at 1 / this); 0.01 is CLIP's cap of 100")
    p.add_argument("--alpha", type=float, default=1.0,
                   help="1 = uniform over targets, 0 = uniform over rows")
    p.add_argument("--lam_pocket", type=float, default=1.0,
                   help="weight of the per-point pocket BCE; 0 lets the regions drift")
    p.add_argument("--lam_site", type=float, default=0.0,
                   help="weight of the within-protein anchor CE (which anchor)")
    p.add_argument("--plip", default="",
                   help="plip_labels.py run output (data/plip), for --lam_plip_*")
    p.add_argument("--lam_plip_sal", type=float, default=0.0,
                   help="weight of the PLIP salience BCE on the site's points "
                        "(plip_loss.py); 0 = off")
    p.add_argument("--lam_plip_type", type=float, default=0.0,
                   help="weight of the PLIP type CE: site types 0-5 tied to "
                        "donor/acceptor/cation/anion/aromatic/hydrophobic; 0 = off")
    p.add_argument("--pos_weight", type=float, default=0.0,
                   help="pocket BCE positive weight; 0 = the running ratio")
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--val_split", default="",
                   help="use the dataset's own split(s) as validation instead "
                        "of carving a group-disjoint one out of these rows")
    p.add_argument("--val_max", type=int, default=400,
                   help="cap the validation set; every protein costs a forward")
    p.add_argument("--val_balance", action="store_true",
                   help="draw the --val_max rows round-robin over target groups "
                        "instead of uniformly over rows (see subsample_val)")
    p.add_argument("--val_set", default="",
                   help="a src/valset.py directory: validate on its fixed rows and "
                        "train on everything outside its held-out targets; the "
                        "same rows across datasets, so runs compare")
    p.add_argument("--freeze_encoder", action="store_true",
                   help="control: freeze the surface encoder below the "
                        "projection, so only projection + ligand train")
    p.add_argument("--split_seed", type=int, default=0)
    p.add_argument("--eval_every", type=int, default=500)
    p.add_argument("--save_every", type=int, default=500)
    p.add_argument("--max_hours", type=float, default=0.0,
                   help="stop cleanly and checkpoint after this long")
    p.add_argument("--resume", default="", help="checkpoint to resume from")
    p.add_argument("--no_auto_resume", dest="auto_resume", action="store_false",
                   help="do not pick up <out>/last.pt automatically")

    p = sub.add_parser("eval", help="container: metrics from a checkpoint")
    add_data_args(p)
    add_region_args(p, defaults=False)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--use_ckpt_split", action="store_true",
                   help="evaluate the checkpoint's own validation rows")
    p.add_argument("--recarve", action="store_true",
                   help="evaluate the checkpoint's FULL group-disjoint val "
                        "pool, recomputed from its val_frac / split_seed")
    p.add_argument("--val_set", default="",
                   help="evaluate on a src/valset.py fixed set (any dataset's "
                        "checkpoint, the same rows)")
    p.add_argument("--val_max", type=int, default=0,
                   help="then subsample to this many rows (0 = all)")
    p.add_argument("--val_balance", action="store_true",
                   help="subsample round-robin over target groups")

    p = sub.add_parser("selftest", help="planted signal; no data, no GPU, no KeOps")
    p.add_argument("--tmp", default="tmp/train_selftest")
    p.add_argument("--steps", type=int, default=180)
    p.add_argument("--batch", type=int, default=12)
    p.add_argument("--min_mrr", type=float, default=0.30)
    p.add_argument("--pool_mode", default="shells")
    p.add_argument("--anchor_norm", default="batch")

    a = ap.parse_args()
    if a.cmd == "prepare":
        from train_data import prepare
        prepare(a)
    else:
        {"cache": cache, "train": train, "eval": evaluate_only,
         "selftest": selftest}[a.cmd](a)


if __name__ == "__main__":
    main()
