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

Stages:
    prepare    CPU venv. index.parquet -> runs/<name>/items.tsv.
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
    python src/train.py prepare --index datasets/dataset_v1/index.parquet \\
        --out runs/v1 --split train
    python src/ligand_features.py --index datasets/dataset_v1/index.parquet \\
        --out runs/v1
    # container (see src/slurm/train.sbatch)
    python src/train.py train --items runs/v1/items.tsv \\
        --lig_feats runs/v1/ligand_feats.npz --out runs/v1 \\
        --head_ckpt encoder_test/head_pretrained_300.pt \\
        --region_method ball --ball_radius 8 --ball_shift 2 \\
        --surface_cache surfaces/ --steps 20000 --max_hours 11

Depends on: numpy, torch; train_data, probe_retrieval, surface_encoder,
            ligand_encoder, anchors, test_encoder, ids.
"""

import argparse
import json
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
        region_method=a.region_method, region_top_fraction=(
            None if a.top_fraction <= 0 else a.top_fraction),
        region_threshold=a.threshold, region_ball_radius=a.ball_radius,
        region_ball_shift=a.ball_shift, region_ball_rank=a.ball_rank,
        region_persistence=a.persistence, region_nms_iou=a.nms_iou,
        region_nms_alpha=a.nms_alpha, region_nms_score=a.nms_score,
        pool_mode=a.pool_mode, n_shells=a.n_shells, anchor_norm=a.anchor_norm,
        out_dim=a.out_dim, dropout=a.enc_dropout,
        resolution=a.resolution, sup_sampling=a.sup_sampling)
    lig_cfg = LigandConfig(arm=a.arm, out_dim=a.out_dim, hidden=a.hidden,
                           dropout=a.lig_dropout, gnn_hidden=a.gnn_hidden,
                           gnn_layers=a.gnn_layers)

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
            # reduce the loss.
            return self.logit_scale.exp().clamp(max=100.0)

        def scores(self, anchors, usable, lig_vec):
            """(B, K, D) anchors x (L, D) ligands -> (B, L), max over anchors."""
            sim = torch.einsum("bkd,ld->bkl", anchors, lig_vec)
            sim = sim.masked_fill(~usable[..., None], -1e4)
            return sim.max(dim=1).values * self.scale()

    model = RetrievalModel().to(device)
    print(f"encoder: {enc_cfg.in_channels} input channels -> {enc_cfg.emb_dims} dims "
          f"-> {enc_cfg.pooled_dims} pooled ({enc_cfg.pool_mode}) -> {a.out_dim}; "
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
    state = ckpt["state_dict"]
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
def build_batch(model, store, labels, table, rows, device):
    """-> (P, labs) for these row indices. Surfaces come from the store."""
    from test_encoder import to_batch

    items, labs = [], []
    for i in rows:
        entry = store.get(model.encoder, table.rows[i])
        labs.append(labels.get(int(i), entry))
        items.append(entry)
    return to_batch(items, device), labs


def anchor_block(model, P, labs, a, generator=None):
    """Forward the proteins and assemble every anchor the loss can use.

    Returns a dict with, padded to a common slot count:
        anchors  (B, K', D)   normalised, the usable slots only
        usable   (B, K')      predicted pocket anchors, which is what a score
                              may be taken over - at training and at inference
        label    (B, K')      1 positive / 0 negative / -1 ignore per region
        fallback (B,)         True where the positive is the fallback ball
        out                   the raw encoder output, for the pocket loss
    """
    import torch

    from anchors import pool_regions
    from train_data import fallback_region, region_labels

    cfg = model.encoder.cfg
    # project=False: the fallback anchor below has to go through the same
    # normalisation as the rest, in one call, so the tail is run here.
    out = model.encoder(P, generator=generator, project=False)
    pooled, valid = out["anchor_pooled"], out["anchor_valid"]
    is_pos = out["anchor_is_pos"]
    B, K = valid.shape
    prob = torch.sigmoid(out["pocket_logit"])

    label = torch.zeros(B, K, dtype=torch.long, device=valid.device)
    extra_vec, extra_at = [], []
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

        pos_here = bool(((label[b] == 1) & valid[b] & is_pos[b]).any())
        if not pos_here and a.fallback and bool(lb["pocket"].any()):
            # The label masks live on the CPU with the cached surface; this
            # indexes the encoder's own xyz, which is on the device, and a
            # boolean mask has to be on the tensor's device to index it.
            pk = lb["pocket"].to(out["xyz"].device)
            reg = fallback_region(out["xyz"][sel], pk, cfg.region_ball_radius)
            if len(reg) >= cfg.min_region_points:
                extra_vec.append(pool_regions(
                    out["point_emb"][sel], prob[sel], [reg.to(sel.device)],
                    mode=cfg.pool_mode, xyz=out["xyz"][sel],
                    n_shells=cfg.n_shells)[0])
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
    # `valid`, not `usable`: the normalisation sees every real anchor, pocket
    # and negative alike, in training and in evaluation, so its statistics mean
    # the same thing in both.
    anchors = model.encoder.project_anchors(pooled, valid)
    return {"anchors": anchors, "usable": valid & is_pos, "label": label,
            "valid": valid, "is_pos": is_pos, "fallback": fallback, "out": out,
            "pooled": pooled}


def pair_mask(table, rows, ligands, device, tanimoto):
    """False-negative mask for an all-vs-all block of these rows."""
    import torch

    from probe_retrieval import mask_pairs

    fp_rows = table.fp_row[rows]
    g = torch.as_tensor(table.group_id[rows], device=device)
    fr = torch.as_tensor(fp_rows, device=device)
    fps = torch.as_tensor(ligands.fp[fp_rows].astype(np.float32), device=device)
    return mask_pairs(g, g, fr, fr, fps, fps, tanimoto)


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
    scores them at inference and a loss over them would be free marks.
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
            P, labs = build_batch(model, store, labels, table, chunk, device)
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
                   fallback_rate=got["fallback_rate"])
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

        # Is the ligand's best anchor the one over its own pocket?
        sim = torch.einsum("bkd,bd->bk", anchors, lig).masked_fill(~usable, -1e4)
        best = sim.argmax(1)
        lb = label.to(device).gather(1, best[:, None]).squeeze(1)
        has_pos = (label.to(device) == 1).any(1)
        met["site"] = (float((lb[has_pos] == 1).float().mean())
                       if bool(has_pos.any()) else float("nan"))
        met["site_cov"] = float(has_pos.float().mean())

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
    met.update(auroc=got["auroc"], ap=got["ap"],
               fallback_rate=got["fallback_rate"])
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
# train
# ---------------------------------------------------------------------------
def train(a):
    import torch

    from ligand_encoder import LigandData
    from probe_retrieval import draw_batch, group_index, info_nce, split_by_group
    from surface_encoder import pocket_loss
    from train_data import ItemTable, LabelCache, SurfaceStore, read_items

    device = a.device
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
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

    model, enc_cfg, lig_cfg = build_model(a, device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr,
                            weight_decay=a.weight_decay)
    # Two stores over ONE disk cache. Validation touches every val protein and
    # would otherwise evict the whole training window on every evaluation, and
    # the training window back out on the next step - the surfaces are the
    # expensive thing, so the two sets get their own residency.
    store = SurfaceStore(a.surface_cache, ram=a.surface_ram, device=device,
                         keep_hydrogens=a.keep_hydrogens)
    val_store = SurfaceStore(a.surface_cache, ram=a.val_ram, device=device,
                             keep_hydrogens=a.keep_hydrogens)
    labels = LabelCache(table, cutoff=a.cutoff, ram=a.label_ram)

    # split, sampler, state
    if a.val_split:
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
    if a.val_max and len(val_rows) > a.val_max:
        rng0 = np.random.default_rng(a.split_seed)
        val_rows = np.sort(rng0.choice(val_rows, a.val_max, replace=False))
        print(f"validation subsampled to {a.val_max} rows (--val_max)")
    print(f"{len(train_rows)} train / {len(val_rows)} val rows "
          f"({len(set(table.group[train_rows]))} / "
          f"{len(set(table.group[val_rows]))} target groups)")
    ligands.set_scaler(table.fp_row[train_rows])
    print("ligand descriptor standardisation fitted on the TRAIN rows only")

    members, counts = group_index(table.group[train_rows])
    rng = np.random.default_rng(a.seed)
    gen = torch.Generator().manual_seed(a.seed)   # negative-region seeding
    history, step0 = [], 0
    pos_neg = [0.0, 0.0]

    resume = a.resume or (str(out / "last.pt") if a.auto_resume
                          and (out / "last.pt").exists() else "")
    if resume:
        ck = torch.load(resume, map_location="cpu", weights_only=False)
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
    t_start = time.time()
    losses, nces, pocks = [], [], []
    mean = lambda v: float(np.mean(v[-a.eval_every:])) if v else float("nan")
    print(f"\nsteps {step0} -> {a.steps}, batch {a.batch}, alpha {a.alpha}, "
          f"lam_pocket {a.lam_pocket}, lam_site {a.lam_site}, tau {a.tau}")
    print(f"region selector: {enc_cfg.region_method} "
          f"R={enc_cfg.region_ball_radius} shift={enc_cfg.region_ball_shift}, "
          f"k_pos {enc_cfg.k_pos} + k_neg {enc_cfg.k_neg}")
    print("fb = fraction of val proteins whose positive anchor had to be the "
          "fallback ball\n")
    hdr = ("step", "sec", "loss", "nce", "pock", "mrr", "chance", "top1",
           "top10", "site", "AUROC", "AP", "fb", "cosdiff")
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
                    met.get("cos_diff", float("nan")))
            # Seconds as an integer: at 8.3f, 1000 s and up is 8+ characters
            # and runs into the step column ("22912.301" was step 2, 2912 s).
            print(f"{step:8d} {met['sec']:7.0f}" + "".join(
                f"{v:8.3f}" if isinstance(v, float) else f"{v:8d}"
                for v in vals[2:]), flush=True)
            print(f"         train {store.report()}", flush=True)
            print(f"         val   {val_store.report()}", flush=True)
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

        rows = train_rows[draw_batch(rng, members, counts, a.batch, a.alpha)]
        P, labs = build_batch(model, store, labels, table, rows, device)
        blk = anchor_block(model, P, labs, a, generator=gen)
        keep = blk["usable"].any(1)
        if int(keep.sum()) < 2:
            continue                       # nothing to contrast this step
        kn = keep.cpu().numpy()
        rows_k = rows[kn]

        lig = encode_ligands(model, ligands, table.fp_row[rows_k], device)
        scores = model.scores(blk["anchors"][keep], blk["usable"][keep], lig)
        nce = info_nce(scores, pair_mask(table, rows_k, ligands, device, a.tanimoto))

        label = torch.cat([lb["pocket"] for lb in labs]).to(device).float()
        pos_neg[0] += float(label.sum())
        pos_neg[1] += float((1 - label).sum())
        pw = a.pos_weight or (pos_neg[1] / max(pos_neg[0], 1.0))
        pock = pocket_loss(blk["out"]["pocket_logit"], label, pw)

        loss = nce + a.lam_pocket * pock
        if a.lam_site:
            per_anchor = (torch.einsum("bkd,bd->bk", blk["anchors"][keep], lig)
                          * model.scale())
            loss = loss + a.lam_site * site_loss(per_anchor, blk["label"][keep])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip)
        opt.step()
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
          "\n                     often the ligand's best anchor is that one"
          "\nAUROC / AP           the per-point pocket head, unchanged in meaning"
          "\n                     from encoder_test, so it is comparable to 0.937 / 0.47"
          "\nfb                   fraction of val proteins needing the fallback ball;"
          "\n                     if this stays high the loop is leaning on labels"
          "\ncosdiff              median cosine between DIFFERENT proteins' top"
          "\n                     anchors. 0.993 before any contrastive training;"
          "\n                     this is the number the whole exercise has to move")


def evaluate_only(a):
    """Validation metrics from a checkpoint; nothing is trained."""
    from ligand_encoder import LigandData
    from train_data import ItemTable, LabelCache, SurfaceStore, read_items

    model, ckpt, saved = load_model_from_ckpt(a.ckpt, a.device, overrides=a)
    table = ItemTable(read_items(a.items, a.limit))
    ligands = LigandData(a.lig_feats)
    store = SurfaceStore(a.surface_cache, ram=a.surface_ram, device=a.device,
                         keep_hydrogens=saved.keep_hydrogens)
    labels = LabelCache(table, cutoff=a.cutoff, ram=a.label_ram)
    rows = (np.asarray(ckpt["val_rows"]) if a.use_ckpt_split
            else np.arange(len(table)))
    rows = rows[rows < len(table)]
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
    """End-to-end check of everything but dMaSIF, on synthetic proteins.

    A stub stands in for the three pieces that need KeOps and a GPU - the
    surface sampler, AtomNet and the convolution. Everything else is the real
    code: anchors.select_regions, pool_regions, train_data's labels / other-site
    masks / fallback, the sampler, the InfoNCE, the metrics, project_anchors,
    and the checkpoint round-trip.

    A signal is planted: each protein's pocket points carry its ligand's latent
    vector, so a learnable projection and ligand encoder CAN recover the
    pairing and MRR must climb well above chance. If it does not, something in
    the loss, the masking, the sampler, the labels or the metrics is wrong -
    which is what cannot be checked on real data without a GPU and hours.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    from anchors import pool_regions, pooled_width, select_regions
    from ligand_encoder import LigandConfig, LigandEncoder
    from probe_retrieval import (draw_batch, group_index, info_nce,
                                 retrieval_metrics, split_by_group)
    from surface_encoder import EncoderConfig, SurfaceEncoder, pocket_loss
    from test_encoder import to_batch
    from train_data import (ItemTable, LabelCache, SurfaceStore,
                            fallback_region, region_labels)

    import shutil

    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    tmp = Path(a.tmp)
    # Cleared, not reused. A surface left over from a previous run would make
    # the first `get` a disk HIT, and the sampling assertions below - which
    # only mean anything on a MISS - would pass without testing anything.
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True, exist_ok=True)
    N_PROT, N_LIG, LATENT, EMB = 120, 60, 6, 16
    fails = []

    def check(name, ok, detail=""):
        print(f"  {'ok  ' if ok else 'FAIL'} {name}" + (f"   {detail}" if detail else ""))
        if not ok:
            fails.append(name)

    # ---- synthetic proteins: a sphere of surface points, a pocket patch -----
    print("\n[1] synthetic proteins, real region selection")
    z = rng.normal(size=(N_LIG, LATENT)).astype(np.float32)
    W_fp = rng.normal(size=(LATENT, 2048))
    bits = ((1 / (1 + np.exp(-(z @ W_fp)))) > rng.random((N_LIG, 2048))).astype(np.uint8)

    # 2400 points on a 20 A sphere, which is ~1.6 A spacing - coarser than
    # dMaSIF's ~1 A but fine enough that a ball_R8 region holds ~99 points and
    # the counts below are not dominated by discretisation.
    def make_protein(seed):
        g = np.random.default_rng(seed)
        v = g.normal(size=(2400, 3))
        xyz = (v / np.linalg.norm(v, axis=1, keepdims=True)) * 20.0
        return xyz.astype(np.float32)

    rows, surfaces = [], {}
    for i in range(N_PROT):
        lig = i % N_LIG
        iid = f"p{i:03d}"
        xyz = make_protein(i)
        # The "ligand": a cloud of atoms just outside one patch of the sphere.
        # Sized to reproduce the real geometry, because the fixture is useless
        # on either side of it. The pocket is every surface point within 4 A of
        # the cloud, and a ball_R8 region holds ~99 of the 2400 points, so:
        #   16 atoms at scale 2.0 -> pocket 64 points (2.7% of the surface,
        #     against 2.8% on real receptors) and ball precision 0.65 (against
        #     0.63 for ball_R8 in the region sweep). Every protein can form a
        #     genuine positive.
        #   a tighter cloud (12 atoms at scale 1.2) -> pocket 22 points and
        #     precision 0.44, with the worst proteins at 0.24. Those never
        #     clear pos_prec 0.3, so they train on the fallback forever and
        #     every `site` assertion goes vacuous.
        #   a looser one (24 at scale 3.0) -> precision 0.84, which clears
        #     everything trivially and would hide a broken label rule.
        g = np.random.default_rng(1000 + i)
        dirn = g.normal(size=3)
        dirn /= np.linalg.norm(dirn)
        lig_xyz = (dirn * 21.0)[None] + g.normal(scale=2.0, size=(16, 3))
        sdf = tmp / f"{iid}.sdf"
        sdf.write_text(_fake_sdf(lig_xyz))
        surfaces[iid] = xyz
        rows.append({"system_id": iid, "input_id": iid, "receptor_key": iid,
                     "split": "train", "input_path": "-", "input_chains": "",
                     "sdf_paths": str(sdf), "fp_row": lig,
                     "group": f"g{lig % 15}", "smiles": ""})
    table = ItemTable(rows)
    check("ItemTable indexes", len(table) == N_PROT
          and len(table.siblings) == N_PROT, table.summary())

    # ---- SurfaceStore: disk round-trip, RNG isolation, determinism, LRU ----
    # Only the structure parse is faked. _sample itself - the seeding, the
    # save/restore of the global RNG, the write-then-rename - is the real one.
    print("\n[2] SurfaceStore")

    class FakeStore(SurfaceStore):
        def _load_atoms(self, row):
            # atoms are the sphere, shrunk; the fake sampler re-expands them,
            # so the surface still comes out of the sampler, not out of here
            xyz = surfaces[row["input_id"]] * 0.9
            n = len(xyz)
            return {"atom_xyz": xyz,
                    "atom_type": np.zeros(n, dtype=np.int64),
                    "atom_groups": np.zeros((n, 6), dtype=np.float32)}

    class FakeSampler:
        """What SurfaceEncoder.sample_surface looks like from the store: a
        STOCHASTIC function of the atoms, which is the whole reason the store
        has to seed it and put the RNG back."""

        def sample_surface(self, atom_xyz, atomtypes, batch_atoms):
            xyz = atom_xyz / 0.9 + 0.05 * torch.randn_like(atom_xyz)
            return xyz, F.normalize(xyz, dim=1), batch_atoms

    sampler = FakeSampler()
    store = FakeStore(tmp / "surfaces", ram=8)
    e1 = store.get(sampler, rows[0])
    e2 = store.get(sampler, rows[0])
    check("RAM hit returns the same entry", e1 is e2, store.report())
    check("disk file written, no .tmp left behind",
          (tmp / "surfaces" / "p000.npz").exists()
          and not list((tmp / "surfaces").glob("*.tmp")))

    store2 = FakeStore(tmp / "surfaces", ram=8)
    e3 = store2.get(sampler, rows[0])
    check("disk round-trip is exact",
          bool(torch.equal(e1["xyz"], e3["xyz"])) and store2.stats["disk"] == 1)

    # Same receptor, a store that has never seen it, and a different amount of
    # randomness consumed first: the surface must be the same anyway, or a
    # cached surface and the labels computed on it would disagree between jobs.
    torch.manual_seed(999)
    torch.randn(17)
    store3 = FakeStore(tmp / "surfaces3", ram=8)
    e4 = store3.get(sampler, rows[0])
    check("a surface depends only on the receptor's identity",
          bool(torch.allclose(e1["xyz"], e4["xyz"], atol=1e-6)),
          f"max difference {float((e1['xyz'] - e4['xyz']).abs().max()):.2e}")

    torch.manual_seed(7)
    want = torch.randn(4)
    torch.manual_seed(7)
    store3.get(sampler, rows[1])                      # a miss: seeds and samples
    got_after = torch.randn(4)
    check("sampling a surface leaves the global RNG untouched",
          bool(torch.equal(want, got_after)),
          "otherwise a cache miss would silently redirect the whole run's "
          "random stream")

    for r in rows[:12]:
        store.get(sampler, r)
    check("LRU bounded to ram", len(store._mem) == 8, f"{len(store._mem)} resident")
    check("region cache travels with the surface",
          "region_cache" in e1 and isinstance(e1["region_cache"], dict))
    # From here on the loop needs every protein resident, or each batch would
    # evict the region caches it is about to reuse. The disk cache is already
    # warm, so this costs a read, not a resampling.
    store = FakeStore(tmp / "surfaces", ram=N_PROT + 4)

    # ---- labels, other sites, fallback -------------------------------------
    print("\n[3] labels, other sites, fallback region")
    labels = LabelCache(table, cutoff=4.0, ram=32)
    lb = labels.get(0, e1)
    n_pk = int(lb["pocket"].sum())
    frac = n_pk / len(e1["xyz"])
    check("pocket is labelled and is a realistic fraction of the surface",
          20 <= n_pk and 0.01 <= frac <= 0.06,
          f"{n_pk} of {len(e1['xyz'])} points = {frac:.1%} "
          "(real receptors: 2.8%)")
    check("no other site for a single-ligand receptor", int(lb["other"].sum()) == 0)

    # two ligands on one receptor: the sibling's site must be ignorable
    shared = [dict(rows[0]), dict(rows[1])]
    shared[1].update(input_id="p000", system_id="p000b")
    t2 = ItemTable(shared)
    l2 = LabelCache(t2, cutoff=4.0, ram=8)
    r0 = l2.get(0, e1)
    check("sibling ligand becomes an other-site mask",
          int(r0["other"].sum()) > 0 and not bool((r0["other"] & r0["pocket"]).any()),
          f"{int(r0['other'].sum())} other-site points, disjoint from the pocket")

    fb = fallback_region(e1["xyz"], lb["pocket"], 8.0)
    centre = e1["xyz"][lb["pocket"]].mean(0)
    R = torch.linalg.qr(torch.randn(3, 3))[0]
    fb_rot = fallback_region(e1["xyz"] @ R.T + 5.0, lb["pocket"], 8.0)
    check("fallback is a ball at the pocket and rigid-motion invariant",
          len(fb) >= 20 and set(fb.tolist()) == set(fb_rot.tolist()),
          f"{len(fb)} points, centre {centre.norm():.1f} A from origin")

    # ---- pooling modes -----------------------------------------------------
    print("\n[4] pooling modes")
    emb = torch.randn(len(e1["xyz"]), EMB)
    prob = torch.rand(len(e1["xyz"]))
    regs = [fb]
    for mode, nsh in (("mean", 3), ("mean_std", 3), ("shells", 3), ("shells", 2)):
        p = pool_regions(emb, prob, regs, mode=mode, xyz=e1["xyz"], n_shells=nsh)
        want = pooled_width(EMB, mode, nsh)
        ok = p.shape == (1, want) and bool(torch.isfinite(p).all())
        detail = f"{mode}(n_shells={nsh}) -> {tuple(p.shape)}"
        if mode == "shells":
            # the shells must be a rigid-motion invariant of the region
            pr = pool_regions(emb, prob, regs, mode=mode,
                              xyz=e1["xyz"] @ R.T + 5.0, n_shells=nsh)
            d = float((p - pr).abs().max())
            ok = ok and d < 1e-4
            detail += f", rotation/translation delta {d:.1e}"
        check(f"pool_regions {mode}", ok, detail)
    perm = torch.randperm(len(regs[0]))
    check("shell pooling is permutation invariant",
          float((pool_regions(emb, prob, [regs[0]], mode="shells", xyz=e1["xyz"])
                 - pool_regions(emb, prob, [regs[0][perm]], mode="shells",
                                xyz=e1["xyz"])).abs().max()) < 1e-5)

    # ---- region labels -----------------------------------------------------
    print("\n[5] region label rules")
    pocket = lb["pocket"]
    idx = pocket.nonzero().squeeze(1)
    far = (~pocket).nonzero().squeeze(1)[:len(idx)]
    lab, cov, pre = region_labels([idx, idx[: len(idx) // 3], far], pocket, None)
    check("exact pocket is positive, a third of it is ignore, elsewhere negative",
          lab.tolist() == [1, -1, 0], f"labels {lab.tolist()}, cover {cov.tolist()}")
    lab_o, _, _ = region_labels([far], pocket, other=torch.zeros_like(pocket)
                                .index_fill_(0, far, True))
    check("a region over another ligand's site is ignore, not negative",
          lab_o.tolist() == [-1])

    # ---- the stub encoder: planted signal ----------------------------------
    print("\n[6] training loop on a planted signal")

    class StubEncoder(nn.Module):
        """SurfaceEncoder's interface with dMaSIF replaced.

        point_emb carries the protein's latent inside its pocket and noise
        outside, so the pooled pocket anchor is recoverable; the pocket logit
        is a learnable readout of it, so pocket BCE has something to train. The
        regions, the pooling, the normalisation and the projection are the real
        implementations.
        """

        def __init__(self, cfg):
            super().__init__()
            self.cfg = cfg
            self.pool_norm = (nn.BatchNorm1d(cfg.pooled_dims)
                              if cfg.anchor_norm == "batch" else nn.Identity())
            self.project = nn.Sequential(
                nn.Linear(cfg.pooled_dims, 64), nn.LeakyReLU(0.2),
                nn.Linear(64, cfg.out_dim))
            self.pocket_head = nn.Linear(EMB, 1)
            self.mix = nn.Linear(EMB, EMB)

        # the real tail, so the normalisation and projection under test are
        # the ones training uses
        project_anchors = SurfaceEncoder.project_anchors

        def sample_surface(self, *args, **kw):      # never reached: store is faked
            raise AssertionError("the stub does not sample surfaces")

        def forward(self, P, forced=None, generator=None, project=True):
            c = self.cfg
            xyz, batch = P["xyz"], P["batch"]
            emb = self.mix(P["planted"])
            logit = self.pocket_head(emb).squeeze(-1)
            prob = torch.sigmoid(logit)
            K = c.k_pos + c.k_neg
            B = int(batch.max()) + 1
            vec = emb.new_zeros(B, K, c.pooled_dims)
            valid = torch.zeros(B, K, dtype=torch.bool)
            is_pos = torch.zeros(B, K, dtype=torch.bool)
            size = torch.zeros(B, K, dtype=torch.long)
            centre = xyz.new_zeros(B, K, 3)
            all_regions = []
            for b in range(B):
                sel = (batch == b).nonzero().squeeze(1)
                regions, pos = select_regions(
                    xyz[sel], prob[sel], c.k_pos, c.k_neg,
                    threshold=c.region_threshold, top_fraction=c.region_top_fraction,
                    link_radius=c.link_radius, smooth_radius=c.smooth_radius,
                    min_points=c.min_region_points, neg_quantile=c.neg_quantile,
                    neg_exclusion=c.neg_exclusion, generator=generator,
                    method=c.region_method, ball_radius=c.region_ball_radius,
                    ball_shift=c.region_ball_shift, ball_rank=c.region_ball_rank,
                    cache=None if P.get("region_cache") is None else P["region_cache"][b])
                n = len(regions)
                if n:
                    vec[b, :n] = pool_regions(emb[sel], prob[sel], regions,
                                              mode=c.pool_mode, xyz=xyz[sel],
                                              n_shells=c.n_shells)
                    centre[b, :n] = pool_regions(xyz[sel], prob[sel].detach(), regions)
                    size[b, :n] = torch.tensor([len(r) for r in regions])
                    is_pos[b, :n] = pos
                    valid[b, :n] = True
                all_regions.append([sel[r] for r in regions])
            return {"xyz": xyz, "normals": P["normals"], "batch": batch,
                    "point_emb": emb, "pocket_logit": logit,
                    "anchor_vec": self.project_anchors(vec, valid) if project else None,
                    "anchor_pooled": vec, "anchor_valid": valid,
                    "anchor_is_pos": is_pos, "anchor_center": centre,
                    "anchor_size": size, "anchor_regions": all_regions}

    cfg = EncoderConfig(emb_dims=EMB, k_pos=4, k_neg=4, region_method="ball",
                        region_ball_radius=8.0, region_ball_shift=2,
                        region_top_fraction=0.05, min_region_points=20,
                        pool_mode=a.pool_mode, n_shells=3,
                        anchor_norm=a.anchor_norm, out_dim=32)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = StubEncoder(cfg)
            self.ligand = LigandEncoder(LigandConfig(arm="fp_linear", out_dim=32))
            self.logit_scale = nn.Parameter(torch.log(torch.tensor(1 / 0.07)))

        def scale(self):
            return self.logit_scale.exp().clamp(max=100.0)

        def scores(self, anchors, usable, lig):
            sim = torch.einsum("bkd,ld->bkl", anchors, lig)
            return sim.masked_fill(~usable[..., None], -1e4).max(1).values * self.scale()

    model = Model()
    ligands = _FakeLigands(bits)

    def batch_for(idx):
        """build_batch's job, with the planted embedding attached."""
        items, labs, planted = [], [], []
        for i in idx:
            entry = store.get(sampler, table.rows[i])
            lab = labels.get(int(i), entry)
            items.append(entry)
            labs.append(lab)
            e = torch.randn(len(entry["xyz"]), EMB) * 0.3
            zi = torch.from_numpy(z[table.fp_row[i]])
            # Two separate things, because they train different parts. Channel
            # LATENT is a CONSTANT pocket marker, so a linear pocket head can
            # find the pocket and the selector gets real regions; channels
            # 0..LATENT-1 carry the protein's identity, which is what the
            # projection and the ligand encoder have to pair up. The identity
            # alone would not do: z changes sign across proteins, so no single
            # linear readout detects every pocket, the head would stay blind
            # and the fallback would fire forever.
            e[lab["pocket"], LATENT] += 4.0
            e[lab["pocket"], :LATENT] += zi * 3.0
            planted.append(e)
        P = to_batch(items, "cpu")
        P["planted"] = torch.cat(planted)
        return P, labs

    args = argparse.Namespace(
        pos_cover=0.5, pos_prec=0.3, ignore_overlap=0.1, fallback=True,
        tanimoto=0.8, eval_batch=16, cutoff=4.0)
    # A fixture guard, not a code test: if a ball at the right place cannot
    # clear pos_prec, no region is ever positive, the loop trains on the
    # fallback forever, and the site / fallback-rate assertions below pass
    # while measuring nothing. Real receptors sit at 0.63 here.
    best_prec = int(lb["pocket"].sum()) / len(fb)
    check("a correctly placed ball clears pos_prec with room to spare",
          best_prec >= args.pos_prec + 0.1,
          f"pocket {int(lb['pocket'].sum())} points, a ball holds {len(fb)}, so "
          f"precision at best {best_prec:.2f} against pos_prec {args.pos_prec} "
          f"(real receptors: 0.63)")
    tr, va = split_by_group(table.group, 0.3, 0)
    check("group-disjoint split", not (set(table.group[tr]) & set(table.group[va])),
          f"{len(tr)} train / {len(va)} val, "
          f"{len(set(table.group[tr]))} / {len(set(table.group[va]))} groups")
    members, counts = group_index(table.group[tr])
    rng_s = np.random.default_rng(0)
    seen, dups = set(), 0
    for _ in range(200):
        b = draw_batch(rng_s, members, counts, 6, 1.0)
        g = [table.group[tr[j]] for j in b]
        seen |= set(g)
        dups += len(g) != len(set(g))
    check("sampler draws one row per target group, and reaches every group",
          dups == 0 and seen == set(table.group[tr]),
          f"{dups} batches with a repeated group, {len(seen)} of "
          f"{len(set(table.group[tr]))} groups seen over 200 batches")

    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    gen = torch.Generator().manual_seed(0)
    mrrs, fbs = [], []

    def val():
        model.eval()
        with torch.no_grad():
            P, labs = batch_for(va)
            blk = anchor_block(model, P, labs, args)
            keep = blk["usable"].any(1)
            lig = model.ligand(ligands.collate(table.fp_row[va[keep.numpy()]]))
            s = model.scores(blk["anchors"][keep], blk["usable"][keep], lig)
            m = pair_mask(table, va[keep.numpy()], ligands, "cpu", 0.8)
            met = retrieval_metrics(s.T, m)
            met["fb"] = float(blk["fallback"].float().mean())
        model.train()
        return met

    m0 = val()
    print(f"    step 0: MRR {m0['mrr']:.3f} (chance {m0['chance']:.3f}), "
          f"fallback {m0['fb']:.2f}")
    rng2 = np.random.default_rng(1)
    for step in range(1, a.steps + 1):
        idx = tr[draw_batch(rng2, members, counts, a.batch, 1.0)]
        P, labs = batch_for(idx)
        blk = anchor_block(model, P, labs, args, generator=gen)
        keep = blk["usable"].any(1)
        if int(keep.sum()) < 2:
            continue
        idx_k = idx[keep.numpy()]
        lig = model.ligand(ligands.collate(table.fp_row[idx_k]))
        s = model.scores(blk["anchors"][keep], blk["usable"][keep], lig)
        loss = info_nce(s, pair_mask(table, idx_k, ligands, "cpu", 0.8))
        pl = torch.cat([lb["pocket"] for lb in labs]).float()
        loss = loss + 0.5 * pocket_loss(blk["out"]["pocket_logit"], pl, 20.0)
        loss = loss + 0.5 * site_loss(
            torch.einsum("bkd,bd->bk", blk["anchors"][keep], lig) * model.scale(),
            blk["label"][keep])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        if step % max(a.steps // 6, 1) == 0:
            m = val()
            mrrs.append(m["mrr"])
            fbs.append(m["fb"])
            print(f"    step {step:4d}: loss {loss.item():7.4f}  MRR {m['mrr']:.3f}  "
                  f"top1 {m['top1']:.3f}  fallback {m['fb']:.2f}", flush=True)

    best_mrr = max(mrrs) if mrrs else 0.0
    floor = max(a.min_mrr, 2 * m0["chance"])
    check("MRR climbs well above chance on the planted signal",
          best_mrr >= floor, f"best {best_mrr:.3f} vs floor {floor:.3f} "
          f"(chance {m0['chance']:.3f}, untrained {m0['mrr']:.3f})")
    check("the fallback rate falls as the head learns",
          fbs[-1] <= m0["fb"] + 0.05,
          f"{m0['fb']:.2f} untrained -> {fbs[-1]:.2f} trained")

    # ---- gradients reach everything ---------------------------------------
    print("\n[7] gradients and checkpoint round-trip")
    dead = sorted(n for n, p in model.named_parameters()
                  if p.grad is None or float(p.grad.abs().sum()) == 0)
    check("every parameter has a gradient", not dead, f"dead: {dead or 'none'}")

    torch.save({"state_dict": model.state_dict(), "step": 7}, tmp / "ck.pt")
    model2 = Model()
    model2.load_state_dict(torch.load(tmp / "ck.pt",
                                      weights_only=False)["state_dict"])

    def score_block(m, P, labs, idx):
        blk = anchor_block(m, P, labs, args)
        lig = m.ligand(ligands.collate(table.fp_row[idx]))
        return m.scores(blk["anchors"], blk["usable"], lig)

    model.eval()
    model2.eval()
    with torch.no_grad():
        idx = va[:8]
        P, labs = batch_for(idx)
        s_a = score_block(model, P, labs, idx)
        s_b = score_block(model2, P, labs, idx)
    delta = float((s_a - s_b).abs().max())
    check("a reloaded checkpoint scores identically", delta < 1e-5,
          f"max delta {delta:.2e}; this is what makes --resume meaningful, and "
          "it covers the BatchNorm running statistics as much as the weights")

    print("\nSELFTEST " + ("PASSED" if not fails else f"FAILED: {fails}"))
    print("Nothing here involves dMaSIF, KeOps or a GPU, so it says nothing about\n"
          "surface sampling or the convolution; chemo_test.sbatch's smoke pass is\n"
          "what exercises those.")
    if fails:
        raise SystemExit(1)


def _fake_sdf(xyz):
    """A minimal V2000 SDF for read_sdf_heavy, so the selftest exercises the
    real ligand path rather than a shortcut around it."""
    head = f"fake\n  selftest\n\n{len(xyz):3d}  0  0  0  0  0  0  0  0  0999 V2000\n"
    body = "".join(f"{x:10.4f}{y:10.4f}{zz:10.4f} C   0  0  0  0  0\n"
                   for x, y, zz in xyz)
    return head + body + "M  END\n$$$$\n"


class _FakeLigands:
    """LigandData's surface, for the selftest's planted fingerprints."""

    def __init__(self, fp):
        import torch

        self.fp = fp
        self._t = torch.from_numpy(fp.astype(np.float32))
        self.failed = np.zeros(len(fp), dtype=bool)

    def __len__(self):
        return len(self.fp)

    def collate(self, rows, device="cpu"):
        rows = np.asarray(rows, dtype=np.int64)
        return {"fp": self._t[rows].to(device), "n_mol": len(rows)}


# ---------------------------------------------------------------------------
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


def add_model_args(p):
    p.add_argument("--arm", default="gnn", help="ligand_encoder.ARMS")
    p.add_argument("--out_dim", type=int, default=128)
    p.add_argument("--hidden", type=int, default=512)
    p.add_argument("--gnn_hidden", type=int, default=128)
    p.add_argument("--gnn_layers", type=int, default=4)
    p.add_argument("--lig_dropout", type=float, default=0.0)
    p.add_argument("--enc_dropout", type=float, default=0.0)
    p.add_argument("--chemo", action="store_true",
                   help="add chemotype.py's 12 pharmacophore channels")
    p.add_argument("--pool_mode", default="mean",
                   help="mean | mean_std | shells (anchors.pool_regions)")
    p.add_argument("--n_shells", type=int, default=3)
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
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--clip", type=float, default=5.0)
    p.add_argument("--tau", type=float, default=0.07)
    p.add_argument("--alpha", type=float, default=1.0,
                   help="1 = uniform over targets, 0 = uniform over rows")
    p.add_argument("--lam_pocket", type=float, default=1.0,
                   help="weight of the per-point pocket BCE; 0 lets the regions drift")
    p.add_argument("--lam_site", type=float, default=0.0,
                   help="weight of the within-protein anchor CE (which anchor)")
    p.add_argument("--pos_weight", type=float, default=0.0,
                   help="pocket BCE positive weight; 0 = the running ratio")
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--val_split", default="",
                   help="use the dataset's own split(s) as validation instead "
                        "of carving a group-disjoint one out of these rows")
    p.add_argument("--val_max", type=int, default=400,
                   help="cap the validation set; every protein costs a forward")
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

    p = sub.add_parser("selftest", help="planted signal; no data, no GPU, no KeOps")
    p.add_argument("--tmp", default="train_selftest")
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
