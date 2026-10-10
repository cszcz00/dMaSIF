"""Anchor selection and pooling on one protein surface.

An anchor is a surface point whose neighbourhood is pooled into one vector for
the representation space. Each protein contributes K = k_pos + k_neg anchors:

    pocket candidates   highest pocket score, with non-maximum suppression:
                        once a point is taken, nothing within nms_radius can
                        be, so one pocket yields one anchor and the k_pos slots
                        spread over k_pos different pocket-like regions.
    negatives           low-score points (below the neg_quantile of this
                        protein's scores), optionally restricted to flat
                        surface, spread by farthest-point sampling so they
                        cover the protein instead of clustering.

Selection is discrete and runs without gradients. The pocket head learns from
its own per-point loss, and the pooled vectors carry gradient back into the
point embeddings, so the network trains through what it selected.

Pure torch, no KeOps: selection runs on a CPU copy of the coordinates, because
the greedy loops would otherwise synchronise the GPU at every step.

Depends on: torch.
Imported by: surface_encoder.py.
"""

import torch


def _gumbel_order(scores, temperature, generator):
    """Indices by descending score; with a temperature, a random order in which
    each point's chance of coming first is softmax(score / temperature)."""
    if not temperature:
        return torch.argsort(scores, descending=True)
    u = torch.rand(scores.shape, generator=generator).clamp_(1e-10, 1 - 1e-10)
    return torch.argsort(scores / temperature - torch.log(-torch.log(u)), descending=True)


def _nms(xyz, order, k, radius, blocked):
    """Walk `order`, keep a point unless within `radius` of one already kept."""
    keep = []
    r2 = radius * radius
    for i in order.tolist():
        if len(keep) == k:
            break
        if blocked[i]:
            continue
        keep.append(i)
        blocked |= ((xyz - xyz[i]) ** 2).sum(-1) < r2
    return keep


def _farthest(xyz, pool, k, start_dist, generator):
    """Farthest-point sampling over the indices in `pool`."""
    if k <= 0 or len(pool) == 0:
        return []
    pts = xyz[pool]
    dist = start_dist[pool].clone()
    if torch.isinf(dist).all():
        first = int(torch.randint(len(pool), (1,), generator=generator))
    else:
        first = int(torch.argmax(dist))
    out = [first]
    dist = torch.minimum(dist, ((pts - pts[first]) ** 2).sum(-1))
    while len(out) < min(k, len(pool)):
        nxt = int(torch.argmax(dist))
        if dist[nxt] <= 0:
            break
        out.append(nxt)
        dist = torch.minimum(dist, ((pts - pts[nxt]) ** 2).sum(-1))
    return pool[torch.tensor(out)].tolist()


@torch.no_grad()
def select_anchors(xyz, scores, k_pos, k_neg, nms_radius, neg_quantile=0.5,
                   neg_allowed=None, temperature=None, forced=None, generator=None):
    """Choose K = k_pos + k_neg anchor points on one protein surface.

    xyz          (N, 3) surface points
    scores       (N,)   pocket logits (any monotone score works)
    nms_radius   A; no two pocket candidates closer than this, and no negative
                 closer than this to a pocket candidate
    neg_quantile negatives come from points scoring at or below this quantile
    neg_allowed  (N,) bool, optional extra restriction on negatives (e.g. flat)
    temperature  None: strict ranking. > 0: sample candidates in proportion to
                 softmax(score / T) before suppression, so training explores
                 beyond the current top-ranked regions
    forced       (F,) point indices taken as pocket candidates first, e.g. the
                 point nearest the true ligand during training

    Returns (idx, is_pos): (K,) long, -1 where fewer than K points qualified,
    and (K,) bool, True for pocket-candidate slots.
    """
    dev = xyz.device
    xyz = xyz.detach().float().cpu()
    scores = scores.detach().float().cpu()
    n = len(xyz)

    blocked = torch.zeros(n, dtype=torch.bool)
    pos = []
    if forced is not None and len(forced):
        pos = _nms(xyz, torch.as_tensor(forced).cpu(), k_pos, nms_radius, blocked)
    pos += _nms(xyz, _gumbel_order(scores, temperature, generator),
                k_pos - len(pos), nms_radius, blocked)

    pool_mask = scores <= torch.quantile(scores, neg_quantile)
    if neg_allowed is not None:
        pool_mask &= neg_allowed.detach().cpu()
    if pos:
        d_pos = torch.cdist(xyz, xyz[pos]).min(1).values
        pool_mask &= d_pos >= nms_radius
        start = d_pos ** 2
    else:
        start = torch.full((n,), float("inf"))
    neg = _farthest(xyz, pool_mask.nonzero().squeeze(1), k_neg, start, generator)

    idx = torch.full((k_pos + k_neg,), -1, dtype=torch.long)
    idx[: len(pos)] = torch.tensor(pos, dtype=torch.long)
    idx[k_pos : k_pos + len(neg)] = torch.tensor(neg, dtype=torch.long)
    is_pos = torch.zeros(k_pos + k_neg, dtype=torch.bool)
    is_pos[:k_pos] = True
    return idx.to(dev), is_pos.to(dev)


def pool_at_anchors(xyz, emb, idx, radius, sigma=None):
    """Gaussian-weighted mean of point embeddings around each anchor.

    Weights exp(-d^2 / 2 sigma^2) within `radius`, zero beyond; sigma defaults
    to radius / 2. Rows for missing anchors (idx == -1) are zero.
    Differentiable in `emb`. Returns (K, E).
    """
    sigma = sigma or radius / 2
    valid = idx >= 0
    d2 = torch.cdist(xyz[idx.clamp(min=0)], xyz) ** 2
    w = torch.exp(-d2 / (2 * sigma * sigma)) * (d2 < radius * radius)
    w = w * valid[:, None]
    w = w / w.sum(1, keepdim=True).clamp_min(1e-8)
    return w @ emb


@torch.no_grad()
def anchor_pocket_overlap(xyz, idx, pocket_mask, radius):
    """Fraction of each anchor's neighbourhood (within radius) that is labelled
    pocket. The training-time link between an anchor and a ligand: the anchor
    that overlaps the ligand's pocket most is that ligand's positive.
    Returns (K,), 0 for missing anchors."""
    d2 = torch.cdist(xyz[idx.clamp(min=0)], xyz) ** 2
    near = d2 < radius * radius
    frac = (near & pocket_mask[None, :]).sum(1) / near.sum(1).clamp_min(1)
    return frac * (idx >= 0)
