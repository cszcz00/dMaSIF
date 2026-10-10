"""Region anchors on one protein surface.

An anchor is a REGION of surface points, pooled into one vector for the
representation space. Regions follow the predicted pocket map, so their size
and shape follow the pocket instead of a fixed radius.

    pocket regions   1. smooth the per-point pocket probability over
                        `smooth_radius`, so noise does not split a pocket
                     2. candidates: this protein's top `top_fraction` of
                        points by smoothed probability (rank-based, so it
                        does not depend on how the head is calibrated - a
                        pos_weighted loss inflates every probability), or
                        those above an absolute `threshold` if one is given
                     3. flood from the highest-probability points down: a
                        candidate joins the region of its strongest
                        already-assigned neighbour within `link_radius`, or
                        starts a region if it has none. Two pockets joined by a
                        lower ridge stay two regions (watershed)
                     4. drop regions under `min_points`; rank by total
                        probability; keep the top k_pos
    negative regions low-probability patches, seeded by farthest-point sampling
                     away from the pocket regions, each the median size of
                     the pocket regions so size cannot tell them apart
    forced regions   given point sets (the labelled pocket during training)
                     inserted first; predicted regions mostly inside one are
                     dropped as duplicates

Pooling weights each point by its pocket probability, so the downstream loss
also trains the pocket head. Region construction itself is discrete and runs
without gradients, on a CPU copy (its loops would stall a GPU).

Depends on: torch.
Imported by: surface_encoder.py.
"""

import torch


# ----------------------------------------------------------------------------
# surface neighbourhoods
# ----------------------------------------------------------------------------
def radius_neighbors(xyz, radius, chunk=1024):
    """CSR neighbour lists within `radius` (self included) -> (offsets, nbrs)."""
    r2 = radius * radius
    rows, cols = [], []
    for s in range(0, len(xyz), chunk):
        d2 = torch.cdist(xyz[s:s + chunk], xyz) ** 2
        r, c = (d2 < r2).nonzero(as_tuple=True)
        rows.append(r + s)
        cols.append(c)
    rows, cols = torch.cat(rows), torch.cat(cols)
    counts = torch.bincount(rows, minlength=len(xyz))
    offsets = torch.zeros(len(xyz) + 1, dtype=torch.long)
    offsets[1:] = torch.cumsum(counts, 0)
    return offsets, cols


def smooth(values, offsets, nbrs):
    """Mean of `values` over each point's neighbour list."""
    seg = torch.repeat_interleave(torch.arange(len(offsets) - 1), offsets.diff())
    out = torch.zeros_like(values).index_add_(0, seg, values[nbrs])
    return out / offsets.diff().clamp_min(1)


# ----------------------------------------------------------------------------
# region construction
# ----------------------------------------------------------------------------
def watershed_regions(xyz, prob, threshold, link_radius, min_points, smooth_radius,
                      top_fraction=None):
    """Pocket regions from a probability map -> list of index tensors, ranked
    by total (smoothed) probability, highest first. top_fraction, when given,
    replaces `threshold` with this protein's (1 - top_fraction) quantile."""
    if smooth_radius > 0:
        prob = smooth(prob, *radius_neighbors(xyz, smooth_radius))
    off, nb = radius_neighbors(xyz, link_radius)
    if top_fraction is not None:
        threshold = float(torch.quantile(prob, 1 - top_fraction))

    label = torch.full((len(xyz),), -1, dtype=torch.long)
    cand = (prob >= threshold).nonzero().squeeze(1)
    order = cand[torch.argsort(prob[cand], descending=True)]
    n_regions = 0
    for i in order.tolist():
        nbr = nb[off[i]:off[i + 1]]
        lab = label[nbr]
        nbr, lab = nbr[lab >= 0], lab[lab >= 0]
        if len(lab):
            label[i] = lab[torch.argmax(prob[nbr])]
        else:
            label[i] = n_regions
            n_regions += 1

    regions = [(label == r).nonzero().squeeze(1) for r in range(n_regions)]
    regions = [r for r in regions if len(r) >= min_points]
    regions.sort(key=lambda r: -float(prob[r].sum()))
    return regions


def negative_regions(xyz, prob, k, size, exclude, neg_quantile, allowed, generator):
    """k low-probability patches of `size` points each, spread over the surface.

    Seeds: farthest-point sampling among points at or below the neg_quantile of
    prob, not in `exclude`, and allowed. Region: the `size` points nearest the
    seed, so each patch is contiguous and matches the pocket regions in size.
    """
    pool = (prob <= torch.quantile(prob, neg_quantile)) & ~exclude
    if allowed is not None:
        pool &= allowed
    pool = pool.nonzero().squeeze(1)
    if k <= 0 or len(pool) == 0:
        return []
    pts = xyz[pool]
    if exclude.any():
        dist = torch.cdist(pts, xyz[exclude]).min(1).values ** 2
        seeds = [int(torch.argmax(dist))]
    else:
        dist = torch.full((len(pool),), float("inf"))
        seeds = [int(torch.randint(len(pool), (1,), generator=generator))]
    dist = torch.minimum(dist, ((pts - pts[seeds[0]]) ** 2).sum(-1))
    while len(seeds) < min(k, len(pool)):
        nxt = int(torch.argmax(dist))
        if dist[nxt] <= 0:
            break
        seeds.append(nxt)
        dist = torch.minimum(dist, ((pts - pts[nxt]) ** 2).sum(-1))
    size = min(size, len(xyz))
    return [torch.cdist(pts[s:s + 1], xyz)[0].topk(size, largest=False).indices
            for s in seeds]


@torch.no_grad()
def select_regions(xyz, prob, k_pos, k_neg, threshold=0.5, top_fraction=None, link_radius=2.0,
                   smooth_radius=2.0, min_points=20, neg_quantile=0.5,
                   neg_size=None, neg_exclusion=4.0, neg_allowed=None,
                   forced=None, dup_overlap=0.5, generator=None):
    """Pocket and negative regions on one protein surface.

    xyz (N, 3), prob (N,) pocket probabilities in [0, 1].
    top_fraction: grow regions from this fraction of the protein's points
    (highest smoothed probability); overrides `threshold` when given.
    forced: list of index tensors taken as pocket regions first.
    neg_size: points per negative region; default the median pocket-region
    size, or 4 * min_points when there are none.
    neg_exclusion: A; negative points must be this far from every pocket region.

    Returns (regions, is_pos): a list of up to k_pos + k_neg index tensors
    (pocket regions first), and a bool tensor marking which are pocket regions.
    """
    xyz = xyz.detach().float().cpu()
    prob = prob.detach().float().cpu()

    pos = [torch.as_tensor(f).long().cpu() for f in (forced or [])][:k_pos]
    taken = torch.zeros(len(xyz), dtype=torch.bool)
    for f in pos:
        taken[f] = True
    for r in watershed_regions(xyz, prob, threshold, link_radius, min_points, smooth_radius,
                               top_fraction):
        if len(pos) == k_pos:
            break
        if taken[r].float().mean() > dup_overlap:
            continue
        pos.append(r)

    in_pos = torch.zeros(len(xyz), dtype=torch.bool)
    for r in pos:
        in_pos[r] = True
    exclude = in_pos.clone()
    if in_pos.any() and neg_exclusion > 0:
        exclude = torch.cdist(xyz, xyz[in_pos]).min(1).values < neg_exclusion
    if neg_size is None:
        neg_size = int(torch.tensor([len(r) for r in pos]).median()) if pos else 4 * min_points
    neg = negative_regions(xyz, prob, k_neg, neg_size, exclude, neg_quantile,
                           None if neg_allowed is None else neg_allowed.detach().cpu(),
                           generator)

    is_pos = torch.tensor([True] * len(pos) + [False] * len(neg), dtype=torch.bool)
    return pos + neg, is_pos


# ----------------------------------------------------------------------------
# pooling and evaluation
# ----------------------------------------------------------------------------
def pool_regions(emb, prob, regions, eps=1e-6):
    """Probability-weighted mean of point embeddings over each region -> (R, E).
    Differentiable in emb and prob."""
    out = []
    for r in regions:
        r = r.to(emb.device)
        w = prob[r] + eps
        out.append((w[:, None] * emb[r]).sum(0) / w.sum())
    return torch.stack(out) if out else emb.new_zeros(0, emb.shape[1])


@torch.no_grad()
def region_overlap(regions, pocket_mask):
    """Per region: (fraction of the region that is pocket, fraction of the
    pocket the region covers). Training uses these to decide which region is a
    ligand's positive; evaluation uses them for pocket recall."""
    pocket_mask = pocket_mask.cpu()
    n_pocket = max(int(pocket_mask.sum()), 1)
    prec = torch.tensor([float(pocket_mask[r.cpu()].float().mean()) for r in regions])
    cover = torch.tensor([float(pocket_mask[r.cpu()].sum()) / n_pocket for r in regions])
    return prec, cover
