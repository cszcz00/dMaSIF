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
                     3. group candidates into regions, by `method`:
                        watershed   flood from the highest-probability points
                                    down: a candidate joins the region of its
                                    strongest already-assigned neighbour within
                                    `link_radius`, or starts a region if it has
                                    none. When a candidate touches two regions,
                                    the one with the lower peak is merged in if
                                    its peak stands less than `persistence`
                                    above the candidate (ToMATo). persistence
                                    is in per-protein quantile units: 0 never
                                    merges (plain watershed), >= top_fraction
                                    merges everything connected
                        components  single-linkage clusters of candidates at
                                    `link_radius` (P2Rank); ranked by sum p^2
                        ball        greedy NMS: centre a ball of `ball_radius`
                                    on the strongest unsuppressed candidate
                                    (optionally moved by `ball_shift` mean-shift
                                    steps over candidates), take every surface
                                    point inside, suppress candidates inside,
                                    repeat. Fixed size; ranked by seed strength
                     4. drop regions under `min_points`; rank by total
                        probability (unless the method says otherwise);
                        keep the top k_pos
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
Imported by: surface_encoder.py, region_sweep.py.
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


# `cache`, where accepted, is a dict reused across calls on the SAME xyz and
# prob (region_sweep.py runs many selectors per protein); None recomputes.
def _neighbors(xyz, radius, cache=None):
    key = ("nbr", float(radius))
    if cache is None or key not in cache:
        nbrs = radius_neighbors(xyz, radius)
        if cache is None:
            return nbrs
        cache[key] = nbrs
    return cache[key]


def smoothed(xyz, prob, radius, cache=None):
    """prob averaged over `radius`; unchanged when radius <= 0."""
    if radius <= 0:
        return prob
    key = ("smooth", float(radius))
    if cache is None or key not in cache:
        out = smooth(prob, *_neighbors(xyz, radius, cache))
        if cache is None:
            return out
        cache[key] = out
    return cache[key]


def candidates(prob, threshold, top_fraction=None):
    """Bool mask of candidate points: the top `top_fraction` by prob, or
    prob >= threshold when top_fraction is None."""
    if top_fraction is not None:
        threshold = float(torch.quantile(prob, 1 - top_fraction))
    return prob >= threshold


# ----------------------------------------------------------------------------
# region construction
# ----------------------------------------------------------------------------
REGION_METHODS = ("watershed", "components", "ball")


def watershed_regions(xyz, prob, threshold, link_radius, min_points, smooth_radius,
                      top_fraction=None, persistence=0.0, rank_by="sum", cache=None):
    """Pocket regions from a probability map -> list of index tensors, ranked
    by total (smoothed) probability, highest first (sum of squares when
    rank_by="sumsq"). top_fraction, when given, replaces `threshold` with this
    protein's (1 - top_fraction) quantile.

    persistence: merge threshold in quantile units of this protein's smoothed
    probability. 0 is plain watershed; float("inf") merges every connected
    candidate (single-linkage connected components)."""
    prob = smoothed(xyz, prob, smooth_radius, cache)
    off, nb = _neighbors(xyz, link_radius, cache)
    cand = candidates(prob, threshold, top_fraction).nonzero().squeeze(1)
    order = cand[torch.argsort(prob[cand], descending=True)]
    # height of each point as its quantile on this protein, so persistence
    # does not depend on how the head is calibrated
    height = torch.empty(len(xyz))
    height[torch.argsort(prob)] = torch.arange(len(xyz), dtype=torch.float) / max(len(xyz), 1)

    label = torch.full((len(xyz),), -1, dtype=torch.long)
    parent, peak = [], []  # union-find over region ids; peak height per region

    def find(r):
        while parent[r] != r:
            parent[r] = parent[parent[r]]
            r = parent[r]
        return r

    for i in order.tolist():
        nbr = nb[off[i]:off[i + 1]]
        lab = label[nbr]
        nbr, lab = nbr[lab >= 0], lab[lab >= 0]
        if not len(lab):
            label[i] = len(parent)
            parent.append(len(parent))
            peak.append(float(height[i]))
            continue
        label[i] = lab[torch.argmax(prob[nbr])]
        if persistence <= 0:
            continue
        h = float(height[i])
        for r in {find(int(x)) for x in lab.tolist()}:
            a, b = find(int(label[i])), find(r)
            if a == b:
                continue
            low, high = (a, b) if peak[a] < peak[b] else (b, a)
            if peak[low] - h < persistence:
                parent[low] = high

    if not parent:
        return []
    root = torch.tensor([find(r) for r in range(len(parent))])
    assigned = label >= 0
    label[assigned] = root[label[assigned]]
    regions = [(label == r).nonzero().squeeze(1) for r in root.unique().tolist()]
    regions = [r for r in regions if len(r) >= min_points]
    weight = prob ** 2 if rank_by == "sumsq" else prob
    regions.sort(key=lambda r: -float(weight[r].sum()))
    return regions


def component_regions(xyz, prob, threshold, link_radius, min_points, smooth_radius,
                      top_fraction=None, cache=None):
    """P2Rank-style pockets: connected components of the candidates at
    link_radius, ranked by sum of squared (smoothed) probability."""
    return watershed_regions(xyz, prob, threshold, link_radius, min_points, smooth_radius,
                             top_fraction, persistence=float("inf"), rank_by="sumsq",
                             cache=cache)


def ball_regions(xyz, prob, threshold, radius, min_points, smooth_radius,
                 top_fraction=None, shift=1, suppress=None, cache=None):
    """Greedy NMS balls -> list of index tensors, in seed order (strongest
    first). Seed: the highest smoothed-probability candidate not yet
    suppressed. Centre: the seed, moved by `shift` mean-shift steps (centroid
    of the candidates within `radius`, weighted by probability), so it can sit
    in the pocket's void rather than on its rim. Region: every surface point
    within `radius` of the centre, candidate or not. Then candidates within
    `suppress` (default `radius`) of the centre are removed.

    Euclidean, not geodesic: a ball can reach across a thin cleft. Pooling
    is probability-weighted, so low-probability points inside count little."""
    prob = smoothed(xyz, prob, smooth_radius, cache)
    cand = candidates(prob, threshold, top_fraction)
    w_cand = prob * cand
    r2 = radius * radius
    s2 = r2 if suppress is None else suppress * suppress
    avail = cand.clone()
    regions = []
    while avail.any():
        idx = avail.nonzero().squeeze(1)
        seed = int(idx[torch.argmax(prob[idx])])
        centre = xyz[seed]
        for _ in range(shift):
            inside = ((xyz - centre) ** 2).sum(-1) < r2
            w = w_cand[inside]
            if float(w.sum()) <= 0:
                break
            centre = (w[:, None] * xyz[inside]).sum(0) / w.sum()
        d2 = ((xyz - centre) ** 2).sum(-1)
        avail &= d2 >= s2
        avail[seed] = False  # always progress, even if the centre moved away
        region = (d2 < r2).nonzero().squeeze(1)
        if len(region) >= min_points:
            regions.append(region)
    return regions


def pocket_regions(xyz, prob, method="watershed", threshold=0.5, top_fraction=None,
                   link_radius=2.0, smooth_radius=2.0, min_points=20, persistence=0.0,
                   ball_radius=10.0, ball_shift=1, cache=None):
    """Ranked pocket regions by `method` (see REGION_METHODS and the module
    docstring). Parameters a method does not use are ignored."""
    if method == "watershed":
        return watershed_regions(xyz, prob, threshold, link_radius, min_points, smooth_radius,
                                 top_fraction, persistence=persistence, cache=cache)
    if method == "components":
        return component_regions(xyz, prob, threshold, link_radius, min_points, smooth_radius,
                                 top_fraction, cache=cache)
    if method == "ball":
        return ball_regions(xyz, prob, threshold, ball_radius, min_points, smooth_radius,
                            top_fraction, shift=ball_shift, cache=cache)
    raise ValueError(f"unknown region method {method!r}; expected one of {REGION_METHODS}")


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
                   forced=None, dup_overlap=0.5, generator=None,
                   method="watershed", persistence=0.0, ball_radius=10.0, ball_shift=1):
    """Pocket and negative regions on one protein surface.

    xyz (N, 3), prob (N,) pocket probabilities in [0, 1].
    top_fraction: grow regions from this fraction of the protein's points
    (highest smoothed probability); overrides `threshold` when given.
    method, persistence, ball_radius, ball_shift: how candidates become
    regions, see pocket_regions().
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
    for r in pocket_regions(xyz, prob, method, threshold, top_fraction, link_radius,
                            smooth_radius, min_points, persistence, ball_radius, ball_shift):
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
