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
                                    (or `ball_rank` sum / sumsq inside the ball)
                        nms         score-then-suppress balls: mean shift from
                                    every candidate gives proposal centres;
                                    each ball scored by the probability inside
                                    it (`nms_score`); greedy NMS by point-set
                                    IoU (`nms_iou`); stop below `nms_alpha` x
                                    the best score
                        compball    components choose and rank the pockets;
                                    each becomes a ball of `ball_radius` at its
                                    weighted centroid
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
also trains the pocket head. `pool_regions` can return one weighted mean per
region (the original), mean + standard deviation, or means in equal-count
radial shells around the region's own centroid, which is the only one of the
three that keeps any of the arrangement INSIDE a region. Region construction
itself is discrete and runs without gradients, on a CPU copy (its loops would
stall a GPU).

Depends on: torch.
Imported by: surface_encoder.py, region_sweep.py.
"""

import time

import torch


# ----------------------------------------------------------------------------
# surface neighbourhoods
# ----------------------------------------------------------------------------
# Where radius_neighbors runs its all-pairs distances. None = on the points' own
# device, which inside select_regions is the CPU - the original behaviour, and
# what region_sweep.py and every earlier result used. train.py --gpu_neighbors
# points it at the GPU: select_regions moves the surface to the CPU, so on a
# training run this cdist was an O(N^2) CPU job on ~30k points every time a
# surface left the RAM cache, which was 88% of loads in runs/v1. The lists come
# back on the CPU either way, so nothing downstream changes. The GPU path takes
# exact distances (no matmul shortcut), so a list can differ from the CPU one
# only by float rounding at the radius boundary; the CPU path is untouched.
NEIGHBOR_DEVICE = None
# Wall-clock seconds spent in radius_neighbors since the caller last reset it;
# train.py reads and zeroes it to report where a step's time goes.
NEIGHBOR_SECONDS = [0.0]


def radius_neighbors(xyz, radius, chunk=1024):
    """CSR neighbour lists within `radius` (self included) -> (offsets, nbrs)."""
    t0 = time.perf_counter()
    home, kw = xyz.device, {}
    if NEIGHBOR_DEVICE is not None:
        xyz, chunk = xyz.to(NEIGHBOR_DEVICE), 4096
        kw = {"compute_mode": "donot_use_mm_for_euclid_dist"}
    r2 = radius * radius
    rows, cols = [], []
    for s in range(0, len(xyz), chunk):
        d2 = torch.cdist(xyz[s:s + chunk], xyz, **kw) ** 2
        r, c = (d2 < r2).nonzero(as_tuple=True)
        rows.append(r + s)
        cols.append(c)
    rows, cols = torch.cat(rows).to(home), torch.cat(cols).to(home)
    counts = torch.bincount(rows, minlength=len(xyz))
    offsets = torch.zeros(len(xyz) + 1, dtype=torch.long, device=home)
    offsets[1:] = torch.cumsum(counts, 0)
    NEIGHBOR_SECONDS[0] += time.perf_counter() - t0
    return offsets, cols


def smooth(values, offsets, nbrs):
    """Mean of `values` over each point's neighbour list."""
    seg = torch.repeat_interleave(torch.arange(len(offsets) - 1), offsets.diff())
    out = torch.zeros_like(values).index_add_(0, seg, values[nbrs])
    return out / offsets.diff().clamp_min(1)


# `cache`, where accepted, is a dict reused across calls on the SAME xyz and
# prob (region_sweep.py runs many selectors per protein); None recomputes.
#
# Only ("nbr", radius) is independent of `prob`: the radius neighbour lists,
# which are the one O(N^2) step here. ("smooth", r) and ("proposals", ...) are
# functions OF prob, so they may be reused across selector settings on a fixed
# map - what region_sweep.py does - but must NOT survive a training step, where
# prob changes as the head learns; a stale smoothed map would drive every
# selection thereafter, silently.
#
# An ALLOWLIST on purpose. select_regions() keeps these keys and drops
# everything else, so any quantity cached here in future is discarded by
# default and has to be added deliberately. A blocklist would persist it.
PROB_INDEPENDENT = ("nbr",)
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
REGION_METHODS = ("watershed", "components", "ball", "nms", "compball")


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
                 top_fraction=None, shift=1, suppress=None, rank_by="seed", cache=None):
    """Greedy NMS balls -> list of index tensors, in seed order (strongest
    first). Seed: the highest smoothed-probability candidate not yet
    suppressed. Centre: the seed, moved by `shift` mean-shift steps (centroid
    of the candidates within `radius`, weighted by probability), so it can sit
    in the pocket's void rather than on its rim. Region: every surface point
    within `radius` of the centre, candidate or not. Then candidates within
    `suppress` (default `radius`) of the centre are removed.

    rank_by: "seed" keeps seed order; "sum" / "sumsq" re-rank the balls by the
    total (squared) smoothed probability inside them.

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
    if rank_by != "seed":
        weight = prob ** 2 if rank_by == "sumsq" else prob
        regions.sort(key=lambda r: -float(weight[r].sum()))
    return regions


def _mean_shift(pts, w, radius, iters=30, tol=0.01):
    """Flat-kernel mean shift started from every point of `pts`, weighted by
    w, run until no centre moves more than `tol` A -> (M, 3) end points."""
    x = pts.clone()
    r2 = radius * radius
    for _ in range(iters):
        k = (torch.cdist(x, pts) ** 2 < r2).float() * w
        s = k.sum(1, keepdim=True)
        new = torch.where(s > 0, (k @ pts) / s.clamp_min(1e-12), x)
        moved = float((new - x).norm(dim=1).max())
        x = new
        if moved < tol:
            break
    return x


def ball_proposals(xyz, prob, threshold, radius, smooth_radius, top_fraction=None,
                   merge=2.0, cache=None):
    """Ball proposals for nms_regions -> (centres (U, 3), members (U, N) bool).

    Mean shift (radius `radius`) from every candidate over the candidates,
    weighted by smoothed probability, so each start climbs to a local centre
    of candidate mass - for a concave pocket, a point in its void. End points
    within `merge` A of an already kept one (strongest start first) are the
    same proposal. Members: every surface point within `radius` of a centre.
    Depends only on the map and these arguments, so it is cached."""
    key = ("proposals", float(radius), float(smooth_radius), top_fraction, float(threshold),
           float(merge))
    if cache is not None and key in cache:
        return cache[key]
    p = smoothed(xyz, prob, smooth_radius, cache)
    cand = candidates(p, threshold, top_fraction).nonzero().squeeze(1)
    if len(cand) == 0:
        out = (xyz.new_zeros(0, 3), torch.zeros(0, len(xyz), dtype=torch.bool))
    else:
        ends = _mean_shift(xyz[cand], p[cand], radius)
        kept = []
        for i in torch.argsort(p[cand], descending=True).tolist():
            if kept and float((torch.stack(kept) - ends[i]).norm(dim=1).min()) < merge:
                continue
            kept.append(ends[i])
        centres = torch.stack(kept)
        out = (centres, torch.cdist(centres, xyz) ** 2 < radius * radius)
    if cache is not None:
        cache[key] = out
    return out


def _nms(members, scores, iou_max, alpha, min_points):
    """Greedy non-maximum suppression over point sets -> list of index tensors.
    Highest score first; a set whose IoU with any kept set exceeds iou_max is
    dropped; stop at the first set scoring below alpha x the best one."""
    sizes = members.sum(1)
    kept, best = [], None
    for j in torch.argsort(scores, descending=True).tolist():
        if int(sizes[j]) < min_points:
            continue
        if best is None:
            best = float(scores[j])
        elif float(scores[j]) < alpha * best:
            break
        if kept:
            inter = (members[kept] & members[j]).sum(1)
            iou = inter.float() / (sizes[kept] + sizes[j] - inter).float()
            if float(iou.max()) > iou_max:
                continue
        kept.append(j)
    return [members[j].nonzero().squeeze(1) for j in kept]


def nms_regions(xyz, prob, threshold, radius, min_points, smooth_radius, top_fraction=None,
                score="sum", iou_max=0.3, alpha=0.0, merge=2.0, cache=None):
    """Score-then-suppress balls -> list of index tensors, best first.

    1. proposals: mean-shift centres of candidate mass (ball_proposals)
    2. score each ball by the total smoothed probability of the surface
       points inside it ("sum"), or of its squares ("sumsq", favours balls of
       confidently high points over balls of many middling ones)
    3. greedy NMS: keep the best ball, drop balls overlapping a kept one by
       IoU > iou_max, stop below alpha x the best score (alpha 0: no stop)."""
    centres, members = ball_proposals(xyz, prob, threshold, radius, smooth_radius,
                                      top_fraction, merge, cache)
    if len(centres) == 0:
        return []
    p = smoothed(xyz, prob, smooth_radius, cache)
    weight = p ** 2 if score == "sumsq" else p
    return _nms(members, members.float() @ weight, iou_max, alpha, min_points)


def component_ball_regions(xyz, prob, threshold, link_radius, radius, min_points,
                           smooth_radius, top_fraction=None, shift=0, iou_max=0.3,
                           cache=None):
    """Balls placed on P2Rank-style components -> list of index tensors.
    Components (component_regions) choose and rank the pockets; each becomes
    a ball of `radius` around its probability-weighted centroid, moved by
    `shift` mean-shift steps over the candidates. A ball overlapping a better
    ranked one by IoU > iou_max is dropped."""
    comps = component_regions(xyz, prob, threshold, link_radius, min_points, smooth_radius,
                              top_fraction, cache=cache)
    if not comps:
        return []
    p = smoothed(xyz, prob, smooth_radius, cache)
    w_cand = p * candidates(p, threshold, top_fraction)
    r2 = radius * radius
    centres = []
    for c in comps:
        centre = (p[c][:, None] * xyz[c]).sum(0) / p[c].sum().clamp_min(1e-12)
        for _ in range(shift):
            inside = ((xyz - centre) ** 2).sum(-1) < r2
            w = w_cand[inside]
            if float(w.sum()) <= 0:
                break
            centre = (w[:, None] * xyz[inside]).sum(0) / w.sum()
        centres.append(centre)
    members = torch.cdist(torch.stack(centres), xyz) ** 2 < r2
    rank = torch.arange(len(centres), 0, -1, dtype=torch.float)  # keep component order
    return _nms(members, rank, iou_max, 0.0, min_points)


def pocket_regions(xyz, prob, method="watershed", threshold=0.5, top_fraction=None,
                   link_radius=2.0, smooth_radius=2.0, min_points=20, persistence=0.0,
                   ball_radius=10.0, ball_shift=1, ball_rank="seed", nms_score="sum",
                   nms_iou=0.3, nms_alpha=0.0, merge_radius=2.0, cache=None):
    """Ranked pocket regions by `method` (see REGION_METHODS and the module
    docstring). Parameters a method does not use are ignored."""
    if method == "nms":
        return nms_regions(xyz, prob, threshold, ball_radius, min_points, smooth_radius,
                           top_fraction, score=nms_score, iou_max=nms_iou, alpha=nms_alpha,
                           merge=merge_radius, cache=cache)
    if method == "compball":
        return component_ball_regions(xyz, prob, threshold, link_radius, ball_radius,
                                      min_points, smooth_radius, top_fraction,
                                      shift=ball_shift, iou_max=nms_iou, cache=cache)
    if method == "watershed":
        return watershed_regions(xyz, prob, threshold, link_radius, min_points, smooth_radius,
                                 top_fraction, persistence=persistence, cache=cache)
    if method == "components":
        return component_regions(xyz, prob, threshold, link_radius, min_points, smooth_radius,
                                 top_fraction, cache=cache)
    if method == "ball":
        return ball_regions(xyz, prob, threshold, ball_radius, min_points, smooth_radius,
                            top_fraction, shift=ball_shift, rank_by=ball_rank, cache=cache)
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
                   method="watershed", persistence=0.0, ball_radius=10.0, ball_shift=1,
                   ball_rank="seed", nms_score="sum", nms_iou=0.3, nms_alpha=0.0,
                   cache=None):
    """Pocket and negative regions on one protein surface.

    xyz (N, 3), prob (N,) pocket probabilities in [0, 1].
    top_fraction: grow regions from this fraction of the protein's points
    (highest smoothed probability); overrides `threshold` when given.
    method, persistence, ball_*, nms_*: how candidates become regions, see
    pocket_regions().
    forced: list of index tensors taken as pocket regions first.
    neg_size: points per negative region; default the median pocket-region
    size, or 4 * min_points when there are none.
    neg_exclusion: A; negative points must be this far from every pocket region.

    cache: a dict OWNED BY THE CALLER and kept alongside this protein's surface,
    so the radius neighbour lists - a chunked all-pairs cdist, the one O(N^2)
    step here - are built once instead of on every forward pass. Entries that
    depend on `prob` are dropped before returning; see PROB_INDEPENDENT. The cache
    is valid for ONE protein's point cloud, so it is checked against the points
    it was built on.

    Returns (regions, is_pos): a list of up to k_pos + k_neg index tensors
    (pocket regions first), and a bool tensor marking which are pocket regions.
    """
    xyz = xyz.detach().float().cpu()
    prob = prob.detach().float().cpu()

    if cache is not None:
        # A cache reused across proteins would return one protein's neighbour
        # lists for another's points, silently. Point count plus the first and
        # last coordinates catches that, and uses exact values rather than a
        # float sum, which over ~37k float32 values lands where the precision
        # is coarser than any sane rounding.
        stamp = (len(xyz), tuple(xyz[0].tolist()), tuple(xyz[-1].tolist()))
        seen = cache.setdefault("_stamp", stamp)
        if seen != stamp:
            raise ValueError(
                f"region cache was built for points {seen} but is being used "
                f"for {stamp}; a cache belongs to exactly one protein surface")

    pos = [torch.as_tensor(f).long().cpu() for f in (forced or [])][:k_pos]
    taken = torch.zeros(len(xyz), dtype=torch.bool)
    for f in pos:
        taken[f] = True
    for r in pocket_regions(xyz, prob, method, threshold, top_fraction, link_radius,
                            smooth_radius, min_points, persistence, ball_radius, ball_shift,
                            ball_rank, nms_score, nms_iou, nms_alpha, cache=cache):
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

    if cache is not None:
        for key in [k for k in cache
                    if not (k == "_stamp"
                            or (isinstance(k, tuple) and k[0] in PROB_INDEPENDENT))]:
            del cache[key]

    is_pos = torch.tensor([True] * len(pos) + [False] * len(neg), dtype=torch.bool)
    return pos + neg, is_pos


# ----------------------------------------------------------------------------
# pooling and evaluation
# ----------------------------------------------------------------------------
POOL_MODES = ("mean", "mean_std", "shells")


def pooled_width(emb_dims, mode="mean", n_shells=3):
    """Width of pool_regions' output for one embedding width."""
    if mode == "mean":
        return emb_dims
    if mode == "mean_std":
        return 2 * emb_dims
    if mode == "shells":
        return n_shells * emb_dims
    raise ValueError(f"unknown pool mode {mode!r}; expected one of {POOL_MODES}")


def pool_regions(emb, prob, regions, eps=1e-6, mode="mean", xyz=None, n_shells=3):
    """Probability-weighted pooling of point embeddings over each region.

    mode:
      mean      one weighted mean per region -> (R, E). The original.
      mean_std  weighted mean and weighted standard deviation -> (R, 2E). A
                mean says what the patch is made of on average; the spread
                says whether it is uniform or mixed, which one mean cannot.
      shells    n_shells weighted means, in equal-COUNT shells by distance
                from the region's weighted centroid -> (R, n_shells * E).
                Needs `xyz`.

    `shells` exists because a single mean over ~290 points discards all spatial
    arrangement inside the region. Pocket DETECTION does not need it - one
    number per point is the answer - but ligand DISCRIMINATION does: the same
    composition arranged rim-out versus centre-out is a different pocket. Three
    radial means are the cheapest ordering that is still rigid-motion
    invariant: distance to the region's own centroid uses no external frame, so
    the output is unchanged by rotating or translating the protein, and the
    shells are indexed inner-to-outer rather than by any point order.

    Equal-count rather than fixed-radius shells: region size and shape vary by
    selector (a watershed region follows the pocket, a ball is fixed at R), so
    fixed edges would leave outer shells empty for small regions. Equal counts
    guarantee every shell is populated for any region of at least n_shells
    points.

    Differentiable in emb and prob under every mode; the shell assignment
    itself is discrete and taken without gradient, like region selection.
    """
    if mode not in POOL_MODES:
        raise ValueError(f"unknown pool mode {mode!r}; expected one of {POOL_MODES}")
    width = pooled_width(emb.shape[1], mode, n_shells)
    if not regions:
        return emb.new_zeros(0, width)
    if mode == "shells" and xyz is None:
        raise ValueError("pool mode 'shells' needs xyz for the region's own distances")

    out = []
    for r in regions:
        r = r.to(emb.device)
        w = prob[r] + eps
        e = emb[r]
        mean = (w[:, None] * e).sum(0) / w.sum()
        if mode == "mean":
            out.append(mean)
        elif mode == "mean_std":
            var = (w[:, None] * (e - mean) ** 2).sum(0) / w.sum()
            # clamped off zero, not at it: sqrt has an infinite derivative at 0,
            # so a region whose embeddings happen to agree exactly would send a
            # NaN back through the whole conv
            out.append(torch.cat([mean, var.clamp_min(1e-12).sqrt()]))
        else:
            p = xyz[r]
            with torch.no_grad():
                c = (w[:, None] * p).sum(0) / w.sum()
                d = (p - c).norm(dim=-1)
                order = torch.argsort(d)
                parts = [t for t in torch.tensor_split(order, n_shells) if len(t)]
            shells = []
            for s in range(n_shells):
                if s < len(parts):
                    ws = w[parts[s]]
                    shells.append((ws[:, None] * e[parts[s]]).sum(0) / ws.sum())
                else:
                    # fewer points than shells: repeat the outermost populated
                    # one rather than emit a structural zero
                    shells.append(shells[-1] if shells else mean)
            out.append(torch.cat(shells))
    return torch.stack(out)


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
