"""Protein-side encoder: receptor atoms -> K anchor vectors in representation space.

    atoms ──(1) dMaSIF surface sampler──> surface points + normals     no grad
          ──(2) curvatures (10) + AtomNet_MP chemistry (6)
                + optional pharmacophore channels (chemotype.py, 12)──> input feats
          ──(3) dMaSIFConv_seg, weights re-learned──> point embeddings (E)
          ──(4) pocket head──> one logit per point, trained as "pocket-ness"
          ──(5) region anchors: k_pos pocket regions grown from the pocket
                map (watershed) + k_neg low-probability regions of matched size
          ──(6) probability-weighted pooling per region (one mean, mean+std,
                or radial shells) + projection──> (K, D)

Steps 1-3 are dMaSIF's own classes, instantiated here with fresh weights, so
everything from the chemistry network onward trains. load_pretrained() can
start them from the published search checkpoint instead. Step 1 is the only
part without parameters; its output can be cached and passed back in.

Batching follows dMaSIF: proteins are concatenated, with batch vectors saying
which protein each atom / point belongs to. Anchor outputs are (B, K, ...).

Runs where dMaSIF runs (GPU, PyKeOps), i.e. inside the container.

Depends on: dmasif_compat, anchors, the dMaSIF clone at ../dMaSIF.
"""

import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import atom_features  # noqa: E402
import chemotype  # noqa: E402
from dir_atomnet import DirAtomNet  # noqa: E402
import dmasif_compat  # noqa: E402
from anchors import pool_regions, pooled_width, select_regions  # noqa: E402
from pair_pool import PairPool, TypedAttnPool  # noqa: E402

# pool modes whose pooling is a learned module, held as SurfaceEncoder.pair_pool
LEARNED_POOLS = ("pairs", "typed_attn")
# input channels the learned-typing branch (cfg.atom_feats) adds
CHEMNET_DIMS = 16

DEFAULT_REPO = HERE.parent / "dMaSIF"


@dataclass
class EncoderConfig:
    # (1) surface sampling, dMaSIF defaults
    resolution: float = 1.0
    sup_sampling: int = 20
    distance: float = 1.05
    smoothness: float = 0.5
    variance: float = 0.1
    nits: int = 4
    # (2) input features
    curvature_scales: tuple = (1.0, 2.0, 3.0, 5.0, 10.0)
    atom_dims: int = 6
    # Pharmacophore channels (chemotype.py): composition, buriedness, proximity.
    # Off by default so every existing checkpoint and result is reproducible.
    chemo: bool = False
    # Learned atom typing (atom_features.py, dir_atomnet.py): a second
    # AtomNet-style branch over 13 per-atom chemistry values (element,
    # hybridisation, H count, charge, donor / acceptor, backbone, burial) and
    # each polar atom's direction, its 16 outputs appended as input channels
    # after everything else. Needs P["atom_feats"] and P["atom_dir"]
    # (SurfaceStore(atom_feats=True)). Off by default.
    atom_feats: bool = False
    chemo_sigmas: tuple = (3.0, 6.0, 9.0)
    dropout: float = 0.0
    # (3) convolution; these match the published search checkpoint
    emb_dims: int = 16
    n_layers: int = 3
    radius: float = 12.0
    orientation_units: int = 16
    # (3b) descriptor branch. 0 = off, the original encoder. > 0 adds a SECOND
    # dMaSIF conv of this width, fresh weights, over the same input features,
    # used ONLY for what a region looks like to a ligand: its output is
    # concatenated to the pretrained 16-dim embedding before pooling. The
    # pretrained conv and its pocket head still decide WHERE the regions are.
    # Why not just widen emb_dims: the head checkpoint and the published dMaSIF
    # weights are 16-wide, so widening discards both, and the regions with them.
    desc_dims: int = 0
    desc_layers: int = 3
    # (4) pocket head
    head_hidden: int = 32
    # (5) region anchors (see anchors.py)
    # WHERE the pocket regions come from. "predicted" is the original: the
    # pocket head's map through the selector below. "external": an outside
    # pocket finder's pockets (fpocket, external_pockets.py) ARE the pocket
    # regions, best first, and the selector is not run. "external_map": that
    # finder's per-point map replaces sigmoid(pocket_logit) as the selector's
    # input, thresholded at region_threshold. Both external sources pool each
    # region as a plain mean, so the head no longer touches the anchors; it is
    # still computed, trained by the pocket BCE and reported. They need the
    # batch to carry P["ext_map"] / P["ext_regions"] (SurfaceStore(pockets=)).
    region_source: str = "predicted"     # predicted | external | external_map
    k_pos: int = 8
    k_neg: int = 8
    region_top_fraction: float | None = 0.05  # grow regions from each protein's top 5% ...
    region_threshold: float = 0.5             # ... or, if top_fraction is None, from p >= this
    region_method: str = "watershed"       # watershed | components | ball | nms | compball
    region_persistence: float = 0.0        # watershed merge threshold, quantile units
    region_ball_radius: float = 10.0       # A; ball, nms, compball
    region_ball_shift: int = 1             # mean-shift steps for the ball centre
    region_ball_rank: str = "seed"         # ball: seed | sum | sumsq
    region_nms_score: str = "sum"          # nms: sum | sumsq of probability in the ball
    region_nms_iou: float = 0.3            # nms / compball: suppress above this IoU
    region_nms_alpha: float = 0.0          # nms: stop below alpha x best score
    link_radius: float = 2.0               # A; points closer are neighbours
    smooth_radius: float = 2.0             # A; probability smoothing
    min_region_points: int = 20
    neg_quantile: float = 0.5
    neg_exclusion: float = 4.0             # A; negatives keep off pocket regions
    neg_flat_quantile: float | None = None # e.g. 0.5: negative seeds on flatter half
    # (6) pooling and output
    pool_mode: str = "mean"      # mean | mean_std | shells | pairs | typed_attn
    n_shells: int = 3            # pool_mode "shells": radial shells per region
    # pool_mode "pairs" (pair_pool.PairPool): learned soft site types x types x
    # distance bins over each region - which kinds of site sit at what
    # distance from each other. pair_feats feeds the raw input channels
    # (curvature, AtomNet, and the pharmacophore ones under `chemo`) to the
    # type head beside the conv embedding, since they are sharp where the
    # 12 A conv is smeared.
    pair_types: int = 8
    pair_bins: int = 8
    pair_rmin: float = 3.0
    pair_rmax: float = 16.0
    pair_exclude: float = 2.0
    pair_feats: bool = True
    # pool_mode "typed_attn" (pair_pool.TypedAttnPool): PairPool's per-point
    # type / salience heads, then attn_heads learned heads, each a (from-type
    # mix, to-type mix, distance profile over attn_bins Gaussians) with a
    # v_dims partner description. Uses pair_types / pair_rmin / pair_rmax /
    # pair_exclude / pair_feats as PairPool does.
    attn_heads: int = 16
    attn_vdim: int = 8
    attn_bins: int = 12
    # Standardisation of the pooled vector before the projection. "none" is the
    # original path; "batch" is a BatchNorm1d over the batch's USABLE anchors,
    # which subtracts the per-feature mean the whole dataset shares. Worth
    # having because ~88% of a pooled vector is one shared direction (different
    # proteins sit at cosine 0.993), so an unnormalised projection has to learn
    # to subtract a constant before it can read the 12% that distinguishes
    # proteins. Learned running statistics rather than statistics fitted once,
    # because the encoder trains and anything fitted up front goes stale.
    anchor_norm: str = "none"    # none | batch
    out_dim: int = 128
    normalize: bool = True
    repo: str = field(default=str(DEFAULT_REPO))

    @property
    def chemo_dims(self):
        # Derived from this config's own scales, so changing chemo_sigmas cannot
        # leave in_channels disagreeing with what surface_features() returns.
        return chemotype.dims(tuple(self.chemo_sigmas)) if self.chemo else 0

    @property
    def in_channels(self):
        return (2 * len(self.curvature_scales) + self.atom_dims + self.chemo_dims
                + (CHEMNET_DIMS if self.atom_feats else 0))

    @property
    def pooled_dims(self):
        """Width of a pooled region vector, which pool_mode decides."""
        return pooled_width(self.emb_dims + self.desc_dims, self.pool_mode,
                            self.n_shells, self.pair_types, self.pair_bins,
                            self.attn_heads, self.attn_vdim)


def _mlp(i, h, o):
    return nn.Sequential(nn.Linear(i, h), nn.LeakyReLU(0.2), nn.Linear(h, o))


class SurfaceEncoder(nn.Module):
    def __init__(self, cfg: EncoderConfig = None):
        super().__init__()
        self.cfg = cfg = cfg or EncoderConfig()
        dmasif_compat.add_repo_to_path(cfg.repo)
        from benchmark_models import dMaSIFConv_seg
        from model import AtomNet_MP

        I, E = cfg.in_channels, cfg.emb_dims
        args = SimpleNamespace(atom_dims=cfg.atom_dims)
        # Same attribute names as dMaSIF's model, so its checkpoint keys load as-is.
        self.atomnet = AtomNet_MP(args)
        # The learned-typing branch: dir_atomnet.DirAtomNet, AtomNet's shape
        # of computation over atom_features' chemistry plus polar direction,
        # fresh weights (its name is not in the checkpoint, so load_pretrained
        # leaves it alone). Only built when asked for, so every other encoder
        # keeps exactly its state dict.
        self.chemnet = (DirAtomNet(n_in=atom_features.DIMS, dim=CHEMNET_DIMS)
                        if cfg.atom_feats else None)
        self.orientation_scores = _mlp(I, cfg.orientation_units, 1)
        self.conv = dMaSIFConv_seg(args, in_channels=I, out_channels=E,
                                   n_layers=cfg.n_layers, radius=cfg.radius)
        if cfg.desc_dims and (cfg.desc_dims < 8 or cfg.desc_dims % 8):
            # dMaSIFConv splits its hidden units into heads of 8 and GroupNorms
            # them in 4 groups; anything else fails deep inside KeOps instead
            raise ValueError(f"desc_dims {cfg.desc_dims}: use a multiple of 8")
        # Only built when asked for, so an encoder without it has exactly the
        # original state dict and every earlier checkpoint loads strictly.
        self.desc_conv = (dMaSIFConv_seg(args, in_channels=I, out_channels=cfg.desc_dims,
                                         n_layers=cfg.desc_layers, radius=cfg.radius)
                          if cfg.desc_dims else None)
        self.dropout = nn.Dropout(cfg.dropout)
        # seconds in region selection during the last forward, for train.py's
        # per-phase timing line
        self.last_select_sec = 0.0
        self.pocket_head = _mlp(E, cfg.head_hidden, 1)
        # Only built for the learned pool modes, so every other encoder keeps
        # exactly its original state dict.
        if cfg.pool_mode == "pairs":
            self.pair_pool = PairPool(E + cfg.desc_dims, I if cfg.pair_feats else 0,
                                      n_types=cfg.pair_types, n_bins=cfg.pair_bins,
                                      r_min=cfg.pair_rmin, r_max=cfg.pair_rmax,
                                      exclude=cfg.pair_exclude)
        elif cfg.pool_mode == "typed_attn":
            self.pair_pool = TypedAttnPool(
                E + cfg.desc_dims, I if cfg.pair_feats else 0, n_types=cfg.pair_types,
                n_heads=cfg.attn_heads, v_dims=cfg.attn_vdim, n_bins=cfg.attn_bins,
                r_min=cfg.pair_rmin, r_max=cfg.pair_rmax, exclude=cfg.pair_exclude)
        else:
            self.pair_pool = None
        Pd = cfg.pooled_dims
        # named pool_norm, not anchor_norm: cfg.anchor_norm is the string that
        # chose it, and two attributes of one name would be a trap
        self.pool_norm = (nn.BatchNorm1d(Pd) if cfg.anchor_norm == "batch"
                          else nn.Identity())
        self.project = _mlp(Pd, max(cfg.out_dim, Pd), cfg.out_dim)

    # ------------------------------------------------------------------ (1)
    @torch.no_grad()
    def sample_surface(self, atom_xyz, atomtypes, batch_atoms):
        """dMaSIF's stochastic surface sampler -> (xyz, normals, batch).
        Cache the result to keep one fixed surface per protein."""
        from geometry_processing import atoms_to_points_normals

        c = self.cfg
        return atoms_to_points_normals(
            atom_xyz, batch_atoms, atomtypes=atomtypes, distance=c.distance,
            smoothness=c.smoothness, resolution=c.resolution, nits=c.nits,
            sup_sampling=c.sup_sampling, variance=c.variance)

    # ------------------------------------------------------------------ (2)
    def input_features(self, xyz, normals, batch, atom_xyz, atomtypes, batch_atoms,
                       atom_groups=None, chemo=None, atom_feats=None, atom_dir=None):
        """-> (N, in_channels): curvatures, AtomNet chemistry, chemo channels.

        `chemo` lets a caller pass precomputed pharmacophore channels, since
        they are deterministic given (atoms, points) and cache with the
        surface; `atom_groups` (M, 6) is what they are computed FROM.
        """
        from geometry_processing import curvatures

        with torch.no_grad():  # geometry has no parameters
            curv = curvatures(xyz, normals=normals, batch=batch,
                              scales=list(self.cfg.curvature_scales))
        chem = self.atomnet(xyz, atom_xyz, atomtypes, batch, batch_atoms)
        parts = [curv, chem]
        if self.cfg.chemo_dims:
            if chemo is None:
                if atom_groups is None:
                    raise ValueError(
                        "cfg.chemo is set but the batch carries neither atom_groups "
                        "nor a precomputed `chemo` block. Call chemotype.annotate() "
                        "on each protein when it is loaded.")
                chemo = chemotype.surface_features(
                    xyz, batch, atom_xyz, atom_groups, batch_atoms,
                    sigmas=tuple(self.cfg.chemo_sigmas))
            parts.append(chemo.to(curv.dtype))
        if self.chemnet is not None:
            if atom_feats is None or atom_dir is None:
                raise ValueError(
                    "cfg.atom_feats is set but the batch carries no atom_feats / "
                    "atom_dir; build the SurfaceStore with atom_feats=True "
                    "(train.py --atom_feats) so the cache provides them")
            parts.append(self.chemnet(xyz, atom_xyz, atom_feats.to(curv.dtype),
                                      batch, batch_atoms, atom_dir.to(curv.dtype)))
        return torch.cat(parts, dim=1).contiguous()

    # ------------------------------------------------------------------ (6)
    def project_anchors(self, pooled, valid):
        """(B, K, P) pooled region vectors -> (B, K, out_dim) anchor vectors.

        Only the USABLE slots go through the normalisation and the projection;
        empty slots come back as zeros. Padding must not reach a BatchNorm,
        whose statistics it would otherwise set.

        Public because the training loop sometimes has one more region than the
        selector produced - the fallback anchor at a missed pocket - and the
        normalisation has to see the batch's anchors ONCE, all together. Hence
        forward(project=False): the caller assembles the full pooled tensor and
        calls this itself, rather than running the tail twice on different data.
        """
        out = pooled.new_zeros(*pooled.shape[:-1], self.cfg.out_dim)
        if not bool(valid.any()):
            return out
        flat = pooled[valid]
        norm = self.pool_norm
        if self.training and len(flat) < 2 and isinstance(norm, nn.BatchNorm1d):
            # A BatchNorm in training mode refuses a single sample, and a whole
            # batch yielding one region is possible on a small smoke run. Use
            # the running statistics for that step rather than dying hours in.
            norm.eval()
            flat = norm(flat)
            norm.train()
        else:
            flat = norm(flat)
        flat = self.project(flat)
        if self.cfg.normalize:
            flat = F.normalize(flat, dim=-1)
        out[valid] = flat
        return out

    # ------------------------------------------------------------- (2)-(6)
    def forward(self, P, forced=None, generator=None, project=True):
        """P: dict with atom_xyz (M,3), atomtypes (M,6) one-hot, batch_atoms (M,),
        and optionally a cached surface: xyz (N,3), normals (N,3), batch (N,).
        P["region_cache"]: optional list of per-protein dicts that persist the
        selector's neighbour lists across steps (see anchors.select_regions).
        forced: optional list, per protein, of lists of point-index tensors
        (LOCAL to that protein) taken as pocket regions first, e.g. the
        labelled pocket during training.
        project: False returns anchor_vec None and leaves the pooled vectors to
        the caller's own project_anchors() call - see that method.

        Returns a dict:
            xyz, normals, batch        the surface used
            input_feats   (N, 16)      curvatures + chemistry
            point_emb     (N, E)       conv output
            pool_emb      (N, E + desc_dims)  what regions are pooled from:
                                       point_emb, plus the descriptor branch
            pocket_logit  (N,)         per-point pocket score
            anchor_vec    (B, K, D)    the representation-space vectors, or
                                       None when project=False
            anchor_pooled (B, K, P)    the pooled region embeddings BEFORE the
                                       normalisation and projection, P =
                                       cfg.pooled_dims. Dumped by
                                       probe_retrieval.py:
                                       the projection is the part the contrastive
                                       loss has to learn, so freezing its current
                                       random weights would freeze noise.
            anchor_valid  (B, K)       False for empty slots
            anchor_is_pos (B, K)       True for pocket regions, False for negatives
            anchor_center (B, K, 3)    probability-weighted region centroid
            anchor_size   (B, K)       points per region
            anchor_regions             per protein, list of GLOBAL index tensors
            pool_weight   (N,)         the per-point weights regions were
                                       pooled with: the pocket probability, or
                                       ones under an external region_source.
                                       Anything pooled later (the training
                                       loop's fallback anchor) must use these.
        """
        c = self.cfg
        if "xyz" not in P:
            P = dict(P)
            P["xyz"], P["normals"], P["batch"] = self.sample_surface(
                P["atom_xyz"], P["atomtypes"], P["batch_atoms"])
        xyz, normals, batch = P["xyz"], P["normals"], P["batch"]

        feats = self.dropout(self.input_features(
            xyz, normals, batch, P["atom_xyz"], P["atomtypes"], P["batch_atoms"],
            atom_groups=P.get("atom_groups"), chemo=P.get("chemo"),
            atom_feats=P.get("atom_feats"), atom_dir=P.get("atom_dir")))
        weights = self.orientation_scores(feats)
        self.conv.load_mesh(xyz, normals=normals, weights=weights, batch=batch)
        emb = self.conv(feats)
        logit = self.pocket_head(emb).squeeze(-1)
        # What gets pooled into a region vector: the pretrained embedding, plus
        # the descriptor branch when there is one. The pocket head reads emb only.
        if self.desc_conv is not None:
            self.desc_conv.load_mesh(xyz, normals=normals, weights=weights, batch=batch)
            pool_emb = torch.cat([emb, self.desc_conv(feats)], dim=1)
        else:
            pool_emb = emb
        prob = torch.sigmoid(logit)

        sp = select_and_pool(c, P, xyz, batch, feats, prob, pool_emb,
                             forced=forced, generator=generator,
                             pair_pool=self.pair_pool)
        self.last_select_sec = sp["select_sec"]
        vec, valid = sp["vec"], sp["valid"]
        out = self.project_anchors(vec, valid) if project else None
        return {
            "xyz": xyz, "normals": normals, "batch": batch,
            "input_feats": feats, "point_emb": emb, "pool_emb": pool_emb,
            "pocket_logit": logit,
            "anchor_vec": out, "anchor_pooled": vec,
            "anchor_valid": valid,
            "anchor_is_pos": sp["is_pos"], "anchor_center": sp["center"],
            "anchor_size": sp["size"], "anchor_regions": sp["regions"],
            "pool_weight": sp["pool_weight"],
        }

    # ------------------------------------------------------------------
    # The only tensors whose shape depends on in_channels. Everything else in
    # the conv stack is (emb_dims -> emb_dims) and loads unchanged.
    INPUT_TENSORS = ("conv.layers.0.net_in.0.weight",
                     "conv.linear_transform.0.weight",
                     "orientation_scores.0.weight")

    def _inflate_inputs(self, take):
        """Widen the checkpoint's three input-side weights to in_channels.

        Pretrained columns are copied and the new ones zeroed, so at step 0 the
        network computes EXACTLY the pretrained function while gradients still
        reach the new columns from the first step. Exact because nothing
        learned normalises the input before these layers (the conv's GroupNorm
        sits on the hidden dims, which do not change).

        A new channel must be finite: it is multiplied by zero, and 0 * inf is
        NaN, not 0.
        """
        target = self.state_dict()
        out, grown = dict(take), []
        for key in self.INPUT_TENSORS:
            if key not in out:
                continue
            old, want = out[key], tuple(target[key].shape)
            if tuple(old.shape) == want:
                continue
            if old.shape[0] != want[0] or old.shape[1] > want[1]:
                raise RuntimeError(
                    f"{key}: cannot inflate {tuple(old.shape)} -> {want}; only the "
                    "input width may grow")
            new = old.new_zeros(want)
            new[:, :old.shape[1]] = old
            out[key] = new
            grown.append(f"{key} {old.shape[1]}->{want[1]}")
        if grown:
            print("inflated input weights (pretrained function preserved at init): "
                  + "; ".join(grown))
        return out

    def load_pretrained(self, ckpt=None, parts=("atomnet", "orientation_scores", "conv"),
                        inflate=True):
        """Start the named parts from the published search checkpoint (its first
        branch). Needs emb_dims, n_layers, radius and curvature_scales at their
        defaults. With `inflate`, extra input channels are zero-padded instead
        of refusing to load; it is a no-op when in_channels is unchanged.
        Returns the list of loaded tensor names."""
        ckpt = ckpt or Path(self.cfg.repo) / "models" / "dMaSIF_search_3layer_12A_16dim"
        state = torch.load(ckpt, map_location="cpu", weights_only=False)["model_state_dict"]
        take = {k: v for k, v in state.items() if k.split(".")[0] in parts}
        if inflate:
            take = self._inflate_inputs(take)
        missing, unexpected = self.load_state_dict(take, strict=False)
        wanted = [k for k in missing if k.split(".")[0] in parts]
        if wanted or unexpected:
            raise RuntimeError(f"checkpoint mismatch: missing {wanted}, unexpected {unexpected}")
        return sorted(take)


REGION_SOURCES = ("predicted", "external", "external_map")


def pool_region_vectors(c, pair_pool, emb, feats, w, regions, xyz):
    """One protein's regions -> (R, cfg.pooled_dims), by cfg.pool_mode. The
    one place a region becomes a vector, so the selector's anchors and the
    training loop's fallback anchor are pooled identically."""
    if c.pool_mode in LEARNED_POOLS:
        return pair_pool(emb, feats if c.pair_feats else None, w, regions, xyz)
    return pool_regions(emb, w, regions, mode=c.pool_mode, xyz=xyz,
                        n_shells=c.n_shells)


def select_and_pool(c, P, xyz, batch, feats, prob, pool_emb, forced=None,
                    generator=None, pair_pool=None):
    """Region selection and pooling for every protein of a batch.

    The tail of SurfaceEncoder.forward, as a function of its tensors so the
    region logic - including the external-pocket sources - can be exercised
    without KeOps (train.py selftest). `c` is the EncoderConfig; `prob` the
    head's per-point probability. Returns vec / center / size / is_pos / valid
    (B, K, ...), the per-protein GLOBAL region index lists, the per-point
    pool_weight, and select_sec. pair_pool: the encoder's PairPool, needed
    for pool_mode "pairs" (with `feats` when cfg.pair_feats).
    """
    if c.pool_mode in LEARNED_POOLS and pair_pool is None:
        raise ValueError(f"pool_mode {c.pool_mode!r} needs the encoder's pair_pool")
    if c.region_source not in REGION_SOURCES:
        raise ValueError(f"unknown region_source {c.region_source!r}; expected "
                         f"one of {REGION_SOURCES}")
    external = c.region_source != "predicted"
    if external:
        if "ext_map" not in P:
            raise ValueError(
                f"region_source is {c.region_source!r} but the batch carries no "
                "external pockets (P['ext_map']); give the SurfaceStore a "
                "PocketSource (train.py --pockets)")
        # The external map seeds negatives (and, for external_map, drives the
        # selector); pooling is a plain mean over the region. Ones rather than
        # the map, so a ball region's points outside the pocket proper still
        # count, and no gradient reaches the head through the weights.
        sel_prob = P["ext_map"].to(device=prob.device, dtype=prob.dtype)
        pool_w = torch.ones_like(prob)
    else:
        sel_prob, pool_w = prob, prob
    # external_map is 0 over ~97% of the surface, so a top-fraction quantile
    # would be 0 and make every point a candidate; threshold it instead.
    top_fraction = None if c.region_source == "external_map" else c.region_top_fraction

    K = c.k_pos + c.k_neg
    n_prot = int(batch.max()) + 1
    flat_col = 2 * (len(c.curvature_scales) - 1)  # mean curvature, largest scale
    vec = pool_emb.new_zeros(n_prot, K, c.pooled_dims)
    center = xyz.new_zeros(n_prot, K, 3)
    size = torch.zeros(n_prot, K, dtype=torch.long, device=xyz.device)
    is_pos = torch.zeros(n_prot, K, dtype=torch.bool, device=xyz.device)
    valid = torch.zeros(n_prot, K, dtype=torch.bool, device=xyz.device)
    # One dict per protein, owned by the caller and kept with its cached
    # surface, so ball_R8's radius neighbour lists (a chunked all-pairs
    # cdist, the only O(N^2) step in selection) are built once rather than
    # on every forward pass.
    caches = P.get("region_cache")
    all_regions = []
    t_sel = time.perf_counter()
    for b in range(n_prot):
        sel = (batch == b).nonzero().squeeze(1)
        neg_ok = None
        if c.neg_flat_quantile is not None:
            h = feats[sel, flat_col].detach().abs()
            neg_ok = h <= torch.quantile(h, c.neg_flat_quantile)
        forced_b = None if forced is None else forced[b]
        if c.region_source == "external":
            forced_b = list(forced_b or []) + [torch.as_tensor(r).long()
                                               for r in P["ext_regions"][b]]
        regions, pos = select_regions(
            xyz[sel], sel_prob[sel], c.k_pos, c.k_neg, threshold=c.region_threshold,
            top_fraction=top_fraction,
            link_radius=c.link_radius, smooth_radius=c.smooth_radius,
            min_points=c.min_region_points, neg_quantile=c.neg_quantile,
            neg_exclusion=c.neg_exclusion, neg_allowed=neg_ok,
            forced=forced_b, generator=generator,
            fill=c.region_source != "external",
            method=c.region_method, persistence=c.region_persistence,
            ball_radius=c.region_ball_radius, ball_shift=c.region_ball_shift,
            ball_rank=c.region_ball_rank, nms_score=c.region_nms_score,
            nms_iou=c.region_nms_iou, nms_alpha=c.region_nms_alpha,
            cache=None if caches is None else caches[b])
        n = len(regions)
        if n:
            vec[b, :n] = pool_region_vectors(c, pair_pool, pool_emb[sel],
                                             None if feats is None else feats[sel],
                                             pool_w[sel], regions, xyz[sel])
            center[b, :n] = pool_regions(xyz[sel], pool_w[sel].detach(), regions)
            size[b, :n] = torch.tensor([len(r) for r in regions], device=xyz.device)
            is_pos[b, :n] = pos.to(xyz.device)
            valid[b, :n] = True
        all_regions.append([sel[r.to(sel.device)] for r in regions])
    return {"vec": vec, "center": center, "size": size, "is_pos": is_pos,
            "valid": valid, "regions": all_regions, "pool_weight": pool_w,
            "select_sec": time.perf_counter() - t_sel}


def pocket_loss(logit, labels, pos_weight=None):
    """Per-point BCE for the pocket head. labels: (N,) 0/1 from labels.py.
    Pockets are ~5% of a surface, so pass pos_weight (e.g. negatives/positives)."""
    pw = None if pos_weight is None else torch.as_tensor(pos_weight, device=logit.device)
    return F.binary_cross_entropy_with_logits(logit, labels.float(), pos_weight=pw)


def config_dict(cfg):
    """For checkpoints and provenance."""
    return asdict(cfg)
