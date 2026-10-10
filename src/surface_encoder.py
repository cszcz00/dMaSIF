"""Protein-side encoder: receptor atoms -> K anchor vectors in representation space.

    atoms ──(1) dMaSIF surface sampler──> surface points + normals     no grad
          ──(2) curvatures (10)  +  AtomNet_MP chemistry (6)──> 16 input feats
          ──(3) dMaSIFConv_seg, weights re-learned──> point embeddings (E)
          ──(4) pocket head──> one logit per point, trained as "pocket-ness"
          ──(5) anchors: k_pos pocket candidates (top logits, suppression)
                       + k_neg negatives (low logits, spread out)
          ──(6) Gaussian pooling around each anchor + projection──> (K, D)

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
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import dmasif_compat  # noqa: E402
from anchors import pool_at_anchors, select_anchors  # noqa: E402

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
    dropout: float = 0.0
    # (3) convolution; these match the published search checkpoint
    emb_dims: int = 16
    n_layers: int = 3
    radius: float = 12.0
    orientation_units: int = 16
    # (4) pocket head
    head_hidden: int = 32
    # (5) anchors
    k_pos: int = 16
    k_neg: int = 16
    nms_radius: float = 8.0
    neg_quantile: float = 0.5
    neg_flat_quantile: float | None = None   # e.g. 0.5: negatives only on flatter half
    select_temperature: float | None = None  # > 0 while training to explore
    # (6) pooling and output
    pool_radius: float = 8.0
    out_dim: int = 128
    normalize: bool = True
    repo: str = field(default=str(DEFAULT_REPO))

    @property
    def in_channels(self):
        return 2 * len(self.curvature_scales) + self.atom_dims


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
        self.orientation_scores = _mlp(I, cfg.orientation_units, 1)
        self.conv = dMaSIFConv_seg(args, in_channels=I, out_channels=E,
                                   n_layers=cfg.n_layers, radius=cfg.radius)
        self.dropout = nn.Dropout(cfg.dropout)
        self.pocket_head = _mlp(E, cfg.head_hidden, 1)
        self.project = _mlp(E, max(cfg.out_dim, E), cfg.out_dim)

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
    def input_features(self, xyz, normals, batch, atom_xyz, atomtypes, batch_atoms):
        from geometry_processing import curvatures

        with torch.no_grad():  # geometry has no parameters
            curv = curvatures(xyz, normals=normals, batch=batch,
                              scales=list(self.cfg.curvature_scales))
        chem = self.atomnet(xyz, atom_xyz, atomtypes, batch, batch_atoms)
        return torch.cat([curv, chem], dim=1).contiguous()

    # ------------------------------------------------------------- (2)-(6)
    def forward(self, P, forced=None, generator=None):
        """P: dict with atom_xyz (M,3), atomtypes (M,6) one-hot, batch_atoms (M,),
        and optionally a cached surface: xyz (N,3), normals (N,3), batch (N,).
        forced: optional list, per protein, of point-index tensors (LOCAL to
        that protein) to take as pocket anchors first, e.g. the true pocket.

        Returns a dict:
            xyz, normals, batch      the surface used
            input_feats  (N, 16)     curvatures + chemistry
            point_emb    (N, E)      conv output
            pocket_logit (N,)        per-point pocket score
            anchor_idx   (B, K)      GLOBAL point indices, -1 = no anchor
            anchor_is_pos(B, K)      True for pocket-candidate slots
            anchor_valid (B, K)
            anchor_vec   (B, K, D)   the representation-space vectors
        """
        c = self.cfg
        if "xyz" not in P:
            P = dict(P)
            P["xyz"], P["normals"], P["batch"] = self.sample_surface(
                P["atom_xyz"], P["atomtypes"], P["batch_atoms"])
        xyz, normals, batch = P["xyz"], P["normals"], P["batch"]

        feats = self.dropout(self.input_features(
            xyz, normals, batch, P["atom_xyz"], P["atomtypes"], P["batch_atoms"]))
        self.conv.load_mesh(xyz, normals=normals,
                            weights=self.orientation_scores(feats), batch=batch)
        emb = self.conv(feats)
        logit = self.pocket_head(emb).squeeze(-1)

        n_prot = int(batch.max()) + 1
        flat_col = 2 * (len(c.curvature_scales) - 1)  # mean curvature, largest scale
        idx_all, pos_all, vec_all = [], [], []
        for b in range(n_prot):
            sel = (batch == b).nonzero().squeeze(1)
            neg_ok = None
            if c.neg_flat_quantile is not None:
                h = feats[sel, flat_col].detach().abs()
                neg_ok = h <= torch.quantile(h, c.neg_flat_quantile)
            local, is_pos = select_anchors(
                xyz[sel], logit[sel], c.k_pos, c.k_neg, c.nms_radius,
                neg_quantile=c.neg_quantile, neg_allowed=neg_ok,
                temperature=c.select_temperature if self.training else None,
                forced=None if forced is None else forced[b], generator=generator)
            vec_all.append(pool_at_anchors(xyz[sel], emb[sel], local, c.pool_radius))
            idx_all.append(torch.where(local >= 0, sel[local.clamp(min=0)], -1))
            pos_all.append(is_pos)

        anchor_idx = torch.stack(idx_all)
        vec = self.project(torch.stack(vec_all))
        if c.normalize:
            vec = F.normalize(vec, dim=-1)
        valid = anchor_idx >= 0
        return {
            "xyz": xyz, "normals": normals, "batch": batch,
            "input_feats": feats, "point_emb": emb, "pocket_logit": logit,
            "anchor_idx": anchor_idx, "anchor_is_pos": torch.stack(pos_all),
            "anchor_valid": valid, "anchor_vec": vec * valid[..., None],
        }

    # ------------------------------------------------------------------
    def load_pretrained(self, ckpt=None, parts=("atomnet", "orientation_scores", "conv")):
        """Start the named parts from the published search checkpoint (its first
        branch). Needs emb_dims, n_layers, radius and curvature_scales at their
        defaults. Returns the list of loaded tensor names."""
        ckpt = ckpt or Path(self.cfg.repo) / "models" / "dMaSIF_search_3layer_12A_16dim"
        state = torch.load(ckpt, map_location="cpu", weights_only=False)["model_state_dict"]
        take = {k: v for k, v in state.items() if k.split(".")[0] in parts}
        missing, unexpected = self.load_state_dict(take, strict=False)
        wanted = [k for k in missing if k.split(".")[0] in parts]
        if wanted or unexpected:
            raise RuntimeError(f"checkpoint mismatch: missing {wanted}, unexpected {unexpected}")
        return sorted(take)


def pocket_loss(logit, labels, pos_weight=None):
    """Per-point BCE for the pocket head. labels: (N,) 0/1 from labels.py.
    Pockets are ~5% of a surface, so pass pos_weight (e.g. negatives/positives)."""
    pw = None if pos_weight is None else torch.as_tensor(pos_weight, device=logit.device)
    return F.binary_cross_entropy_with_logits(logit, labels.float(), pos_weight=pw)


def config_dict(cfg):
    """For checkpoints and provenance."""
    return asdict(cfg)
