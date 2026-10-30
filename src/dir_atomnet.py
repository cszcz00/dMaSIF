"""DirAtomNet: dMaSIF's AtomNet, given chemistry and polar direction.

The learned-typing branch of the protein encoder (SurfaceEncoder.chemnet,
cfg.atom_feats). Same shape of computation as dMaSIF's AtomNet_MP, which the
pretrained element branch keeps using unchanged:

    atom features (atom_features.DIMS = 13) -> Linear -> width D (16)
    3 rounds atom -> atom   over each atom's 16 nearest atoms
    3 rounds atom -> point  over each surface point's 16 nearest atoms
    -> (N_points, D) input channels for the conv

Three differences, each for a reason:

  * The input width is decoupled from D by the first Linear. dMaSIF's AtomNet
    maps D -> D, so its input count IS its output width; here 13 chemistry
    values feed 16 channels.
  * Distances enter as d / 4 A, not dMaSIF's raw squared distance (up to
    ~40 A^2 into a small MLP): same information, better conditioned.
  * Every atom -> point message also carries
        cos_ap = u_a . (p - x_a) / |p - x_a|
    where u_a is the atom's polar direction (atom_features.atom_direction:
    the N-H / lone-pair bisector, the C->O axis, ...; 0 for non-polar atoms).
    "Does this acceptor's lone pair point at this patch, or into the protein?"
    is the H-bond availability that distances alone cannot express.

Invariance: u_a rotates with the atoms (it is built from coordinate
differences) and p - x_a likewise, so their dot product - and every distance
- is invariant to rotation and translation. No frame, axis or cross product
is formed, so nothing flips sign or degenerates. Conditioning: |p - x_a| is
at least ~1 A for surface points outside the atoms, and the divisor is
clamped at 0.5 A regardless; |u_a| <= 1, so cos_ap is in [-1, 1].

Neighbours come from torch.cdist + topk per protein (chunked), not KeOps, so
the module runs - and its self-test runs - on a CPU without compiling
anything. ~32 proteins x ~10k points x ~5k atoms per step is a few GB-flop of
cdist on the GPU.

Example:
    python src/dir_atomnet.py        # self-test, CPU torch, no KeOps

Depends on: torch, atom_features (DIMS).
Imported by: surface_encoder.py.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import atom_features  # noqa: E402


@torch.no_grad()
def knn(x, y, x_batch, y_batch, k, chunk=2048):
    """For every row of x, its k nearest rows of y in the same batch item.

    -> idx (N, k) into y, dist (N, k), valid (N, k) bool. An item with fewer
    than k candidates fills the rest with valid False (idx 0, dist 0), which
    callers mask out of their sums."""
    n = len(x)
    idx = torch.zeros(n, k, dtype=torch.long, device=x.device)
    dist = torch.zeros(n, k, dtype=x.dtype, device=x.device)
    valid = torch.zeros(n, k, dtype=torch.bool, device=x.device)
    for b in torch.unique(x_batch):
        xi = (x_batch == b).nonzero().squeeze(1)
        yi = (y_batch == b).nonzero().squeeze(1)
        kk = min(k, len(yi))
        if kk == 0:
            continue
        ys = y[yi]
        for s in range(0, len(xi), chunk):
            rows = xi[s:s + chunk]
            d, j = torch.cdist(x[rows], ys).topk(kk, dim=1, largest=False)
            idx[rows, :kk] = yi[j]
            dist[rows, :kk] = d
            valid[rows, :kk] = True
    return idx, dist, valid


def _mlp(i, o):
    return nn.Sequential(nn.Linear(i, i), nn.LeakyReLU(0.2), nn.Linear(i, o))


class DirAtomNet(nn.Module):
    def __init__(self, n_in=atom_features.DIMS, dim=16, k=16, n_layers=3,
                 dist_scale=4.0):
        super().__init__()
        if dim % 2:
            raise ValueError(f"dim {dim}: GroupNorm(2, dim) needs an even width")
        self.dim, self.k, self.dist_scale = dim, k, dist_scale
        self.transform = nn.Sequential(nn.Linear(n_in, dim), nn.LeakyReLU(0.2),
                                       nn.Linear(dim, dim))
        # atom -> atom: [self, neighbour, distance]
        self.atom_mlp = nn.ModuleList([_mlp(2 * dim + 1, dim) for _ in range(n_layers)])
        self.atom_norm = nn.ModuleList([nn.GroupNorm(2, dim) for _ in range(n_layers)])
        # atom -> point: [point, atom, distance, cos(polar direction, atom -> point)]
        self.point_mlp = nn.ModuleList([_mlp(2 * dim + 2, dim) for _ in range(n_layers)])
        self.point_norm = nn.ModuleList([nn.GroupNorm(2, dim) for _ in range(n_layers)])

    @staticmethod
    def direction_cos(xyz, atom_xyz, atom_dir, idx, dist):
        """(N, k) cos between each neighbour atom's polar direction and the
        vector from that atom to the point; 0 for directionless atoms."""
        diff = xyz[:, None, :] - atom_xyz[idx]                     # (N, k, 3)
        return (atom_dir[idx] * diff).sum(-1) / dist.clamp_min(0.5)

    def forward(self, xyz, atom_xyz, atom_feats, batch, atom_batch, atom_dir):
        """xyz (N, 3) surface points, atom_xyz (M, 3), atom_feats (M, n_in),
        batch (N,), atom_batch (M,), atom_dir (M, 3) -> (N, dim)."""
        k = self.k
        h = self.transform(atom_feats)
        relu = nn.functional.leaky_relu

        # atom -> atom; the first neighbour of an atom is itself, dropped
        ia, da, va = knn(atom_xyz, atom_xyz, atom_batch, atom_batch, k + 1)
        ia, da, va = ia[:, 1:], da[:, 1:] / self.dist_scale, va[:, 1:, None]
        for mlp, norm in zip(self.atom_mlp, self.atom_norm):
            msg = mlp(torch.cat([h[:, None, :].expand(-1, k, -1), h[ia], da[..., None]], -1))
            h = h + relu(norm((msg * va).sum(1)), 0.2)

        # atom -> point, with the polar direction
        ip, dp, vp = knn(xyz, atom_xyz, batch, atom_batch, k)
        cos = self.direction_cos(xyz, atom_xyz, atom_dir, ip, dp)
        edge = torch.stack([dp / self.dist_scale, cos], -1)        # (N, k, 2)
        pe = torch.ones(len(xyz), self.dim, dtype=h.dtype, device=h.device)
        for mlp, norm in zip(self.point_mlp, self.point_norm):
            msg = mlp(torch.cat([pe[:, None, :].expand(-1, k, -1), h[ip], edge], -1))
            pe = pe + relu(norm((msg * vp[..., None]).sum(1)), 0.2)
        return pe


# ---------------------------------------------------------------------------
def _selftest():
    torch.manual_seed(0)
    fails = []

    def check(name, ok, detail=""):
        print(f"  {'ok  ' if ok else 'FAIL'} {name}" + (f"   {detail}" if detail else ""))
        if not ok:
            fails.append(name)

    # two proteins: atoms with random chemistry and directions, points around them
    M1, M2, N1, N2 = 300, 200, 500, 400
    atom_xyz = torch.cat([torch.randn(M1, 3) * 6, torch.randn(M2, 3) * 5 + 40])
    atom_batch = torch.cat([torch.zeros(M1), torch.ones(M2)]).long()
    feats = torch.rand(M1 + M2, atom_features.DIMS)
    d = F.normalize(torch.randn(M1 + M2, 3), dim=1)
    d[torch.rand(M1 + M2) < 0.6] = 0                               # non-polar
    xyz = torch.cat([torch.randn(N1, 3) * 8, torch.randn(N2, 3) * 7 + 40])
    batch = torch.cat([torch.zeros(N1), torch.ones(N2)]).long()
    net = DirAtomNet().eval()
    out = net(xyz, atom_xyz, feats, batch, atom_batch, d)
    check("shape (N, 16), finite", tuple(out.shape) == (N1 + N2, 16)
          and bool(torch.isfinite(out).all()))

    # knn against a brute-force reference, and never across proteins
    idx, dist, valid = knn(xyz, atom_xyz, batch, atom_batch, 16)
    ref = torch.cdist(xyz[:N1], atom_xyz[:M1]).topk(16, largest=False)
    check("knn matches brute force and stays inside each protein",
          torch.allclose(dist[:N1], ref.values, atol=1e-5)
          and bool((atom_batch[idx] == batch[:, None]).all()) and bool(valid.all()))

    # invariance: rotate + translate atoms, points and directions together
    R = torch.linalg.qr(torch.randn(3, 3))[0]
    R = R * torch.sign(torch.det(R))
    t = torch.tensor([5.0, -3.0, 9.0])
    out_r = net(xyz @ R.T + t, atom_xyz @ R.T + t, feats, batch, atom_batch, d @ R.T)
    check("rotation / translation invariant", torch.allclose(out, out_r, atol=1e-4),
          f"max diff {float((out - out_r).abs().max()):.1e}")
    perm = torch.randperm(M1 + M2)
    out_p = net(xyz, atom_xyz[perm], feats[perm], batch, atom_batch[perm], d[perm])
    check("atom order does not matter", torch.allclose(out, out_p, atol=1e-4))

    # the direction is seen: flip every polar direction, the output changes;
    # with no polar atoms, flipping does nothing
    out_f = net(xyz, atom_xyz, feats, batch, atom_batch, -d)
    check("flipping polar directions changes the output",
          float((out - out_f).abs().max()) > 1e-3)

    # cos is bounded even for a point sitting on an atom (clamp at 0.5 A)
    on = atom_xyz[:3].clone()
    ii, dd, _ = knn(on, atom_xyz, torch.zeros(3).long(), atom_batch, 16)
    c = DirAtomNet.direction_cos(on, atom_xyz, d, ii, dd)
    check("cos stays in [-1, 1] and finite, even for a point on an atom",
          bool(torch.isfinite(c).all()) and float(c.abs().max()) <= 1 + 1e-5)

    # gradients reach every parameter and the atom features
    net.train()
    fg = feats.clone().requires_grad_(True)
    net(xyz, atom_xyz, fg, batch, atom_batch, d).pow(2).mean().backward()
    check("gradient reaches every parameter and the atom features",
          float(fg.grad.abs().sum()) > 0
          and all(p.grad is not None and float(p.grad.abs().sum()) > 0
                  for p in net.parameters()),
          str([n for n, p in net.named_parameters()
               if p.grad is None or float(p.grad.abs().sum()) == 0]))

    # a protein with fewer atoms than k: padded neighbours are masked, not NaN
    small = DirAtomNet().eval()
    o = small(torch.randn(20, 3), torch.randn(5, 3), torch.rand(5, atom_features.DIMS),
              torch.zeros(20).long(), torch.zeros(5).long(), torch.zeros(5, 3))
    check("fewer atoms than k: finite (padding masked)", bool(torch.isfinite(o).all()))

    print("\nPASS" if not fails else f"\nFAIL: {fails}")
    raise SystemExit(1 if fails else 0)


if __name__ == "__main__":
    _selftest()
