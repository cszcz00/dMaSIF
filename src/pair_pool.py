"""Pairwise type-distance pooling: a learned 3D pharmacophore pair fingerprint.

Why. Binding specificity is a handful of sites in a particular ARRANGEMENT -
an H-bond donor ~7 A from an acceptor, both buried, beside an aromatic patch -
matched by complementary ligand groups. The other pooling modes cannot say
that: `mean` keeps what the pocket is made of, `mean_std` how uniform it is,
`shells` one radial order around the centroid (two acceptors on the same side
and on opposite sides look alike). This keeps WHICH kinds of site sit at WHAT
distance from each other.

For one region of n surface points, with embeddings h_i (the conv output, plus
optionally the raw input channels - curvature, AtomNet chemistry and, with
--chemo, the pharmacophore composition / buriedness / proximity, which are
sharp where the 12 A conv is smeared) and positions x_i:

    a_i = softmax(type_head(z_i))[:T]     soft site type; T learned types plus
                                          one "not a site" type that is dropped
    s_i = sigmoid(sal_head(z_i)) * w_i    salience: only a few polar atoms of a
                                          pocket matter; w_i is the encoder's
                                          pool weight (ones for external pockets)
    m_i = s_i a_i                         per-point site-type mass (T,)

    mean  = sum_i w_i h_i / sum_i w_i                          (E,)
    mass  = log1p(sum_i m_i / mass_scale)                      (T,)   how many
    H     = sum_{i,j} m_i[t1] m_j[t2] K_b(|x_i - x_j|)         (T, T, B)
    pairs = log1p(H[t1 <= t2] / pair_scale)                    (T(T+1)/2 * B,)

    out = [mean, mass, pairs]

K_b are Gaussian distance bins with centres evenly from r_min to r_max and
width = their spacing, zeroed below `exclude` A: one polar atom shows up as a
cluster of 10-20 surface points ~1 A apart (dMaSIF samples ~1 point/A^2), and
without the exclusion an atom's own cluster would fill the short bins. Counts
are kept rather than normalised away, because "two donors" and "five donors"
are different pockets; log1p keeps them on the scale of the other channels.

Properties, all checked by the self-test:
  rigid-motion invariant   only distances inside the region are used
  permutation invariant    sums over points; the large-region subsample (below)
                           is chosen by distance to the centroid, not by order
  differentiable           in h (and the input channels) through a_i, s_i and
                           the mean; distances and bins carry no gradient
  arrangement-sensitive    same composition, different geometry -> different
                           `pairs`, identical `mean` and `mass`

Cost: the bin tensor is n x n x B per region, ~20 MB at n = 800, B = 8. It is
built inside torch.utils.checkpoint in training, so it is recomputed in the
backward pass instead of being held for it; with 32 proteins x 8 regions that
is the difference between ~5 GB and nothing. Regions above `max_points` (4096)
are thinned to every k-th point by distance from the centroid and H rescaled
by (n / kept)^2, so a huge merged site cannot exhaust the GPU.

Depends on: torch. Self-test: `python src/pair_pool.py` (CPU torch, no KeOps).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def pair_width(emb_dims, n_types=8, n_bins=8):
    """Output width for an embedding width: mean + mass + upper-triangle pairs."""
    return emb_dims + n_types + n_types * (n_types + 1) // 2 * n_bins


class PairPool(nn.Module):
    def __init__(self, emb_dims, feat_dims=0, n_types=8, n_bins=8, r_min=3.0,
                 r_max=16.0, exclude=2.0, hidden=32, mass_scale=10.0,
                 pair_scale=100.0, max_points=4096):
        super().__init__()
        if n_bins < 2 or r_max <= r_min:
            raise ValueError(f"pair bins: need n_bins >= 2 and r_max > r_min, got "
                             f"{n_bins}, {r_min}..{r_max}")
        self.emb_dims, self.feat_dims = emb_dims, feat_dims
        self.n_types, self.n_bins = n_types, n_bins
        self.exclude, self.max_points = exclude, max_points
        self.mass_scale, self.pair_scale = mass_scale, pair_scale
        z = emb_dims + feat_dims
        self.type_head = nn.Sequential(nn.Linear(z, hidden), nn.LeakyReLU(0.2),
                                       nn.Linear(hidden, n_types + 1))
        self.sal_head = nn.Sequential(nn.Linear(z, hidden), nn.LeakyReLU(0.2),
                                      nn.Linear(hidden, 1))
        self.register_buffer("centres", torch.linspace(r_min, r_max, n_bins),
                             persistent=False)
        self.width_A = (r_max - r_min) / (n_bins - 1)
        iu = torch.triu_indices(n_types, n_types)
        self.register_buffer("iu", iu, persistent=False)

    @property
    def out_dims(self):
        return pair_width(self.emb_dims, self.n_types, self.n_bins)

    def _bins(self, x):
        """(n, 3) -> (n, n, B) Gaussian distance bins, no gradient."""
        with torch.no_grad():
            d = torch.cdist(x, x)
            k = torch.exp(-0.5 * ((d[..., None] - self.centres) / self.width_A) ** 2)
            return k * (d >= self.exclude)[..., None]

    def _pairs(self, m, x):
        k = self._bins(x)
        a = torch.einsum("it,ijb->tjb", m, k)
        return torch.einsum("tjb,ju->tub", a, m)

    def site_types(self, emb, feats=None):
        """-> (a (n, T), s (n,)) for points, e.g. to inspect what a type learned."""
        z = emb if feats is None else torch.cat([emb, feats.to(emb.dtype)], dim=1)
        a = F.softmax(self.type_head(z), dim=-1)[:, :self.n_types]
        s = torch.sigmoid(self.sal_head(z)).squeeze(-1)
        return a, s

    def forward(self, emb, feats, w, regions, xyz, eps=1e-6):
        """emb (N, E), feats (N, F) or None, w (N,), regions: list of index
        tensors into these N points, xyz (N, 3) -> (R, out_dims)."""
        if not regions:
            return emb.new_zeros(0, self.out_dims)
        out = []
        for r in regions:
            r = r.to(emb.device)
            h = emb[r]
            ww = w[r]
            a, s = self.site_types(h, None if feats is None else feats[r])
            m = (s * ww)[:, None] * a
            mean = (ww[:, None] * h).sum(0) / (ww.sum() + eps)
            mass = m.sum(0)
            x = xyz[r].detach().float()
            scale = 1.0
            if len(r) > self.max_points:
                with torch.no_grad():
                    c = (ww[:, None] * x).sum(0) / (ww.sum() + eps)
                    order = torch.argsort((x - c).norm(dim=1))
                    keep = order[::-(-len(r) // self.max_points)]
                scale = (len(r) / len(keep)) ** 2
                m_p, x = m[keep], x[keep]
            else:
                m_p = m
            if self.training and m_p.requires_grad:
                H = checkpoint(self._pairs, m_p.float(), x, use_reentrant=False)
            else:
                H = self._pairs(m_p.float(), x)
            H = H[self.iu[0], self.iu[1]].reshape(-1) * scale
            out.append(torch.cat([mean, torch.log1p(mass / self.mass_scale),
                                  torch.log1p(H.to(mean.dtype) / self.pair_scale)]))
        return torch.stack(out)


# ---------------------------------------------------------------------------
def _selftest():
    torch.manual_seed(0)
    fails = []

    def check(name, ok, detail=""):
        print(f"  {'ok  ' if ok else 'FAIL'} {name}" + (f"   {detail}" if detail else ""))
        if not ok:
            fails.append(name)

    E, Fd, n = 16, 12, 300
    pool = PairPool(E, Fd).eval()
    emb, feats = torch.randn(n, E), torch.randn(n, Fd)
    xyz = torch.randn(n, 3) * 6
    w = torch.ones(n)
    regs = [torch.arange(n), torch.arange(0, n, 2)]
    v = pool(emb, feats, w, regs, xyz)
    check("shape = pair_width", tuple(v.shape) == (2, pair_width(E)) == (2, pool.out_dims),
          f"{tuple(v.shape)}")
    R = torch.linalg.qr(torch.randn(3, 3))[0]
    v_rot = pool(emb, feats, w, regs, xyz @ R.T + torch.tensor([5.0, -3.0, 9.0]))
    check("rigid-motion invariant", torch.allclose(v, v_rot, atol=1e-4),
          f"max diff {float((v - v_rot).abs().max()):.1e}")
    perm = torch.randperm(n)
    inv = torch.argsort(perm)
    v_p = pool(emb[perm], feats[perm], w[perm], [inv[r] for r in regs], xyz[perm])
    check("permutation invariant", torch.allclose(v, v_p, atol=1e-4))

    # same points, same features, rearranged: two "sites" 5 A vs 12 A apart
    site = torch.randn(1, E).repeat(20, 1)
    other = torch.randn(1, E).repeat(20, 1)
    blob = torch.randn(20, 3) * 0.7
    e2 = torch.cat([site, other])
    near = torch.cat([blob, blob + torch.tensor([5.0, 0, 0])])
    far = torch.cat([blob, blob + torch.tensor([12.0, 0, 0])])
    p0 = PairPool(E, 0).eval()
    a_near = p0(e2, None, torch.ones(40), [torch.arange(40)], near)[0]
    a_far = p0(e2, None, torch.ones(40), [torch.arange(40)], far)[0]
    k = E + p0.n_types
    check("same composition: identical mean and mass", torch.allclose(a_near[:k], a_far[:k]))
    check("different arrangement: different pairs",
          float((a_near[k:] - a_far[k:]).abs().max()) > 1e-3)
    nb = p0.n_bins
    Hn = a_near[k:].view(-1, nb).sum(0)
    Hf = a_far[k:].view(-1, nb).sum(0)
    b5 = int(torch.argmin((p0.centres - 5).abs()))
    b12 = int(torch.argmin((p0.centres - 12).abs()))
    check("the 5 A arrangement fills the bin nearest 5 A, the 12 A one nearest 12 A",
          bool(Hn[b5] > Hf[b5]) and bool(Hf[b12] > Hn[b12]))

    pool.train()
    emb_g = emb.clone().requires_grad_(True)
    feats_g = feats.clone().requires_grad_(True)
    out = pool(emb_g, feats_g, w, regs, xyz)
    out[:, E + pool.n_types:].sum().backward()      # the pair block alone
    check("gradient reaches the embeddings, the input channels and both heads "
          "through the pair block (checkpointed)",
          emb_g.grad is not None and float(emb_g.grad.abs().sum()) > 0
          and float(feats_g.grad.abs().sum()) > 0
          and all(p.grad is not None for p in pool.parameters()))

    small = PairPool(E, Fd, max_points=100).eval()
    small.load_state_dict(pool.state_dict())
    vs = small(emb, feats, w, regs, xyz)
    big = pool.eval()(emb, feats, w, regs, xyz)
    k = E + pool.n_types
    cos = float(F.cosine_similarity(vs[0, k:], big[0, k:], dim=0))
    check("a thinned large region stays close to the full one, and stays invariant",
          cos > 0.95 and torch.allclose(vs, small(emb[perm], feats[perm], w[perm],
                                                  [inv[r] for r in regs], xyz[perm]), atol=1e-4),
          f"pair-block cosine to the unthinned region {cos:.3f}")
    check("no regions -> empty", pool(emb, feats, w, [], xyz).shape == (0, pool.out_dims))
    print("\nPASS" if not fails else f"\nFAIL: {fails}")
    raise SystemExit(1 if fails else 0)


if __name__ == "__main__":
    _selftest()
