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

TypedAttnPool (below) is the learned-heads generalisation, pool_mode
"typed_attn".

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

    def heads(self, emb, feats=None):
        """-> (type logits (n, T + 1), salience logits (n,)): the raw head
        outputs, which plip_loss.py supervises."""
        z = emb if feats is None else torch.cat([emb, feats.to(emb.dtype)], dim=1)
        return self.type_head(z), self.sal_head(z).squeeze(-1)

    def site_types(self, emb, feats=None):
        """-> (a (n, T), s (n,)) for points, e.g. to inspect what a type learned."""
        tl, sl = self.heads(emb, feats)
        return F.softmax(tl, dim=-1)[:, :self.n_types], torch.sigmoid(sl)

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


def attn_width(emb_dims, n_types=8, n_heads=16, v_dims=8):
    """TypedAttnPool's output width: mean + mass + per head (log count, value)."""
    return emb_dims + n_types + n_heads * (1 + v_dims)


class TypedAttnPool(PairPool):
    """Typed distance attention: PairPool with learned heads (the v8 ladder rung P3).

    PairPool counts, for every PAIR of site types and every distance bin, how
    much type-t1 surface sits at that distance from type-t2 surface: 36 type
    pairs x 8 bins, all with equal room, carrying a count and nothing else.
    This keeps the per-point heads (type softmax a_i, salience s_i, so
    PairPool.heads / site_types and the PLIP losses work unchanged) and makes
    three things learned, per head h:

        g_i = a_i . softplus(alpha_h)       which site types the head starts FROM
        k_j = a_j . softplus(beta_h)        which site types it looks AT
        psi_h(d) = sum_b softplus(c_hb) K_b(d)   its own distance profile over
                                            n_bins Gaussians (r_min..r_max)
        W_ij = m_i g_i * m_j k_j * psi_h(d_ij),  m = s * w, 0 below `exclude`
        M_h  = sum_ij W_ij                  how much of the arrangement there is
        o_h  = sum_ij W_ij v_j / (M_h + eps)    what the partners look like,
                                            v_j = Linear(z_j) of v_dims

    out = [mean h, log1p(mass / mass_scale), per head log1p(M_h / count_scale)
    and o_h]. PairPool is the special case of one head per type pair with
    one-hot gates, one fixed bin each and v = 1. Sums, not a softmax, so
    counts survive ("two donors" vs "five donors"); distances only, so it is
    rigid-motion invariant; heads read as (from types, to types, distance
    profile), so channel_viz can print them.

    Cost: the same no-grad n x n x n_bins bin tensor as PairPool, contracted
    once against (n, heads, 1 + v_dims) - a matmul, recomputed in the backward
    pass under checkpoint as PairPool's is. Regions above max_points are
    thinned the same way, counts rescaled by (n / kept)^2.
    """

    def __init__(self, emb_dims, feat_dims=0, n_types=8, n_heads=16, v_dims=8,
                 n_bins=12, r_min=3.0, r_max=16.0, exclude=2.0, hidden=32,
                 mass_scale=10.0, count_scale=100.0, max_points=4096, eps=1e-6):
        super().__init__(emb_dims, feat_dims, n_types=n_types, n_bins=n_bins,
                         r_min=r_min, r_max=r_max, exclude=exclude, hidden=hidden,
                         mass_scale=mass_scale, max_points=max_points)
        self.n_heads, self.v_dims = n_heads, v_dims
        self.count_scale, self.eps = count_scale, eps
        # Random, not zero: identical heads would get identical gradients and
        # never separate. softplus keeps every gate and profile >= 0, so W and
        # M are counts.
        self.q_aff = nn.Parameter(torch.randn(n_heads, n_types) * 0.5)
        self.k_aff = nn.Parameter(torch.randn(n_heads, n_types) * 0.5)
        self.dist_w = nn.Parameter(torch.randn(n_heads, n_bins) * 0.5)
        self.value = nn.Linear(emb_dims + feat_dims, v_dims)

    @property
    def out_dims(self):
        return attn_width(self.emb_dims, self.n_types, self.n_heads, self.v_dims)

    def _attn(self, q, xv, c, x):
        """q (n, H), xv (n, H, C), c (H, B), x (n, 3) -> (H, C)."""
        k = self._bins(x)                                     # (n, n, B), no grad
        a = torch.einsum("ijb,jhc->ibhc", k, xv)              # (n, B, H, C)
        z = torch.einsum("ibhc,hb->ihc", a, c)
        return torch.einsum("ih,ihc->hc", q, z)

    def forward(self, emb, feats, w, regions, xyz, eps=1e-6):
        """emb (N, E), feats (N, F) or None, w (N,), regions: list of index
        tensors into these N points, xyz (N, 3) -> (R, out_dims)."""
        if not regions:
            return emb.new_zeros(0, self.out_dims)
        qa, ka = F.softplus(self.q_aff), F.softplus(self.k_aff)
        c = F.softplus(self.dist_w).float()
        out = []
        for r in regions:
            r = r.to(emb.device)
            h, ww = emb[r], w[r]
            f = None if feats is None else feats[r]
            z = h if f is None else torch.cat([h, f.to(h.dtype)], dim=1)
            tl, sl = self.type_head(z), self.sal_head(z).squeeze(-1)
            a = F.softmax(tl, dim=-1)[:, :self.n_types]
            m = torch.sigmoid(sl) * ww
            mean = (ww[:, None] * h).sum(0) / (ww.sum() + eps)
            mass = (m[:, None] * a).sum(0)
            q = (m[:, None] * (a @ qa.T)).float()                       # (n, H)
            v = torch.cat([torch.ones_like(m)[:, None], self.value(z)], 1)
            xv = ((m[:, None] * (a @ ka.T))[:, :, None] * v[:, None, :]).float()
            x = xyz[r].detach().float()
            scale = 1.0
            if len(r) > self.max_points:
                with torch.no_grad():
                    cen = (ww[:, None] * x).sum(0) / (ww.sum() + eps)
                    order = torch.argsort((x - cen).norm(dim=1))
                    keep = order[::-(-len(r) // self.max_points)]
                scale = (len(r) / len(keep)) ** 2
                q, xv, x = q[keep], xv[keep], x[keep]
            if self.training and (q.requires_grad or xv.requires_grad):
                o = checkpoint(self._attn, q, xv, c, x, use_reentrant=False)
            else:
                o = self._attn(q, xv, c, x)
            o = o * scale
            cnt = o[:, 0]
            part = o[:, 1:] / (cnt[:, None] + self.eps)
            out.append(torch.cat([mean, torch.log1p(mass / self.mass_scale),
                                  torch.log1p(cnt.to(mean.dtype) / self.count_scale),
                                  part.to(mean.dtype).reshape(-1)]))
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
          f"max diff {float((v - v_rot).detach().abs().max()):.1e}")
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

    # ---- TypedAttnPool -------------------------------------------------
    print("  TypedAttnPool:")
    ta = TypedAttnPool(E, Fd).eval()
    v = ta(emb, feats, w, regs, xyz)
    check("typed_attn: shape = attn_width",
          tuple(v.shape) == (2, attn_width(E)) == (2, ta.out_dims), f"{tuple(v.shape)}")
    check("typed_attn: finite", bool(torch.isfinite(v).all()))
    v_rot = ta(emb, feats, w, regs, xyz @ R.T + torch.tensor([5.0, -3.0, 9.0]))
    check("typed_attn: rigid-motion invariant", torch.allclose(v, v_rot, atol=1e-4),
          f"max diff {float((v - v_rot).abs().max()):.1e}")
    v_p = ta(emb[perm], feats[perm], w[perm], [inv[r] for r in regs], xyz[perm])
    check("typed_attn: permutation invariant", torch.allclose(v, v_p, atol=1e-4))
    t0 = TypedAttnPool(E, 0).eval()
    a_near = t0(e2, None, torch.ones(40), [torch.arange(40)], near)[0]
    a_far = t0(e2, None, torch.ones(40), [torch.arange(40)], far)[0]
    k = E + t0.n_types
    check("typed_attn: same composition -> identical mean and mass",
          torch.allclose(a_near[:k], a_far[:k]))
    check("typed_attn: different arrangement -> different head block",
          float((a_near[k:] - a_far[k:]).abs().max()) > 1e-3)
    # counts survive: the same site twice over (a copy 30 A away, beyond every
    # bin) doubles every head's count and leaves the partner means alone
    two = torch.cat([near, near + torch.tensor([0.0, 30.0, 0.0])])
    a_two = t0(torch.cat([e2, e2]), None, torch.ones(80), [torch.arange(80)], two)[0]
    H, V = t0.n_heads, t0.v_dims
    blk1, blk2 = a_near[k:].view(H, 1 + V), a_two[k:].view(H, 1 + V)
    cnt1 = torch.expm1(blk1[:, 0]) * t0.count_scale
    cnt2 = torch.expm1(blk2[:, 0]) * t0.count_scale
    check("typed_attn: two copies of a site -> twice the count, same partners",
          torch.allclose(cnt2, 2 * cnt1, rtol=1e-3)
          and torch.allclose(blk2[:, 1:], blk1[:, 1:], atol=1e-4))
    ta.train()
    emb_g = emb.clone().requires_grad_(True)
    feats_g = feats.clone().requires_grad_(True)
    out = ta(emb_g, feats_g, w, regs, xyz)
    out[:, E + ta.n_types:].sum().backward()
    check("typed_attn: gradient reaches embeddings, input channels and every "
          "parameter through the head block (checkpointed)",
          float(emb_g.grad.abs().sum()) > 0 and float(feats_g.grad.abs().sum()) > 0
          and all(p.grad is not None and float(p.grad.abs().sum()) > 0
                  for p in ta.parameters()),
          str([n for n, p in ta.named_parameters()
               if p.grad is None or float(p.grad.abs().sum()) == 0]))
    tsmall = TypedAttnPool(E, Fd, max_points=100).eval()
    tsmall.load_state_dict(ta.state_dict())
    vs = tsmall(emb, feats, w, regs, xyz)
    full = ta.eval()(emb, feats, w, regs, xyz)
    cos = float(F.cosine_similarity(vs[0, k:], full[0, k:], dim=0))
    check("typed_attn: a thinned large region stays close to the full one",
          cos > 0.95, f"head-block cosine {cos:.3f}")
    check("typed_attn: heads() / site_types() are PairPool's (PLIP-compatible)",
          ta.heads(emb, feats)[0].shape == (n, ta.n_types + 1))
    check("typed_attn: no regions -> empty", ta(emb, feats, w, [], xyz).shape == (0, ta.out_dims))
    print("\nPASS" if not fails else f"\nFAIL: {fails}")
    raise SystemExit(1 if fails else 0)


if __name__ == "__main__":
    _selftest()
