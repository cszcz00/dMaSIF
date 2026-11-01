"""Tanimoto regulariser on the ligand side of the joint space (the v8 run).

Why. InfoNCE only ever asks a ligand vector to sit near ITS protein's anchors
and away from the batch's other proteins. Nothing says two ligands of the same
chemotype should sit near each other, and with ~19 visits per target group
(dataset_v2) the ligand encoder is free to place close analogues far apart -
which is memorisation on the ligand side. This adds a LIGHT pull: pairs of
training ligands at ECFP4 Tanimoto >= `thr` are asked to have a cosine of at
least their Tanimoto, in the same normalised space the anchors are scored in:

    loss = mean over pairs (i, j) with T_ij >= thr of  relu(T_ij - cos(z_i, z_j))^2

One-sided (a pair already closer than its Tanimoto is left alone) and bounded
(cos >= T, never cos -> 1), so it cannot by itself collapse the space; the
`random cos` diagnostic (mean cosine of unrelated in-batch ligands) is the
collapse alarm and should stay well below the pulled pairs' cosine.

Pairs per step. A batch draws one row per target group, so in-batch pairs at
Tanimoto >= 0.5 are rare. Each batch ligand therefore also brings one partner
drawn from its precomputed neighbours (`Neighbours`, TRAIN ligands only, so
validation ligands never enter), and the partner goes through the ligand
encoder with the batch. Ligands are cheap next to proteins (a ~30-atom graph
against thousands of surface points), so this costs little. The ligand
encoder has no batch-dependent normalisation (GINE + LayerNorm), so extra
ligands in the forward do not change the batch's own vectors. Partners are
drawn from a stream of their own, so switching this on does not change which
batches a run sees.

Interplay with InfoNCE's false-negative mask: pairs at Tanimoto >= --tanimoto
(0.8) are already masked out of the denominator; pairs in [thr, 0.8) are
negatives there AND pulled here. The weight is meant to be small (0.05-0.1)
so this shapes the ligand space without overruling the contrastive loss.

    python src/lig_sim.py selftest      # numpy part always; torch part if torch imports

Depends on: numpy; torch for `pair_loss` only.
"""

import argparse
import sys

import numpy as np


# ---------------------------------------------------------------------------
# neighbours (numpy)
# ---------------------------------------------------------------------------
def tanimoto_np(a, b):
    """0/1 fingerprints (A, F), (B, F) -> (A, B) float32 Tanimoto."""
    a = np.asarray(a, np.float32)
    b = np.asarray(b, np.float32)
    inter = a @ b.T
    union = a.sum(1)[:, None] + b.sum(1)[None, :] - inter
    return inter / np.maximum(union, 1.0)


class Neighbours:
    """Each ligand's training-set neighbours at Tanimoto >= thr, CSR-packed.

    fp:       the full fingerprint table (ligands.fp), indexed by fp_row
    fp_rows:  the fp_rows allowed as ligands AND partners (the train rows'),
              duplicates fine
    k:        keep at most the k most similar per ligand (ties broken by
              fp_row, so the table is deterministic)
    A ligand is never its own neighbour; distinct fp_rows with identical
    fingerprints (T = 1) are kept, since they are the same chemotype.

    Chunked numpy matmul: ~20k distinct ligands x 2048 bits is ~10 s on a
    few cores, done once at the start of a run.
    """

    def __init__(self, fp, fp_rows, thr=0.5, k=32, chunk=1024):
        self.thr, self.k = float(thr), int(k)
        rows = np.unique(np.asarray(fp_rows, np.int64))
        self.rows = rows
        F = np.asarray(fp[rows], np.float32)
        ptr, idx, sim = [0], [], []
        for s in range(0, len(rows), chunk):
            T = tanimoto_np(F[s:s + chunk], F)
            T[np.arange(len(T)), np.arange(s, s + len(T))] = -1.0   # not self
            for t in T:
                j = np.nonzero(t >= self.thr)[0]
                if len(j) > self.k:
                    # most similar first; stable sort keeps fp_row order on ties
                    j = j[np.argsort(-t[j], kind="stable")[:self.k]]
                idx.append(rows[j])
                sim.append(t[j])
                ptr.append(ptr[-1] + len(j))
        self.ptr = np.asarray(ptr, np.int64)
        self.idx = np.concatenate(idx) if idx else np.zeros(0, np.int64)
        self.sim = (np.concatenate(sim) if sim else np.zeros(0)).astype(np.float32)
        self._pos = {int(r): i for i, r in enumerate(rows)}

    def of(self, fp_row):
        """-> (neighbour fp_rows, their Tanimoto) for one ligand."""
        i = self._pos.get(int(fp_row))
        if i is None:
            return self.idx[:0], self.sim[:0]
        a, b = self.ptr[i], self.ptr[i + 1]
        return self.idx[a:b], self.sim[a:b]

    def sample(self, rng, fp_rows):
        """One neighbour per query ligand, uniformly -> (query positions,
        partner fp_rows, Tanimoto). Ligands without neighbours are skipped."""
        at, part, t = [], [], []
        for q, f in enumerate(fp_rows):
            nb, s = self.of(f)
            if len(nb):
                j = int(rng.integers(len(nb)))
                at.append(q)
                part.append(int(nb[j]))
                t.append(float(s[j]))
        return (np.asarray(at, np.int64), np.asarray(part, np.int64),
                np.asarray(t, np.float32))

    def describe(self):
        n = np.diff(self.ptr)
        return (f"{len(self.rows)} train ligands, {float((n > 0).mean()):.2f} with "
                f">= 1 neighbour at Tanimoto >= {self.thr} (median "
                f"{int(np.median(n[n > 0])) if (n > 0).any() else 0}, capped at {self.k})")


def in_batch_pairs(fp, fp_rows, thr):
    """Upper-triangle pairs of a batch at Tanimoto >= thr, different ligands
    -> (i, j, T)."""
    fp_rows = np.asarray(fp_rows, np.int64)
    T = tanimoto_np(fp[fp_rows], fp[fp_rows])
    i, j = np.triu_indices(len(fp_rows), 1)
    keep = (T[i, j] >= thr) & (fp_rows[i] != fp_rows[j])
    return i[keep], j[keep], T[i, j][keep].astype(np.float32)


# ---------------------------------------------------------------------------
# loss (torch)
# ---------------------------------------------------------------------------
def pair_loss(za, zb, t):
    """Normalised (P, D) vectors in pairs and their Tanimoto (P,) -> (loss,
    mean cosine of the pairs). relu(t - cos)^2 averaged; 0 with grad when P = 0."""
    import torch
    import torch.nn.functional as F

    if not len(za):
        z = za.sum() * 0.0
        return z, float("nan")
    t = torch.as_tensor(t, device=za.device, dtype=za.dtype)
    cos = (za * zb).sum(-1)
    return (F.relu(t - cos) ** 2).mean(), float(cos.detach().mean())


def random_cos(z):
    """Mean off-diagonal cosine of a batch of normalised vectors: the
    collapse alarm."""
    n = len(z)
    if n < 2:
        return float("nan")
    g = (z @ z.T).detach()
    return float((g.sum() - g.diagonal().sum()) / (n * (n - 1)))


# ---------------------------------------------------------------------------
def selftest():
    fails = []

    def check(name, ok, detail=""):
        print(f"  {'ok  ' if ok else 'FAIL'} {name}" + (f"   {detail}" if detail else ""))
        if not ok:
            fails.append(name)

    rng = np.random.default_rng(0)
    # planted chemotypes: 6 families of 10, members share most bits
    n_fam, per, F = 6, 10, 512
    fp = np.zeros((n_fam * per + 5, F), np.uint8)
    for f in range(n_fam):
        core = rng.choice(F, 60, replace=False)
        for m in range(per):
            on = np.concatenate([core[rng.random(60) < 0.9], rng.choice(F, 8)])
            fp[f * per + m, on] = 1
    fp[-5:] = (rng.random((5, F)) < 0.1)                       # unrelated singletons
    fp[-1] = fp[0]                                              # duplicate of ligand 0
    fam = np.concatenate([np.repeat(np.arange(n_fam), per), -np.ones(5, int)])
    fam[-1] = 0

    T = tanimoto_np(fp, fp)
    same = fam[:, None] == fam[None, :]
    off = ~np.eye(len(fp), dtype=bool)
    check("planted: within-family Tanimoto well above between-family",
          T[same & off & (fam[:, None] >= 0)].mean() > 0.6 and T[~same].mean() < 0.2,
          f"{T[same & off & (fam[:, None] >= 0)].mean():.2f} vs {T[~same].mean():.2f}")

    allowed = np.arange(len(fp) - 3)               # 3 ligands are "val": never partners
    nb = Neighbours(fp, np.concatenate([allowed, allowed[:5]]), thr=0.5, k=4, chunk=7)
    ok_self = all(r not in nb.of(r)[0] for r in nb.rows)
    check("never its own neighbour", ok_self)
    check("partners come only from the allowed (train) rows",
          set(nb.idx.tolist()) <= set(allowed.tolist()))
    check("every neighbour at Tanimoto >= thr, at most k per ligand",
          bool((nb.sim >= 0.5).all()) and int(np.diff(nb.ptr).max()) <= 4)
    check("neighbours are family members",
          all(fam[j] == fam[r] for r in nb.rows for j in nb.of(r)[0]))
    nb0, s0 = nb.of(0)
    check("k cap keeps the MOST similar (descending)", bool((np.diff(s0) <= 1e-7).all()),
          " ".join(f"{v:.2f}" for v in s0))
    full = Neighbours(fp, allowed, thr=0.5, k=1000, chunk=1000)
    check("chunking does not change the table",
          all(set(Neighbours(fp, allowed, thr=0.5, k=1000, chunk=3).of(r)[0].tolist())
              == set(full.of(r)[0].tolist()) for r in full.rows))
    check("unknown (val) ligand has no neighbours", len(nb.of(len(fp) - 1)[0]) == 0)
    at, part, t = nb.sample(np.random.default_rng(1), [0, 15, len(fp) - 2, 33])
    check("sample: one partner per ligand that has any, Tanimoto matches",
          list(at) == [0, 1, 3]
          and all(abs(T[q, p] - tt) < 1e-6 for q, p, tt in zip([0, 15, 33], part, t)))
    i, j, tb = in_batch_pairs(fp, [0, 1, 25, 2, 0], 0.5)
    check("in-batch pairs: same-family pairs only, never a ligand with itself",
          set(zip(i.tolist(), j.tolist())) == {(0, 1), (0, 3), (1, 3), (1, 4), (3, 4)},
          str(list(zip(i.tolist(), j.tolist()))))
    print("  " + nb.describe())

    try:
        import torch
    except ImportError:
        print("  skip torch part (no torch here): run this where torch imports")
        print("\nPASS (numpy part)" if not fails else f"\nFAIL: {fails}")
        raise SystemExit(1 if fails else 0)

    import torch.nn.functional as Fn

    za = Fn.normalize(torch.tensor([[1.0, 0.0], [1.0, 0.0]]), dim=1)
    zb = Fn.normalize(torch.tensor([[1.0, 1.0], [0.0, 1.0]]), dim=1)
    loss, c = pair_loss(za, zb, [0.5, 0.5])
    # cos 0.707 >= 0.5 -> 0; cos 0 -> 0.25; mean 0.125
    check("pair_loss: one-sided hinge, by hand", abs(float(loss) - 0.125) < 1e-6
          and abs(c - (0.7071 + 0) / 2) < 1e-3, f"{float(loss):.4f}")
    z0, c0 = pair_loss(za[:0], zb[:0], [])
    check("no pairs -> 0 that backprops, cos NaN", float(z0) == 0 and np.isnan(c0))
    # planted: training free vectors with the pull alone raises family cosine
    # to at least the Tanimoto and does NOT collapse unrelated ligands
    torch.manual_seed(0)
    emb = torch.nn.Parameter(torch.randn(len(fp), 32))
    opt = torch.optim.Adam([emb], lr=0.05)
    gen = np.random.default_rng(2)
    for _ in range(300):
        q = gen.choice(full.rows, 16, replace=False)
        at, part, t = full.sample(gen, q)
        z = Fn.normalize(emb, dim=1)
        l, _ = pair_loss(z[torch.as_tensor(q[at])], z[torch.as_tensor(part)], t)
        opt.zero_grad()
        l.backward()
        opt.step()
    with torch.no_grad():
        z = Fn.normalize(emb, dim=1)
        G = (z @ z.T).numpy()
    pairs = full.idx, np.repeat(full.rows, np.diff(full.ptr))
    viol = np.maximum(full.sim - G[pairs[1], pairs[0]], 0)
    rnd = G[~same & off].mean()
    check("planted: after training, pulled pairs reach cos ~>= Tanimoto",
          float(viol.mean()) < 0.05, f"mean shortfall {viol.mean():.3f}")
    check("planted: unrelated ligands not collapsed (mean cos < 0.3)",
          abs(rnd) < 0.3, f"{rnd:.3f}")
    check("random_cos matches the definition",
          abs(random_cos(z[:10]) - (G[:10, :10].sum() - np.trace(G[:10, :10])) / 90) < 1e-5)

    print("\nPASS" if not fails else f"\nFAIL: {fails}")
    raise SystemExit(1 if fails else 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")
    ap.parse_args()
    selftest()


if __name__ == "__main__":
    sys.exit(main())
