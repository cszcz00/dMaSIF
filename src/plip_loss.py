"""PLIP supervision of pair pooling's site types and salience (the v6 run).

Why. channel_viz.py on C (RESULTS, 2026-10-29) found the 8 learned site types
NOT collapsed but coarse - two polar, two apolar, two shape types, and no
type separating donor from acceptor - and the salience head tracking
concavity (PLIP-contact AUROC 0.55 inside the site) rather than which points
a ligand actually uses. v3d showed that more input chemistry alone does not
sharpen them: the contrastive loss is what keeps them coarse. PLIP says, per
training complex, which protein atoms the ligand H-bonds, salt-bridges,
pi-stacks or packs against; this module turns that into two auxiliary losses
on the points of the row's SITE region (the pocket the ligand sits in):

    salience   BCE(sal_logit, contacted)
               contacted = any PLIP class at the point, for ANY ligand of the
               receptor (the union over sibling rows is denser, and a point
               another ligand uses is a usable point, not a negative).
               Positive-weighted by the batch's negative / positive ratio.
    types      cross-entropy of the type softmax (T + 1 incl. "not a site")
               against the PLIP classes, on contacted points only:
               q = uniform over the point's classes, mapped type t <- class t
               for t < 6 (donor, acceptor, cation, anion, aromatic,
               hydrophobic). Types 6.. and "not a site" get no target, so they
               stay free for whatever retrieval wants. Uncontacted points get
               no type loss: PLIP labels are positive-unlabelled (a donor this
               ligand does not use is still a donor).

Hard or soft labels (`soft`, train.py --plip_soft). Hard (0, v6): a point is
class c when it lies within 3 A of a class-c protein atom (4 A of a ring /
charge centre), a step from 1 to 0. That step lands between surface points
~1 A apart whose embeddings are nearly identical (the conv sees ~12 A), so
salience is asked to separate the inseparable, and v6's salience BCE sat at
its constant-prediction value (~0.93, RESULTS). Soft (width s A): a linear
ramp from 1 at r - s/2 to 0 at r + s/2, max over the class's interactions -
the same total label mass as the step, spread over the ~1 A uncertainty of
where the surface sits over an atom. The salience target is then the
point's max class weight (a soft BCE target), the type target q is the class
weights normalised (the nearer interaction dominates a two-class point), and
each point's type CE is weighted by its salience target. With hard labels
every one of these reduces exactly to the v6 losses.

Inference needs none of this: the labels are a training signal only.

Split so the label side runs without torch:
    point_labels(xyz, recs)   numpy: per-point (N, 6) class labels, union over
                              PLIP records (plip_labels.site_labels each, or
                              soft_site_labels with soft > 0)
    receptor_labels(...)      numpy: this row's labels + the receptor union,
                              loaded from data/plip/<system_id>.npz
    plip_losses(...)          torch: the two losses for a batch

    python src/plip_loss.py selftest     # numpy part always; torch part if
                                         # torch imports (login node, container)

Depends on: numpy, plip_labels (numpy); torch for plip_losses only.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

N_CLASSES = 6          # plip_labels.CLASSES: donor acceptor cation anion aromatic hydrophobic


# ---------------------------------------------------------------------------
# labels (numpy)
# ---------------------------------------------------------------------------
def soft_site_labels(xyz, rec, soft, atom_radius=3.0, centre_radius=4.0,
                     chunk=8192):
    """Surface points -> (N, 6) float32 in [0, 1]: plip_labels.site_labels with
    the step at radius r replaced by a linear ramp from 1 at r - soft/2 to 0 at
    r + soft/2, max over each class's interactions."""
    from plip_labels import CENTRE_KINDS, KIND_CLASS

    xyz = np.asarray(xyz, np.float32)
    out = np.zeros((len(xyz), N_CLASSES), np.float32)
    if not len(rec["kind"]):
        return out
    kind = rec["kind"].astype(np.int64)
    r = np.where(CENTRE_KINDS[kind], centre_radius, atom_radius).astype(np.float32)
    cls = KIND_CLASS[kind]
    P = rec["prot_xyz"].astype(np.float32)
    for s in range(0, len(xyz), chunk):
        d = np.sqrt(((xyz[s:s + chunk, None] - P[None]) ** 2).sum(-1))       # (n, k)
        w = np.clip((r[None] + soft / 2 - d) / soft, 0.0, 1.0)
        for c in range(N_CLASSES):
            m = cls == c
            if m.any():
                out[s:s + chunk, c] = w[:, m].max(1)
    return out


def point_labels(xyz, recs, soft=0.0, **kw):
    """(N, 3) surface points + PLIP records -> (N, 6) labels, union over
    records: bool (OR) when soft is 0, float32 (max) when soft > 0."""
    from plip_labels import site_labels

    if soft > 0:
        out = np.zeros((len(xyz), N_CLASSES), np.float32)
        for rec in recs:
            if rec is not None and len(rec["kind"]):
                np.maximum(out, soft_site_labels(xyz, rec, soft, **kw), out=out)
        return out
    out = np.zeros((len(xyz), N_CLASSES), bool)
    for rec in recs:
        if rec is not None and len(rec["kind"]):
            out |= site_labels(xyz, rec, **kw)
    return out


def receptor_labels(xyz, system_id, sibling_ids, plip_dir, **kw):
    """-> (this row's (N, 6) labels or None, receptor-union (N, 6) labels or
    None). None where no PLIP file exists for the row / any sibling."""
    from plip_labels import load_plip

    def load(sid):
        f = Path(plip_dir) / f"{sid}.npz"
        return load_plip(f) if f.exists() else None

    own = load(system_id)
    others = [load(s) for s in sibling_ids if s != system_id]
    recs = [r for r in [own] + others if r is not None]
    row = point_labels(xyz, [own], **kw) if own is not None else None
    union = point_labels(xyz, recs, **kw) if recs else None
    return row, union


# ---------------------------------------------------------------------------
# losses (torch)
# ---------------------------------------------------------------------------
def plip_losses(type_logits, sal_logits, union_labels, max_pos_weight=20.0):
    """Per protein lists -> (salience loss, type loss, stats dict), scalars.

    type_logits[b]   (n_b, T + 1)  pair_pool type head on the site points
    sal_logits[b]    (n_b,)        salience head on the same points
    union_labels[b]  (n_b, 6)      receptor-union PLIP classes there, bool
                                   (hard) or float in [0, 1] (soft), or None

    Salience target t = max over classes; the positive weight is the batch's
    (n - sum t) / sum t. The type CE runs on points with t > 0 against the
    class weights normalised, each point weighted by t. Bool labels give
    exactly the hard-label losses. Proteins without labels are skipped; a
    loss is 0 (with grad) when no protein in the batch has any contacted
    point."""
    import torch
    import torch.nn.functional as F

    sal_terms, type_terms = [], []
    n_pos = n_all = 0.0
    for tl, sl, y in zip(type_logits, sal_logits, union_labels):
        if y is None or not len(sl):
            continue
        y = torch.as_tensor(np.asarray(y), device=sl.device).float()
        t = y.max(1).values
        hit = t > 0
        n_pos += float(t.sum())
        n_all += len(t)
        sal_terms.append((sl, t))
        if hit.any():
            T1 = tl.shape[1]
            if T1 - 1 < N_CLASSES:
                raise ValueError(f"{T1 - 1} site types; PLIP supervision needs >= {N_CLASSES}")
            yy, wt = y[hit], t[hit]
            q = torch.zeros(len(yy), T1, device=tl.device)
            q[:, :N_CLASSES] = yy / yy.sum(1, keepdim=True)
            ce = -(q * F.log_softmax(tl[hit], dim=1)).sum(1)
            type_terms.append((ce * wt).sum() / wt.sum())
    zero = (type_logits[0].sum() * 0) if len(type_logits) else torch.zeros(())
    if not sal_terms:
        return zero, zero, {"plip_pos": 0.0}
    pw = min(max(n_all - n_pos, 1.0) / max(n_pos, 1.0), max_pos_weight)
    sal = torch.stack([F.binary_cross_entropy_with_logits(
        sl, t, pos_weight=torch.tensor(pw, device=sl.device)) for sl, t in sal_terms]).mean()
    typ = torch.stack(type_terms).mean() if type_terms else zero
    return sal, typ, {"plip_pos": n_pos / max(n_all, 1)}


def site_points(out, b, labs_b, site_select):
    """The row's SITE region as (global indices, local indices) into protein
    b's points, or None: under --pocket_select site, slot 0 when it is the
    site and usable (the same test channel_viz.dump and anchor_block use)."""
    if not site_select or not labs_b.get("n_site", 1):
        return None
    if not (bool(out["anchor_valid"][b, 0]) and bool(out["anchor_is_pos"][b, 0])):
        return None
    sel = (out["batch"] == b).nonzero().squeeze(1)
    glob = out["anchor_regions"][b][0].to(sel.device)
    return glob, glob - sel[0]


# ---------------------------------------------------------------------------
def selftest():
    fails = []

    def check(name, ok, detail=""):
        print(f"  {'ok  ' if ok else 'FAIL'} {name}" + (f"   {detail}" if detail else ""))
        if not ok:
            fails.append(name)

    import tempfile

    from plip_labels import KINDS

    rng = np.random.default_rng(0)
    xyz = rng.normal(size=(2000, 3)).astype(np.float32) * 8
    don = xyz[:5]                                    # protein-side donor atoms
    hyd = xyz[100:104]
    rec_a = {"kind": np.full(5, KINDS.index("hbond_pdon"), np.int8),
             "prot_xyz": don, "lig_xyz": don + 2.9}
    rec_b = {"kind": np.full(4, KINDS.index("hydrophobic"), np.int8),
             "prot_xyz": hyd, "lig_xyz": hyd + 3.5}
    la = point_labels(xyz, [rec_a])
    lab = point_labels(xyz, [rec_a, rec_b])
    check("point_labels: donor class only for the donor record",
          la[:, 0].any() and not la[:, 1:].any())
    check("point_labels: the union adds the second record's hydrophobic class "
          "and keeps the first",
          (lab[:, 0] == la[:, 0]).all() and lab[:, 5].any() and lab.sum() > la.sum())
    with tempfile.TemporaryDirectory() as td:
        np.savez(Path(td) / "rowA.npz", **rec_a)
        np.savez(Path(td) / "rowB.npz", **rec_b)
        row, union = receptor_labels(xyz, "rowA", ["rowA", "rowB", "rowC"], td)
        check("receptor_labels: row = own record, union = own + siblings, "
              "a sibling without a file is skipped",
              (row == la).all() and (union == lab).all())
        none_row, none_u = receptor_labels(xyz, "rowX", ["rowX"], td)
        check("receptor_labels: no files -> (None, None)",
              none_row is None and none_u is None)
        _, soft_u = receptor_labels(xyz, "rowA", ["rowA", "rowB"], td, soft=1.5)
        check("receptor_labels: soft= reaches the labels (float, same support "
              "as the hard union where the ramp is 1)",
              soft_u.dtype == np.float32 and (soft_u[lab] > 0).all())

    # soft ramp: exact values at known distances from one donor atom
    probe = np.array([[0, 0, 2.0], [0, 0, 2.25], [0, 0, 3.0], [0, 0, 3.6],
                      [0, 0, 3.75], [0, 0, 5.0]], np.float32)
    one = {"kind": np.array([KINDS.index("hbond_pdon")], np.int8),
           "prot_xyz": np.zeros((1, 3), np.float32), "lig_xyz": np.ones((1, 3), np.float32)}
    w = soft_site_labels(probe, one, 1.5)[:, 0]
    check("soft ramp: 1 inside r - s/2, 0.5 at r, 0 beyond r + s/2 (r 3 A, s 1.5 A)",
          np.allclose(w, [1, 1, 0.5, 0.1, 0, 0], atol=1e-5), " ".join(f"{v:.2f}" for v in w))
    cen = dict(one, kind=np.array([KINDS.index("pistack")], np.int8))
    wc = soft_site_labels(np.array([[0, 0, 4.0]], np.float32), cen, 1.5)
    check("soft ramp: ring / charge centres ramp around 4 A, into their own class",
          abs(wc[0, 4] - 0.5) < 1e-5 and wc[0, [0, 1, 2, 3, 5]].sum() == 0)
    dense = rng.uniform(-6, 6, size=(200000, 3)).astype(np.float32)
    hard_n = point_labels(dense, [one])[:, 0].sum()
    soft_m = point_labels(dense, [one], soft=1.5)[:, 0].sum()
    check("soft ramp keeps roughly the hard label's mass (within 15%)",
          abs(soft_m / hard_n - 1) < 0.15, f"{soft_m / hard_n:.3f}")
    two = {"kind": np.array([KINDS.index("hbond_pdon")] * 2, np.int8),
           "prot_xyz": np.array([[0, 0, 0], [0, 0, 1.0]], np.float32),
           "lig_xyz": np.ones((2, 3), np.float32)}
    check("soft: max (not sum) over a class's interactions, so stays in [0, 1]",
          soft_site_labels(np.array([[0, 0, 0.5]], np.float32), two, 1.5)[0, 0] == 1.0)
    check("soft=0 is exactly the hard boolean labels",
          point_labels(xyz, [rec_a, rec_b], soft=0.0).dtype == bool
          and (point_labels(xyz, [rec_a, rec_b], soft=0.0) == lab).all())

    try:
        import torch
    except ImportError:
        print("  skip torch part (no torch here): run this where torch imports")
        print("\nPASS (numpy part)" if not fails else f"\nFAIL: {fails}")
        raise SystemExit(1 if fails else 0)

    import torch.nn.functional as F

    torch.manual_seed(0)
    T = 8
    # hand-computed losses on a tiny case
    tl = torch.zeros(4, T + 1, requires_grad=True)
    sl = torch.zeros(4, requires_grad=True)
    y = np.zeros((4, 6), bool)
    y[0, 0] = True                     # donor
    y[1, [1, 5]] = True                # acceptor + hydrophobic
    sal, typ, st = plip_losses([tl], [sl], [y])
    # uniform softmax over 9: CE = log 9 for every labelled point
    pw = 2 / 2
    exp_sal = float(F.binary_cross_entropy_with_logits(
        torch.zeros(4), torch.tensor([1., 1, 0, 0]), pos_weight=torch.tensor(pw)))
    check("losses at zero logits: type CE = log(T + 1), salience BCE as by hand",
          abs(float(typ) - np.log(T + 1)) < 1e-5 and abs(float(sal) - exp_sal) < 1e-6
          and abs(st["plip_pos"] - 0.5) < 1e-9)
    (sal + typ).backward()
    check("gradient reaches both heads' logits",
          float(tl.grad.abs().sum()) > 0 and float(sl.grad.abs().sum()) > 0)
    check("type loss puts gradient only on labelled points",
          float(tl.grad[2:].abs().sum()) == 0)
    tl2 = torch.randn(50, T + 1)
    sl2 = torch.randn(50)
    yb = rng.uniform(size=(50, 6)) < 0.15
    hb = plip_losses([tl2], [sl2], [yb])
    hf = plip_losses([tl2], [sl2], [yb.astype(np.float16)])
    check("hard labels as bool and as 0/1 floats give identical losses",
          abs(float(hb[0]) - float(hf[0])) < 1e-6 and abs(float(hb[1]) - float(hf[1])) < 1e-6)
    hit = yb.any(1)
    q = np.zeros((hit.sum(), T + 1))
    q[:, :6] = yb[hit] / yb[hit].sum(1, keepdims=True)
    ref_t = float(-(torch.as_tensor(q).float() * F.log_softmax(tl2[torch.as_tensor(hit)], 1)).sum(1).mean())
    check("hard type CE is v6's: mean over contacted points of CE against "
          "uniform-over-classes", abs(float(hb[1]) - ref_t) < 1e-5)
    # soft: a point at weight 0.25 counts a quarter as much as one at 1
    ys = np.zeros((2, 6), np.float32)
    ys[0, 0], ys[1, 1] = 1.0, 0.25
    tl3 = torch.randn(2, T + 1)
    ce = -F.log_softmax(tl3, 1)
    _, ts, sts = plip_losses([tl3], [torch.zeros(2)], [ys])
    check("soft type CE weights each point by its salience target",
          abs(float(ts) - float((ce[0, 0] + 0.25 * ce[1, 1]) / 1.25)) < 1e-5
          and abs(sts["plip_pos"] - 0.625) < 1e-6)
    z, z2, _ = plip_losses([torch.zeros(3, T + 1, requires_grad=True)],
                           [torch.zeros(3, requires_grad=True)], [None])
    check("no labels in the batch -> zero losses that still backprop",
          float(z) == 0 and z.requires_grad)
    try:
        plip_losses([torch.zeros(2, 5)], [torch.zeros(2)], [np.ones((2, 6), bool)])
        check("fewer than 6 types is refused", False)
    except ValueError:
        check("fewer than 6 types is refused", True)

    # planted: features carry the class; training the real PairPool heads on
    # the losses must make type t pick out class t and salience the contacts
    from pair_pool import PairPool
    E, n = 16, 3000
    cls = rng.integers(-1, 6, n)                     # -1 = not contacted
    feats = rng.normal(size=(n, E)).astype(np.float32)
    feats[np.arange(n), (cls + 1) % E] += 3.0        # a learnable class signature
    lab = np.zeros((n, 6), bool)
    lab[cls >= 0, cls[cls >= 0]] = True
    pool = PairPool(E, 0, n_types=T)
    opt = torch.optim.Adam(pool.parameters(), lr=3e-3)
    emb = torch.from_numpy(feats)
    for _ in range(300):
        tl_, sl_ = pool.heads(emb)
        s_, t_, _ = plip_losses([tl_], [sl_], [lab])
        opt.zero_grad()
        (s_ + t_).backward()
        opt.step()
    with torch.no_grad():
        a, s = pool.site_types(emb)
    a, s = a.numpy(), s.numpy()

    def auc(score, y):
        o = np.argsort(score)
        r = np.empty(len(score))
        r[o] = np.arange(1, len(score) + 1)
        n1 = y.sum()
        return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1))
    aucs = [auc(a[:, c], cls == c) for c in range(6)]
    check("planted: type t separates PLIP class t (AUROC > 0.9 for all 6)",
          min(aucs) > 0.9, " ".join(f"{v:.2f}" for v in aucs))
    check("planted: salience separates contacted points (AUROC > 0.9)",
          auc(s, cls >= 0) > 0.9, f"{auc(s, cls >= 0):.2f}")

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
    main()
