"""train.py's self-test: the whole pipeline on synthetic proteins, no GPU.

Extracted from train.py (refactor stage 2, docs/REFACTOR.md): it was 990 of
that module's 2,436 lines, and nothing imports it. The command is unchanged -
`python src/train.py selftest` delegates here - and the body below is that
code, moved verbatim.

A stub stands in for the three pieces needing KeOps and a GPU: the surface
sampler, AtomNet and the convolution. Everything else is the real code, which
is the point. Eleven groups:

    [1]  synthetic proteins, real region selection
    [2]  SurfaceStore: disk round-trip, RNG isolation, determinism, LRU
    [3]  labels, other sites, the fallback region
    [4]  pooling modes
    [5]  region label rules
    [6]  the training loop on a planted signal (MRR must climb)
    [7]  gradients and the checkpoint round-trip
    [8]  the memory bank, extra ligands, cross mask, lr schedule
    [9]  external pockets (region_source external / external_map)
    [10] the v3 stack: --pocket_select site + --pool_mode pairs
    [11] site decoys (--n_decoys) and the site metric

What it says nothing about: dMaSIF, KeOps, the convolution or surface
sampling. `chemo_test.sbatch`'s smoke pass is what exercises those.

Run it in the container (CPU torch is enough, no GPU, no KeOps):
    apptainer exec --cleanenv --env OMP_NUM_THREADS=8 $WORK/dmasif_sandbox \\
        python src/train.py selftest

One wrinkle worth knowing. Run as `python src/train.py selftest`, train.py is
`__main__`, so the `from train import ...` below imports train.py a SECOND
time under the name `train`. That is harmless here and was checked rather than
assumed: train.py's module level is only imports, defs, constants and the
`__main__` guard, so re-executing it has no side effects. The one consequence
is that `train.ProteinBank` is a different class object from
`__main__.ProteinBank`; nothing below compares types across that boundary, it
only calls the implementations. If train.py ever gains module-level state, this
stops being free.

Depends on: numpy, torch; train (the pieces under test), plus the modules the
            body has always imported function-locally.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# The pieces under test. train.py does NOT import this module at load time -
# its selftest() delegates here from inside the function - so there is no cycle.
from train import (ProteinBank, anchor_block, build_batch, cross_mask, lr_at,
                   nce_extended, pair_mask, site_decoys, site_loss, site_rate)


def selftest(a):
    """End-to-end check of everything but dMaSIF, on synthetic proteins.

    A stub stands in for the three pieces that need KeOps and a GPU - the
    surface sampler, AtomNet and the convolution. Everything else is the real
    code: anchors.select_regions, pool_regions, train_data's labels / other-site
    masks / fallback, the sampler, the InfoNCE, the metrics, project_anchors,
    and the checkpoint round-trip.

    A signal is planted: each protein's pocket points carry its ligand's latent
    vector, so a learnable projection and ligand encoder CAN recover the
    pairing and MRR must climb well above chance. If it does not, something in
    the loss, the masking, the sampler, the labels or the metrics is wrong -
    which is what cannot be checked on real data without a GPU and hours.
    """
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    from anchors import pool_regions, pooled_width, select_regions
    from ligand_encoder import LigandConfig, LigandEncoder
    from probe_retrieval import (draw_batch, group_index, info_nce,
                                 retrieval_metrics, split_by_group)
    from pair_pool import PairPool, pair_width
    from surface_encoder import (EncoderConfig, SurfaceEncoder, pocket_loss,
                                 select_and_pool)
    from test_encoder import to_batch
    from train_data import (ItemTable, LabelCache, SurfaceStore,
                            fallback_region, region_labels)

    import shutil

    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    tmp = Path(a.tmp)
    # Cleared, not reused. A surface left over from a previous run would make
    # the first `get` a disk HIT, and the sampling assertions below - which
    # only mean anything on a MISS - would pass without testing anything.
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True, exist_ok=True)
    N_PROT, N_LIG, LATENT, EMB = 120, 60, 6, 16
    fails = []

    def check(name, ok, detail=""):
        print(f"  {'ok  ' if ok else 'FAIL'} {name}" + (f"   {detail}" if detail else ""))
        if not ok:
            fails.append(name)

    # ---- synthetic proteins: a sphere of surface points, a pocket patch -----
    print("\n[1] synthetic proteins, real region selection")
    z = rng.normal(size=(N_LIG, LATENT)).astype(np.float32)
    W_fp = rng.normal(size=(LATENT, 2048))
    bits = ((1 / (1 + np.exp(-(z @ W_fp)))) > rng.random((N_LIG, 2048))).astype(np.uint8)

    # 2400 points on a 20 A sphere, which is ~1.6 A spacing - coarser than
    # dMaSIF's ~1 A but fine enough that a ball_R8 region holds ~99 points and
    # the counts below are not dominated by discretisation.
    def make_protein(seed):
        g = np.random.default_rng(seed)
        v = g.normal(size=(2400, 3))
        xyz = (v / np.linalg.norm(v, axis=1, keepdims=True)) * 20.0
        return xyz.astype(np.float32)

    rows, surfaces = [], {}
    for i in range(N_PROT):
        lig = i % N_LIG
        iid = f"p{i:03d}"
        xyz = make_protein(i)
        # The "ligand": a cloud of atoms just outside one patch of the sphere.
        # Sized to reproduce the real geometry, because the fixture is useless
        # on either side of it. The pocket is every surface point within 4 A of
        # the cloud, and a ball_R8 region holds ~99 of the 2400 points, so:
        #   16 atoms at scale 2.0 -> pocket 64 points (2.7% of the surface,
        #     against 2.8% on real receptors) and ball precision 0.65 (against
        #     0.63 for ball_R8 in the region sweep). Every protein can form a
        #     genuine positive.
        #   a tighter cloud (12 atoms at scale 1.2) -> pocket 22 points and
        #     precision 0.44, with the worst proteins at 0.24. Those never
        #     clear pos_prec 0.3, so they train on the fallback forever and
        #     every `site` assertion goes vacuous.
        #   a looser one (24 at scale 3.0) -> precision 0.84, which clears
        #     everything trivially and would hide a broken label rule.
        g = np.random.default_rng(1000 + i)
        dirn = g.normal(size=3)
        dirn /= np.linalg.norm(dirn)
        lig_xyz = (dirn * 21.0)[None] + g.normal(scale=2.0, size=(16, 3))
        sdf = tmp / f"{iid}.sdf"
        sdf.write_text(_fake_sdf(lig_xyz))
        surfaces[iid] = xyz
        rows.append({"system_id": iid, "input_id": iid, "receptor_key": iid,
                     "split": "train", "input_path": "-", "input_chains": "",
                     "sdf_paths": str(sdf), "fp_row": lig,
                     "group": f"g{lig % 15}", "smiles": ""})
    table = ItemTable(rows)
    check("ItemTable indexes", len(table) == N_PROT
          and len(table.siblings) == N_PROT, table.summary())

    # ---- SurfaceStore: disk round-trip, RNG isolation, determinism, LRU ----
    # Only the structure parse is faked. _sample itself - the seeding, the
    # save/restore of the global RNG, the write-then-rename - is the real one.
    print("\n[2] SurfaceStore")

    class FakeStore(SurfaceStore):
        def _load_atoms(self, row):
            # atoms are the sphere, shrunk; the fake sampler re-expands them,
            # so the surface still comes out of the sampler, not out of here
            xyz = surfaces[row["input_id"]] * 0.9
            n = len(xyz)
            return {"atom_xyz": xyz,
                    "atom_type": np.zeros(n, dtype=np.int64),
                    "atom_groups": np.zeros((n, 6), dtype=np.float32)}

    class FakeSampler:
        """What SurfaceEncoder.sample_surface looks like from the store: a
        STOCHASTIC function of the atoms, which is the whole reason the store
        has to seed it and put the RNG back."""

        def sample_surface(self, atom_xyz, atomtypes, batch_atoms):
            xyz = atom_xyz / 0.9 + 0.05 * torch.randn_like(atom_xyz)
            return xyz, F.normalize(xyz, dim=1), batch_atoms

    sampler = FakeSampler()
    store = FakeStore(tmp / "surfaces", ram=8)
    e1 = store.get(sampler, rows[0])
    e2 = store.get(sampler, rows[0])
    check("RAM hit returns the same entry", e1 is e2, store.report())
    check("disk file written, no .tmp left behind",
          (tmp / "surfaces" / "p000.npz").exists()
          and not list((tmp / "surfaces").glob("*.tmp")))

    store2 = FakeStore(tmp / "surfaces", ram=8)
    e3 = store2.get(sampler, rows[0])
    check("disk round-trip is exact",
          bool(torch.equal(e1["xyz"], e3["xyz"])) and store2.stats["disk"] == 1)

    # Same receptor, a store that has never seen it, and a different amount of
    # randomness consumed first: the surface must be the same anyway, or a
    # cached surface and the labels computed on it would disagree between jobs.
    torch.manual_seed(999)
    torch.randn(17)
    store3 = FakeStore(tmp / "surfaces3", ram=8)
    e4 = store3.get(sampler, rows[0])
    check("a surface depends only on the receptor's identity",
          bool(torch.allclose(e1["xyz"], e4["xyz"], atol=1e-6)),
          f"max difference {float((e1['xyz'] - e4['xyz']).abs().max()):.2e}")

    torch.manual_seed(7)
    want = torch.randn(4)
    torch.manual_seed(7)
    store3.get(sampler, rows[1])                      # a miss: seeds and samples
    got_after = torch.randn(4)
    check("sampling a surface leaves the global RNG untouched",
          bool(torch.equal(want, got_after)),
          "otherwise a cache miss would silently redirect the whole run's "
          "random stream")

    for r in rows[:12]:
        store.get(sampler, r)
    check("LRU bounded to ram", len(store._mem) == 8, f"{len(store._mem)} resident")
    check("region cache travels with the surface",
          "region_cache" in e1 and isinstance(e1["region_cache"], dict))
    # From here on the loop needs every protein resident, or each batch would
    # evict the region caches it is about to reuse. The disk cache is already
    # warm, so this costs a read, not a resampling.
    store = FakeStore(tmp / "surfaces", ram=N_PROT + 4)

    # ---- labels, other sites, fallback -------------------------------------
    print("\n[3] labels, other sites, fallback region")
    labels = LabelCache(table, cutoff=4.0, ram=32)
    lb = labels.get(0, e1)
    n_pk = int(lb["pocket"].sum())
    frac = n_pk / len(e1["xyz"])
    check("pocket is labelled and is a realistic fraction of the surface",
          20 <= n_pk and 0.01 <= frac <= 0.06,
          f"{n_pk} of {len(e1['xyz'])} points = {frac:.1%} "
          "(real receptors: 2.8%)")
    check("no other site for a single-ligand receptor", int(lb["other"].sum()) == 0)

    # two ligands on one receptor: the sibling's site must be ignorable
    shared = [dict(rows[0]), dict(rows[1])]
    shared[1].update(input_id="p000", system_id="p000b")
    t2 = ItemTable(shared)
    l2 = LabelCache(t2, cutoff=4.0, ram=8)
    r0 = l2.get(0, e1)
    check("sibling ligand becomes an other-site mask",
          int(r0["other"].sum()) > 0 and not bool((r0["other"] & r0["pocket"]).any()),
          f"{int(r0['other'].sum())} other-site points, disjoint from the pocket")

    fb = fallback_region(e1["xyz"], lb["pocket"], 8.0)
    centre = e1["xyz"][lb["pocket"]].mean(0)
    R = torch.linalg.qr(torch.randn(3, 3))[0]
    fb_rot = fallback_region(e1["xyz"] @ R.T + 5.0, lb["pocket"], 8.0)
    check("fallback is a ball at the pocket and rigid-motion invariant",
          len(fb) >= 20 and set(fb.tolist()) == set(fb_rot.tolist()),
          f"{len(fb)} points, centre {centre.norm():.1f} A from origin")

    # ---- pooling modes -----------------------------------------------------
    print("\n[4] pooling modes")
    emb = torch.randn(len(e1["xyz"]), EMB)
    prob = torch.rand(len(e1["xyz"]))
    regs = [fb]
    for mode, nsh in (("mean", 3), ("mean_std", 3), ("shells", 3), ("shells", 2)):
        p = pool_regions(emb, prob, regs, mode=mode, xyz=e1["xyz"], n_shells=nsh)
        want = pooled_width(EMB, mode, nsh)
        ok = p.shape == (1, want) and bool(torch.isfinite(p).all())
        detail = f"{mode}(n_shells={nsh}) -> {tuple(p.shape)}"
        if mode == "shells":
            # the shells must be a rigid-motion invariant of the region
            pr = pool_regions(emb, prob, regs, mode=mode,
                              xyz=e1["xyz"] @ R.T + 5.0, n_shells=nsh)
            d = float((p - pr).abs().max())
            ok = ok and d < 1e-4
            detail += f", rotation/translation delta {d:.1e}"
        check(f"pool_regions {mode}", ok, detail)
    perm = torch.randperm(len(regs[0]))
    check("shell pooling is permutation invariant",
          float((pool_regions(emb, prob, [regs[0]], mode="shells", xyz=e1["xyz"])
                 - pool_regions(emb, prob, [regs[0][perm]], mode="shells",
                                xyz=e1["xyz"])).abs().max()) < 1e-5)

    # ---- region labels -----------------------------------------------------
    print("\n[5] region label rules")
    pocket = lb["pocket"]
    idx = pocket.nonzero().squeeze(1)
    far = (~pocket).nonzero().squeeze(1)[:len(idx)]
    lab, cov, pre = region_labels([idx, idx[: len(idx) // 3], far], pocket, None)
    check("exact pocket is positive, a third of it is ignore, elsewhere negative",
          lab.tolist() == [1, -1, 0], f"labels {lab.tolist()}, cover {cov.tolist()}")
    lab_o, _, _ = region_labels([far], pocket, other=torch.zeros_like(pocket)
                                .index_fill_(0, far, True))
    check("a region over another ligand's site is ignore, not negative",
          lab_o.tolist() == [-1])

    # ---- the stub encoder: planted signal ----------------------------------
    print("\n[6] training loop on a planted signal")

    class StubEncoder(nn.Module):
        """SurfaceEncoder's interface with dMaSIF replaced.

        point_emb carries the protein's latent inside its pocket and noise
        outside, so the pooled pocket anchor is recoverable; the pocket logit
        is a learnable readout of it, so pocket BCE has something to train. The
        regions, the pooling, the normalisation and the projection are the real
        implementations.
        """

        def __init__(self, cfg):
            super().__init__()
            self.cfg = cfg
            self.pool_norm = (nn.BatchNorm1d(cfg.pooled_dims)
                              if cfg.anchor_norm == "batch" else nn.Identity())
            self.project = nn.Sequential(
                nn.Linear(cfg.pooled_dims, 64), nn.LeakyReLU(0.2),
                nn.Linear(64, cfg.out_dim))
            self.pocket_head = nn.Linear(EMB, 1)
            self.mix = nn.Linear(EMB, EMB)
            # the real PairPool when asked for, as SurfaceEncoder builds it
            self.pair_pool = (PairPool(EMB, 0, n_types=cfg.pair_types,
                                       n_bins=cfg.pair_bins, r_min=cfg.pair_rmin,
                                       r_max=cfg.pair_rmax, exclude=cfg.pair_exclude)
                              if cfg.pool_mode == "pairs" else None)

        # the real tail, so the normalisation and projection under test are
        # the ones training uses
        project_anchors = SurfaceEncoder.project_anchors

        def sample_surface(self, *args, **kw):      # never reached: store is faked
            raise AssertionError("the stub does not sample surfaces")

        def forward(self, P, forced=None, generator=None, project=True):
            c = self.cfg
            xyz, batch = P["xyz"], P["batch"]
            emb = self.mix(P["planted"])
            logit = self.pocket_head(emb).squeeze(-1)
            prob = torch.sigmoid(logit)
            if c.region_source != "predicted" or c.pool_mode == "pairs":
                # the external sources and pair pooling go through the REAL
                # select_and_pool (groups [9], [10]); every earlier group keeps
                # the loop below, unchanged
                sp = select_and_pool(c, P, xyz, batch, None, prob, emb, forced=forced,
                                     generator=generator, pair_pool=self.pair_pool)
                return {"xyz": xyz, "normals": P["normals"], "batch": batch,
                        "point_emb": emb, "pocket_logit": logit,
                        "anchor_vec": (self.project_anchors(sp["vec"], sp["valid"])
                                       if project else None),
                        "anchor_pooled": sp["vec"], "anchor_valid": sp["valid"],
                        "anchor_is_pos": sp["is_pos"], "anchor_center": sp["center"],
                        "anchor_size": sp["size"], "anchor_regions": sp["regions"],
                        "pool_weight": sp["pool_weight"]}
            K = c.k_pos + c.k_neg
            B = int(batch.max()) + 1
            vec = emb.new_zeros(B, K, c.pooled_dims)
            valid = torch.zeros(B, K, dtype=torch.bool)
            is_pos = torch.zeros(B, K, dtype=torch.bool)
            size = torch.zeros(B, K, dtype=torch.long)
            centre = xyz.new_zeros(B, K, 3)
            all_regions = []
            for b in range(B):
                sel = (batch == b).nonzero().squeeze(1)
                regions, pos = select_regions(
                    xyz[sel], prob[sel], c.k_pos, c.k_neg,
                    threshold=c.region_threshold, top_fraction=c.region_top_fraction,
                    link_radius=c.link_radius, smooth_radius=c.smooth_radius,
                    min_points=c.min_region_points, neg_quantile=c.neg_quantile,
                    neg_exclusion=c.neg_exclusion, generator=generator,
                    method=c.region_method, ball_radius=c.region_ball_radius,
                    ball_shift=c.region_ball_shift, ball_rank=c.region_ball_rank,
                    cache=None if P.get("region_cache") is None else P["region_cache"][b])
                n = len(regions)
                if n:
                    vec[b, :n] = pool_regions(emb[sel], prob[sel], regions,
                                              mode=c.pool_mode, xyz=xyz[sel],
                                              n_shells=c.n_shells)
                    centre[b, :n] = pool_regions(xyz[sel], prob[sel].detach(), regions)
                    size[b, :n] = torch.tensor([len(r) for r in regions])
                    is_pos[b, :n] = pos
                    valid[b, :n] = True
                all_regions.append([sel[r] for r in regions])
            return {"xyz": xyz, "normals": P["normals"], "batch": batch,
                    "point_emb": emb, "pocket_logit": logit,
                    "anchor_vec": self.project_anchors(vec, valid) if project else None,
                    "anchor_pooled": vec, "anchor_valid": valid,
                    "anchor_is_pos": is_pos, "anchor_center": centre,
                    "anchor_size": size, "anchor_regions": all_regions}

    cfg = EncoderConfig(emb_dims=EMB, k_pos=4, k_neg=4, region_method="ball",
                        region_ball_radius=8.0, region_ball_shift=2,
                        region_top_fraction=0.05, min_region_points=20,
                        pool_mode=a.pool_mode, n_shells=3,
                        anchor_norm=a.anchor_norm, out_dim=32)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = StubEncoder(cfg)
            self.ligand = LigandEncoder(LigandConfig(arm="fp_linear", out_dim=32))
            self.logit_scale = nn.Parameter(torch.log(torch.tensor(1 / 0.07)))

        def scale(self):
            return self.logit_scale.exp().clamp(max=100.0)

        def scores(self, anchors, usable, lig):
            sim = torch.einsum("bkd,ld->bkl", anchors, lig)
            return sim.masked_fill(~usable[..., None], -1e4).max(1).values * self.scale()

    model = Model()
    ligands = _FakeLigands(bits)

    def batch_for(idx):
        """build_batch's job, with the planted embedding attached."""
        items, labs, planted = [], [], []
        for i in idx:
            entry = store.get(sampler, table.rows[i])
            lab = labels.get(int(i), entry)
            items.append(entry)
            labs.append(lab)
            e = torch.randn(len(entry["xyz"]), EMB) * 0.3
            zi = torch.from_numpy(z[table.fp_row[i]])
            # Two separate things, because they train different parts. Channel
            # LATENT is a CONSTANT pocket marker, so a linear pocket head can
            # find the pocket and the selector gets real regions; channels
            # 0..LATENT-1 carry the protein's identity, which is what the
            # projection and the ligand encoder have to pair up. The identity
            # alone would not do: z changes sign across proteins, so no single
            # linear readout detects every pocket, the head would stay blind
            # and the fallback would fire forever.
            e[lab["pocket"], LATENT] += 4.0
            e[lab["pocket"], :LATENT] += zi * 3.0
            planted.append(e)
        P = to_batch(items, "cpu")
        P["planted"] = torch.cat(planted)
        return P, labs

    args = argparse.Namespace(
        pos_cover=0.5, pos_prec=0.3, ignore_overlap=0.1, fallback=True,
        tanimoto=0.8, eval_batch=16, cutoff=4.0)
    # A fixture guard, not a code test: if a ball at the right place cannot
    # clear pos_prec, no region is ever positive, the loop trains on the
    # fallback forever, and the site / fallback-rate assertions below pass
    # while measuring nothing. Real receptors sit at 0.63 here.
    best_prec = int(lb["pocket"].sum()) / len(fb)
    check("a correctly placed ball clears pos_prec with room to spare",
          best_prec >= args.pos_prec + 0.1,
          f"pocket {int(lb['pocket'].sum())} points, a ball holds {len(fb)}, so "
          f"precision at best {best_prec:.2f} against pos_prec {args.pos_prec} "
          f"(real receptors: 0.63)")
    tr, va = split_by_group(table.group, 0.3, 0)
    check("group-disjoint split", not (set(table.group[tr]) & set(table.group[va])),
          f"{len(tr)} train / {len(va)} val, "
          f"{len(set(table.group[tr]))} / {len(set(table.group[va]))} groups")
    members, counts = group_index(table.group[tr])
    rng_s = np.random.default_rng(0)
    seen, dups = set(), 0
    for _ in range(200):
        b = draw_batch(rng_s, members, counts, 6, 1.0)
        g = [table.group[tr[j]] for j in b]
        seen |= set(g)
        dups += len(g) != len(set(g))
    check("sampler draws one row per target group, and reaches every group",
          dups == 0 and seen == set(table.group[tr]),
          f"{dups} batches with a repeated group, {len(seen)} of "
          f"{len(set(table.group[tr]))} groups seen over 200 batches")

    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    gen = torch.Generator().manual_seed(0)
    mrrs, fbs = [], []

    def val():
        model.eval()
        with torch.no_grad():
            P, labs = batch_for(va)
            blk = anchor_block(model, P, labs, args)
            keep = blk["usable"].any(1)
            lig = model.ligand(ligands.collate(table.fp_row[va[keep.numpy()]]))
            s = model.scores(blk["anchors"][keep], blk["usable"][keep], lig)
            m = pair_mask(table, va[keep.numpy()], ligands, "cpu", 0.8)
            met = retrieval_metrics(s.T, m)
            met["fb"] = float(blk["fallback"].float().mean())
        model.train()
        return met

    m0 = val()
    print(f"    step 0: MRR {m0['mrr']:.3f} (chance {m0['chance']:.3f}), "
          f"fallback {m0['fb']:.2f}")
    rng2 = np.random.default_rng(1)
    for step in range(1, a.steps + 1):
        idx = tr[draw_batch(rng2, members, counts, a.batch, 1.0)]
        P, labs = batch_for(idx)
        blk = anchor_block(model, P, labs, args, generator=gen)
        keep = blk["usable"].any(1)
        if int(keep.sum()) < 2:
            continue
        idx_k = idx[keep.numpy()]
        lig = model.ligand(ligands.collate(table.fp_row[idx_k]))
        s = model.scores(blk["anchors"][keep], blk["usable"][keep], lig)
        loss = info_nce(s, pair_mask(table, idx_k, ligands, "cpu", 0.8))
        pl = torch.cat([lb["pocket"] for lb in labs]).float()
        loss = loss + 0.5 * pocket_loss(blk["out"]["pocket_logit"], pl, 20.0)
        loss = loss + 0.5 * site_loss(
            torch.einsum("bkd,bd->bk", blk["anchors"][keep], lig) * model.scale(),
            blk["label"][keep])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        if step % max(a.steps // 6, 1) == 0:
            m = val()
            mrrs.append(m["mrr"])
            fbs.append(m["fb"])
            print(f"    step {step:4d}: loss {loss.detach().item():7.4f}  MRR {m['mrr']:.3f}  "
                  f"top1 {m['top1']:.3f}  fallback {m['fb']:.2f}", flush=True)

    best_mrr = max(mrrs) if mrrs else 0.0
    floor = max(a.min_mrr, 2 * m0["chance"])
    check("MRR climbs well above chance on the planted signal",
          best_mrr >= floor, f"best {best_mrr:.3f} vs floor {floor:.3f} "
          f"(chance {m0['chance']:.3f}, untrained {m0['mrr']:.3f})")
    check("the fallback rate falls as the head learns",
          fbs[-1] <= m0["fb"] + 0.05,
          f"{m0['fb']:.2f} untrained -> {fbs[-1]:.2f} trained")

    # ---- gradients reach everything ---------------------------------------
    print("\n[7] gradients and checkpoint round-trip")
    dead = sorted(n for n, p in model.named_parameters()
                  if p.grad is None or float(p.grad.abs().sum()) == 0)
    check("every parameter has a gradient", not dead, f"dead: {dead or 'none'}")

    torch.save({"state_dict": model.state_dict(), "step": 7}, tmp / "ck.pt")
    model2 = Model()
    model2.load_state_dict(torch.load(tmp / "ck.pt",
                                      weights_only=False)["state_dict"])

    def score_block(m, P, labs, idx):
        blk = anchor_block(m, P, labs, args)
        lig = m.ligand(ligands.collate(table.fp_row[idx]))
        return m.scores(blk["anchors"], blk["usable"], lig)

    model.eval()
    model2.eval()
    with torch.no_grad():
        idx = va[:8]
        P, labs = batch_for(idx)
        s_a = score_block(model, P, labs, idx)
        s_b = score_block(model2, P, labs, idx)
    delta = float((s_a - s_b).abs().max())
    check("a reloaded checkpoint scores identically", delta < 1e-5,
          f"max delta {delta:.2e}; this is what makes --resume meaningful, and "
          "it covers the BatchNorm running statistics as much as the weights")

    # ---- v2 pieces: extended InfoNCE, the bank, the masks, the schedule ----
    print("\n[8] memory bank, extra ligands, cross mask, lr schedule")
    g = torch.Generator().manual_seed(1)
    S = torch.randn(6, 6, generator=g)
    M = torch.zeros(6, 6, dtype=torch.bool)
    M[0, 3] = M[3, 0] = True
    d0 = float((nce_extended(S, M) - info_nce(S, M)).abs())
    check("nce_extended with no extras IS info_nce", d0 < 1e-6, f"delta {d0:.2e}")
    # An extra candidate that is masked must change nothing; an unmasked one
    # that outscores the answer must raise the loss.
    xb, xl = torch.full((1, 6), 50.0), torch.full((6, 1), 50.0)
    on, off = torch.ones(1, 6, dtype=torch.bool), torch.zeros(1, 6, dtype=torch.bool)
    d1 = float((nce_extended(S, M, xb, on, xl, on.T) - info_nce(S, M)).abs())
    check("fully masked extras leave the loss unchanged", d1 < 1e-6, f"delta {d1:.2e}")
    check("an unmasked strong extra raises the loss",
          float(nce_extended(S, M, xb, off, xl, off.T)) > float(info_nce(S, M)) + 1.0)

    rq, rc = va[:4], np.concatenate([va[:4], tr[:4]])
    cm = cross_mask(table, rq, rc, ligands, "cpu", args.tanimoto)
    same = (table.group_id[rq][:, None] == table.group_id[rc][None, :])
    check("cross_mask masks every same-group pair, the 'diagonal' included",
          bool(cm[torch.as_tensor(same)].all()) and bool(cm[:, :4].diagonal().all()),
          "a row is its own false negative when it reappears as a bank entry")
    eq = cross_mask(table, va[:4], va[4:8], ligands, "cpu", args.tanimoto)
    want = torch.as_tensor(table.group_id[va[:4]][:, None] == table.group_id[va[4:8]][None, :])
    check("cross_mask between equal-sized disjoint sets clears no diagonal",
          bool(eq[want].all()))

    c8 = model.encoder.cfg
    bk = ProteinBank(5, c8.k_pos + c8.k_neg + 1, c8.pooled_dims, "cpu")
    check("an empty bank contributes nothing", bk.anchors(model) is None)
    P8, labs8 = batch_for(tr[:3])
    with torch.no_grad():
        blk8 = anchor_block(model, P8, labs8, args)
    k8 = blk8["usable"].any(1)
    for _ in range(5):           # 5 x 1-3 rows into 5 slots: it must fill and wrap
        bk.push(blk8["pooled"][k8], blk8["valid"][k8], blk8["usable"][k8],
                tr[:3][k8.numpy()])
    got8 = bk.anchors(model)
    ref = model.encoder.project_anchors(blk8["pooled"][k8], blk8["valid"][k8]).detach()
    j = int(np.nonzero(bk.rows == tr[:3][k8.numpy()][0])[0][0])
    kk = blk8["pooled"].shape[1]
    check("the bank wraps at its size and re-projects with the current weights",
          bk.n == 5 and got8[0].shape[0] == 5 and not got8[0].requires_grad
          and float((got8[0][j, :kk] - ref[0]).abs().max()) < 1e-5)

    a8 = argparse.Namespace(lr=1.0, warmup=10, lr_schedule="cosine", steps=110,
                            min_lr_frac=0.1)
    lrs = [lr_at(a8, t) for t in (0, 9, 10, 60, 110, 500)]
    check("warmup rises, cosine falls to the floor and stays there",
          abs(lrs[0] - 0.1) < 1e-9 and abs(lrs[1] - 1.0) < 1e-9
          and abs(lrs[2] - 1.0) < 1e-9 and abs(lrs[3] - 0.55) < 1e-9
          and abs(lrs[4] - 0.1) < 1e-9 and abs(lrs[5] - 0.1) < 1e-9,
          f"{[round(x, 3) for x in lrs]}")
    a8c = argparse.Namespace(lr=0.3, warmup=0, lr_schedule="const")
    check("const with no warmup is the old constant rate",
          lr_at(a8c, 0) == lr_at(a8c, 10 ** 6) == 0.3)

    # ---- external pockets: the real select_and_pool, no KeOps -------------
    print("\n[9] external pockets (region_source external / external_map)")
    import dataclasses

    from external_pockets import PocketSource, save_pockets

    # Two fake fpocket pockets per protein: a decoy on the far side, ranked
    # FIRST by score, and the true one over the labelled pocket, ranked first
    # by volume. One r=8 sphere each, its centre 25 A out, so it covers a cap
    # of ~45 points against the ~64-point labelled pocket.
    pdir = tmp / "pockets"
    pdir.mkdir(exist_ok=True)
    ids9 = list(range(4))
    for i in ids9:
        e = store.get(sampler, rows[i])
        d = e["xyz"][labels.get(i, e)["pocket"]].mean(0).numpy()
        d = d / np.linalg.norm(d)
        save_pockets(pdir / f"{rows[i]['input_id']}.npz", {
            "sphere_xyz": np.stack([-25 * d, 25 * d]).astype(np.float32),
            "sphere_r": np.full(2, 8.0, np.float32),
            "sphere_pocket": np.asarray([0, 1], np.int32),
            "pocket_score": np.asarray([0.9, 0.2], np.float32),
            "pocket_drug": np.asarray([0.1, 0.2], np.float32),
            "pocket_volume": np.asarray([100.0, 900.0], np.float32)})
    st9 = FakeStore(tmp / "surfaces", ram=8, pockets=PocketSource(
        pdir, k=cfg.k_pos, min_points=cfg.min_region_points))
    items9 = [st9.get(sampler, rows[i]) for i in ids9]
    P9 = to_batch(items9, "cpu")
    labs9 = [labels.get(i, it) for i, it in zip(ids9, items9)]
    check("SurfaceStore + PocketSource -> to_batch carries ext_map / ext_regions",
          "ext_map" in P9 and len(P9["ext_regions"]) == 4
          and all(len(r) == 2 for r in P9["ext_regions"]), st9.report())

    N9 = len(P9["xyz"])
    emb9 = torch.randn(N9, EMB, requires_grad=True)
    logit9 = torch.randn(N9, requires_grad=True)
    c9 = dataclasses.replace(cfg, region_source="external", pool_mode="mean")
    sp9 = select_and_pool(c9, P9, P9["xyz"], P9["batch"], None,
                          torch.sigmoid(logit9), emb9,
                          generator=torch.Generator().manual_seed(0))
    ok_regions = ok_label = ok_mean = True
    for b in range(4):
        sel = (P9["batch"] == b).nonzero().squeeze(1)
        npos = int(sp9["is_pos"][b].sum())
        want = [sel[r] for r in P9["ext_regions"][b]]
        got = sp9["regions"][b][:npos]
        ok_regions &= npos == 2 and all(torch.equal(g, w) for g, w in zip(got, want))
        lab, _, _ = region_labels([g - int(sel[0]) for g in got], labs9[b]["pocket"],
                                  labs9[b]["other"])
        ok_label &= lab.tolist()[1] == 1 and lab.tolist()[0] != 1
        ok_mean &= bool(torch.allclose(sp9["vec"][b, 0], emb9[got[0]].mean(0), atol=1e-5))
    check("external: the pocket regions ARE the external pockets, in rank order",
          ok_regions)
    check("external: the decoy is not a positive, the true pocket is", ok_label)
    check("external: pooled as a plain mean", ok_mean)
    sp9["vec"].sum().backward()
    check("external: no gradient reaches the pocket head through the anchors",
          logit9.grad is None and emb9.grad is not None
          and bool(sp9["pool_weight"].eq(1).all()))

    st9v = FakeStore(tmp / "surfaces", ram=8, pockets=PocketSource(
        pdir, k=cfg.k_pos, rank="volume", min_points=cfg.min_region_points))
    P9v = to_batch([st9v.get(sampler, rows[i]) for i in ids9], "cpu")
    sp9v = select_and_pool(c9, P9v, P9v["xyz"], P9v["batch"], None,
                           torch.sigmoid(logit9.detach()), emb9.detach(),
                           generator=torch.Generator().manual_seed(0))
    first = [sp9v["regions"][b][0] - int((P9v["batch"] == b).nonzero()[0])
             for b in range(4)]
    check("rank volume puts the true pocket first",
          all(region_labels([f], labs9[b]["pocket"], labs9[b]["other"])[0].item() == 1
              for b, f in enumerate(first)))

    c9m = dataclasses.replace(cfg, region_source="external_map", region_threshold=0.5)
    sp9m = select_and_pool(c9m, P9, P9["xyz"], P9["batch"], None,
                           torch.sigmoid(logit9.detach()), emb9.detach(),
                           generator=torch.Generator().manual_seed(0))
    hits, ok_in = 0, True
    for b in range(4):
        sel = (P9["batch"] == b).nonzero().squeeze(1)
        regs = [r - int(sel[0]) for r, pos in zip(sp9m["regions"][b], sp9m["is_pos"][b])
                if pos]
        lab, _, _ = region_labels(regs, labs9[b]["pocket"], labs9[b]["other"])
        hits += int((lab == 1).any())
        ok_in &= all(float((P9["ext_map"][sel][r] > 0).float().mean()) > 0.3
                     for r in regs)
    check("external_map: the selector on the external map finds the pocket",
          hits == 4 and ok_in, f"{hits}/4 proteins with a positive region")

    P9n = to_batch([store.get(sampler, rows[i]) for i in ids9], "cpu")
    try:
        select_and_pool(c9, P9n, P9n["xyz"], P9n["batch"], None,
                        torch.sigmoid(logit9.detach()), emb9.detach())
        raised = False
    except ValueError:
        raised = True
    check("an external source without external pockets refuses loudly", raised)

    # The refactor out of forward(): the predicted source must select exactly
    # what select_regions selects on the head's map.
    cp = dataclasses.replace(cfg, pool_mode="mean")
    prob_p = torch.sigmoid(logit9.detach())
    spp = select_and_pool(cp, P9n, P9n["xyz"], P9n["batch"], None, prob_p,
                          emb9.detach(), generator=torch.Generator().manual_seed(0))
    gen_p, same = torch.Generator().manual_seed(0), True
    for b in range(4):
        sel = (P9n["batch"] == b).nonzero().squeeze(1)
        ref, _ = select_regions(
            P9n["xyz"][sel], prob_p[sel], cp.k_pos, cp.k_neg,
            threshold=cp.region_threshold, top_fraction=cp.region_top_fraction,
            link_radius=cp.link_radius, smooth_radius=cp.smooth_radius,
            min_points=cp.min_region_points, neg_quantile=cp.neg_quantile,
            neg_exclusion=cp.neg_exclusion, generator=gen_p,
            method=cp.region_method, ball_radius=cp.region_ball_radius,
            ball_shift=cp.region_ball_shift, ball_rank=cp.region_ball_rank,
            cache=P9n["region_cache"][b])
        same &= len(ref) == len(spp["regions"][b]) and all(
            torch.equal(sel[r], g) for r, g in zip(ref, spp["regions"][b]))
    check("predicted source: select_and_pool == select_regions on the head's map",
          same and torch.equal(spp["pool_weight"], prob_p))

    # ---- v3: the site pocket, pair-pooled, through the real loop pieces ----
    print("\n[10] v3 stack: --pocket_select site + --pool_mode pairs")
    from types import SimpleNamespace

    cfg10 = dataclasses.replace(cfg, region_source="external", pool_mode="pairs",
                                pair_feats=False, anchor_norm="none")
    model10 = SimpleNamespace(encoder=StubEncoder(cfg10))
    check("pairs width: config, PairPool and projection agree",
          cfg10.pooled_dims == pair_width(EMB) == model10.encoder.pair_pool.out_dims
          == model10.encoder.project[0].in_features, f"{cfg10.pooled_dims} dims")
    # rows 0-3 have a pocket file (group [9]); row 4 has none, so fpocket
    # "found nothing" there and the step must fall back. The fake pocket is one
    # sphere 25 A out against a ligand ~21 A out, so the site cutoff is 6 A
    # here rather than the real 3 A between alpha-sphere centres and atoms.
    ids10 = ids9 + [4]
    table10 = ItemTable([rows[i] for i in ids10])
    labels10 = LabelCache(table10, cutoff=4.0, ram=16)
    st10 = FakeStore(tmp / "surfaces", ram=16, pockets=PocketSource(
        pdir, k=cfg10.k_pos, min_points=cfg10.min_region_points, select="site",
        site_cutoff=6.0))
    P10, labs10 = build_batch(model10, st10, labels10, table10, np.arange(5), "cpu")
    P10["planted"] = torch.randn(len(P10["xyz"]), EMB)
    check("build_batch: exactly the true pocket per row, nothing for the row "
          "without pockets",
          [len(r) for r in P10["ext_regions"]] == [1, 1, 1, 1, 0]
          and all(torch.equal(torch.sort(P10["ext_regions"][b][0]).values,
                              torch.sort(P9["ext_regions"][b][1]).values)
                  for b in range(4)), st10.report())
    args10 = argparse.Namespace(**vars(args))
    args10.pocket_select = "site"
    model10.encoder.train()
    blk10 = anchor_block(model10, P10, labs10, args10,
                         generator=torch.Generator().manual_seed(0))
    usable10 = blk10["usable"].sum(1).tolist()
    check("anchor_block: one usable anchor per protein, labelled positive; the "
          "pocketless row falls back to the ball",
          usable10 == [1, 1, 1, 1, 1] and blk10["fallback"].tolist() == [False] * 4 + [True]
          and all(int(blk10["label"][b][blk10["usable"][b]][0]) == 1 for b in range(5)),
          f"usable {usable10}, fallback {blk10['fallback'].tolist()}")
    blk10["anchors"][blk10["usable"]].sum().backward()
    pp = model10.encoder.pair_pool
    check("the loss reaches the pair pool's heads and the embedding",
          all(p.grad is not None and float(p.grad.abs().sum()) > 0
              for p in pp.parameters())
          and model10.encoder.mix.weight.grad is not None)

    # ---- site decoys: the receptor's other pockets as within-protein negatives
    print("\n[11] site decoys (--n_decoys) and the site metric")
    srt = lambda t: torch.sort(t).values
    # Rows 0-3 have two pockets each (group [9]): the far-side decoy (rank 0
    # by score) and the true one, which is the site. Every row is its own
    # receptor, so there are no sibling sites to keep clear of.
    P11, labs11 = build_batch(model10, st10, labels10, table10, np.arange(5), "cpu",
                              n_decoys=3)
    P11["planted"] = torch.randn(len(P11["xyz"]), EMB)
    check("build_batch: the site, then the far-side pocket as the one decoy; the "
          "true pocket is never its own decoy",
          [len(r) for r in P11["ext_regions"]] == [2, 2, 2, 2, 0]
          and all(torch.equal(srt(P11["ext_regions"][b][0]), srt(P10["ext_regions"][b][0]))
                  and torch.equal(srt(P11["ext_regions"][b][1]),
                                  srt(P9["ext_regions"][b][0]))
                  for b in range(4))
          and [lb["n_site"] for lb in labs11] == [1, 1, 1, 1, 0]
          and [lb["n_decoy"] for lb in labs11] == [1, 1, 1, 1, 0])
    e11 = st10.get(model10.encoder, table10.rows[0])
    check("build_batch leaves the shared label record alone",
          "n_site" not in labels10.get(0, e11))
    cands = list(e11["ext_regions"])                       # [far side, true]
    blank = {"pocket": torch.zeros(len(e11["xyz"]), dtype=torch.bool), "other": None}
    by_site = site_decoys(cands, [cands[1]], blank, 5)
    check("site_decoys: a pocket overlapping the site region is dropped even with "
          "no label to catch it; n caps the count",
          len(by_site) == 1 and torch.equal(by_site[0], torch.as_tensor(cands[0]).long())
          and len(site_decoys(cands, [], blank, 5)) == 2
          and len(site_decoys(cands, [], blank, 1)) == 1
          and site_decoys(cands, [cands[1]], blank, 0) == [])

    blk11 = anchor_block(model10, P11, labs11, args10,
                         generator=torch.Generator().manual_seed(0))
    check("anchor_block: the decoy is a negative and never usable; still one "
          "usable positive per protein",
          blk11["usable"].sum(1).tolist() == [1] * 5
          and blk11["decoy"].sum(1).tolist() == [1, 1, 1, 1, 0]
          and all(int(blk11["label"][b, 0]) == 1 and int(blk11["label"][b, 1]) == 0
                  and bool(blk11["decoy"][b, 1]) for b in range(4))
          and blk11["fallback"].tolist() == [False] * 4 + [True],
          f"usable {blk11['usable'].sum(1).tolist()}, "
          f"decoys {blk11['decoy'].sum(1).tolist()}")

    model10.encoder.zero_grad(set_to_none=True)
    lig11 = F.normalize(torch.randn(5, cfg10.out_dim), dim=1)
    per11 = torch.einsum("bkd,bd->bk", blk11["anchors"], lig11) * 10.0
    sl11 = site_loss(per11, blk11["label"])
    sl11_none = site_loss(per11, torch.where(blk11["decoy"], -1, blk11["label"]))
    sl11.backward()
    check("lam_site's term trains the site against its decoys, and is 0 without them",
          bool(torch.isfinite(sl11)) and float(sl11) > 0 and float(sl11_none) == 0.0
          and any(p.grad is not None and float(p.grad.abs().sum()) > 0
                  for p in pp.parameters()),
          f"site_loss {float(sl11):.3f}")

    # fpocket finds nothing at the site (fractions above 1 cannot be met):
    # slot 0 is then a DECOY, and labelling it the site would train the
    # ligand onto the wrong pocket.
    st11 = FakeStore(tmp / "surfaces", ram=16, pockets=PocketSource(
        pdir, k=cfg10.k_pos, min_points=cfg10.min_region_points, select="site",
        site_frac=1.01, site_lig_frac=1.01))
    P11m, labs11m = build_batch(model10, st11, labels10, table10, np.arange(5),
                                "cpu", n_decoys=3)
    P11m["planted"] = torch.randn(len(P11m["xyz"]), EMB)
    blk11m = anchor_block(model10, P11m, labs11m, args10,
                          generator=torch.Generator().manual_seed(0))
    check("site missed: slot 0 is a decoy (negative, unusable), the true pocket "
          "is not a decoy, the fallback ball is the positive",
          [len(r) for r in P11m["ext_regions"]] == [1, 1, 1, 1, 0]
          and [lb["n_site"] for lb in labs11m] == [0] * 5
          and blk11m["fallback"].tolist() == [True] * 5
          and blk11m["usable"].sum(1).tolist() == [1] * 5
          and all(int(blk11m["label"][b, 0]) == 0 and bool(blk11m["decoy"][b, 0])
                  for b in range(4)),
          f"fallback {blk11m['fallback'].tolist()}")

    sim_r = torch.tensor([[0.9, 0.1, 0.5], [0.1, 0.9, 0.0], [0.2, 0.8, 0.9]])
    lab_r = torch.tensor([[1, 0, -1], [1, 0, -1], [1, -1, -1]])
    use_r = torch.tensor([[True, False, False]] * 3)
    r_site, c_site = site_rate(sim_r, lab_r, use_r, True)
    r_sel, c_sel = site_rate(sim_r, lab_r, use_r, False)
    check("site_rate: under site select, the site beating its decoys (rows with "
          "decoys only, NaN with none); otherwise the old argmax over usable",
          r_site == 0.5 and abs(c_site - 2 / 3) < 1e-6 and r_sel == 1.0
          and c_sel == 1.0
          and np.isnan(site_rate(sim_r[2:], lab_r[2:], use_r[2:], True)[0]),
          f"site {r_site} over {c_site:.2f}; selector {r_sel}")

    print("\nSELFTEST " + ("PASSED" if not fails else f"FAILED: {fails}"))
    print("Nothing here involves dMaSIF, KeOps or a GPU, so it says nothing about\n"
          "surface sampling or the convolution; chemo_test.sbatch's smoke pass is\n"
          "what exercises those.")
    if fails:
        raise SystemExit(1)


def _fake_sdf(xyz):
    """A minimal V2000 SDF for read_sdf_heavy, so the selftest exercises the
    real ligand path rather than a shortcut around it."""
    head = f"fake\n  selftest\n\n{len(xyz):3d}  0  0  0  0  0  0  0  0  0999 V2000\n"
    body = "".join(f"{x:10.4f}{y:10.4f}{zz:10.4f} C   0  0  0  0  0\n"
                   for x, y, zz in xyz)
    return head + body + "M  END\n$$$$\n"


class _FakeLigands:
    """LigandData's surface, for the selftest's planted fingerprints."""

    def __init__(self, fp):
        import torch

        self.fp = fp
        self._t = torch.from_numpy(fp.astype(np.float32))
        self.failed = np.zeros(len(fp), dtype=bool)

    def __len__(self):
        return len(self.fp)

    def collate(self, rows, device="cpu"):
        rows = np.asarray(rows, dtype=np.int64)
        return {"fp": self._t[rows].to(device), "n_mol": len(rows)}
