"""What does the model actually retrieve? Top-k ligands per receptor, inspected.

Why this exists. Every number we have is a scalar over a val set: mrr, top1,
top10, site, cosdiff. They say HOW WELL the ranking does, never WHAT it ranks.
A model at mrr 0.18 could be
  - finding the right chemotype and missing the exact molecule (nearly right),
  - matching ligand SIZE to pocket size and nothing else (the "size baseline"
    question docs/STATUS.md has had open since the probe), or
  - returning the same few popular ligands for every receptor (a ligand prior,
    the classic contrastive-retrieval collapse, which an aggregate mrr hides
    completely because it is mostly driven by the rows that work).
These are different diagnoses with different fixes, and the top-k lists tell
them apart. Run on TRAIN and VAL receptors together, they also localise the
memorisation `v3c+` showed as a scalar (val mrr flat while train nce fell):
if train rows retrieve their own ligand while val rows fall back to the hub
ligands, the model has memorised pairs and has a popularity prior for
everything else.

Two stages, the house pattern (dump once on the GPU, sweep offline on CPU):

    dump    container, GPU. A checkpoint + items.tsv -> one .npz holding the
            receptor anchors and every ligand's vector, for a val slice and a
            group-balanced train slice. Nothing is scored here, so one dump
            serves every question `report` can ask.
    report  numpy only (CPU venv). Scores, ranks and the diagnostics below.
    selftest numpy only, no dump: planted scores whose right answer is known.

What `report` prints, per split, and what each line is for:

  retrieval       mrr / top1 / top10 over the full ligand set, masked like
                  training (same target group, same ligand, Tanimoto >= 0.8
                  leave the ranking). Cross-check against the eval log: it
                  should agree with --pocket_select's own number, modulo the
                  candidate set being all ligands rather than the val rows'.
  chemotype rank  the best rank among candidates with Tanimoto >= t to the
                  TRUE ligand. If molecule-level mrr is 0.18 but
                  chemotype mrr at t=0.5 is 0.45, the model finds the right
                  class and misses the molecule - which is a success, and
                  argues for reporting (and maybe training) at chemotype
                  level rather than chasing exact-molecule rank.
  hubs            how concentrated the top-k slots are: the share held by the
                  20 most-retrieved ligands, the normalised entropy, and how
                  many distinct ligands cover half the slots. A low entropy
                  is a ligand prior, not retrieval. The train/val Jaccard of
                  the hub sets says whether both splits lean on the SAME
                  ligands (one global prior) or different ones.
  coherence       mean pairwise Tanimoto inside each top-k, against a
                  random-k baseline, and the top-k's mean Tanimoto to the
                  true ligand. High internal coherence with low similarity to
                  truth = a consistent but wrong hypothesis, i.e. something
                  systematic to find. Both low = noise.
  properties      correlation between the TRUE ligand's descriptors and the
                  top-1's, for size / polarity / shape columns, plus a
                  SIZE-ONLY BASELINE: rank candidates by |heavy atoms -
                  heavy atoms of truth| and report its mrr. If the model's
                  mrr is near the size baseline's, the signal is size
                  matching. This is docs/STATUS.md "Next steps" item 6.

Files written: <out>.rows.tsv (one line per receptor: the true ligand, its
rank, the chemotype ranks, the top-k ids) and <out>.hubs.tsv (the most
retrieved ligands with SMILES and how many receptors retrieved each). Both are
for eyeballing - the aggregate lines above are what to read first.

Deliberately NOT here: anything that changes train.py. This module imports
from it and never writes to it, so it is safe to run while a training job
holds the same files.

Example:
    # container, GPU (~2 min for 400 + 400 receptors)
    python src/retrieval_report.py dump \\
        --ckpt out/runs/v3c_site_pairs_bn/best.pt \\
        --items out/runs/v1/items.tsv --lig_feats out/runs/v1/ligand_feats.npz \\
        --surface_cache data/surfaces --pocket_select topk --pocket_rank drug \\
        --n_train 400 --out out/reports/v3c5000_blind.npz
    # CPU venv
    python src/retrieval_report.py report --dump out/reports/v3c5000_blind.npz \\
        --items out/runs/v1/items.tsv --lig_feats out/runs/v1/ligand_feats.npz \\
        --out out/reports/v3c5000_blind

Dump both --pocket_select site and topk from the same checkpoint: the first is
what training optimised, the second what a screen would see, and the two
reports side by side say how much of the retrieval survives losing the site.

Depends on: numpy; train (dump only: load_model_from_ckpt, load_rows,
            build_batch, anchor_block, encode_ligands, pocket_source),
            train_data, ligand_encoder. report/selftest need numpy alone.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# descriptor columns worth correlating, by name in ligand_features.DESCRIPTORS
PROPS = ("HeavyAtomCount", "MolWt", "MolLogP", "TPSA", "NumHAcceptors",
         "NumHDonors", "NumRotatableBonds", "NumAromaticRings")


# ---------------------------------------------------------------------------
# numpy helpers
# ---------------------------------------------------------------------------
def tanimoto(a, B):
    """(F,) or (A, F) 0/1 fingerprints against (B, F) -> (A, B) or (B,)."""
    a = np.atleast_2d(np.asarray(a, np.float32))
    B = np.asarray(B, np.float32)
    inter = a @ B.T
    union = a.sum(1)[:, None] + B.sum(1)[None, :] - inter
    out = np.divide(inter, np.maximum(union, 1e-9))
    return out[0] if out.shape[0] == 1 else out


def rank_of(scores, target, mask):
    """1-based rank of `target` among unmasked candidates. mask True = drop.

    The target itself is never dropped, whatever the mask says: it is the
    answer, and masking it would make the row unscoreable rather than hard.
    """
    m = mask.copy()
    m[target] = False
    better = (scores > scores[target]) & ~m
    return int(better.sum()) + 1


def spearman(x, y):
    """Rank correlation without scipy: Pearson on the rank transforms."""
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return float("nan")
    rx = np.argsort(np.argsort(x[ok])).astype(np.float64)
    ry = np.argsort(np.argsort(y[ok])).astype(np.float64)
    rx -= rx.mean()
    ry -= ry.mean()
    d = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / d) if d > 0 else float("nan")


def entropy_norm(counts):
    """Shannon entropy of a count vector / log(n_nonzero_max) -> [0, 1].

    Normalised by the entropy of spreading the SAME number of slots as evenly
    as possible, so 1.0 means "no ligand retrieved more than any other" at
    this many slots and 0 means every slot went to one ligand.
    """
    c = np.asarray(counts, np.float64)
    c = c[c > 0]
    if len(c) < 2:
        return 0.0
    p = c / c.sum()
    return float(-(p * np.log(p)).sum() / np.log(len(c)))


# ---------------------------------------------------------------------------
# dump: anchors and ligand vectors, nothing scored
# ---------------------------------------------------------------------------
def dump_split(model, store, labels, table, rows, a, device):
    """-> dict of per-row arrays for these receptor rows (anchors padded)."""
    import torch

    from train import anchor_block, build_batch

    chunks, use, fb, n_use, size0, keep = [], [], [], [], [], []
    gen = torch.Generator().manual_seed(a.seed)
    model.eval()
    with torch.no_grad():
        for s in range(0, len(rows), a.eval_batch):
            idx = rows[s:s + a.eval_batch]
            P, labs = build_batch(model, store, labels, table, idx, device,
                                  getattr(a, "n_decoys", 0) or 0)
            blk = anchor_block(model, P, labs, a, generator=gen)
            chunks.append(blk["anchors"].cpu())
            use.append(blk["usable"].cpu())
            fb.append(blk["fallback"].cpu().numpy())
            n_use.append(blk["usable"].sum(1).cpu().numpy())
            # slot 0 is the site under --pocket_select site, else the
            # best-ranked external pocket; a covariate, not a metric
            sz = blk["out"].get("anchor_size")
            size0.append(np.zeros(len(idx), np.int64) if sz is None
                         else sz[:, 0].cpu().numpy())
            keep.append(np.asarray(idx))
            print(f"    {s + len(idx)}/{len(rows)}", end="\r", flush=True)
    model.train()
    # Chunks differ in slot count when one needed the fallback anchor.
    K = max(t.shape[1] for t in chunks)
    pad = lambda t, f: (t if t.shape[1] == K else torch.cat(
        [t, torch.full((t.shape[0], K - t.shape[1], *t.shape[2:]), f,
                       dtype=t.dtype)], dim=1))
    return {"anchors": torch.cat([pad(t, 0) for t in chunks]).numpy(),
            "usable": torch.cat([pad(t, False) for t in use]).numpy(),
            "rows": np.concatenate(keep), "fallback": np.concatenate(fb),
            "n_usable": np.concatenate(n_use), "site_size": np.concatenate(size0)}


def dump(a):
    import torch

    from train import (POCKET_DEFAULTS, encode_ligands, load_model_from_ckpt,
                       load_rows, pocket_source, subsample_val,
                       use_neighbor_device)
    from train_data import LabelCache, SurfaceStore

    use_neighbor_device(a, a.device)
    model, ckpt, saved = load_model_from_ckpt(a.ckpt, a.device, overrides=a)
    table, ligands = load_rows(a)
    for k, v in POCKET_DEFAULTS.items():
        if getattr(a, k, None) is None:
            setattr(a, k, getattr(saved, k, None) or v)
    store = SurfaceStore(a.surface_cache, ram=a.surface_ram, device=a.device,
                         keep_hydrogens=saved.keep_hydrogens,
                         pockets=pocket_source(a, model.encoder.cfg),
                         atom_feats=getattr(model.encoder.cfg, "atom_feats", False))
    labels = LabelCache(table, cutoff=a.cutoff, ram=a.label_ram)
    tr = np.asarray(ckpt["train_rows"])
    va = np.asarray(ckpt["val_rows"])
    tr, va = tr[tr < len(table)], va[va < len(table)]
    # The scaler training used, so the ligand branch sees its own inputs.
    ligands.set_scaler(table.fp_row[tr])
    if a.n_val and len(va) > a.n_val:
        va = subsample_val(va, table.group, a.n_val, saved.split_seed, True)
    if a.n_train and len(tr) > a.n_train:
        tr = subsample_val(tr, table.group, a.n_train, saved.split_seed, True)
    print(f"{a.ckpt}: step {ckpt['step']}; {len(tr)} train / {len(va)} val "
          f"receptors, pocket_select {a.pocket_select}")

    out = {}
    for tag, rows in (("train", tr), ("val", va)):
        print(f"  {tag}: {len(rows)} receptors")
        for k, v in dump_split(model, store, labels, table, rows, a, a.device).items():
            out[f"{tag}_{k}"] = v
    # every distinct ligand, in fp_row order, with the SAME encoder
    vecs = []
    model.eval()
    with torch.no_grad():
        for s in range(0, len(ligands), a.lig_batch):
            fp_rows = np.arange(s, min(s + a.lig_batch, len(ligands)))
            vecs.append(encode_ligands(model, ligands, fp_rows, a.device).cpu().numpy())
            print(f"    ligands {fp_rows[-1] + 1}/{len(ligands)}", end="\r", flush=True)
    model.train()
    out["lig_vec"] = np.concatenate(vecs)
    out["step"] = np.asarray(ckpt["step"])
    out["pocket_select"] = np.asarray(a.pocket_select)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, **out)
    print(f"\nwrote {a.out}: {out['lig_vec'].shape[0]} ligand vectors, "
          f"anchors {out['val_anchors'].shape}")
    print("report it with: python src/retrieval_report.py report --dump "
          f"{a.out} --items {a.items} --lig_feats {a.lig_feats}")


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def score_rows(anchors, usable, lig_vec):
    """(R, K, D) x (L, D) -> (R, L) max over the usable anchors."""
    sim = np.einsum("rkd,ld->rkl", anchors, lig_vec)
    sim = np.where(usable[:, :, None], sim, -1e4)
    return sim.max(1)


def group_ligand_sets(table_group, fp_row):
    """group -> set of fp_rows seen in it, for the same-group mask."""
    out = {}
    for g, f in zip(table_group, fp_row):
        out.setdefault(g, set()).add(int(f))
    return out


def analyse(tag, d, table, ligands, desc_names, a):
    """One split -> (summary dict, per-row records, top-k count vector)."""
    rows = d["rows"]
    scores = score_rows(d["anchors"], d["usable"], d["lig_vec"])
    truth = table.fp_row[rows]
    groups = table.group[rows]
    gl = group_ligand_sets(table.group, table.fp_row)
    L = scores.shape[1]
    k = a.topk

    recs, counts = [], np.zeros(L, np.int64)
    ranks, chem = [], {t: [] for t in a.tanimoto_levels}
    coh, coh_truth, size_ranks = [], [], []
    prop_true, prop_top1 = [], []
    hv = ligands.n_heavy.astype(np.float64)

    for i, (row, tr_lig) in enumerate(zip(rows, truth)):
        s = scores[i]
        tsim = tanimoto(ligands.fp[tr_lig], ligands.fp)
        # training's mask: same target group, same ligand, Tanimoto >= 0.8
        mask = np.zeros(L, bool)
        mask[list(gl.get(groups[i], ()))] = True
        mask |= tsim >= a.mask_tanimoto
        ranks.append(rank_of(s, tr_lig, mask))
        for t in a.tanimoto_levels:
            near = np.union1d(np.nonzero((tsim >= t) & ~mask)[0], [tr_lig])
            # the best-scoring member of the chemotype has its best rank, so
            # one rank_of on the argmax is the best rank over the class
            chem[t].append(rank_of(s, int(near[np.argmax(s[near])]), mask))
        # top-k over the unmasked candidates, the list a screen would return.
        # `mask` holds the row's own ligand (it shares its own target group),
        # so unmask the answer first or the list that every statistic below
        # is computed on would systematically exclude the correct one.
        shown = mask.copy()
        shown[tr_lig] = False
        order = np.argsort(-np.where(shown, -np.inf, s))[:k]
        counts[order] += 1
        tk = ligands.fp[order]
        pair = tanimoto(tk, tk)
        iu = np.triu_indices(len(order), 1)
        coh.append(float(pair[iu].mean()) if len(iu[0]) else float("nan"))
        coh_truth.append(float(tsim[order].mean()))
        # size-only baseline: rank by closeness in heavy-atom count
        size_ranks.append(rank_of(-np.abs(hv - hv[tr_lig]), tr_lig, mask))
        prop_true.append(ligands.desc_raw[tr_lig])
        prop_top1.append(ligands.desc_raw[order[0]])
        recs.append({"row": int(row), "group": groups[i],
                     "true_fp": int(tr_lig), "rank": ranks[-1],
                     "size_rank": size_ranks[-1],
                     "fallback": int(d["fallback"][i]),
                     "site_size": int(d["site_size"][i]),
                     "top_coh": coh[-1], "top_tanimoto_true": coh_truth[-1],
                     "chem": {t: chem[t][-1] for t in a.tanimoto_levels},
                     "topk": order.tolist()})

    ranks = np.asarray(ranks, np.float64)
    rng = np.random.default_rng(0)
    rand = [tanimoto(ligands.fp[r], ligands.fp[r])[np.triu_indices(k, 1)].mean()
            for r in rng.integers(0, L, size=(64, k))]
    summ = {
        "n": len(rows), "mrr": float((1 / ranks).mean()),
        "top1": float((ranks == 1).mean()), "top10": float((ranks <= 10).mean()),
        "median_rank": float(np.median(ranks)), "cands": L,
        "size_mrr": float((1 / np.asarray(size_ranks, np.float64)).mean()),
        "hub_share20": float(np.sort(counts)[::-1][:20].sum() / max(counts.sum(), 1)),
        "hub_entropy": entropy_norm(counts),
        "n_cover_half": int((np.cumsum(np.sort(counts)[::-1])
                             <= counts.sum() / 2).sum() + 1),
        "distinct_retrieved": int((counts > 0).sum()),
        "coh_topk": float(np.nanmean(coh)), "coh_random": float(np.mean(rand)),
        "tanimoto_to_true": float(np.mean(coh_truth)),
        "fallback_rate": float(d["fallback"].mean()),
    }
    for t in a.tanimoto_levels:
        summ[f"chem_mrr_{t}"] = float((1 / np.asarray(chem[t], np.float64)).mean())
    pt, p1 = np.asarray(prop_true), np.asarray(prop_top1)
    for name in PROPS:
        if name in desc_names:
            j = desc_names.index(name)
            summ[f"rho_{name}"] = spearman(pt[:, j], p1[:, j])
    return summ, recs, counts


def load_ligand_arrays(path):
    """The ligand arrays `report` needs, WITHOUT importing ligand_encoder.

    LigandData lives in ligand_encoder, which imports torch at module level,
    and the whole point of the report stage is that it runs in the CPU venv on
    numpy alone. Everything used here is a plain array in the same .npz.
    """
    from types import SimpleNamespace

    with np.load(path, allow_pickle=False) as f:
        d = {k: f[k] for k in f.files}
    fp = d["fp"]
    return SimpleNamespace(
        fp=fp, desc_raw=d["desc"].astype(np.float32),
        desc_names=[str(x) for x in d["desc_names"]],
        failed=d.get("failed", np.zeros(len(fp), bool)),
        smiles=d.get("smiles", np.array([""] * len(fp))),
        n_heavy=d.get("n_heavy", np.diff(d["atom_ptr"]).astype(np.int32)),
    )


def report(a):
    from train_data import ItemTable, read_items

    dump_npz = {k: v for k, v in np.load(a.dump, allow_pickle=False).items()}
    table = ItemTable(read_items(a.items))
    ligands = load_ligand_arrays(a.lig_feats)
    desc_names = ligands.desc_names
    bad = ligands.failed[table.fp_row]
    if bad.any():
        table = ItemTable([r for r, b in zip(table.rows, bad) if not b])
    print(f"{len(table)} rows after dropping unfeaturised ligands")
    print(f"{a.dump}: step {int(dump_npz['step'])}, pocket_select "
          f"{str(dump_npz['pocket_select'])}, {len(ligands)} candidate ligands\n")

    out, counts = {}, {}
    for tag in ("train", "val"):
        d = {k[len(tag) + 1:]: v for k, v in dump_npz.items() if k.startswith(tag + "_")}
        if not d:
            continue
        d["lig_vec"] = dump_npz["lig_vec"]
        out[tag], recs, counts[tag] = analyse(tag, d, table, ligands, desc_names, a)
        out[tag]["_recs"] = recs

    keys = [k for k in next(iter(out.values())) if not k.startswith("_")]
    w = max(len(k) for k in keys) + 2
    print(f"{'':{w}}" + "".join(f"{t:>12s}" for t in out))
    for k in keys:
        print(f"{k:{w}s}" + "".join(
            f"{out[t][k]:>12.4f}" if isinstance(out[t][k], float)
            else f"{out[t][k]:>12d}" for t in out))
    if len(counts) == 2:
        hub = {t: set(np.argsort(c)[::-1][:20].tolist()) for t, c in counts.items()}
        j = len(hub["train"] & hub["val"]) / len(hub["train"] | hub["val"])
        print(f"\ntrain/val hub-set Jaccard (top 20 retrieved): {j:.3f}  "
              "- high means ONE global ligand prior, not per-receptor retrieval")

    if a.out:
        base = Path(a.out)
        base.parent.mkdir(parents=True, exist_ok=True)
        cols = ["split", "row", "group", "true_fp", "rank", "size_rank",
                "fallback", "site_size", "top_coh", "top_tanimoto_true"]
        lines = ["\t".join(cols + [f"chem_{t}" for t in a.tanimoto_levels]
                           + ["topk"])]
        for tag, o in out.items():
            for r in o["_recs"]:
                lines.append("\t".join(
                    [tag] + [str(r[c]) for c in cols[1:]]
                    + [str(r["chem"][t]) for t in a.tanimoto_levels]
                    + [",".join(map(str, r["topk"]))]))
        (base.with_suffix(".rows.tsv")).write_text("\n".join(lines) + "\n")
        hl = ["\t".join(["split", "fp_row", "n_receptors", "share", "smiles"])]
        for tag, c in counts.items():
            for i in np.argsort(c)[::-1][:a.n_hubs]:
                if not c[i]:
                    break
                hl.append(f"{tag}\t{i}\t{c[i]}\t{c[i] / max(c.sum(), 1):.4f}\t"
                          f"{str(ligands.smiles[i])}")
        (base.with_suffix(".hubs.tsv")).write_text("\n".join(hl) + "\n")
        print(f"\nper-receptor: {base.with_suffix('.rows.tsv')}"
              f"\nmost retrieved: {base.with_suffix('.hubs.tsv')}")


# ---------------------------------------------------------------------------
def selftest(a):
    """Planted scores whose right answer is known. numpy only, no dump."""
    fails = []

    def check(name, ok, detail=""):
        print(f"  {'ok  ' if ok else 'FAIL'} {name}" + (f"   {detail}" if detail else ""))
        if not ok:
            fails.append(name)

    print("[1] tanimoto, ranks, entropy")
    fp = np.zeros((4, 8), np.uint8)
    fp[0, :4] = 1
    fp[1, :4] = 1            # identical to 0
    fp[2, 2:6] = 1           # 2 of 6 shared with 0
    fp[3, 4:] = 1            # disjoint from 0
    t = tanimoto(fp[0], fp)
    check("tanimoto: identical 1, disjoint 0, partial 2/6",
          np.allclose(t, [1.0, 1.0, 2 / 6, 0.0]), f"{np.round(t, 3)}")
    check("tanimoto matrix is symmetric with a unit diagonal",
          np.allclose(tanimoto(fp, fp), tanimoto(fp, fp).T)
          and np.allclose(np.diag(tanimoto(fp, fp)), 1.0))

    s = np.array([0.9, 0.5, 0.7, 0.1])
    none = np.zeros(4, bool)
    check("rank_of: best is 1, worst is n, ties count as better-or-equal",
          rank_of(s, 0, none) == 1 and rank_of(s, 3, none) == 4
          and rank_of(s, 2, none) == 2)
    m = np.zeros(4, bool)
    m[0] = m[2] = True
    check("rank_of: masked candidates leave the ranking", rank_of(s, 1, m) == 1)
    mt = np.ones(4, bool)
    check("rank_of: the target is never masked away", rank_of(s, 1, mt) == 1)

    check("entropy: one hub 0, flat 1",
          entropy_norm([10, 0, 0]) == 0.0
          and abs(entropy_norm([5, 5, 5, 5]) - 1.0) < 1e-12)
    check("spearman: monotone 1, reversed -1",
          abs(spearman([1, 2, 3, 4], [10, 20, 30, 40]) - 1) < 1e-9
          and abs(spearman([1, 2, 3, 4], [4, 3, 2, 1]) + 1) < 1e-9)

    print("\n[2] score_rows: max over usable anchors only")
    anc = np.zeros((2, 3, 4), np.float32)
    lig = np.zeros((5, 4), np.float32)
    anc[0, 0, 0] = anc[0, 1, 1] = anc[0, 2, 2] = 1.0
    anc[1, 0, 3] = 1.0
    anc[1, 1, 0] = 1.0       # matches ligand 0, but row 1 marks it unusable
    lig[0, 0] = lig[1, 1] = lig[2, 2] = lig[3, 3] = lig[4, 0] = 1.0
    use = np.array([[True, False, True], [True, False, False]])
    sc = score_rows(anc, use, lig)
    allsc = score_rows(anc, np.ones_like(use), lig)
    # anchor 1 is the only one matching ligand 1, and it is NOT usable, so
    # row 0 falls to the next usable anchor (0) rather than to -1e4
    check("an unusable anchor cannot contribute its ligand's score",
          sc[0, 1] == 0 and allsc[0, 1] == 1 and sc[0, 0] == 1 and sc[0, 2] == 1,
          f"row 0 {np.round(sc[0], 2)} against all-usable {np.round(allsc[0], 2)}")
    check("a row scores only through its usable anchors",
          sc[1, 3] == 1 and sc[1, 0] == 0 and allsc[1, 0] == 1)
    check("a row with NO usable anchor scores -1e4 everywhere",
          (score_rows(anc, np.zeros_like(use), lig) < -1e3).all())

    print("\n[3] a planted retrieval: receptor i prefers ligand i")
    L, R, D = 40, 12, 16
    rng = np.random.default_rng(0)
    V = rng.normal(size=(L, D)).astype(np.float32)
    V /= np.linalg.norm(V, axis=1, keepdims=True)
    d = {"anchors": V[:R][:, None, :].copy(), "usable": np.ones((R, 1), bool),
         "rows": np.arange(R), "fallback": np.zeros(R, bool),
         "n_usable": np.ones(R, np.int64), "site_size": np.full(R, 100),
         "lig_vec": V}

    class T:
        pass
    table, lg = T(), T()
    table.fp_row = np.arange(R)
    table.group = np.asarray([f"g{i}" for i in range(R)])
    lg.fp = (rng.random((L, 64)) < 0.2).astype(np.uint8)
    lg.n_heavy = np.full(L, 20, np.int32)
    lg.desc_raw = np.zeros((L, 1), np.float32)
    lg.smiles = np.asarray(["C"] * L)
    args = argparse.Namespace(topk=5, tanimoto_levels=[0.5], mask_tanimoto=0.8)
    summ, recs, counts = analyse("t", d, table, lg, ["HeavyAtomCount"], args)
    check("a perfectly planted ranking gives mrr 1 and top1 1",
          summ["mrr"] == 1.0 and summ["top1"] == 1.0, f"mrr {summ['mrr']}")
    check("each receptor's own ligand takes a top-k slot",
          all(r["topk"][0] == r["true_fp"] for r in recs))
    check("top-k slots are spread, so the hub share is low",
          summ["hub_entropy"] > 0.9 and summ["distinct_retrieved"] >= R,
          f"entropy {summ['hub_entropy']:.3f}, "
          f"{summ['distinct_retrieved']} distinct")

    print("\n[4] the hub failure: every receptor ranks ligand 0 first")
    d2 = dict(d)
    d2["anchors"] = np.repeat(V[0][None, None, :], R, axis=0)
    s2, r2, c2 = analyse("t", d2, table, lg, ["HeavyAtomCount"], args)
    check("one ligand taking every slot reads as a collapsed hub share",
          s2["hub_share20"] == 1.0 and s2["distinct_retrieved"] <= args.topk
          and s2["mrr"] < summ["mrr"],
          f"share {s2['hub_share20']:.2f}, mrr {s2['mrr']:.3f} "
          f"against {summ['mrr']:.3f}")
    check("identical anchors make every receptor's top-k the same list",
          len({tuple(r["topk"]) for r in r2}) == 1)

    print("\nSELFTEST " + ("PASSED" if not fails else f"FAILED: {fails}"))
    if fails:
        raise SystemExit(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("dump", help="container, GPU: anchors + ligand vectors")
    from train import add_data_args, add_region_args
    add_data_args(p)
    add_region_args(p, defaults=False)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True, help=".npz to write")
    p.add_argument("--n_val", type=int, default=400,
                   help="val receptors, group-balanced (0 = the checkpoint's own)")
    p.add_argument("--n_train", type=int, default=400,
                   help="train receptors, group-balanced (0 = skip the split)")
    p.add_argument("--lig_batch", type=int, default=512)

    p = sub.add_parser("report", help="CPU: the diagnostics")
    p.add_argument("--dump", required=True)
    p.add_argument("--items", required=True)
    p.add_argument("--lig_feats", required=True)
    p.add_argument("--out", default="", help="base path for the two TSVs")
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--mask_tanimoto", type=float, default=0.8,
                   help="training's false-negative cut; candidates this close "
                        "to the true ligand leave the ranking")
    p.add_argument("--tanimoto_levels", type=float, nargs="+",
                   default=[0.4, 0.5, 0.7],
                   help="chemotype thresholds for the best-of-class rank")
    p.add_argument("--n_hubs", type=int, default=30)

    p = sub.add_parser("selftest", help="numpy only, no dump")

    a = ap.parse_args()
    {"dump": dump, "report": report, "selftest": selftest}[a.cmd](a)


if __name__ == "__main__":
    main()
