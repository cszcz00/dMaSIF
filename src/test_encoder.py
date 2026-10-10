"""Staged tests of the protein-side encoder (surface_encoder.py) on real receptors.

    prepare   CPU venv. Pick receptors from a built dataset, write items.tsv:
              structure path, chains, ligand SDF paths, system_id, split.
    smoke     GPU container. Run the encoder on each item once: does it load,
              how many atoms / points / pocket points, is the ligand in the
              receptor's frame, how many regions, time and memory, and does
              backward reach every part.
    overfit   GPU container. Train on the pocket labels alone for a few hundred
              steps. Passing means the stack can learn: training AUROC near 1
              and predicted regions recovering the labelled pockets. Optional
              held-out items show whether anything generalises.

Pocket labels: surface points within --cutoff A of a ligand heavy atom, the
same rule as labels.py. Surfaces are sampled once per item and reused, so the
labels stay attached to the points they were computed on.

Example:
    # CPU venv
    python src/test_encoder.py prepare --index datasets/dataset_v1/index.parquet \\
        --n 48 --out encoder_test/
    # GPU container (see src/slurm/encoder_test.sbatch)
    python src/test_encoder.py smoke   --items encoder_test/items.tsv --n 8
    python src/test_encoder.py overfit --items encoder_test/items.tsv --n 32 --holdout 16
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

COLS = ["input_path", "input_chains", "sdf_paths", "system_id", "split"]


# ----------------------------------------------------------------------------
# prepare (CPU venv: pandas)
# ----------------------------------------------------------------------------
def prepare(a):
    import pandas as pd

    from ids import ligand_chain, ligand_sdf_paths

    df = pd.read_parquet(a.index)
    if a.split:
        df = df[df["split"] == a.split]
    # one interaction per receptor, so items are distinct surfaces
    df = df.drop_duplicates("input_id").sample(frac=1, random_state=a.seed)
    rows = []
    for r in df.itertuples():
        sdfs = ligand_sdf_paths(a.systems, r.system_id, ligand_chain(r.ligand_id))
        if not sdfs:
            continue
        rows.append([r.input_path, r.input_chains, ",".join(map(str, sdfs)),
                     r.system_id, r.split])
        if len(rows) == a.n:
            break
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "items.tsv").write_text("\t".join(COLS) + "\n" +
                                   "".join("\t".join(x) + "\n" for x in rows))
    print(f"{out / 'items.tsv'}: {len(rows)} items")


# ----------------------------------------------------------------------------
# loading (container: numpy, Biopython, torch)
# ----------------------------------------------------------------------------
def read_sdf_heavy(path):
    """Heavy-atom coordinates from an SDF, V2000 or V3000, without RDKit."""
    lines = Path(path).read_text().splitlines()
    xyz = []
    if any("V3000" in ln for ln in lines[:5]):
        inside = False
        for ln in lines:
            if ln.startswith("M  V30 BEGIN ATOM"):
                inside = True
            elif ln.startswith("M  V30 END ATOM"):
                break
            elif inside:
                f = ln.split()
                if f[3].upper() not in ("H", "D"):
                    xyz.append([float(f[4]), float(f[5]), float(f[6])])
    else:
        n = int(lines[3][:3])
        for ln in lines[4:4 + n]:
            if ln[31:34].strip().upper() not in ("H", "D"):
                xyz.append([float(ln[0:10]), float(ln[10:20]), float(ln[20:30])])
    return np.asarray(xyz, np.float32)


def load_items(path, n, keep_h):
    from extract import load_protein_atoms
    from ids import parse_chains

    lines = Path(path).read_text().splitlines()
    header = lines[0].split("\t")
    items = []
    for ln in lines[1:1 + n]:
        r = dict(zip(header, ln.split("\t")))
        prot = load_protein_atoms(r["input_path"], parse_chains(r["input_chains"]),
                                  keep_hydrogens=keep_h)
        lig = np.concatenate([read_sdf_heavy(p) for p in r["sdf_paths"].split(",")])
        items.append({"id": r["system_id"], "split": r["split"],
                      "atom_xyz": prot["atom_xyz"], "atom_type": prot["atom_type"],
                      "lig": lig})
    return items


def to_batch(items, device, with_surface=True):
    import torch

    P = {"atom_xyz": [], "atomtypes": [], "batch_atoms": []}
    if with_surface:
        P.update(xyz=[], normals=[], batch=[])
    for i, it in enumerate(items):
        P["atom_xyz"].append(torch.from_numpy(it["atom_xyz"]))
        P["atomtypes"].append(torch.nn.functional.one_hot(
            torch.from_numpy(it["atom_type"]), 6).float())
        P["batch_atoms"].append(torch.full((len(it["atom_type"]),), i))
        if with_surface:
            P["xyz"].append(it["xyz"])
            P["normals"].append(it["normals"])
            P["batch"].append(torch.full((len(it["xyz"]),), i))
    return {k: torch.cat(v).to(device) for k, v in P.items()}


def attach_surfaces(enc, items, device, cutoff):
    """Sample each surface once; label pocket points; record frame distance."""
    import torch

    for it in items:
        P = to_batch([it], device, with_surface=False)
        xyz, normals, _ = enc.sample_surface(P["atom_xyz"], P["atomtypes"], P["batch_atoms"])
        it["xyz"], it["normals"] = xyz.cpu(), normals.cpu()
        d = torch.cdist(it["xyz"], torch.from_numpy(it["lig"])).min(1).values
        it["pocket"] = d < cutoff
        it["min_dist"] = float(d.min())


def auroc(score, label):
    """Rank-based AUROC; nan if one class is absent."""
    import torch

    label = label.bool()
    n_pos, n_neg = int(label.sum()), int((~label).sum())
    if not n_pos or not n_neg:
        return float("nan")
    ranks = torch.empty_like(score).scatter_(0, torch.argsort(score),
                                             torch.arange(1, len(score) + 1, dtype=score.dtype))
    return float((ranks[label].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def region_hit(out, items, min_cover=0.5, min_prec=0.3):
    """Per item: rank (1-based) of the first PREDICTED pocket region covering
    >= min_cover of the labelled pocket with precision >= min_prec, else 0."""
    from anchors import region_overlap

    ranks = []
    for b, it in enumerate(items):
        base = int((out["batch"] == b).nonzero()[0])
        regs = [r - base for r, p in zip(out["anchor_regions"][b], out["anchor_is_pos"][b]) if p]
        hit = 0
        if regs:
            prec, cover = region_overlap([r.cpu() for r in regs], it["pocket"])
            ok = ((cover >= min_cover) & (prec >= min_prec)).nonzero()
            hit = int(ok[0]) + 1 if len(ok) else 0
        ranks.append(hit)
    return ranks


def make_encoder(a, device):
    import torch

    from surface_encoder import EncoderConfig, SurfaceEncoder

    torch.manual_seed(a.seed)
    enc = SurfaceEncoder(EncoderConfig(k_pos=a.k_pos, k_neg=a.k_neg)).to(device)
    if a.pretrained:
        print(f"loaded {len(enc.load_pretrained())} pretrained tensors")
    return enc


# ----------------------------------------------------------------------------
def smoke(a):
    import torch

    from surface_encoder import pocket_loss

    dev = a.device
    enc = make_encoder(a, dev)
    items = load_items(a.items, a.n, a.keep_hydrogens)
    attach_surfaces(enc, items, dev, a.cutoff)
    got_grad = set()
    print(f"\n{'system_id':40s} {'atoms':>7s} {'points':>7s} {'pocket':>7s} "
          f"{'lig-surf':>8s} {'regions':>8s} {'sec':>6s} {'GB':>6s}")
    for it in items:
        if dev.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        t = time.time()
        P = to_batch([it], dev)
        forced = [[it["pocket"].nonzero().squeeze(1)]] if it["pocket"].any() else None
        out = enc(P, forced=forced)
        loss = pocket_loss(out["pocket_logit"], it["pocket"].to(dev).float(), 20.0) \
            + 1e-3 * out["anchor_vec"][..., 0].sum()  # touch the projection too
        loss.backward()
        if dev.startswith("cuda"):
            torch.cuda.synchronize()
        gb = torch.cuda.max_memory_allocated() / 1e9 if dev.startswith("cuda") else 0
        n_pos = int(out["anchor_is_pos"][0][out["anchor_valid"][0]].sum())
        n_neg = int(out["anchor_valid"][0].sum()) - n_pos
        print(f"{it['id'][:40]:40s} {len(it['atom_type']):7d} {len(it['xyz']):7d} "
              f"{int(it['pocket'].sum()):7d} {it['min_dist']:8.2f} {n_pos:>4d}+{n_neg:<3d} "
              f"{time.time() - t:6.2f} {gb:6.2f}")
        got_grad |= {n.split(".")[0] for n, p in enc.named_parameters()
                     if p.grad is not None and p.grad.abs().sum() > 0}
        enc.zero_grad()
    no_grad = sorted({n.split(".")[0] for n, _ in enc.named_parameters()} - got_grad)
    print(f"\nparameters without gradient: {no_grad or 'none'}")
    print("lig-surf = closest ligand atom to surface (A); > ~4 means frame or chain problem")
    print("regions = pocket+negative anchors from the UNTRAINED head (+ the forced label)")


def overfit(a):
    import torch

    from surface_encoder import pocket_loss

    dev = a.device
    enc = make_encoder(a, dev)
    items = load_items(a.items, a.n + a.holdout, a.keep_hydrogens)
    attach_surfaces(enc, items, dev, a.cutoff)
    items = [it for it in items if it["pocket"].any()]
    train, held = items[:a.n], items[a.n:]
    frac = float(torch.cat([it["pocket"] for it in train]).float().mean())
    pos_weight = (1 - frac) / max(frac, 1e-6)
    print(f"{len(train)} train / {len(held)} held-out items; pocket fraction {frac:.3f}, "
          f"pos_weight {pos_weight:.1f}")
    opt = torch.optim.Adam(enc.parameters(), lr=a.lr)

    def evaluate(group):
        enc.eval()
        aucs, hits = [], []
        with torch.no_grad():
            for s in range(0, len(group), a.batch):
                chunk = group[s:s + a.batch]
                out = enc(to_batch(chunk, dev))
                for b, it in enumerate(chunk):
                    m = out["batch"] == b
                    aucs.append(auroc(out["pocket_logit"][m].cpu(), it["pocket"]))
                hits += region_hit(out, chunk)
        enc.train()
        hits = np.array(hits)
        return (np.nanmean(aucs), (hits > 0).mean(), (hits == 1).mean())

    print(f"\n{'step':>5s} {'loss':>8s} {'train AUROC':>12s} {'region hit':>11s} {'top-1':>6s}"
          + (f" {'held AUROC':>11s} {'hit':>6s} {'top-1':>6s}" if held else ""))
    rng = np.random.default_rng(a.seed)
    for step in range(a.steps + 1):
        if step % a.eval_every == 0:
            row = f"{step:5d} {loss_val if step else float('nan'):8.4f} "
            auc, hit, top1 = evaluate(train)
            row += f"{auc:12.3f} {hit:11.2f} {top1:6.2f}"
            if held:
                auc, hit, top1 = evaluate(held)
                row += f" {auc:11.3f} {hit:6.2f} {top1:6.2f}"
            print(row, flush=True)
        if step == a.steps:
            break
        chunk = [train[i] for i in rng.choice(len(train), min(a.batch, len(train)), replace=False)]
        P = to_batch(chunk, dev)
        out = enc(P)
        label = torch.cat([it["pocket"] for it in chunk]).to(dev).float()
        loss = pocket_loss(out["pocket_logit"], label, pos_weight)
        opt.zero_grad()
        loss.backward()
        opt.step()
        loss_val = loss.detach().item()
    print("\nregion hit = a predicted pocket region covers >=50% of the labelled pocket at "
          ">=30% precision; top-1 = it is the highest-ranked region")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--index", required=True)
    p.add_argument("--systems", default="systems")
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=48)
    p.add_argument("--split", default="train")
    p.add_argument("--seed", type=int, default=0)
    for name in ("smoke", "overfit"):
        p = sub.add_parser(name)
        p.add_argument("--items", required=True)
        p.add_argument("--n", type=int, default=8 if name == "smoke" else 32)
        p.add_argument("--device", default="cuda:0")
        p.add_argument("--cutoff", type=float, default=4.0)
        p.add_argument("--keep_hydrogens", action="store_true",
                       help="default strips H so every structure is treated alike")
        p.add_argument("--pretrained", action="store_true",
                       help="start features + conv from the published checkpoint")
        p.add_argument("--k_pos", type=int, default=8)
        p.add_argument("--k_neg", type=int, default=8)
        p.add_argument("--seed", type=int, default=0)
        if name == "overfit":
            p.add_argument("--holdout", type=int, default=0)
            p.add_argument("--steps", type=int, default=300)
            p.add_argument("--batch", type=int, default=4)
            p.add_argument("--lr", type=float, default=1e-3)
            p.add_argument("--eval_every", type=int, default=50)
    a = ap.parse_args()
    {"prepare": prepare, "smoke": smoke, "overfit": overfit}[a.cmd](a)


if __name__ == "__main__":
    main()
