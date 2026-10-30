"""The trained ligand encoder on its own: every ligand -> its learned vector.

Why this exists. The ligand half of a checkpoint is plain torch (ligand_encoder
imports neither KeOps nor torch_geometric), so unlike the surface encoder it
runs on any CPU, a laptop included. That makes the LEARNED ligand space
something we can look at directly - which molecules the model puts together,
and whether that agrees with chemistry (Tanimoto) or with something else (size,
target) - without a GPU job or the container.

Only the `ligand.*` tensors of the checkpoint are loaded; the surface encoder's
weights are skipped, so its config and dependencies never come into it.

    embed     a checkpoint + the ligand_feats.npz it trained on -> one .npz:
                  lig_vec    (n, out_dim) float32, L2-normalised as in training
                  smiles     (n,)
                  failed     (n,) bool; failed rows are left at zero
                  in_train   (n,) bool; the ligand is in some training row
                  in_val     (n,) bool; ... some validation row
                  n_heavy    (n,)
              --items is optional and only fills in_train / in_val (and fits
              the descriptor scaler, which the desc arms need, on exactly the
              rows training fitted it on). Prints the nearest neighbours of a
              few ligands next to their Tanimoto, as a sanity check.
    selftest  a fake checkpoint round trip; no data.

Example (CPU venv + `pip install torch`):
    python src/ligand_embed.py embed --ckpt out/runs/v3c_site_pairs_bn/best.pt \\
        --lig_feats out/runs/v1/ligand_feats.npz --items out/runs/v1/items.tsv \\
        --out out/reports/v3c_site_pairs_bn_ligands.npz

The checkpoint, ligand_feats.npz and items.tsv must belong together: fp_row is
re-indexed per dataset (docs/DATA_LAYOUT.md), and train_rows in the checkpoint
are positions in that run's items.tsv.

Depends on: numpy, torch (CPU is enough); ligand_encoder, ligand_features,
            train_data (read_items only).
"""

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def load_ligand_encoder(path):
    """-> (LigandEncoder in eval mode, the checkpoint dict). CPU only."""
    import torch

    from ligand_encoder import LigandConfig, LigandEncoder

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    enc = LigandEncoder(LigandConfig(**ckpt["ligand_config"]))
    state = {k[len("ligand."):]: v for k, v in ckpt["state_dict"].items()
             if k.startswith("ligand.")}
    enc.load_state_dict(state)          # strict: a missing tensor is an error
    enc.eval()
    return enc, ckpt


def split_ligands(items, ckpt, ligands):
    """-> (in_train, in_val) per ligand, after fitting the descriptor scaler.

    Rebuilds the row table the way train.load_rows does (same --limit, failed
    ligands dropped), because the checkpoint's train_rows / val_rows index THAT
    table, not the raw file.
    """
    from train_data import read_items

    rows = read_items(items, ckpt["args"].get("limit"))
    fp_row = np.asarray([r["fp_row"] for r in rows], dtype=np.int64)
    if fp_row.max() >= len(ligands):
        raise SystemExit(f"fp_row up to {fp_row.max()} but the features hold "
                         f"{len(ligands)} ligands; items and features are different datasets")
    fp_row = fp_row[~ligands.failed[fp_row]]
    tr = fp_row[np.asarray(ckpt["train_rows"], dtype=np.int64)]
    va = fp_row[np.asarray(ckpt["val_rows"], dtype=np.int64)]
    ligands.set_scaler(tr)
    in_train = np.zeros(len(ligands), dtype=bool)
    in_val = np.zeros(len(ligands), dtype=bool)
    in_train[tr], in_val[va] = True, True
    return in_train, in_val


def encode_all(enc, ligands, batch=512):
    import torch

    ok = np.flatnonzero(~ligands.failed)
    emb = np.zeros((len(ligands), enc.cfg.out_dim), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, len(ok), batch):
            idx = ok[s:s + batch]
            emb[idx] = enc(ligands.collate(idx)).numpy()
    return emb


def tanimoto(fp, i, js):
    a, b = fp[i].astype(bool), fp[js].astype(bool)
    inter = (a & b).sum(1)
    union = (a | b).sum(1)
    return inter / np.maximum(union, 1)


def neighbours(emb, ligands, picks, k=5):
    """Print each pick's k nearest ligands by cosine, with Tanimoto to it."""
    for i in picks:
        sim = emb @ emb[i]
        sim[i] = -np.inf
        sim[ligands.failed] = -np.inf
        top = np.argsort(-sim)[:k]
        tan = tanimoto(ligands.fp, i, top)
        print(f"\n  [{i}] {ligands.smiles[i]}  ({ligands.n_heavy[i]} heavy atoms)")
        for j, s, t in zip(top, sim[top], tan):
            print(f"     cos {s:+.3f}  tanimoto {t:.2f}  {ligands.n_heavy[j]:3d} ha  "
                  f"{ligands.smiles[j]}")


def embed(a):
    from ligand_encoder import LigandData

    enc, ckpt = load_ligand_encoder(a.ckpt)
    c = enc.cfg
    print(f"{a.ckpt}: step {ckpt['step']}, ligand arm {c.arm} -> {c.out_dim} dims, "
          f"{enc.n_params():,} parameters")
    ligands = LigandData(a.lig_feats)
    in_train = in_val = np.zeros(len(ligands), dtype=bool)
    if a.items:
        in_train, in_val = split_ligands(a.items, ckpt, ligands)
    elif c.arm in ("desc", "fp_desc"):
        raise SystemExit(f"arm {c.arm} needs --items to fit the descriptor scaler "
                         "on the checkpoint's training rows")

    emb = encode_all(enc, ligands)
    n_ok = int((~ligands.failed).sum())
    print(f"{len(ligands)} ligands, {n_ok} encoded, {len(ligands) - n_ok} failed featurisation")
    if a.items:
        print(f"{int(in_train.sum())} in training rows, {int(in_val.sum())} in validation "
              f"rows, {int((in_train & in_val).sum())} in both")

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as fh:          # a file handle: savez would append .npz
        np.savez_compressed(fh, lig_vec=emb, smiles=ligands.smiles, failed=ligands.failed,
                            in_train=in_train, in_val=in_val, n_heavy=ligands.n_heavy)
    print(f"wrote {out}")

    rng = np.random.default_rng(a.seed)
    pool = np.flatnonzero(~ligands.failed)
    picks = rng.choice(pool, size=min(a.show, len(pool)), replace=False)
    print(f"\nnearest neighbours in the learned space (cosine), with Tanimoto to the query:")
    neighbours(emb, ligands, picks)


def selftest():
    import tempfile

    import torch

    from ligand_encoder import LigandConfig, LigandEncoder, _fake_data, config_dict

    data = _fake_data()
    with tempfile.TemporaryDirectory() as tmp:
        for arm in ("gnn", "fp", "desc"):
            cfg = LigandConfig(arm=arm, out_dim=16, hidden=32, gnn_hidden=16, gnn_layers=2)
            ref = LigandEncoder(cfg).eval()
            # a checkpoint shaped like train.save_ckpt's, with a surface
            # encoder tensor that must be ignored
            state = {f"ligand.{k}": v for k, v in ref.state_dict().items()}
            state["encoder.conv.weight"] = torch.zeros(3)
            path = Path(tmp) / f"{arm}.pt"
            torch.save({"step": 1, "state_dict": state, "ligand_config": config_dict(cfg),
                        "args": {}, "train_rows": [0, 1, 2], "val_rows": [4]}, path)
            enc, _ = load_ligand_encoder(path)
            data.set_scaler([0, 1, 2])
            got = encode_all(enc, data, batch=3)
            ok = ~data.failed
            with torch.no_grad():
                want = ref(data.collate(np.flatnonzero(ok))).numpy()
            assert np.allclose(got[ok], want, atol=1e-6), arm
            assert not got[data.failed].any(), arm
            assert np.allclose(np.linalg.norm(got[ok], axis=1), 1, atol=1e-5), arm
            print(f"  {arm}: checkpoint round trip matches, batching invariant, "
                  f"failed row left at zero")
    print("PASS")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("embed", help="CPU: every ligand -> its learned vector")
    p.add_argument("--ckpt", required=True, help="train.py best.pt / last.pt")
    p.add_argument("--lig_feats", required=True, help="the run's ligand_feats.npz")
    p.add_argument("--items", default="", help="the run's items.tsv (train/val flags)")
    p.add_argument("--out", required=True)
    p.add_argument("--show", type=int, default=3, help="ligands to print neighbours for")
    p.add_argument("--seed", type=int, default=0)

    sub.add_parser("selftest", help="fake checkpoint, no data")

    a = ap.parse_args()
    if a.cmd == "embed":
        embed(a)
    else:
        selftest()


if __name__ == "__main__":
    main()
