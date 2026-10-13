"""Ligand-side encoders: one molecule -> one normalised vector, several ways.

The protein side is a SMALL encoder that is already trained on something:
dMaSIFConv_seg at 16 dimensions over 3 layers, started from the published
search checkpoint, and an anchor is a probability-weighted pool of 16-dim
point embeddings over a surface region. That sets the brief for this side:

  * Match its capacity. dataset_v1 holds 8,378 distinct ligands with heavy
    target concentration (PDE10A 747 rows, Mpro 538). A ligand branch with far
    more capacity than the protein branch can drive the contrastive loss down
    by memorising which ligands exist, and `max` over anchors lets it ignore
    the protein entirely. Every arm here stays near 10^5-10^6 parameters.
  * Spend nothing on speed. A step is thousands of surface points through
    KeOps plus region selection on the CPU; a ligand is ~30 heavy atoms. There
    is no efficiency case for freezing this side, so every arm trains
    end-to-end and we keep the gradient.
  * Encode what a SURFACE can be complementary to: size, shape, polar area,
    flexibility, hydrogen-bonding chemistry. The protein embedding sees
    curvature and element-based chemistry within ~12 A. That is why `desc`
    exists next to `fp`: ECFP4 identifies substructures, which is a strong
    retrieval baseline but says nothing about extent or polarity.

Arms, all mapping to the same `out_dim` and all L2-normalised, so they are
interchangeable in the loss and differ by one config field:

    fp          ECFP4 2048 -> MLP(hidden) -> out            (~1.1 M)
    fp_linear   ECFP4 2048 -> Linear -> out                 (~0.26 M)
    desc        physchem descriptors -> MLP -> out           (~0.04 M)
    fp_desc     [ECFP4, descriptors] -> MLP -> out           (~1.1 M)
    gnn         GINE over the heavy-atom graph -> pool -> out  (~0.32 M)
    frozen      a precomputed embedding -> Linear/MLP -> out

`fp`'s parameters are almost all in the first layer of a 2048-wide input, so
fp-vs-gnn is not a capacity-matched comparison on its own; `fp_linear` is the
control that separates "more parameters" from "better features".

The graph arm implements GINE directly on index_add_ / scatter_reduce rather
than importing PyG. The container has PyG, but the layer is ~30 lines, and
keeping it out means this module (and every test of it) runs anywhere plain
torch runs - which is how the loss, the metrics and the whole probe get tested
without a GPU.

Features come from ligand_features.py as packed arrays; LigandData holds them
and collates a batch. Nothing here imports rdkit.

Example:
    python src/ligand_encoder.py            # self-test: shapes, gradients, sizes

Depends on: torch, ligand_features (schema constants only).
Imported by: probe_retrieval.py, train.py.
"""

import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from ligand_features import ATOM_DIM, BOND_DIM, DESC_DIM, FP_BITS  # noqa: E402

ARMS = ("fp", "fp_linear", "desc", "fp_desc", "gnn", "frozen")


@dataclass
class LigandConfig:
    arm: str = "fp"
    out_dim: int = 128
    hidden: int = 512          # fp / desc / fp_desc MLP width
    dropout: float = 0.0
    # gnn
    gnn_hidden: int = 128
    gnn_layers: int = 4
    gnn_pool: str = "mean_max"  # mean_max | mean
    # frozen
    frozen_dim: int = 0        # width of the precomputed embedding
    frozen_head: str = "linear"  # linear | mlp
    normalize: bool = True

    def __post_init__(self):
        if self.arm not in ARMS:
            raise ValueError(f"unknown ligand arm {self.arm!r}; expected one of {ARMS}")
        if self.arm == "frozen" and self.frozen_dim <= 0:
            raise ValueError("arm 'frozen' needs frozen_dim = the embedding width")


# ---------------------------------------------------------------------------
# features -> batches
# ---------------------------------------------------------------------------
class LigandData:
    """The arrays from ligand_features.py, with a collate for a row list.

    Row i is ligand i of the file, i.e. fp_row i of the dataset. `desc` is
    returned standardised when set_scaler() has been called (fit it on YOUR
    training rows, never on all of them).
    """

    def __init__(self, path=None, **arrays):
        if path is not None:
            with np.load(path, allow_pickle=False) as d:
                arrays = {k: d[k] for k in d.files if k != "schema"}
        self.fp = arrays["fp"]
        self.desc = arrays["desc"].astype(np.float32)
        self.atom_feat = arrays["atom_feat"]
        self.atom_ptr = arrays["atom_ptr"]
        self.bond_feat = arrays["bond_feat"]
        self.bond_index = arrays["bond_index"]
        self.bond_ptr = arrays["bond_ptr"]
        self.failed = arrays.get("failed", np.zeros(len(self.fp), dtype=bool))
        self.smiles = arrays.get("smiles", np.array([""] * len(self.fp)))
        self.n_heavy = arrays.get("n_heavy", np.diff(self.atom_ptr).astype(np.int32))
        if self.atom_feat.shape[1] != ATOM_DIM or self.bond_feat.shape[1] != BOND_DIM:
            raise ValueError(
                f"feature widths {self.atom_feat.shape[1]}/{self.bond_feat.shape[1]} "
                f"do not match ligand_features ({ATOM_DIM}/{BOND_DIM}); the .npz was "
                "written by a different schema")
        self.desc_mean = np.zeros(self.desc.shape[1], dtype=np.float32)
        self.desc_std = np.ones(self.desc.shape[1], dtype=np.float32)

    def __len__(self):
        return len(self.fp)

    def set_scaler(self, rows):
        """Fit descriptor standardisation on these rows only -> (mean, std).

        Descriptors span MolWt in the hundreds and FractionCSP3 in [0, 1]; an
        unstandardised input makes the first layer's job needlessly hard. A
        zero-variance column is left at std 1 so it maps to a constant 0.
        """
        d = self.desc[np.asarray(rows)]
        self.desc_mean = d.mean(0)
        std = d.std(0)
        std[std < 1e-6] = 1.0
        self.desc_std = std.astype(np.float32)
        return self.desc_mean, self.desc_std

    def collate(self, rows, device="cpu"):
        """-> batch dict for LigandEncoder.forward, with rows in the given order."""
        rows = np.asarray(rows, dtype=np.int64)
        fp = torch.from_numpy(self.fp[rows].astype(np.float32))
        desc = torch.from_numpy((self.desc[rows] - self.desc_mean) / self.desc_std)

        atom_feat, atom_batch, edge_src, edge_dst, bond_feat = [], [], [], [], []
        offset = 0
        for i, r in enumerate(rows):
            a0, a1 = int(self.atom_ptr[r]), int(self.atom_ptr[r + 1])
            b0, b1 = int(self.bond_ptr[r]), int(self.bond_ptr[r + 1])
            n_atoms = a1 - a0
            atom_feat.append(self.atom_feat[a0:a1])
            atom_batch.append(np.full(n_atoms, i, dtype=np.int64))
            if b1 > b0:
                e = self.bond_index[:, b0:b1].astype(np.int64) + offset
                # each stored bond becomes both directions; messages flow
                # src -> dst, so an undirected graph needs the pair
                edge_src.append(np.concatenate([e[0], e[1]]))
                edge_dst.append(np.concatenate([e[1], e[0]]))
                bond_feat.append(np.concatenate([self.bond_feat[b0:b1]] * 2))
            offset += n_atoms

        cat = lambda chunks, empty: (np.concatenate(chunks) if chunks else empty)
        edge_index = np.stack([
            cat(edge_src, np.zeros(0, dtype=np.int64)),
            cat(edge_dst, np.zeros(0, dtype=np.int64))])
        return {
            "fp": fp.to(device),
            "desc": desc.to(device),
            "n_mol": len(rows),
            "atom_feat": torch.from_numpy(
                cat(atom_feat, np.zeros((0, ATOM_DIM), dtype=np.uint8)).astype(np.float32)
            ).to(device),
            "atom_batch": torch.from_numpy(
                cat(atom_batch, np.zeros(0, dtype=np.int64))).to(device),
            "edge_index": torch.from_numpy(edge_index).to(device),
            "bond_feat": torch.from_numpy(
                cat(bond_feat, np.zeros((0, BOND_DIM), dtype=np.uint8)).astype(np.float32)
            ).to(device),
        }


# ---------------------------------------------------------------------------
# arms
# ---------------------------------------------------------------------------
def _mlp(i, h, o, dropout=0.0):
    layers = [nn.Linear(i, h), nn.LeakyReLU(0.2)]
    if dropout:
        layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(h, o))
    return nn.Sequential(*layers)


class GINEStack(nn.Module):
    """GINE over a heavy-atom graph, on plain scatter ops.

        h_i <- MLP( (1 + eps) h_i + sum_j ReLU(h_j + W_e e_ij) )

    with a LayerNorm and a residual per layer. eps is learnable, as in the
    paper's GIN-eps. A molecule with no bonds (a lone ion) simply receives no
    messages and is carried by the residual path, which is why the aggregation
    must not assume every atom has a neighbour.
    """

    def __init__(self, atom_dim, bond_dim, hidden, n_layers, dropout=0.0):
        super().__init__()
        self.embed = nn.Linear(atom_dim, hidden)
        self.eps = nn.Parameter(torch.zeros(n_layers))
        self.edge = nn.ModuleList(nn.Linear(bond_dim, hidden) for _ in range(n_layers))
        self.mlps = nn.ModuleList(
            _mlp(hidden, 2 * hidden, hidden, dropout) for _ in range(n_layers))
        self.norms = nn.ModuleList(nn.LayerNorm(hidden) for _ in range(n_layers))

    def forward(self, atom_feat, edge_index, bond_feat):
        h = self.embed(atom_feat)
        src, dst = edge_index[0], edge_index[1]
        for i, (mlp, norm) in enumerate(zip(self.mlps, self.norms)):
            agg = torch.zeros_like(h)
            if src.numel():
                msg = F.relu(h[src] + self.edge[i](bond_feat))
                agg = agg.index_add(0, dst, msg)
            h = h + norm(mlp((1 + self.eps[i]) * h + agg))
        return h


def pool_atoms(h, atom_batch, n_mol, how="mean_max"):
    """Per-molecule pooling of atom embeddings -> (n_mol, H) or (n_mol, 2H).

    A molecule with no atoms (an unparseable row) pools to zeros rather than
    NaN: index_add leaves its sum at 0 and the count is clamped, and the max
    uses include_self=False so an untouched row keeps its 0.
    """
    total = torch.zeros(n_mol, h.shape[1], dtype=h.dtype, device=h.device)
    count = torch.zeros(n_mol, dtype=h.dtype, device=h.device)
    if h.numel():
        total = total.index_add(0, atom_batch, h)
        count = count.index_add(0, atom_batch, torch.ones_like(atom_batch, dtype=h.dtype))
    mean = total / count.clamp_min(1.0)[:, None]
    if how == "mean":
        return mean
    mx = torch.zeros_like(total)
    if h.numel():
        mx = mx.scatter_reduce(0, atom_batch[:, None].expand_as(h), h,
                               reduce="amax", include_self=False)
    return torch.cat([mean, mx], dim=1)


class LigandEncoder(nn.Module):
    """One arm of LigandConfig. forward(batch) -> (n_mol, out_dim), normalised."""

    def __init__(self, cfg: LigandConfig = None):
        super().__init__()
        self.cfg = cfg = cfg or LigandConfig()
        D, H = cfg.out_dim, cfg.hidden
        if cfg.arm == "fp":
            self.net = _mlp(FP_BITS, H, D, cfg.dropout)
        elif cfg.arm == "fp_linear":
            self.net = nn.Linear(FP_BITS, D)
        elif cfg.arm == "desc":
            self.net = _mlp(DESC_DIM, min(H, 256), D, cfg.dropout)
        elif cfg.arm == "fp_desc":
            self.net = _mlp(FP_BITS + DESC_DIM, H, D, cfg.dropout)
        elif cfg.arm == "gnn":
            self.gnn = GINEStack(ATOM_DIM, BOND_DIM, cfg.gnn_hidden,
                                 cfg.gnn_layers, cfg.dropout)
            width = cfg.gnn_hidden * (2 if cfg.gnn_pool == "mean_max" else 1)
            self.net = _mlp(width, cfg.gnn_hidden, D, cfg.dropout)
        elif cfg.arm == "frozen":
            self.net = (nn.Linear(cfg.frozen_dim, D) if cfg.frozen_head == "linear"
                        else _mlp(cfg.frozen_dim, H, D, cfg.dropout))

    def forward(self, batch):
        c = self.cfg
        if c.arm in ("fp", "fp_linear"):
            x = self.net(batch["fp"])
        elif c.arm == "desc":
            x = self.net(batch["desc"])
        elif c.arm == "fp_desc":
            x = self.net(torch.cat([batch["fp"], batch["desc"]], dim=1))
        elif c.arm == "gnn":
            h = self.gnn(batch["atom_feat"], batch["edge_index"], batch["bond_feat"])
            x = self.net(pool_atoms(h, batch["atom_batch"], batch["n_mol"], c.gnn_pool))
        else:
            x = self.net(batch["frozen"])
        return F.normalize(x, dim=-1) if c.normalize else x

    def n_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build(arm, out_dim=128, **kwargs):
    """LigandEncoder for one arm name; kwargs override LigandConfig fields."""
    return LigandEncoder(LigandConfig(arm=arm, out_dim=out_dim, **kwargs))


def config_dict(cfg):
    return asdict(cfg)


# ---------------------------------------------------------------------------
# self-test
# ---------------------------------------------------------------------------
def _fake_data(n=7, seed=0):
    """Packed arrays shaped like ligand_features.py output, including the two
    edge cases that have bitten real molecules: a one-atom row with no bonds
    (an ion) and an unparseable all-zero row."""
    rng = np.random.default_rng(seed)
    sizes = [12, 1, 20, 0, 5, 30, 8]  # row 3 = unparseable
    atom_ptr = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
    atom_feat = rng.integers(0, 2, (atom_ptr[-1], ATOM_DIM), dtype=np.uint8)
    idx, bfeat, bond_ptr = [], [], [0]
    for s in sizes:
        n_bonds = max(s - 1, 0)
        if n_bonds:
            e = np.stack([np.arange(n_bonds), np.arange(1, n_bonds + 1)])
            idx.append(e.astype(np.int32))
            bfeat.append(rng.integers(0, 2, (n_bonds, BOND_DIM), dtype=np.uint8))
        bond_ptr.append(bond_ptr[-1] + n_bonds)
    failed = np.zeros(n, dtype=bool)
    failed[3] = True
    return LigandData(
        fp=rng.integers(0, 2, (n, FP_BITS), dtype=np.uint8),
        desc=rng.normal(50, 100, (n, DESC_DIM)).astype(np.float32),
        atom_feat=atom_feat, atom_ptr=atom_ptr,
        bond_feat=(np.concatenate(bfeat) if bfeat else np.zeros((0, BOND_DIM), np.uint8)),
        bond_index=(np.concatenate(idx, axis=1) if idx else np.zeros((2, 0), np.int32)),
        bond_ptr=np.asarray(bond_ptr, dtype=np.int64), failed=failed)


def selftest():
    torch.manual_seed(0)
    data = _fake_data()
    data.set_scaler(np.arange(len(data)))
    rows = np.arange(len(data))
    batch = data.collate(rows)
    print(f"{len(data)} fake molecules, {int(data.failed.sum())} flagged unparseable; "
          f"{batch['atom_feat'].shape[0]} atoms, {batch['edge_index'].shape[1]} directed edges")
    assert abs(float(batch["desc"].mean())) < 1e-4, "descriptors not centred by the scaler"

    print(f"\n{'arm':12s} {'params':>10s} {'out':>12s} {'|v|':>7s} {'grad':>6s}")
    for arm in ARMS:
        enc = build(arm, frozen_dim=256 if arm == "frozen" else 0)
        b = dict(batch)
        if arm == "frozen":
            b["frozen"] = torch.randn(len(rows), 256)
        v = enc(b)
        assert v.shape == (len(rows), 128), f"{arm}: {v.shape}"
        assert torch.isfinite(v).all(), f"{arm}: non-finite output"
        norms = v.norm(dim=-1)
        # the unparseable row has no atoms and an all-zero fingerprint path;
        # it must still come back finite, which is what normalize(0) risks
        v.sum().backward()
        n_grad = sum(1 for p in enc.parameters() if p.grad is not None
                     and torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0)
        n_all = sum(1 for _ in enc.parameters())
        print(f"{arm:12s} {enc.n_params():10d} {str(tuple(v.shape)):>12s} "
              f"{float(norms.mean()):7.3f} {n_grad:3d}/{n_all:<3d}")
        assert n_grad > 0, f"{arm}: no parameter received a gradient"

    # a molecule's vector must not depend on which batch it rode in
    enc = build("gnn").eval()
    with torch.no_grad():
        alone = enc(data.collate([2]))
        together = enc(data.collate([0, 2, 5]))
    delta = float((alone[0] - together[1]).abs().max())
    print(f"\ngnn batch invariance: max |alone - in batch| = {delta:.2e}")
    assert delta < 1e-5, "gnn output depends on batch composition"

    # permuting the batch must permute the output, nothing more
    with torch.no_grad():
        fwd = enc(data.collate([0, 2, 5]))
        rev = enc(data.collate([5, 2, 0]))
    assert float((fwd - rev.flip(0)).abs().max()) < 1e-5, "gnn not permutation-consistent"
    print("gnn permutation consistency: ok")
    print("\nselftest passed")


if __name__ == "__main__":
    selftest()
