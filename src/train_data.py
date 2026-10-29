"""Data layer for contrastive training: items, surfaces, pocket labels, sampling.

One ROW is one interaction: a receptor structure plus the ligand bound in it.
One SURFACE belongs to a receptor, and several rows can share it (a receptor
with two bound ligands is two rows, one surface). So the two are cached
separately: surfaces by receptor, labels by row.

    SurfaceStore   atoms -> dMaSIF surface, once per receptor, kept on disk and
                   in a bounded host-RAM cache. Sampling is stochastic, so a
                   surface is seeded from the receptor's own identity
                   (ids.seed_for) and is therefore the same in every job, on
                   any batching, which is what lets it be cached at all.
    pocket labels  surface points within `cutoff` of a ligand heavy atom,
                   recomputed against the surface in hand from the ligand's
                   COORDINATES. Never a saved index: labels.py's own warning is
                   that an index addresses the point cloud it was written on.
    other sites    a receptor's OTHER ligands, from the sibling rows of the
                   item table itself. A region over one of them is a genuine
                   pocket and must be ignored rather than trained as a
                   negative, and the table already knows where they are, so
                   nothing external has to be built for it.
    sampling       target-balanced and group-disjoint, reusing
                   probe_retrieval's sample_weights / group_index / draw_batch /
                   split_by_group so the full loop and the frozen-anchor probe
                   share one definition of a batch and a split.

Memory. A 37k-point surface is ~0.9 MB of xyz + normals and its atoms are a
further ~0.2 MB, so the whole of dataset_v1's 19,782 receptors is ~20 GB on
disk and cannot sit in RAM. The disk cache is the scale path and the RAM cache
is a window onto it; `--surface_ram` sets how wide. A receptor's region cache
(the selector's radius neighbour lists, the one O(N^2) step in selection) lives
with its RAM entry, so it survives as long as the surface does and is rebuilt
with it.

Depends on: numpy; torch; pandas (prepare only); extract, chemotype, ids,
            probe_retrieval, test_encoder.
"""

import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# input_id is the surface's identity: build_dataset's receptor input, which is
# what the structure path and chain filter were derived from. receptor_key is
# coarser (it ignores the chain filter), so two inputs of one receptor_key
# would share a cache entry they must not share.
ITEM_COLS = ["system_id", "input_id", "receptor_key", "split", "input_path",
             "input_chains", "sdf_paths", "fp_row", "group", "smiles"]


# ---------------------------------------------------------------------------
# prepare (CPU venv: pandas)
# ---------------------------------------------------------------------------
def prepare(a):
    """Dataset index -> items.tsv, one row per usable interaction.

    Unlike probe_retrieval.prepare this keeps EVERY interaction, not one per
    receptor: a receptor with several ligands is several training pairs, and
    they share one cached surface, so the extra rows are nearly free. They are
    never in-batch negatives for each other either - draw_batch allows one row
    per target group, and sibling rows of a receptor share a group.
    """
    import pandas as pd

    from ids import ligand_chain, ligand_sdf_paths
    from probe_retrieval import group_of

    df = pd.read_parquet(a.index)
    n_all = len(df)
    if a.split:
        keep = a.split.split(",")
        df = df[df["split"].isin(keep)]
        print(f"splits {keep}: {n_all} -> {len(df)} rows")

    from ids import receptor_key as rkey

    excluded = set()
    for path in a.exclude or []:
        lines = Path(path).read_text().splitlines()
        header = lines[0].split("\t")
        if "system_id" not in header:
            raise SystemExit(f"{path}: no system_id column to exclude on")
        col = header.index("system_id")
        excluded |= {rkey(ln.split("\t")[col]) for ln in lines[1:] if ln.strip()}
    if excluded:
        before = len(df)
        df = df[~df["receptor_key"].isin(excluded)]
        print(f"excluded {len(excluded)} receptors from {len(a.exclude)} file(s): "
              f"{before} -> {len(df)} rows")

    df = df.sample(frac=1, random_state=a.seed)
    rows, n_no_sdf = [], 0
    for r in df.itertuples():
        d = r._asdict()
        sdfs = ligand_sdf_paths(a.systems, r.system_id, ligand_chain(r.ligand_id))
        if not sdfs:
            n_no_sdf += 1
            continue
        rows.append({
            "system_id": r.system_id, "input_id": r.input_id,
            "receptor_key": r.receptor_key, "split": r.split,
            "input_path": r.input_path, "input_chains": r.input_chains,
            "sdf_paths": ",".join(map(str, sdfs)), "fp_row": int(r.fp_row),
            "group": group_of(d),
            "smiles": str(d.get("ligand_rdkit_canonical_smiles", "")),
        })
        if a.n and len(rows) == a.n:
            break
    if not rows:
        raise SystemExit("no usable rows; are systems/ and the index in place?")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "items.tsv"
    path.write_text("\t".join(ITEM_COLS) + "\n"
                    + "".join("\t".join(str(r[c]) for c in ITEM_COLS) + "\n"
                              for r in rows))

    groups = np.asarray([r["group"] for r in rows])
    sizes = np.unique(groups, return_counts=True)[1]
    n_surf = len({r["input_id"] for r in rows})
    print(f"\n{path}: {len(rows)} interactions "
          f"({n_no_sdf} skipped for a missing ligand SDF)")
    print(f"{n_surf} distinct receptor inputs = surfaces to sample and cache")
    print(f"{len(sizes)} target groups; largest {sorted(sizes.tolist())[-5:]}, "
          f"median {int(np.median(sizes))}")
    print(f"{len({r['fp_row'] for r in rows})} distinct ligands")
    print("\nnext: src/ligand_features.py for ligand_feats.npz, then train.py "
          "cache / train in the container")


# ---------------------------------------------------------------------------
# the item table
# ---------------------------------------------------------------------------
def read_items(path, n=None):
    """items.tsv -> list of row dicts, fp_row as int."""
    lines = Path(path).read_text().splitlines()
    header = lines[0].split("\t")
    missing = [c for c in ITEM_COLS if c not in header]
    if missing:
        raise SystemExit(f"{path}: missing columns {missing}; rebuild it with "
                         "`train.py prepare`")
    body = lines[1:] if n is None else lines[1:1 + n]
    rows = []
    for ln in body:
        if not ln.strip():
            continue
        r = dict(zip(header, ln.split("\t")))
        r["fp_row"] = int(r["fp_row"])
        rows.append(r)
    return rows


class ItemTable:
    """The rows, plus the two indexes training asks for repeatedly.

    `siblings` is every row of the same receptor input, which is where the
    other-site ignore mask comes from. `group_id` is the integer target group,
    used by the sampler and by the false-negative mask.
    """

    def __init__(self, rows):
        self.rows = rows
        self.group = np.asarray([r["group"] for r in rows])
        _, self.group_id = np.unique(self.group, return_inverse=True)
        self.group_id = self.group_id.astype(np.int64)
        self.fp_row = np.asarray([r["fp_row"] for r in rows], dtype=np.int64)
        self.siblings = {}
        for i, r in enumerate(rows):
            self.siblings.setdefault(r["input_id"], []).append(i)

    def __len__(self):
        return len(self.rows)

    def summary(self):
        return (f"{len(self.rows)} interactions, {len(set(self.group))} target "
                f"groups, {len(self.siblings)} surfaces, "
                f"{len(set(self.fp_row.tolist()))} distinct ligands")


# ---------------------------------------------------------------------------
# surfaces
# ---------------------------------------------------------------------------
SURFACE_KEYS = ("xyz", "normals", "atom_xyz", "atom_type", "atom_groups")


class SurfaceStore:
    """Receptor input -> its surface and atoms, from RAM, then disk, then dMaSIF.

    A miss on both costs a Biopython parse plus a surface sampling pass, so the
    first epoch over a large item table is dominated by this and every later one
    reads the disk cache. `cache_dir` can be prefilled in parallel by
    `train.py cache --shard i/n`, which is the same code path.

    The RAM cache is an LRU of whole entries including each receptor's region
    cache, so evicting a surface evicts the neighbour lists computed on it -
    they are only valid for those points and anchors.select_regions would
    refuse them anyway.
    """

    def __init__(self, cache_dir=None, ram=256, device="cpu",
                 keep_hydrogens=False, pockets=None):
        self.dir = Path(cache_dir) if cache_dir else None
        # external_pockets.PocketSource or None. Mapped onto each surface when
        # its RAM entry is built, so the regions live and die with the surface.
        self.pockets = pockets
        if self.dir:
            self.dir.mkdir(parents=True, exist_ok=True)
        self.ram = max(int(ram), 1)
        self.device = device
        self.keep_hydrogens = keep_hydrogens
        self._mem = OrderedDict()
        self.stats = {"ram": 0, "disk": 0, "sampled": 0, "failed": 0}

    # --- the three layers -------------------------------------------------
    def path_for(self, row):
        return None if self.dir is None else self.dir / f"{row['input_id']}.npz"

    def _load_atoms(self, row):
        import chemotype
        from extract import load_protein_atoms
        from ids import parse_chains

        prot = load_protein_atoms(row["input_path"],
                                  parse_chains(row["input_chains"]),
                                  keep_hydrogens=self.keep_hydrogens)
        # Always typed, cheap, and cached: a run with cfg.chemo on must not need
        # a different cache from one with it off.
        return chemotype.annotate(prot)

    def _sample(self, enc, row):
        """Atoms -> surface, seeded from this receptor's own identity.

        The seed is set and the global RNG state put back afterwards. Without
        that, a cache MISS mid-training would reseed the process from a
        receptor's id and every later draw of the run - batches, dropout,
        negative-region seeds - would depend on which surfaces happened to be
        resident. Two jobs with the same --seed would then diverge on cache
        state alone, which is the hardest kind of irreproducibility to notice.
        """
        import torch

        from ids import seed_for
        from test_encoder import to_batch

        prot = self._load_atoms(row)
        item = {"atom_xyz": prot["atom_xyz"], "atom_type": prot["atom_type"],
                "atom_groups": prot["atom_groups"]}
        cpu_state = torch.get_rng_state()
        cuda_state = (torch.cuda.get_rng_state_all()
                      if torch.cuda.is_available() else None)
        try:
            torch.manual_seed(seed_for(row["input_id"]))
            P = to_batch([item], self.device, with_surface=False)
            with torch.no_grad():
                xyz, normals, _ = enc.sample_surface(P["atom_xyz"], P["atomtypes"],
                                                     P["batch_atoms"])
        finally:
            torch.set_rng_state(cpu_state)
            if cuda_state is not None:
                torch.cuda.set_rng_state_all(cuda_state)
        return {"xyz": xyz.cpu().numpy().astype(np.float32),
                "normals": normals.cpu().numpy().astype(np.float32),
                "atom_xyz": prot["atom_xyz"], "atom_type": prot["atom_type"],
                "atom_groups": prot["atom_groups"]}

    def write(self, row, arrays):
        path = self.path_for(row)
        if path is None:
            return
        # Write-then-rename, so a job killed mid-write (a walltime limit is the
        # normal end of a training job) cannot leave a truncated .npz that the
        # next job loads as a surface.
        #
        # Through an open file handle, NOT by passing the path: np.savez appends
        # ".npz" to any filename that does not already end in it, so savez(path)
        # on "x.npz.tmp" writes "x.npz.tmp.npz" and the rename below then finds
        # nothing. A file object is left alone.
        tmp = path.with_suffix(".npz.tmp")
        with open(tmp, "wb") as fh:
            np.savez(fh, **{k: arrays[k] for k in SURFACE_KEYS})
        tmp.replace(path)

    def ensure(self, enc, row):
        """Make sure this receptor is in the disk cache -> True if it was built."""
        path = self.path_for(row)
        if path is not None and path.exists():
            return False
        self.write(row, self._sample(enc, row))
        self.stats["sampled"] += 1
        return True

    def get(self, enc, row):
        """-> the entry dict: torch surface + atoms + this receptor's region cache."""
        import torch

        key = row["input_id"]
        if key in self._mem:
            self._mem.move_to_end(key)
            self.stats["ram"] += 1
            return self._mem[key]

        path = self.path_for(row)
        arrays = None
        if path is not None and path.exists():
            try:
                with np.load(path, allow_pickle=False) as d:
                    arrays = {k: d[k] for k in SURFACE_KEYS}
                self.stats["disk"] += 1
            except Exception as e:                       # truncated or stale file
                print(f"[warn] {path}: {type(e).__name__}: {e}; resampling")
                arrays = None
        if arrays is None:
            arrays = self._sample(enc, row)
            self.stats["sampled"] += 1
            self.write(row, arrays)

        entry = {"input_id": key,
                 "xyz": torch.from_numpy(np.ascontiguousarray(arrays["xyz"])),
                 "normals": torch.from_numpy(np.ascontiguousarray(arrays["normals"])),
                 "atom_xyz": np.ascontiguousarray(arrays["atom_xyz"]),
                 "atom_type": np.ascontiguousarray(arrays["atom_type"]),
                 "atom_groups": np.ascontiguousarray(arrays["atom_groups"]),
                 "region_cache": {}}
        if self.pockets is not None:
            # ext_regions / ext_map: the receptor's top pockets; ext_raw: the
            # spheres themselves, for the per-row `site` select
            ext = self.pockets.map_entry(key, arrays["xyz"])
            entry["ext_regions"] = [torch.from_numpy(r) for r in ext["ext_regions"]]
            entry["ext_map"] = torch.from_numpy(ext["ext_map"])
            entry["ext_raw"] = ext["ext_raw"]
        self._mem[key] = entry
        while len(self._mem) > self.ram:
            self._mem.popitem(last=False)
        return entry

    def report(self):
        s = self.stats
        total = s["ram"] + s["disk"] + s["sampled"]
        return (f"surfaces: {total} requests, {s['ram']} RAM / {s['disk']} disk / "
                f"{s['sampled']} sampled; {len(self._mem)}/{self.ram} resident"
                + ("" if self.pockets is None else "; " + self.pockets.report()))


# ---------------------------------------------------------------------------
# pocket labels and other sites
# ---------------------------------------------------------------------------
def ligand_xyz(row):
    """This row's ligand heavy atoms, from its SDF(s)."""
    from test_encoder import read_sdf_heavy

    return np.concatenate([read_sdf_heavy(p) for p in row["sdf_paths"].split(",")])


def min_dist(xyz, lig, chunk=8192):
    """Per surface point, the distance to the nearest ligand heavy atom.

    Chunked over the surface: a 37k x 300 cdist is fine, but a receptor with
    several large ligands against a big surface is not, and this is called once
    per row per epoch.
    """
    import torch

    lig = torch.as_tensor(lig, dtype=torch.float32)
    out = torch.empty(len(xyz))
    for s in range(0, len(xyz), chunk):
        out[s:s + chunk] = torch.cdist(xyz[s:s + chunk], lig).min(1).values
    return out


class LabelCache:
    """Pocket and other-site masks per ROW, keyed by row index.

    Bounded like the surfaces and for the same reason: a mask is one bool per
    surface point, so a few thousand rows is a few hundred MB. Evicting a mask
    only costs the SDF read and one chunked cdist to rebuild.
    """

    def __init__(self, table, cutoff=4.0, ram=512, plip=None):
        self.table = table
        self.cutoff = cutoff
        # plip: a plip_labels.py output directory. When set, every record also
        # carries `plip_union`, the (N, 6) PLIP class labels on this surface
        # from every ligand of the receptor (plip_loss.receptor_labels), or
        # None when no PLIP file exists for any of them.
        self.plip = plip
        self.ram = max(int(ram), 1)
        self._mem = OrderedDict()
        self.failed = {}

    def get(self, i, entry):
        """-> dict(pocket, other, lig, min_dist) for row i on this surface.

        `other` is the union of the receptor's OTHER ligands' neighbourhoods,
        with this ligand's own pocket removed: an overlapping site is this
        row's positive, not someone else's.
        """
        import torch

        key = (i, entry["input_id"], len(entry["xyz"]))
        if key in self._mem:
            self._mem.move_to_end(key)
            return self._mem[key]

        row = self.table.rows[i]
        lig = ligand_xyz(row)
        d = min_dist(entry["xyz"], lig)
        pocket = d < self.cutoff
        other = torch.zeros(len(entry["xyz"]), dtype=torch.bool)
        n_other = 0
        for j in self.table.siblings.get(row["input_id"], []):
            if j == i or self.table.rows[j]["sdf_paths"] == row["sdf_paths"]:
                continue
            try:
                other |= min_dist(entry["xyz"], ligand_xyz(self.table.rows[j])) < self.cutoff
                n_other += 1
            except Exception as e:
                self.failed.setdefault(j, f"{type(e).__name__}: {e}")
        other &= ~pocket

        rec = {"pocket": pocket, "other": other, "lig": lig,
               "min_dist": float(d.min()), "n_other": n_other}
        if self.plip:
            from plip_loss import receptor_labels
            sibs = [self.table.rows[j]["system_id"]
                    for j in self.table.siblings.get(row["input_id"], [])]
            _, rec["plip_union"] = receptor_labels(
                np.asarray(entry["xyz"], np.float32), row["system_id"], sibs, self.plip)
        self._mem[key] = rec
        while len(self._mem) > self.ram:
            self._mem.popitem(last=False)
        return rec


# ---------------------------------------------------------------------------
# region labels and the fallback anchor
# ---------------------------------------------------------------------------
def region_labels(regions, pocket, other, pos_cover=0.5, pos_prec=0.3,
                  ignore_overlap=0.1):
    """-> (label, cover, prec) per region: 1 positive, 0 negative, -1 ignore.

    The same rule probe_retrieval uses, so the frozen-anchor probe and the full
    loop agree on what a positive anchor is. A near-duplicate of the pocket is
    an IGNORE, not a negative: ball_R8 produces about four regions per protein
    that are at least 30% pocket, and training the other three against the
    ligand would teach the encoder that its own correct answers are wrong.
    """
    import torch

    from anchors import region_overlap

    if not regions:
        return (torch.zeros(0, dtype=torch.long), torch.zeros(0), torch.zeros(0))
    prec, cover = region_overlap(regions, pocket)
    label = torch.zeros(len(regions), dtype=torch.long)
    label[(cover >= ignore_overlap) | (prec >= ignore_overlap)] = -1
    label[(cover >= pos_cover) & (prec >= pos_prec)] = 1
    if other is not None and bool(other.any()):
        o_prec, o_cover = region_overlap(regions, other)
        touches = (o_cover >= ignore_overlap) | (o_prec >= ignore_overlap)
        label[touches & (label == 0)] = -1
    return label, cover, prec


def fallback_region(xyz, pocket, radius):
    """A selector-shaped region at the labelled pocket's centroid -> indices.

    Used when the predicted regions miss the pocket entirely, so the step still
    has a positive to train on. Deliberately a BALL of the selector's own
    radius rather than the labelled point set itself: the point set has a
    different size and shape from anything ball_R8 produces, so a model that
    saw it as the answer would be learning on an anchor that cannot exist at
    inference. A ball at the right place is the same object the selector would
    have handed over had the head been better.

    `pocket` must be on the same device as `xyz`: a boolean mask cannot index
    a tensor on another device, and these two come from different places - the
    mask from the CPU label cache, the points from the encoder's output.
    """
    centre = xyz[pocket].mean(0)
    return (((xyz - centre) ** 2).sum(-1) < radius * radius).nonzero().squeeze(1)
