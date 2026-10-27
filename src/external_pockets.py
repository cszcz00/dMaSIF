"""Pocket regions from an external pocket finder (fpocket), instead of the head.

Why this exists. Every region the contrastive loop has scored so far came from
the learned pocket head: top 5% of its smoothed map, grown into balls
(ball_R8). The head is the known limit - its candidates miss a third of the
labelled pocket (cover 0.68 at top 5%, AP ~0.47) - and region selection has no
gradient, so a weak map means the loop scores the wrong places and the
contrastive loss cannot fix that. This module takes the head out of WHERE the
regions are: a geometric pocket finder picks the pockets, and the encoder only
has to describe them.

    run     container (Biopython for the atoms; the fpocket binary).
            items.tsv -> one <input_id>.npz per receptor surface, holding
            fpocket's alpha spheres grouped by pocket plus the per-pocket
            score / druggability / volume. Shardable across array tasks, like
            `train.py cache`. fpocket runs on EXACTLY the atoms the surface is
            built from (extract.load_protein_atoms, same chain filter, no
            hydrogens), written to a temporary PDB.
    score   numpy only (CPU venv or container). How good are those pockets as
            REGIONS, against the labelled pocket on the cached dMaSIF
            surfaces, with region_sweep.py's metrics, so the numbers sit next
            to ball_R8's (held-out: hit 0.84-0.86, top1 0.64, dccA 0.77).
            Run it before training: it says whether fpocket can do better than
            the head, and which --pocket_margin / --pocket_shape to use.
    selftest numpy only: the PQR / info parsers on a planted fpocket output,
            and the sphere -> surface mapping on a synthetic surface.

What is stored: alpha spheres, not surface points. A sphere is a centre and a
radius, so it is independent of any particular surface sampling - the same
file serves every resampling, sup_sampling or apo/holo surface of the receptor,
which a saved point index would not (labels.py's warning). The mapping onto a
surface is done when the surface is loaded (`surface_pockets`).

The mapping. An alpha sphere touches four atom CENTRES and contains none, so
its boundary runs through the lining atoms' centres, and the dMaSIF surface,
which sits roughly one van der Waals radius out from the atoms, lies INSIDE the
sphere where it lines the pocket. A surface point therefore belongs to pocket k
when, for some alpha sphere s of k,

    |x - c_s| <= r_s + margin

`margin` (A, default 0) widens it: a positive margin takes in the pocket's rim,
a negative one only the deepest lining. `score` sweeps it.

Two ways to use the pockets in training (EncoderConfig.region_source):

    external      fpocket's pockets ARE the pocket regions, in fpocket's rank
                  order (or by druggability / volume, --pocket_rank), top k_pos
                  of them. --pocket_shape picks each region's shape:
                    pocket  the surface points the pocket's spheres cover
                    ball    a selector-shaped ball (--ball_radius) centred on
                            those points, the same object the fallback anchor
                            and ball_R8 produce
    external_map  a per-point map (0 outside every pocket, 0.5..1 inside, the
                  better-ranked pocket higher) handed to the existing selector
                  (--region_method etc.) in place of sigmoid(pocket_logit).
                  Candidates are thresholded at --threshold, not by top
                  fraction: a map that is 0 on 97% of the surface has no
                  meaningful 95th percentile.

Either way the pocket head is still computed (and its BCE still trains it if
--lam_pocket > 0, and its AUROC is still reported), it just no longer decides
where the regions are; pooling inside a region is a plain mean rather than
probability-weighted, so no gradient reaches the head through the weights.

The RATE at which external pockets miss the ligand is what `fb` reports in
training: the fallback ball is used exactly when no fpocket region clears the
positive rule.

Building fpocket: `bash src/slurm/build_fpocket.sh` on the login node, once.
It clones fpocket 4.2.3 into $WORK/tools/fpocket-src, builds it inside the
container (so the binary links against the libc it runs with) and runs it on
fpocket's own 1ATP sample through parse_fpocket as an end-to-end check.

Example:
    # container, CPU is enough (no KeOps, no GPU); src/slurm/fpocket.sbatch
    # runs this as a 16-task array
    python src/external_pockets.py run --items out/runs/v1/items.tsv \\
        --out data/pockets/fpocket --fpocket $WORK/tools/fpocket-src/bin/fpocket \\
        --shard 0/16
    # CPU venv
    python src/external_pockets.py score --items out/runs/v1/items.tsv \\
        --surface_cache data/surfaces --pockets data/pockets/fpocket --n 500 \\
        --out data/pockets/fpocket_score.tsv
    # training: src/slurm/train_fpocket.sbatch, i.e. train.sbatch plus
    #   --region_source external --pockets data/pockets/fpocket

Depends on: numpy; extract, ids (run only); test_encoder.read_sdf_heavy and
            train_data.read_items (score only, both torch-free).
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

POCKET_KEYS = ("sphere_xyz", "sphere_r", "sphere_pocket", "pocket_score",
               "pocket_drug", "pocket_volume")
RANKS = ("score", "drug", "volume")
SHAPES = ("pocket", "ball")
SOURCES = ("predicted", "external", "external_map")
SELECTS = ("topk", "site")


# ---------------------------------------------------------------------------
# fpocket: input, invocation, output
# ---------------------------------------------------------------------------
_CHAIN_CHARS = ("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789")
_ELEMENT = {0: "C", 1: "H", 2: "O", 3: "N", 4: "S", 5: "SE"}   # extract.ELE2NUM


def write_pdb(prot, path):
    """load_protein_atoms() output -> a plain PDB of ATOM records.

    Chain labels are assembly labels such as "1.A" and "2.A", which a PDB's one
    character cannot hold, so each distinct label gets its own character in
    order of appearance. fpocket only uses chains to group residues, so all
    that matters is that two labels never share one. Everything is written as
    ATOM, MSE included: fpocket treats HETATM as a ligand candidate.
    """
    labels = list(dict.fromkeys(prot["atom_chain"].tolist()))
    if len(labels) > len(_CHAIN_CHARS):
        print(f"[warn] {path}: {len(labels)} chains; chain ids repeat after "
              f"{len(_CHAIN_CHARS)}, so fpocket may merge residues across them")
    cid = {lab: _CHAIN_CHARS[i % len(_CHAIN_CHARS)] for i, lab in enumerate(labels)}
    span = np.abs(prot["atom_xyz"]).max() if len(prot["atom_xyz"]) else 0.0
    if not span < 1000.0:
        # %8.3f overflows its 8 columns and shifts every later field
        raise ValueError(f"{path}: coordinate {span:.1f} does not fit a PDB column")
    lines = []
    for i, (x, t, ch, rn, ic, res, name) in enumerate(zip(
            prot["atom_xyz"], prot["atom_type"], prot["atom_chain"],
            prot["atom_resnum"], prot["atom_icode"], prot["atom_resname"],
            prot["atom_name"])):
        el = _ELEMENT[int(t)]
        name = str(name)[:4]
        # PDB convention: a 1-letter element's atom name starts in column 14
        nm = f" {name:<3s}" if len(name) < 4 and len(el) == 1 else f"{name:<4s}"
        lines.append(
            f"ATOM  {(i + 1) % 100000:5d} {nm}{' '}{str(res)[:3]:>3s} {cid[ch]}"
            f"{int(rn) % 10000:4d}{(str(ic) or ' ')[:1]}   "
            f"{x[0]:8.3f}{x[1]:8.3f}{x[2]:8.3f}{1.0:6.2f}{0.0:6.2f}"
            f"          {el:>2s}")
    lines.append("END")
    Path(path).write_text("\n".join(lines) + "\n")


def parse_pqr(path):
    """fpocket's <name>_pockets.pqr -> (xyz (S, 3), radius (S,), pocket (S,) 0-based).

    Coordinates by fixed column, as in a PDB: a large negative coordinate runs
    into its neighbour ("-100.123-200.456") and a whitespace split breaks. The
    residue number is the 1-based pocket id; charge and radius follow the
    coordinates, whitespace separated.
    """
    xyz, rad, pk = [], [], []
    for ln in Path(path).read_text().splitlines():
        if not ln.startswith(("ATOM", "HETATM")):
            continue
        xyz.append([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])])
        rest = ln[54:].split()
        rad.append(float(rest[1]))
        pk.append(int(ln[22:26]) - 1)
    return (np.asarray(xyz, np.float32).reshape(-1, 3), np.asarray(rad, np.float32),
            np.asarray(pk, np.int32))


def parse_info(path):
    """fpocket's <name>_info.txt -> {1-based pocket id: {field: value}}."""
    out, cur = {}, None
    for ln in Path(path).read_text().splitlines():
        s = ln.strip()
        if s.startswith("Pocket") and s.endswith(":"):
            cur = out.setdefault(int(s.split()[1]), {})
        elif cur is not None and ":" in s:
            k, v = s.split(":", 1)
            try:
                cur[k.strip()] = float(v.strip())
            except ValueError:
                pass
    return out


def parse_fpocket(out_dir, name):
    """An fpocket output directory -> the arrays of POCKET_KEYS.

    Pockets keep fpocket's own numbering, which IS its ranking (pocket 1 is
    the best by fpocket's score), as 0-based indices.
    """
    out_dir = Path(out_dir)
    xyz, rad, pk = parse_pqr(out_dir / f"{name}_pockets.pqr")
    info = parse_info(out_dir / f"{name}_info.txt")
    n = max(int(pk.max()) + 1 if len(pk) else 0, max(info, default=0))
    col = lambda key: np.asarray([info.get(i + 1, {}).get(key, np.nan)
                                  for i in range(n)], np.float32)
    return {"sphere_xyz": xyz, "sphere_r": rad, "sphere_pocket": pk,
            "pocket_score": col("Score"), "pocket_drug": col("Druggability Score"),
            "pocket_volume": col("Volume")}


def run_fpocket(prot, fpocket, workdir, args=(), timeout=600):
    """Atoms -> fpocket's pockets (the arrays of POCKET_KEYS), in a temp dir."""
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="fp_", dir=workdir))
    try:
        pdb = tmp / "rec.pdb"
        write_pdb(prot, pdb)
        res = subprocess.run([str(fpocket), "-f", str(pdb), *args], cwd=tmp,
                             capture_output=True, text=True, timeout=timeout)
        out = tmp / "rec_out"
        if res.returncode != 0 or not (out / "rec_pockets.pqr").exists():
            tail = (res.stdout + res.stderr).strip().splitlines()[-3:]
            raise RuntimeError(f"fpocket exit {res.returncode}: {' | '.join(tail)}")
        return parse_fpocket(out, "rec")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def save_pockets(path, arrays):
    """Write-then-rename through a file handle, as SurfaceStore.write does
    (np.savez on a path appends .npz to a name that lacks it)."""
    path = Path(path)
    tmp = path.with_suffix(".npz.tmp")
    with open(tmp, "wb") as fh:
        np.savez(fh, **{k: arrays[k] for k in POCKET_KEYS})
    tmp.replace(path)


def load_pockets(path):
    with np.load(path, allow_pickle=False) as d:
        return {k: d[k] for k in POCKET_KEYS}


# ---------------------------------------------------------------------------
# pockets -> surface regions
# ---------------------------------------------------------------------------
def pocket_order(pk, rank="score"):
    """Pocket indices, best first. `score` is fpocket's own order."""
    n = len(pk["pocket_score"])
    if rank == "score":
        return np.arange(n)
    key = {"drug": "pocket_drug", "volume": "pocket_volume"}.get(rank)
    if key is None:
        raise ValueError(f"unknown pocket rank {rank!r}; expected one of {RANKS}")
    v = np.nan_to_num(pk[key], nan=-np.inf)
    return np.argsort(-v, kind="stable")      # ties keep fpocket's order


def pocket_points(xyz, centres, radii, margin=0.0):
    """Indices of surface points within (r_s + margin) of any of these spheres.

    Box-filtered first: a pocket spans ~20 A of a ~37k-point surface, so the
    exact distances only run on the few thousand points near it. Per pocket,
    so this costs O(N) comparisons per pocket, not O(N x spheres).
    """
    if not len(centres):
        return np.zeros(0, np.int64)
    reach = radii + margin
    lo = (centres - reach[:, None]).min(0)
    hi = (centres + reach[:, None]).max(0)
    near = np.nonzero(((xyz >= lo) & (xyz <= hi)).all(1))[0]
    if not len(near):
        return near.astype(np.int64)
    inside = np.zeros(len(near), bool)
    pts = xyz[near]
    for s in range(0, len(centres), 256):
        c, r = centres[s:s + 256], reach[s:s + 256]
        d2 = ((pts[:, None, :] - c[None]) ** 2).sum(-1)
        inside |= (d2 <= (np.maximum(r, 0) ** 2)[None]).any(1)
    return near[inside].astype(np.int64)


def surface_pockets(xyz, pk, k=None, rank="score", margin=0.0, shape="pocket",
                    ball_radius=8.0, min_points=20, dup_overlap=0.5):
    """External pockets on one surface -> (regions, point_map).

    regions    list of int64 index arrays into xyz, best pocket first, at most
               k of them (None = all). A pocket with fewer than min_points
               surface points is dropped, as the selectors drop small regions;
               a region more than dup_overlap inside an earlier one is
               dropped as a duplicate (balls of neighbouring sub-pockets).
    point_map  (N,) float32: 0 outside every kept pocket, inside one
               0.5 + 0.5 * (1 - rank / n_kept), so the best pocket reads 1.0
               and every pocket point stays above 0.5 - the map the
               `external_map` source hands to the selector.

    shape "pocket" is the covered points; "ball" a ball of ball_radius at their
    centroid (train_data.fallback_region's construction).
    """
    if shape not in SHAPES:
        raise ValueError(f"unknown pocket shape {shape!r}; expected one of {SHAPES}")
    xyz = np.asarray(xyz, np.float32)
    regions = []
    taken = np.zeros(len(xyz), bool)
    for p in pocket_order(pk, rank):
        if k is not None and len(regions) >= k:
            break
        m = pk["sphere_pocket"] == p
        r = pocket_points(xyz, pk["sphere_xyz"][m], pk["sphere_r"][m], margin)
        if len(r) < min_points:
            continue
        if shape == "ball":
            c = xyz[r].mean(0)
            r = np.nonzero(((xyz - c) ** 2).sum(-1) < ball_radius ** 2)[0]
            if len(r) < min_points:
                continue
        if taken[r].mean() > dup_overlap:
            continue
        taken[r] = True
        regions.append(r.astype(np.int64))
    pmap = np.zeros(len(xyz), np.float32)
    n = len(regions)
    for i in reversed(range(n)):              # better pockets written last, win
        pmap[regions[i]] = 0.5 + 0.5 * (1.0 - i / n)
    return regions, pmap


def site_pockets(pk, lig, cutoff=3.0, frac=0.25, lig_frac=0.5):
    """Pocket ids fpocket found AT this ligand's binding site -> int array.

    Alpha-sphere centres sit in the void a bound ligand fills, so a pocket is
    at the site when the ligand occupies it:
        >= frac of its sphere centres lie within `cutoff` A of a ligand heavy
        atom, or
        >= lig_frac of the ligand's heavy atoms lie within `cutoff` A of its
        sphere centres (a big pocket the ligand fills only part of).
    Several pockets can qualify when fpocket splits one site; the caller
    takes their union. Uses the ligand's coordinates only, never the surface.
    On fpocket's 1ATP sample: the ATP pocket has 0.97 of its centres within
    3 A of the ligand, the next pocket 0.00.
    """
    lig = np.asarray(lig, np.float32)
    n = len(pk["pocket_score"])
    if not len(lig) or not len(pk["sphere_xyz"]) or not n:
        return np.zeros(0, np.int64)
    d = np.sqrt(((pk["sphere_xyz"][:, None] - lig[None]) ** 2).sum(-1))   # (S, L)
    near = d < cutoff
    out = []
    for p in range(n):
        m = pk["sphere_pocket"] == p
        if not m.any():
            continue
        if near[m].any(1).mean() >= frac or near[m].any(0).mean() >= lig_frac:
            out.append(p)
    return np.asarray(out, np.int64)


def site_region(xyz, pk, lig, margin=0.0, shape="pocket", ball_radius=8.0,
                min_points=20, cutoff=3.0, frac=0.25, lig_frac=0.5):
    """The surface region of the pocket(s) at this ligand's site -> (indices or
    None, number of fpocket pockets merged).

    The union of every site pocket's covered points (`pocket_points`), or a
    ball at that union's centroid for shape "ball". None when fpocket found no
    pocket at the site, or it maps onto fewer than min_points points.
    """
    if shape not in SHAPES:
        raise ValueError(f"unknown pocket shape {shape!r}; expected one of {SHAPES}")
    xyz = np.asarray(xyz, np.float32)
    ids = site_pockets(pk, lig, cutoff, frac, lig_frac)
    if not len(ids):
        return None, 0
    m = np.isin(pk["sphere_pocket"], ids)
    r = pocket_points(xyz, pk["sphere_xyz"][m], pk["sphere_r"][m], margin)
    if shape == "ball" and len(r):
        c = xyz[r].mean(0)
        r = np.nonzero(((xyz - c) ** 2).sum(-1) < ball_radius ** 2)[0]
    if len(r) < min_points:
        return None, len(ids)
    return r.astype(np.int64), len(ids)


class PocketSource:
    """A directory of <input_id>.npz pocket files, mapped onto surfaces.

    SurfaceStore holds one of these and calls `map` whenever it builds a RAM
    entry, so the mapping is done once per surface load and lives as long as
    the surface does. A receptor with no file (fpocket never ran on it, or
    failed) gets NO external regions rather than an exception: training then
    leans on the fallback ball for it and evaluation drops it, and the
    `missing` count is reported with the store's statistics.

    select:
      topk  the receptor's best k pockets, the same for every row of it. What
            a screen sees: no ligand, so the score is a max over them.
      site  per ROW, the pocket(s) at that row's own ligand (`site_region`),
            as the single pocket region. The known-site setting (DrugCLIP's):
            training is told WHERE the ligand binds and learns only what the
            pocket there looks like. Needs the ligand, so it is resolved in
            train.build_batch via `site`, not here; the entry keeps the raw
            spheres (`ext_raw`) for it. Evaluate a site-trained model blind
            with `train.py eval --pocket_select topk`.
    """

    def __init__(self, directory, k=None, rank="score", margin=0.0, shape="pocket",
                 ball_radius=8.0, min_points=20, select="topk", site_cutoff=3.0,
                 site_frac=0.25, site_lig_frac=0.5):
        self.dir = Path(directory)
        if not self.dir.is_dir():
            raise SystemExit(f"--pockets {directory}: no such directory; run "
                             "`external_pockets.py run` first")
        if select not in SELECTS:
            raise ValueError(f"unknown pocket select {select!r}; expected one of {SELECTS}")
        self.kw = dict(k=k, rank=rank, margin=margin, shape=shape,
                       ball_radius=ball_radius, min_points=min_points)
        self.select = select
        self.site_kw = dict(cutoff=site_cutoff, frac=site_frac, lig_frac=site_lig_frac)
        self.stats = {"mapped": 0, "missing": 0, "empty": 0}
        self.site_stats = {"rows": 0, "found": 0, "merged": 0}
        self._warned = 0

    def describe(self):
        kw = self.kw
        sk = self.site_kw
        how = (f"the pocket at each row's ligand (site: >= {sk['frac']:g} of its "
               f"alpha spheres or >= {sk['lig_frac']:g} of the ligand within "
               f"{sk['cutoff']:g} A)" if self.select == "site"
               else f"top {kw['k'] or 'all'} by {kw['rank']}")
        return (f"external pockets from {self.dir}: {how}, shape {kw['shape']}"
                + (f" R={kw['ball_radius']:g}" if kw["shape"] == "ball" else "")
                + f", margin {kw['margin']:g} A, min {kw['min_points']} points")

    def map_entry(self, input_id, xyz):
        """-> the SurfaceStore entry fields, numpy (the store makes the first
        two torch): ext_regions (list of index arrays), ext_map ((N,)),
        ext_raw (the pocket arrays, or None)."""
        path = self.dir / f"{input_id}.npz"
        raw = load_pockets(path) if path.exists() else None
        regions, pmap = self.map(input_id, xyz, raw)
        return {"ext_regions": regions, "ext_map": pmap, "ext_raw": raw}

    def site(self, xyz, raw, lig):
        """This row's site region as a one-element list (or [] if fpocket found
        nothing there) - the `site` select."""
        self.site_stats["rows"] += 1
        if raw is None:
            return []
        kw = self.kw
        r, n = site_region(np.asarray(xyz, np.float32), raw, lig, kw["margin"],
                           kw["shape"], kw["ball_radius"], kw["min_points"],
                           **self.site_kw)
        if r is None:
            return []
        self.site_stats["found"] += 1
        self.site_stats["merged"] += n > 1
        return [r]

    def map(self, input_id, xyz, raw=None):
        path = self.dir / f"{input_id}.npz"
        if raw is not None:
            regions, pmap = surface_pockets(xyz, raw, **self.kw)
            self.stats["mapped"] += 1
            self.stats["empty"] += not regions
            return regions, pmap
        if not path.exists():
            self.stats["missing"] += 1
            if self._warned < 5:
                self._warned += 1
                print(f"[warn] no external pockets for {input_id} ({path}); it "
                      "gets no pocket regions" + (" (further warnings suppressed)"
                                                  if self._warned == 5 else ""))
            return [], np.zeros(len(xyz), np.float32)
        regions, pmap = surface_pockets(xyz, load_pockets(path), **self.kw)
        self.stats["mapped"] += 1
        self.stats["empty"] += not regions
        return regions, pmap

    def report(self):
        s = self.stats
        out = (f"external pockets: {s['mapped']} mapped ({s['empty']} with no "
               f"usable pocket), {s['missing']} without a file")
        t = self.site_stats
        if self.select == "site" and t["rows"]:
            out += (f"; site found for {t['found']}/{t['rows']} rows "
                    f"({t['found'] / t['rows']:.0%}), {t['merged']} merged "
                    "from several fpocket pockets")
        return out


# ---------------------------------------------------------------------------
# run: fpocket over every receptor of an item table
# ---------------------------------------------------------------------------
def run(a):
    from extract import load_protein_atoms
    from ids import parse_chains
    from train_data import ItemTable, read_items

    fpocket = Path(a.fpocket or os.environ.get("FPOCKET", "fpocket"))
    if not (fpocket.exists() or shutil.which(str(fpocket))):
        raise SystemExit(f"fpocket binary {fpocket} not found; see the module "
                         "docstring for the two-line build, or pass --fpocket")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    table = ItemTable(read_items(a.items, a.limit))
    # One row per surface, in the same fixed order as `train.py cache`, so
    # shards are disjoint and a rerun of shard i covers the same receptors.
    firsts = sorted(table.siblings.items())
    tag = ""
    if a.shard:
        i, n = (int(x) for x in a.shard.split("/"))
        firsts = firsts[i::n]
        tag = f"_{i}of{n}"
        print(f"shard {i}/{n}: {len(firsts)} receptors")
    work = Path(a.workdir or os.environ.get("TMPDIR", "/tmp"))
    extra = a.fpocket_args.split() if a.fpocket_args else []
    t0, built, failed, n_pk = time.time(), 0, [], []
    for j, (input_id, rows) in enumerate(firsts, 1):
        dest = out / f"{input_id}.npz"
        if dest.exists() and not a.overwrite:
            continue
        row = table.rows[rows[0]]
        try:
            prot = load_protein_atoms(row["input_path"], parse_chains(row["input_chains"]),
                                      keep_hydrogens=False)
            arrays = run_fpocket(prot, fpocket, work, extra, a.timeout)
            save_pockets(dest, arrays)
            built += 1
            n_pk.append(len(arrays["pocket_score"]))
        except Exception as e:
            failed.append((input_id, f"{type(e).__name__}: {e}"))
        if j % 50 == 0 or j == len(firsts):
            print(f"  {j}/{len(firsts)}  {built} built  {len(failed)} failed  "
                  f"{(time.time() - t0) / j:.2f}s/receptor"
                  + (f"  median {int(np.median(n_pk))} pockets" if n_pk else ""),
                  flush=True)
    print(f"\n{built} pocket files written to {out}, {len(failed)} failed")
    if failed:
        log = out / f"failed{tag}.tsv"
        log.write_text("".join(f"{k}\t{m}\n" for k, m in failed))
        for k, m in failed[:10]:
            print(f"    {k}: {m[:160]}")
        print(f"all failures: {log}")


# ---------------------------------------------------------------------------
# score: external pockets as regions, against the labelled pocket
# ---------------------------------------------------------------------------
METRICS = ["n", "size", "iou", "cover", "prec", "hit", "top1", "iou1",
           "dcc1", "dccA", "on", "share"]


def region_metrics(regions, xyz, pocket, lig, k, min_cover=0.5, min_prec=0.3, dcc=4.0):
    """region_sweep.score's definitions, with uniform weights (external regions
    are pooled as plain means, so their centre is the plain centroid)."""
    regs = regions[:k]
    st = dict.fromkeys(METRICS, 0.0)
    st.update(n=len(regs), size=np.nan)
    if not regs:
        return st
    n_pocket = float(pocket.sum())
    sizes = np.asarray([len(r) for r in regs], float)
    inter = np.asarray([float(pocket[r].sum()) for r in regs])
    iou = inter / (sizes + n_pocket - inter)
    cover, prec = inter / max(n_pocket, 1.0), inter / sizes
    ok = (cover >= min_cover) & (prec >= min_prec)
    centre = np.stack([xyz[r].mean(0) for r in regs])
    d = np.linalg.norm(centre - lig.mean(0), axis=1)
    b = int(np.argmax(iou))
    st.update(size=float(sizes.mean()), iou=float(iou[b]), cover=float(cover[b]),
              prec=float(prec[b]), hit=float(ok.any()), top1=float(ok[0]),
              iou1=float(iou[0]), dcc1=float(d[0] < dcc), dccA=float((d < dcc).any()),
              on=float((prec >= min_prec).sum()))
    return st


SITE_METRICS = ["found", "size", "iou", "cover", "prec", "dcc", "merged"]


def site_metrics(r, n_merged, xyz, pocket, lig, dcc=4.0):
    """The site region against the labelled pocket. `dcc`: its centroid within
    4 A of the ligand's; `merged`: fpocket split the site into several."""
    st = {"found": float(r is not None), "merged": float(n_merged > 1)}
    if r is None:
        return st
    inter = float(pocket[r].sum())
    n_pocket = float(pocket.sum())
    st.update(size=float(len(r)), iou=inter / (len(r) + n_pocket - inter),
              cover=inter / max(n_pocket, 1.0), prec=inter / len(r),
              dcc=float(np.linalg.norm(xyz[r].mean(0) - lig.mean(0)) < dcc))
    return st


def score(a):
    from test_encoder import read_sdf_heavy
    from train_data import read_items

    rows = read_items(a.items)
    if a.split:
        rows = [r for r in rows if r["split"] in a.split.split(",")]
    surf_dir, pk_dir = Path(a.surface_cache), Path(a.pockets)
    have = [r for r in rows if (surf_dir / f"{r['input_id']}.npz").exists()
            and (pk_dir / f"{r['input_id']}.npz").exists()]
    print(f"{len(rows)} rows; {len(have)} have both a cached surface and a pocket file")
    if not have:
        raise SystemExit("nothing to score")
    rng = np.random.default_rng(a.seed)
    if a.n and len(have) > a.n:
        have = [have[i] for i in np.sort(rng.choice(len(have), a.n, replace=False))]
    margins = [float(x) for x in a.margins.split(",")]
    shapes, ranks = a.shapes.split(","), a.ranks.split(",")
    settings = [(s, r, m) for s in shapes for r in ranks for m in margins]
    per = {st: [] for st in settings}
    site_settings = [(sh, m) for sh in shapes for m in margins]
    per_site = {st: [] for st in site_settings}
    per_item = []
    t0 = time.time()
    for j, row in enumerate(have, 1):
        with np.load(surf_dir / f"{row['input_id']}.npz", allow_pickle=False) as d:
            xyz = d["xyz"].astype(np.float32)
        lig = np.concatenate([read_sdf_heavy(p) for p in row["sdf_paths"].split(",")])
        dmin = np.full(len(xyz), np.inf, np.float32)
        for s in range(0, len(lig), 64):
            dmin = np.minimum(dmin, np.sqrt(
                ((xyz[:, None, :] - lig[None, s:s + 64]) ** 2).sum(-1)).min(1))
        pocket = dmin < a.cutoff
        if not pocket.any():
            continue
        pk = load_pockets(pk_dir / f"{row['input_id']}.npz")
        for st in settings:
            shape, rank, margin = st
            regs, pmap = surface_pockets(xyz, pk, None, rank, margin, shape,
                                         a.ball_radius, a.min_points)
            m = region_metrics(regs, xyz, pocket, lig, a.k_pos, a.min_cover, a.min_prec)
            m["share"] = float((pmap > 0).mean())
            m["n_pockets"] = len(pk["pocket_score"])
            per[st].append(m)
            per_item.append({"system_id": row["system_id"], "shape": shape, "rank": rank,
                             "margin": margin, **m})
        for st in site_settings:
            shape, margin = st
            r, n_merged = site_region(xyz, pk, lig, margin, shape, a.ball_radius,
                                      a.min_points, a.site_cutoff, a.site_frac,
                                      a.site_lig_frac)
            per_site[st].append(site_metrics(r, n_merged, xyz, pocket, lig))
        if j % 100 == 0 or j == len(have):
            print(f"  {j}/{len(have)}  {(time.time() - t0) / j:.2f}s/row", flush=True)

    W = 26
    hdr = f"{'setting':{W}s}" + "".join(f"{m:>7s}" for m in METRICS)
    print(f"\nexternal pockets as regions, top {a.k_pos} per protein, "
          f"{len(next(iter(per.values())))} rows")
    print(hdr)
    lines = ["\t".join(["shape", "rank", "margin", *METRICS])]
    for st, ms in per.items():
        mean = {m: float(np.nanmean([x[m] for x in ms])) for m in METRICS}
        print(f"{f'{st[0]} {st[1]} m={st[2]:g}':{W}s}" + "".join(
            f"{mean[m]:7.2f}" if m not in ("size",) else f"{mean[m]:7.0f}"
            for m in METRICS))
        lines.append("\t".join([st[0], st[1], f"{st[2]:g}"]
                               + [f"{mean[m]:.4f}" for m in METRICS]))
    print("\nthe pocket AT the ligand's site (--pocket_select site, known-site "
          "training): found = fpocket has a pocket there; the rest over found rows")
    site_lines = ["\t".join(["shape", "margin", *SITE_METRICS])]
    print(f"{'setting':{W}s}" + "".join(f"{m:>7s}" for m in SITE_METRICS))
    for st, ms in per_site.items():
        found = [x for x in ms if x["found"]]
        mean = {m: float(np.mean([x[m] for x in (ms if m == "found" else found)]))
                if (found or m == "found") else np.nan for m in SITE_METRICS}
        print(f"{f'site {st[0]} m={st[1]:g}':{W}s}" + "".join(
            f"{mean[m]:7.0f}" if m == "size" else f"{mean[m]:7.2f}" for m in SITE_METRICS))
        site_lines.append("\t".join([st[0], f"{st[1]:g}"]
                                    + [f"{mean[m]:.4f}" for m in SITE_METRICS]))
    n_pk = [x["n_pockets"] for x in next(iter(per.values()))]
    print(f"\nfpocket pockets per receptor: median {np.median(n_pk):.0f}, "
          f"none on {int(np.sum(np.asarray(n_pk) == 0))}")
    print("\nReference, ball_R8 on the head's map (docs/STATUS.md, held-out): "
          "iou 0.53, cover 0.70, prec 0.63, hit 0.84, top1 0.64, dcc1 0.44, "
          "dccA 0.77, size ~290 points.")
    print("hit / dccA are what max-over-anchors scoring needs (the right region "
          "is SOMEWHERE in the top k);\ntop1 / dcc1 matter only if scoring ever "
          "becomes top-1. share = fraction of the surface inside a kept pocket.")
    if a.out:
        Path(a.out).write_text("\n".join(lines) + "\n")
        site_out = Path(a.out).with_suffix(".site.tsv")
        site_out.write_text("\n".join(site_lines) + "\n")
        print(f"\nwrote {a.out} and {site_out}")
    if a.per_item:
        keys = ["system_id", "shape", "rank", "margin", *METRICS, "n_pockets"]
        Path(a.per_item).write_text("\n".join(
            ["\t".join(keys)] + ["\t".join(str(r[k]) for k in keys) for r in per_item])
            + "\n")
        print(f"wrote {a.per_item}")


# ---------------------------------------------------------------------------
# selftest: parsers and mapping, numpy only
# ---------------------------------------------------------------------------
_PQR = """HEADER This is a pqr format file writen by the programm fpocket.
ATOM      1    O STP     1      12.696  10.125   0.572    0.00     4.14
ATOM      2    C STP     1    -100.123-200.456   0.615    0.00     3.50
ATOM      3    O STP     2      15.748   7.295   2.281    0.00     3.68
TER
END
"""
_INFO = """Pocket 1 :
\tScore : \t0.977
\tDruggability Score : \t0.959
\tVolume : \t590.402

Pocket 2 :
\tScore : \t0.127
\tDruggability Score : \t0.990
\tVolume : \t78.453
"""


def selftest(a):
    fails = []

    def check(name, ok, detail=""):
        print(f"  {'ok  ' if ok else 'FAIL'} {name}" + (f"   {detail}" if detail else ""))
        if not ok:
            fails.append(name)

    tmp = Path(tempfile.mkdtemp())
    try:
        print("[1] fpocket output parsing")
        (tmp / "x_pockets.pqr").write_text(_PQR)
        (tmp / "x_info.txt").write_text(_INFO)
        pk = parse_fpocket(tmp, "x")
        check("3 spheres, pockets 0/0/1", pk["sphere_pocket"].tolist() == [0, 0, 1])
        check("fixed-column coordinates survive run-together negatives",
              np.allclose(pk["sphere_xyz"][1], [-100.123, -200.456, 0.615]))
        check("radius column", np.allclose(pk["sphere_r"], [4.14, 3.50, 3.68]))
        check("scores", np.allclose(pk["pocket_score"], [0.977, 0.127])
              and np.allclose(pk["pocket_drug"], [0.959, 0.990]))
        check("rank by drug puts pocket 2 first",
              pocket_order(pk, "drug").tolist() == [1, 0])
        save_pockets(tmp / "r.npz", pk)
        back = load_pockets(tmp / "r.npz")
        check("npz round trip", all(np.array_equal(back[k], pk[k]) for k in POCKET_KEYS))

        print("[2] write_pdb: fixed columns fpocket can read")
        prot = {"atom_xyz": np.asarray([[1.0, -22.5, 333.25], [-999.5, 0, 0]], np.float32),
                "atom_type": np.asarray([0, 5]), "atom_chain": np.asarray(["1.A", "2.A"]),
                "atom_resnum": np.asarray([5, 12345]), "atom_icode": np.asarray(["", "B"]),
                "atom_resname": np.asarray(["ALA", "MSE"]),
                "atom_name": np.asarray(["CA", "SE"])}
        write_pdb(prot, tmp / "p.pdb")
        ln = (tmp / "p.pdb").read_text().splitlines()
        check("x/y/z in columns 31-54",
              float(ln[0][30:38]) == 1.0 and float(ln[0][38:46]) == -22.5
              and float(ln[0][46:54]) == 333.25 and float(ln[1][30:38]) == -999.5)
        check("distinct chain characters for distinct labels", ln[0][21] != ln[1][21])
        check("element columns 77-78", ln[0][76:78] == " C" and ln[1][76:78] == "SE")
        check("residue number wraps into 4 columns", ln[1][22:26] == "2345")

        print("[3] spheres -> surface points")
        g = np.random.default_rng(0)
        v = g.normal(size=(4000, 3))
        xyz = (20 * v / np.linalg.norm(v, axis=1, keepdims=True)).astype(np.float32)
        # two pockets: spheres just outside the surface at two poles, the
        # second bigger, so 'volume' ranks it first
        c1 = np.asarray([[0, 0, 22.0], [1.5, 0, 22.0]], np.float32)
        c2 = np.asarray([[0, 0, -23.0], [0, 2.0, -23.0], [2.0, 0, -23.0]], np.float32)
        pk = {"sphere_xyz": np.concatenate([c1, c2]),
              "sphere_r": np.full(5, 6.0, np.float32),
              "sphere_pocket": np.asarray([0, 0, 1, 1, 1], np.int32),
              "pocket_score": np.asarray([0.9, 0.5], np.float32),
              "pocket_drug": np.asarray([0.1, 0.8], np.float32),
              "pocket_volume": np.asarray([100.0, 300.0], np.float32)}
        brute = lambda cs, m: np.nonzero(
            (np.linalg.norm(xyz[:, None] - cs[None], axis=-1) <= 6.0 + m).any(1))[0]
        for m in (0.0, 2.0, -1.0):
            got = pocket_points(xyz, c1, np.full(2, 6.0, np.float32), m)
            check(f"box-filtered == brute force at margin {m:g}",
                  np.array_equal(np.sort(got), brute(c1, m)), f"{len(got)} points")
        regs, pmap = surface_pockets(xyz, pk, min_points=5)
        check("two regions in score order, north pole first",
              len(regs) == 2 and xyz[regs[0]][:, 2].mean() > 0
              and xyz[regs[1]][:, 2].mean() < 0)
        check("map: best pocket 1.0, second 0.75, rest 0",
              np.allclose(pmap[regs[0]], 1.0) and np.allclose(pmap[regs[1]], 0.75)
              and (pmap > 0).sum() == len(regs[0]) + len(regs[1]))
        regs_v, _ = surface_pockets(xyz, pk, rank="volume", min_points=5)
        check("rank volume puts the south pocket first", xyz[regs_v[0]][:, 2].mean() < 0)
        check("k caps the region count", len(surface_pockets(xyz, pk, k=1, min_points=5)[0]) == 1)
        regs_b, _ = surface_pockets(xyz, pk, shape="ball", ball_radius=8.0, min_points=5)
        c = xyz[regs[0]].mean(0)
        check("ball shape = every point within R of the pocket's centroid",
              np.array_equal(regs_b[0], np.nonzero(((xyz - c) ** 2).sum(-1) < 64)[0]))
        check("min_points drops small pockets",
              len(surface_pockets(xyz, pk, min_points=10 ** 6)[0]) == 0)
        dup = dict(pk, sphere_pocket=np.asarray([0, 0, 1, 1, 1], np.int32),
                   sphere_xyz=np.concatenate([c1, c1 + 0.1, c1[:1]]))
        check("a near-copy of an earlier pocket is dropped as a duplicate",
              len(surface_pockets(xyz, dup, min_points=5)[0]) == 1)

        print("[4] PocketSource")
        (tmp / "pk").mkdir()
        save_pockets(tmp / "pk" / "rec1.npz", pk)
        src = PocketSource(tmp / "pk", k=8, min_points=5)
        r1, _ = src.map("rec1", xyz)
        r2, m2 = src.map("nope", xyz)
        check("a mapped receptor and a missing one",
              len(r1) == 2 and r2 == [] and not m2.any()
              and src.stats == {"mapped": 1, "missing": 1, "empty": 0})

        print("[5] the pocket at a ligand's site")
        lig_n = np.asarray([[0, 0, 21.5], [1.0, 0, 21.0], [0, 1.0, 21.5]], np.float32)
        check("a ligand in the north pocket selects pocket 0 only",
              site_pockets(pk, lig_n).tolist() == [0])
        far = np.asarray([[30.0, 0, 0]], np.float32)
        check("a ligand nowhere near a pocket selects none",
              site_pockets(pk, far).tolist() == [] and site_region(xyz, pk, far)[0] is None)
        both = np.concatenate([lig_n, [[0, 0, -22.5], [0, 1.5, -22.5]]]).astype(np.float32)
        r_both, n_both = site_region(xyz, pk, both, min_points=5)
        r_n, _ = site_region(xyz, pk, lig_n, min_points=5)
        r_s = surface_pockets(xyz, pk, min_points=5)[0][1]
        check("a ligand spanning two pockets gets their union",
              n_both == 2 and set(r_both.tolist()) == set(r_n.tolist()) | set(r_s.tolist()))
        check("the site region is the pocket's own covered points",
              np.array_equal(np.sort(r_n), np.sort(surface_pockets(xyz, pk, min_points=5)[0][0])))
        big = dict(pk, sphere_xyz=np.concatenate([c1, c2 + [0, 0, 60.0]]))
        check("lig_frac: a pocket holding the ligand counts even if mostly empty",
              0 in site_pockets(big, lig_n[:1], frac=0.9, lig_frac=0.5).tolist())
        src = PocketSource(tmp / "pk", min_points=5, select="site")
        e = src.map_entry("rec1", xyz)
        got = src.site(xyz, e["ext_raw"], lig_n)
        check("PocketSource.site: one region, counted",
              len(got) == 1 and np.array_equal(got[0], r_n)
              and src.site(xyz, e["ext_raw"], far) == []
              and src.site_stats == {"rows": 2, "found": 1, "merged": 0})
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n{'PASS' if not fails else 'FAIL: ' + ', '.join(fails)}")
    sys.exit(1 if fails else 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("run", help="container: fpocket on every receptor")
    p.add_argument("--items", required=True)
    p.add_argument("--out", required=True, help="directory of <input_id>.npz")
    p.add_argument("--fpocket", default="", help="binary; default $FPOCKET or PATH")
    p.add_argument("--fpocket_args", default="",
                   help="extra fpocket flags, e.g. '-m 3.2 -M 6.4 -i 20'")
    p.add_argument("--workdir", default="", help="scratch; default $TMPDIR or /tmp")
    p.add_argument("--timeout", type=int, default=600, help="seconds per receptor")
    p.add_argument("--shard", default="", help="i/n, for an array job")
    p.add_argument("--limit", type=int)
    p.add_argument("--overwrite", action="store_true")

    p = sub.add_parser("score", help="numpy: pockets vs the labelled pocket")
    p.add_argument("--items", required=True)
    p.add_argument("--surface_cache", required=True)
    p.add_argument("--pockets", required=True)
    p.add_argument("--split", default="", help="only these splits (comma list)")
    p.add_argument("--n", type=int, default=500, help="rows to score (0 = all)")
    p.add_argument("--k_pos", type=int, default=8)
    p.add_argument("--margins", default="-1,0,1,2")
    p.add_argument("--shapes", default="pocket,ball")
    p.add_argument("--ranks", default="score,drug")
    p.add_argument("--ball_radius", type=float, default=8.0)
    p.add_argument("--min_points", type=int, default=20)
    p.add_argument("--cutoff", type=float, default=4.0)
    p.add_argument("--site_cutoff", type=float, default=3.0)
    p.add_argument("--site_frac", type=float, default=0.25)
    p.add_argument("--site_lig_frac", type=float, default=0.5)
    p.add_argument("--min_cover", type=float, default=0.5)
    p.add_argument("--min_prec", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="", help="TSV of per-setting means")
    p.add_argument("--per_item", default="", help="TSV of every row x setting")

    sub.add_parser("selftest", help="numpy: parsers and mapping")

    a = ap.parse_args()
    {"run": run, "score": score, "selftest": selftest}[a.cmd](a)


if __name__ == "__main__":
    main()
