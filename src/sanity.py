"""Preflight checks. Read-only, no GPU, no torch - runs on a login node.

Answers the questions that are cheap now and expensive after a 100 GB transfer
or a full extraction:

  A  module shadowing      affinity/ sits INSIDE the dMaSIF repo, and
                           add_repo_to_path puts the repo root ahead of it on
                           sys.path. Any affinity module whose name matches a
                           repo module can be silently replaced.
  B  manifest <-> disk     do receptors.txt, structures/ and systems/ agree
                           with the manifest, and how much is actually present
  C  --merge_models        does a receptor.cif hold several MODELs with
                           repeated chain IDs (flag ON), or one MODEL with
                           pre-dotted labels (flag OFF)? Getting this wrong
                           embeds one subunit of a multimer and every pocket
                           completed by a neighbour has a hole in it.
  D  chain specs           do the labels in extract_inputs.txt match what the
                           parser yields? A mismatch rejects every chain,
                           raises "No protein atoms found", and prints [skip] -
                           so a run that dropped most of its input looks fine.
  E  ligand source         what is actually in ligand_files/
  F  FRAME                 THE GATE. Is the ligand in the same coordinate frame
                           as the receptor? If not, every label is garbage and
                           nothing downstream means anything.

Example:
    python affinity/sanity.py --root /mnt/home/cjs2301/dmasif
    python affinity/sanity.py --root /mnt/home/cjs2301/dmasif --n 20
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

PASS, WARN, FAIL, INFO = "PASS", "WARN", "FAIL", "    "
_counts = {PASS: 0, WARN: 0, FAIL: 0}


def say(level, msg):
    if level in _counts:
        _counts[level] += 1
    print(f"  [{level}] {msg}" if level in _counts else f"        {msg}")


def section(name):
    print(f"\n{'=' * 70}\n{name}\n{'=' * 70}")


# ----------------------------------------------------------------------------
def check_modules(repo):
    section("A. module shadowing")
    repo_mods = {p.stem for p in repo.glob("*.py")}
    mine = {p.stem for p in HERE.glob("*.py")}
    clash = sorted(repo_mods & mine - {"sanity"})
    say(INFO, f"repo root  {repo}")
    say(INFO, f"our code   {HERE}")
    nested = repo in HERE.parents
    if nested:
        say(WARN, "our code lives INSIDE the vendored clone. Workable, but every "
                  "name below is then reserved, and the clone cannot be pulled "
                  "or committed to cleanly. A sibling directory avoids both.")
        say(INFO, f"reserved: {sorted(repo_mods)}")
    if clash:
        say(FAIL, f"name collision with the repo: {clash}")
        say(INFO, "add_repo_to_path inserts the repo root at sys.path[0], AHEAD of")
        say(INFO, "our directory, so these resolve to our copy only while it is")
        say(INFO, "already cached in sys.modules. Rename them (data -> datasets).")
    else:
        say(PASS, "no module of ours shares a name with a repo module")

    # Mirror extract.py: parents[1] of its own location, then "dMaSIF"
    default_repo = HERE.parent / "dMaSIF"
    if default_repo.resolve() == repo.resolve():
        say(PASS, f"extract.py --repo default resolves to the repo root")
    else:
        say(WARN, f"--repo default is {default_repo}, not {repo}; pass --repo explicitly")

    ckpt = repo / "models" / "dMaSIF_search_3layer_12A_16dim"
    say(PASS if ckpt.exists() else FAIL, f"checkpoint {'found' if ckpt.exists() else 'MISSING'}: {ckpt}")


def check_manifest(root, meta, structures, systems, feats):
    section("B. manifest vs disk")
    import pandas as pd
    from ids import read_receptors, surface_tag, parse_chains, system_cif

    mf = meta / "manifest.parquet"
    if not mf.exists():
        say(FAIL, f"no manifest at {mf}")
        return None, None
    df = pd.read_parquet(mf)
    say(INFO, f"manifest: {len(df)} rows, {df['receptor_key'].nunique()} receptors")
    if "split" in df.columns:
        say(INFO, f"splits:   {df['split'].value_counts().to_dict()}")
    for c in ["system_id", "receptor_key", "split", "ligand_ccd_code", "fp_row"]:
        if c not in df.columns:
            say(WARN, f"manifest lacks column {c}")

    tags = read_receptors(meta / "receptors.txt")
    say(INFO, f"receptors.txt: {len(tags)} receptors")
    missing_tag = set(df["receptor_key"]) - set(tags)
    if missing_tag:
        say(FAIL, f"{len(missing_tag)} manifest receptors absent from receptors.txt")
    else:
        say(PASS, "every manifest receptor appears in receptors.txt")

    have_struct = {p.stem for p in structures.glob("*.cif")} if structures.is_dir() else set()
    say(INFO, f"structures/: {len(have_struct)} .cif")
    frac = len(have_struct & set(tags)) / max(len(tags), 1)
    say(PASS if frac > 0.99 else WARN,
        f"{frac:.1%} of receptors have a structure "
        f"({len(set(tags) - have_struct)} missing)")

    have_feats = {p.stem for p in feats.glob("*.npz")} if feats.is_dir() else set()
    say(INFO, f"feats/: {len(have_feats)} .npz")
    want_tags = set(tags.values())
    if have_feats:
        f2 = len(have_feats & want_tags) / max(len(want_tags), 1)
        say(PASS if f2 > 0.99 else WARN, f"{f2:.1%} of receptors have a surface")
        stray = have_feats - want_tags
        if stray:
            say(WARN, f"{len(stray)} npz do not match any receptor tag, e.g. "
                      f"{sorted(stray)[:3]} - directory-mode extraction?")
    else:
        say(WARN, "no surfaces extracted yet")

    # Keep this small: stat() on a GPFS directory of 300k entries is not free.
    probe = df["system_id"].head(100)
    n_sys = sum(1 for s in probe if system_cif(systems, s))
    say(PASS if n_sys else FAIL,
        f"{n_sys}/{len(probe)} sampled systems resolve under {systems}")
    return df, tags


def check_models(structures, n):
    section("C. --merge_models")
    from Bio.PDB import MMCIFParser

    files = sorted(structures.glob("*.cif"))[:n]
    if not files:
        say(FAIL, f"no .cif in {structures}")
        return
    parser = MMCIFParser(QUIET=True)
    multi_model, dotted = 0, 0
    for f in files:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            st = parser.get_structure(f.stem, str(f))
        models = list(st)
        chains = [c.id for c in models[0]]
        multi_model += len(models) > 1
        dotted += any("." in c for c in chains)
        say(INFO, f"{f.name}: {len(models)} model(s), chains {chains[:6]}")

    if multi_model and not dotted:
        say(FAIL, "several MODELs with plain chain IDs -> extract.py NEEDS "
                  "--merge_models, or you embed one subunit only")
    elif dotted and not multi_model:
        say(PASS, "one MODEL, chain IDs already dotted -> leave --merge_models OFF")
    elif dotted and multi_model:
        say(FAIL, "several MODELs AND dotted chain IDs -> --merge_models would "
                  "produce '1.1.A'; inspect before extracting")
    else:
        say(WARN, "one MODEL, plain chain IDs -> check this matches receptors.txt")


def check_chain_specs(structures, n):
    section("D. chain specs in extract_inputs.txt")
    from Bio.PDB import MMCIFParser
    from ids import parse_chains, surface_tag

    inp = structures / "extract_inputs.txt"
    if not inp.exists():
        say(FAIL, f"no {inp}; run fetch_structures.py")
        return
    lines = [l.split() for l in inp.read_text().splitlines() if l.strip()]
    say(INFO, f"{len(lines)} lines")
    if lines and not Path(lines[0][0]).exists():
        say(WARN, f"paths look relative to another cwd, e.g. {lines[0][0]}")

    parser = MMCIFParser(QUIET=True)
    bad = 0
    for parts in lines[:n]:
        path = Path(parts[0])
        if not path.is_absolute():
            path = structures / path.name
        if not path.exists():
            continue
        want = parse_chains(parts[1]) if len(parts) > 1 else None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            have = {c.id for c in next(iter(parser.get_structure(path.stem, str(path))))}
        hit = want & have if want else have
        if want and not hit:
            bad += 1
            say(FAIL, f"{path.name}: wants {sorted(want)}, file has {sorted(have)[:6]}")
        else:
            say(INFO, f"{path.name}: {len(hit)}/{len(want) if want else len(have)} "
                      f"chains matched -> {surface_tag(path.stem, want)}.npz")
    say(PASS if not bad else FAIL,
        f"{bad} of {min(n, len(lines))} sampled specs match no chain "
        "(each would be a silent [skip])")


def check_ligand_files(systems, df, n):
    section("E. ligand source and chain resolution")
    from ids import ligand_chain, ligand_chains, ligand_sdf_paths

    if "ligand_id" not in df.columns:
        say(FAIL, "manifest has no ligand_id. system_id field 4 names EVERY "
                  "ligand chain, so labelling would merge other ligands' sites "
                  "into the label. Rebuild the manifest with ligand_id.")
        return
    multi = df["system_id"].str.split("__").str[3].str.contains("_")
    say(INFO, f"{int(multi.sum())}/{len(df)} systems ({multi.mean():.0%}) name "
              "more than one ligand chain")
    say(PASS if multi.mean() < 0.01 else WARN,
        "ligand_id is therefore the required key, not system_id field 4")

    # Prefer the multi-chain rows: they are the ones that go wrong.
    pool = df[multi] if multi.any() else df
    seen = agree = 0
    for r in pool.head(n * 20).itertuples():
        d = Path(systems) / r.system_id
        if not d.is_dir():
            continue
        lf = d / "ligand_files"
        contents = sorted(p.name for p in lf.iterdir()) if lf.is_dir() else []
        want = ligand_chain(r.ligand_id)
        chosen = ligand_sdf_paths(systems, r.system_id, want)
        all_chains = ligand_chains(r.system_id)
        ok = bool(chosen) and len(chosen) == 1
        agree += ok
        say(PASS if ok else WARN,
            f"{r.system_id}  ligand_id->{sorted(want)} of {sorted(all_chains)}  "
            f"files={contents[:4]}  -> {[p.name for p in chosen] or 'cif fallback'}")
        seen += 1
        if seen >= n:
            break
    if not seen:
        say(FAIL, f"no system directories found under {systems}")
    else:
        say(PASS if agree == seen else WARN,
            f"{agree}/{seen} resolved to exactly one SDF named for the chain")


def check_frame(structures, systems, df, tags, n):
    section("F. FRAME  (the gate - everything downstream depends on this)")
    from extract import load_protein_atoms
    from ids import ligand_chains, ligand_sdf_paths, parse_chains, system_cif
    from labels import ligand_from_cif, ligand_from_sdf

    rows, checked = df.head(n * 20), 0
    for r in rows.itertuples():
        rec = structures / f"{r.receptor_key}.cif"
        if not rec.exists():
            continue
        chains = ligand_chains(r.system_id)
        try:
            sdfs = ligand_sdf_paths(systems, r.system_id, chains)
            if sdfs:
                src, (name, lig) = "sdf", ligand_from_sdf(sdfs)
                prot_sys = None
            else:
                cif = system_cif(systems, r.system_id)
                if cif is None:
                    continue
                name, lig, prot_sys = ligand_from_cif(cif, chains,
                                                      getattr(r, "ligand_ccd_code", None))
                src = "cif"
            from ids import read_receptors  # noqa
            spec = parse_chains(",".join(r.receptor_key.split("__")[2].split("_")))
            prot = load_protein_atoms(rec, spec)["atom_xyz"]
        except Exception as e:
            say(WARN, f"{r.system_id}: {type(e).__name__}: {str(e)[:90]}")
            continue

        d = np.linalg.norm(prot[:, None, :] - lig[None, :, :], axis=2)
        near = float(d.min())
        n_close = int((d.min(0) < 4.5).sum())
        ok = near < 5.0 and n_close > 0.5 * len(lig)
        say(PASS if ok else FAIL,
            f"{r.system_id} [{src}:{name}] min receptor-ligand {near:.2f} A, "
            f"{n_close}/{len(lig)} ligand atoms in contact")
        if prot_sys is not None and len(prot_sys):
            off = float(np.linalg.norm(prot.mean(0) - prot_sys.mean(0)))
            say(PASS if off < 1.0 else FAIL,
                f"    receptor.cif vs system.cif centroid offset {off:.3f} A")
        checked += 1
        if checked >= n:
            break
    if not checked:
        say(FAIL, "nothing checked - fetch structures and systems first")
    else:
        say(INFO, "A min distance of 2-4 A is a real contact. Tens of A means the "
                  "files are in different frames and labels are meaningless.")


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="e.g. /mnt/home/cjs2301/dmasif")
    ap.add_argument("--repo", help="dMaSIF repo root; default = ../dMaSIF beside src/")
    ap.add_argument("--meta"); ap.add_argument("--structures")
    ap.add_argument("--systems"); ap.add_argument("--feats")
    ap.add_argument("--n", type=int, default=5, help="samples per check")
    a = ap.parse_args()

    root = Path(a.root)
    # Sibling layout: src/ beside dMaSIF/. If src/ still lives INSIDE the clone,
    # fall back to its parent so the shadowing check still has something to
    # compare against - and that check will then fire on any colliding name.
    repo = Path(a.repo) if a.repo else (
        HERE.parent / "dMaSIF" if (HERE.parent / "dMaSIF").is_dir() else HERE.parent)
    meta = Path(a.meta or root / "meta")
    structures = Path(a.structures or root / "structures")
    systems = Path(a.systems or root / "systems")
    feats = Path(a.feats or root / "feats")

    for fn, args in [
        (check_modules, (repo,)),
        (check_manifest, (root, meta, structures, systems, feats)),
        (check_models, (structures, a.n)),
        (check_chain_specs, (structures, a.n)),
    ]:
        try:
            res = fn(*args)
        except Exception as e:
            say(FAIL, f"{fn.__name__} crashed: {type(e).__name__}: {e}")
            res = None
        if fn is check_manifest:
            df, tags = res if res else (None, None)

    if df is not None:
        for fn, args in [(check_ligand_files, (systems, df, a.n)),
                         (check_frame, (structures, systems, df, tags, a.n))]:
            try:
                fn(*args)
            except Exception as e:
                say(FAIL, f"{fn.__name__} crashed: {type(e).__name__}: {e}")

    section("summary")
    print(f"  {_counts[PASS]} pass, {_counts[WARN]} warn, {_counts[FAIL]} FAIL")
    if _counts[FAIL]:
        print("\n  Fix the FAILs before extracting or labelling. A frame mismatch or a"
              "\n  chain-spec mismatch produces output that looks healthy and is not.")
    sys.exit(1 if _counts[FAIL] else 0)


if __name__ == "__main__":
    main()