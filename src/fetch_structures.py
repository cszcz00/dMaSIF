"""Fetch only the PLINDER structures the manifest needs.

PLINDER's public download moved to Cloudflare R2 (https://plinderdata.org) and
the official plinder_download CLI is all-or-nothing - hundreds of GB. But the
R2 endpoint is plain HTTP with predictable paths, so we can fetch just the
systems/{two_char_code}.zip archives our receptors fall into, pull the members
we need out of each, and drop the archive.

Reads meta/receptors.txt (receptor_key, chains, two_char_code, system_id) and,
with --systems_out, meta/manifest.parquet for the full list of systems. Writes:

    <out>/<receptor_key>.cif        one receptor per file (protein only, no
                                    ligand) - the input to extract.py
    <out>/extract_inputs.txt        "path CHAINS" lines, ready for extract.py
    <systems_out>/<system_id>/system.cif         receptor WITH its ligands
    <systems_out>/<system_id>/ligand_files/...   one file per ligand chain

Every system in the manifest gets its own system.cif, not just the one
representative per receptor: several systems share a receptor, and each one's
ligand is a separate training pair that needs its own pocket label.

Archives are streamed to a temporary file on disk, not held in memory, so
parallel downloads do not multiply RAM use. Every extracted file is written to
a .part file and renamed when complete, so an interrupted run never leaves a
truncated file that the next run would skip as done. system.cif is written
last in each system folder and marks it complete.

--where COL [COL ...] restricts both receptors and systems to manifest rows
where every listed boolean column is true, e.g. `--where strict_rep` for the
strict subset, or `--where pass_mw pass_interactions` for a middle tier.

Example:
    python fetch_structures.py --receptors meta/receptors.txt --check
    python fetch_structures.py --receptors meta/receptors.txt --out structures/ \\
        --manifest meta/manifest.parquet --systems_out systems/ --limit 5
"""

import argparse
import concurrent.futures as cf
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path

BASE = "https://plinderdata.org/2024-06/v2"
# receptor.cif is preferred over receptor.pdb: PDB format allows only a single
# character for a chain ID, which mangles PLINDER's assembly labels (1.A, 2.A).
MEMBER = "receptor.cif"
# receptor.cif holds protein chains and waters only - no ligand. system.cif is
# the same receptor WITH the ligands, in the same frame, which is what pocket
# labelling needs. ligand_files/ holds each ligand chain separately.
SYSTEM_MEMBER = "system.cif"
LIGAND_DIR = "ligand_files/"


# ----------------------------------------------------------------------------
# Inputs
# ----------------------------------------------------------------------------
def read_receptors(path):
    """-> {two_char: [(receptor_key, chains, system_id), ...]}"""
    by_zip = defaultdict(list)
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 4:
            raise ValueError(
                f"expected 4 tab-separated fields, got {len(parts)}: {line!r}. "
                "Rebuild receptors.txt with the current build_manifest.py."
            )
        key, chains, two_char, system_id = parts[:4]
        by_zip[two_char].append((key, chains, system_id))
    return by_zip


def read_manifest(path, where):
    """Manifest rows passing every `where` column -> (receptor keys, {two_char: [system_id]})."""
    import pandas as pd

    cols = ["system_id", "receptor_key"] + list(where)
    df = pd.read_parquet(path, columns=cols)
    for c in where:
        df = df[df[c].fillna(False).astype(bool)]
    systems = defaultdict(list)
    for sid in sorted(df["system_id"].unique()):
        systems[sid[1:3]].append(sid)
    return set(df["receptor_key"]), systems


# ----------------------------------------------------------------------------
# Download
# ----------------------------------------------------------------------------
def head_size(two_char, base):
    url = f"{base}/systems/{two_char}.zip"
    req = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return two_char, int(r.headers.get("content-length", 0)), None
    except urllib.error.HTTPError as e:
        return two_char, 0, f"HTTP {e.code}"
    except Exception as e:  # network flake, DNS, timeout
        return two_char, 0, str(e)


def check(archives, n_rec, n_sys, base, workers):
    """HEAD every needed archive: total bytes, and any that are missing."""
    total, sizes, missing = 0, [], []
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        for two_char, size, err in ex.map(lambda t: head_size(t, base), sorted(archives)):
            total += size
            sizes.append(size)
            if err:
                missing.append((two_char, err))
    print(f"{len(archives)} archives for {n_rec} receptors and {n_sys} systems")
    print(f"total download: {total / 1e9:.1f} GB "
          f"(largest archive {max(sizes, default=0) / 1e9:.2f} GB)")
    if missing:
        print(f"\n{len(missing)} archives unreachable:")
        for t, e in missing[:10]:
            print(f"  {t}.zip  {e}")
    return total, missing


def download(url, dest, retries=3):
    """Stream url to dest via a .part file. Returns None or an error string."""
    part = dest.with_name(dest.name + ".part")
    err = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=600) as r, open(part, "wb") as f:
                shutil.copyfileobj(r, f, 1 << 20)
            os.replace(part, dest)
            return None
        except Exception as e:  # retry network flakes with backoff
            err = f"{type(e).__name__}: {e}"
            time.sleep(10 * (attempt + 1))
    part.unlink(missing_ok=True)
    return f"download failed after {retries} tries: {err}"


def extract_member(zf, member, dest):
    """Copy one zip member to dest via a .part file, so dest is always complete."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    with zf.open(member) as src, open(part, "wb") as dst:
        shutil.copyfileobj(src, dst)
    os.replace(part, dest)


def index_archive(names, two_char):
    """{system_id: {relative path: member name}}. Archives may or may not nest
    their system folders under a two_char directory; handle both."""
    idx = defaultdict(dict)
    for n in names:
        if n.endswith("/"):
            continue
        parts = n.split("/")
        if parts[0] == two_char and len(parts) > 2:
            parts = parts[1:]
        if len(parts) < 2:
            continue
        idx[parts[0]]["/".join(parts[1:])] = n
    return idx


def fetch_one(two_char, receptors, systems, base, out, systems_dir, tmp_dir, keep_zip):
    """Fetch one archive, extract the receptors and systems it holds, discard it.

    Returns (two_char, rec_got, rec_missed, sys_got, sys_missed, error).
    """
    todo_rec = [(k, c, s) for k, c, s in receptors if not (out / f"{k}.cif").exists()]
    todo_sys = [s for s in systems
                if not (systems_dir / s / SYSTEM_MEMBER).exists()] if systems_dir else []
    if not todo_rec and not todo_sys:
        return two_char, 0, 0, 0, 0, None

    archive = tmp_dir / f"{two_char}.zip"
    err = download(f"{base}/systems/{two_char}.zip", archive)
    if err:
        return two_char, 0, len(todo_rec), 0, len(todo_sys), err

    rec_got = sys_got = 0
    miss = []
    try:
        with zipfile.ZipFile(archive) as zf:
            idx = index_archive(zf.namelist(), two_char)

            for key, _chains, sid in todo_rec:
                member = idx.get(sid, {}).get(MEMBER)
                if member is None:
                    miss.append(f"{sid}/{MEMBER}")
                    continue
                extract_member(zf, member, out / f"{key}.cif")
                rec_got += 1

            for sid in todo_sys:
                files = idx.get(sid, {})
                if SYSTEM_MEMBER not in files:
                    miss.append(f"{sid}/{SYSTEM_MEMBER}")
                    continue
                for rel, member in files.items():
                    if rel.startswith(LIGAND_DIR):
                        extract_member(zf, member, systems_dir / sid / rel)
                # last: its presence marks the system folder complete
                extract_member(zf, files[SYSTEM_MEMBER], systems_dir / sid / SYSTEM_MEMBER)
                sys_got += 1
    except zipfile.BadZipFile as e:
        return two_char, rec_got, len(todo_rec) - rec_got, sys_got, len(todo_sys) - sys_got, f"bad zip: {e}"
    finally:
        if keep_zip and archive.exists():
            shutil.move(str(archive), keep_zip / archive.name)
        else:
            archive.unlink(missing_ok=True)

    err = f"{len(miss)} members not found, e.g. {miss[0]}" if miss else None
    return two_char, rec_got, len(todo_rec) - rec_got, sys_got, len(todo_sys) - sys_got, err


def write_extract_inputs(by_zip, out):
    """`path CHAINS` lines for extract.py's list-file input mode."""
    lines = []
    for wanted in by_zip.values():
        for key, chains, _sid in wanted:
            p = out / f"{key}.cif"
            if p.exists():
                lines.append(f"{p} {chains}" if chains else str(p))
    path = out / "extract_inputs.txt"
    path.write_text("\n".join(sorted(lines)) + "\n")
    print(f"{path}: {len(lines)} structures ready for extract.py")
    return len(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--receptors", required=True, help="meta/receptors.txt")
    ap.add_argument("--out", default="structures")
    ap.add_argument("--manifest", help="meta/manifest.parquet; required by --systems_out and --where")
    ap.add_argument("--systems_out",
                    help="also extract system.cif + ligand_files/ for EVERY system in the "
                         "manifest, one folder per system_id; needed for pocket labelling")
    ap.add_argument("--where", nargs="+", default=[],
                    help="only manifest rows where all these boolean columns are true "
                         "(e.g. strict_rep, or pass_mw pass_interactions)")
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--workers", type=int, default=8,
                    help="parallel archive downloads; be polite to the mirror")
    ap.add_argument("--tmp", help="where archives are staged while being read "
                                  "(default <out>/.tmp; needs workers x largest archive)")
    ap.add_argument("--check", action="store_true",
                    help="HEAD the archives and report total size, then exit")
    ap.add_argument("--keep_zips", help="also save the raw archives to this directory")
    ap.add_argument("--limit", type=int, help="only the first N archives (for a trial run)")
    a = ap.parse_args()

    if (a.systems_out or a.where) and not a.manifest:
        ap.error("--systems_out and --where need --manifest")

    by_zip = read_receptors(a.receptors)
    systems = defaultdict(list)
    if a.manifest:
        keys, systems = read_manifest(a.manifest, a.where)
        by_zip = {tc: [r for r in rs if r[0] in keys] for tc, rs in by_zip.items()}
        by_zip = {tc: rs for tc, rs in by_zip.items() if rs}
        if not a.systems_out:
            systems = defaultdict(list)
    archives = sorted(set(by_zip) | set(systems))
    if a.limit:
        archives = archives[: a.limit]
    n_rec = sum(len(by_zip.get(t, [])) for t in archives)
    n_sys = sum(len(systems.get(t, [])) for t in archives)

    if a.check:
        check(archives, n_rec, n_sys, a.base, a.workers)
        return

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(a.tmp) if a.tmp else out / ".tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    keep = Path(a.keep_zips) if a.keep_zips else None
    if keep:
        keep.mkdir(parents=True, exist_ok=True)
    systems_dir = Path(a.systems_out) if a.systems_out else None
    if systems_dir:
        systems_dir.mkdir(parents=True, exist_ok=True)

    print(f"{len(archives)} archives: {n_rec} receptors"
          + (f", {n_sys} systems" if systems_dir else ""))
    rec = sysn = failed = 0
    errors = []
    with cf.ThreadPoolExecutor(max_workers=a.workers) as ex:
        futures = [
            ex.submit(fetch_one, tc, by_zip.get(tc, []), systems.get(tc, []),
                      a.base, out, systems_dir, tmp_dir, keep)
            for tc in archives
        ]
        for i, fut in enumerate(cf.as_completed(futures), 1):
            two_char, rg, rm, sg, sm, err = fut.result()
            rec += rg
            sysn += sg
            failed += rm + sm
            if err:
                errors.append((two_char, err))
            print(f"\r[{i}/{len(futures)}] {two_char}.zip  "
                  f"{rec} receptors, {sysn} systems extracted, {failed} failed",
                  end="", file=sys.stderr)
    print(file=sys.stderr)

    if errors:
        print(f"\n{len(errors)} archives had problems:")
        for t, e in errors[:20]:
            print(f"  {t}.zip  {e}")
        print("Re-run to retry; completed files are skipped.")

    write_extract_inputs({t: by_zip.get(t, []) for t in archives}, out)


if __name__ == "__main__":
    main()