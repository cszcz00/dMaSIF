"""Fetch only the PLINDER receptor structures the manifest needs.

PLINDER's public download moved to Cloudflare R2 (https://plinderdata.org) and
the official plinder_download CLI is all-or-nothing - hundreds of GB. But the
R2 endpoint is plain HTTP with predictable paths, so we can fetch just the
systems/{two_char_code}.zip archives our filtered receptors fall into, pull the
one receptor.cif we need out of each, and drop the archive.

Reads meta/receptors.txt (receptor_key, chains, two_char_code, system_id) and
writes:

    <out>/<receptor_key>.cif    one receptor structure per file (no ligand)
    <out>/extract_inputs.txt    "path CHAINS" lines, ready for extract.py
    <systems_out>/<system_id>.cif   receptor WITH ligand, for pocket labelling

Resumable: receptors whose .cif already exists are skipped, so re-running after
an interrupted transfer picks up where it stopped.

Example:
    python fetch_structures.py --receptors meta/receptors.txt --check
    python fetch_structures.py --receptors meta/receptors.txt --out structures/
"""

import argparse
import concurrent.futures as cf
import io
import shutil
import sys
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
# the same receptor WITH the ligand, in the same frame, which is what pocket
# labelling needs. Pulling it in the same pass avoids a second full download.
SYSTEM_MEMBER = "system.cif"


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


def check(by_zip, base, workers):
    """HEAD every needed archive: total bytes, and any that are missing."""
    total, missing = 0, []
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        for two_char, size, err in ex.map(lambda t: head_size(t, base), sorted(by_zip)):
            total += size
            if err:
                missing.append((two_char, err))
    n_rec = sum(len(v) for v in by_zip.values())
    print(f"{len(by_zip)} archives for {n_rec} receptors")
    print(f"total download: {total / 1e9:.1f} GB")
    if missing:
        print(f"\n{len(missing)} archives unreachable:")
        for t, e in missing[:10]:
            print(f"  {t}.zip  {e}")
    return total, missing


def fetch_one(two_char, wanted, base, out, keep_zip, systems_dir=None):
    """Download one archive into memory, extract the members we need, discard it.

    The archives are small enough to hold in memory one at a time, which avoids
    writing hundreds of GB of zips to disk just to read a few files from each.
    """
    url = f"{base}/systems/{two_char}.zip"
    def missing(k, s):
        if not (out / f"{k}.cif").exists():
            return True
        return systems_dir is not None and not (systems_dir / f"{s}.cif").exists()

    todo = [(k, c, s) for k, c, s in wanted if missing(k, s)]
    if not todo:
        return two_char, 0, 0, None

    try:
        with urllib.request.urlopen(url, timeout=600) as r:
            blob = r.read()
    except Exception as e:
        return two_char, 0, len(todo), f"download failed: {e}"

    if keep_zip:
        (keep_zip / f"{two_char}.zip").write_bytes(blob)

    got, miss = 0, []
    try:
        zf = zipfile.ZipFile(io.BytesIO(blob))
    except zipfile.BadZipFile as e:
        return two_char, 0, len(todo), f"bad zip: {e}"

    names = set(zf.namelist())
    for key, _chains, system_id in todo:
        # archives may or may not nest under the two_char directory
        candidates = [f"{system_id}/{MEMBER}", f"{two_char}/{system_id}/{MEMBER}"]
        member = next((c for c in candidates if c in names), None)
        if member is None:
            miss.append(system_id)
            continue
        dest = out / f"{key}.cif"
        if not dest.exists():
            with zf.open(member) as src, open(dest, "wb") as dst:
                shutil.copyfileobj(src, dst)
        if systems_dir is not None:
            sdest = systems_dir / f"{system_id}.cif"
            if not sdest.exists():
                smember = member.replace(MEMBER, SYSTEM_MEMBER)
                if smember in names:
                    with zf.open(smember) as src, open(sdest, "wb") as dst:
                        shutil.copyfileobj(src, dst)
                else:
                    miss.append(smember)
        got += 1
    err = None
    if miss:
        err = f"{len(miss)} members not found, e.g. {miss[0]}"
    return two_char, got, len(todo) - got, err


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
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--workers", type=int, default=8,
                    help="parallel archive downloads; be polite to the mirror")
    ap.add_argument("--check", action="store_true",
                    help="HEAD the archives and report total size, then exit")
    ap.add_argument("--keep_zips", help="also save the raw archives to this directory")
    ap.add_argument("--limit", type=int, help="only the first N archives (for a trial run)")
    ap.add_argument("--systems_out",
                    help="also extract system.cif (receptor WITH ligand) here; needed "
                         "for pocket labelling, since receptor.cif has no ligand")
    a = ap.parse_args()

    by_zip = read_receptors(a.receptors)
    if a.limit:
        by_zip = {k: by_zip[k] for k in sorted(by_zip)[: a.limit]}

    if a.check:
        check(by_zip, a.base, a.workers)
        return

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    keep = Path(a.keep_zips) if a.keep_zips else None
    if keep:
        keep.mkdir(parents=True, exist_ok=True)
    systems_dir = Path(a.systems_out) if a.systems_out else None
    if systems_dir:
        systems_dir.mkdir(parents=True, exist_ok=True)

    done = failed = 0
    errors = []
    with cf.ThreadPoolExecutor(max_workers=a.workers) as ex:
        futures = {
            ex.submit(fetch_one, tc, w, a.base, out, keep, systems_dir): tc
            for tc, w in sorted(by_zip.items())
        }
        for i, fut in enumerate(cf.as_completed(futures), 1):
            two_char, got, missed, err = fut.result()
            done += got
            failed += missed
            if err:
                errors.append((two_char, err))
            print(f"\r[{i}/{len(futures)}] {two_char}.zip  "
                  f"{done} extracted, {failed} failed", end="", file=sys.stderr)
    print(file=sys.stderr)

    if errors:
        print(f"\n{len(errors)} archives had problems:")
        for t, e in errors[:20]:
            print(f"  {t}.zip  {e}")
        print("Re-run to retry; existing .cif files are skipped.")

    write_extract_inputs(by_zip, out)


if __name__ == "__main__":
    main()