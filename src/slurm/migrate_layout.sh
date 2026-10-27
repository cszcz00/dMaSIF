#!/bin/bash
# One-time move of the data and output directories into data/ and out/
# (refactor stage 6, docs/REFACTOR.md). Run ONCE on the login node, from $WORK,
# with NO JOBS RUNNING. Code for the new paths is already pushed, so pull first.
#
#   cd $WORK && git pull && bash src/slurm/migrate_layout.sh
#
# Everything moved is gitignored, so git does not see any of this; it is a
# plain `mv` of directories that live only on the HPC. Moves are same-filesystem
# renames, so it is instant regardless of how large surfaces/ is.
#
# NOT moved, deliberately:
#   structures/ systems/   their paths are PERSISTED inside every items.tsv and
#                          index.parquet (input_path, sdf_paths) and resolved
#                          against $WORK. They move with dataset_v2, which
#                          rewrites those columns anyway. See docs/DATA_LAYOUT.md.
#   keops_cache/           $KEOPS_CACHE_FOLDER points at it from your shell.
#   tools/ envs/           build inputs, not data.
set -e
cd ${WORK:?set WORK first}
echo "migrating in $PWD"

mkdir -p data out
move() {                       # move() <dir> <dest>
    if [ -e "$2/$1" ]; then echo "  $1 -> $2/ (already there, skipping)"; return; fi
    if [ ! -e "$1" ];  then echo "  $1 (absent, skipping)"; return; fi
    mv "$1" "$2/"; echo "  $1 -> $2/"
}
for d in surfaces pockets plip meta datasets; do move "$d" data; done
for d in runs reports encoder_test apo_test probe train_check; do move "$d" out; done

# Stale selftest scratch: the defaults now write to tmp/ (stage 7).
for d in probe_selftest train_selftest; do
    [ -d "$d" ] && { rm -rf "$d"; echo "  removed stale scratch $d/"; }
done

echo
echo "now at the root:"
ls -d */ 2>/dev/null | tr -d '/' | tr '\n' ' '; echo
echo
echo "verify, then run nothing else until it reads clean:"
echo "  python src/inventory.py report --items out/runs/v1/items.tsv"
echo
echo "coverage should show data/surfaces and data/pockets/fpocket at 19,293"
echo "receptors. If either reads 0, the move did not land and the job defaults"
echo "are pointing somewhere empty - check with: ls data out"
