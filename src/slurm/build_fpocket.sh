#!/bin/bash
# Build fpocket (https://github.com/Discngine/fpocket) for external_pockets.py.
# Login node, a minute or two, no queue:
#   bash src/slurm/build_fpocket.sh
# -> $WORK/tools/fpocket-src/bin/fpocket, which src/slurm/fpocket.sbatch uses.
#
# Built INSIDE the container so the binary links against the libc it will run
# with (external_pockets.py run executes in the container). Plain `make`, not
# `make -j`: the bundled qhull sub-build races under -j and fails on a missing
# libqhull header. Pinned to 4.2.3, the version the parser was checked against
# (its 1ATP sample: 28 pockets, the top one on the ATP site).
set -e
module load apptainer 2>/dev/null || true
cd $WORK
SRC=$WORK/tools/fpocket-src
TAG=${FPOCKET_TAG:-4.2.3}
if [ ! -d "$SRC" ]; then
    mkdir -p $WORK/tools
    git clone --depth 1 --branch $TAG https://github.com/Discngine/fpocket.git $SRC
fi
apptainer exec --cleanenv $WORK/dmasif_sandbox make -C $SRC
BIN=$SRC/bin/fpocket
[ -x "$BIN" ] || { echo "build finished but $BIN is missing"; exit 1; }
# A real run on fpocket's own sample, parsed by our parser: the end-to-end check.
apptainer exec --cleanenv $WORK/dmasif_sandbox python - <<PY
import sys, shutil, tempfile, subprocess
sys.path.insert(0, "$WORK/src")
from external_pockets import parse_fpocket
d = tempfile.mkdtemp()
shutil.copy("$SRC/data/sample/1ATP.pdb", d)
subprocess.run(["$BIN", "-f", "1ATP.pdb"], cwd=d, check=True, capture_output=True)
pk = parse_fpocket(d + "/1ATP_out", "1ATP")
print(f"fpocket OK: 1ATP -> {len(pk['pocket_score'])} pockets, "
      f"{len(pk['sphere_xyz'])} alpha spheres, top score {pk['pocket_score'][0]:.3f}")
shutil.rmtree(d)
PY
echo "binary: $BIN"
