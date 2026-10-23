#!/bin/bash
# Install PLIP for src/plip_labels.py, layered on the existing container: no new
# container, no conda. Login node, once, about a minute:
#   bash src/slurm/setup_plip.sh
#
# PLIP 3.0.1 and OpenBabel 3.2.1 both ship prebuilt manylinux wheels for the
# container's Python 3.12, so they install with the container's own pip into a
# plain package directory ($PLIP_PKGS, default $WORK/envs/plip_pkgs) that jobs
# put on PYTHONPATH. --no-deps keeps the container's numpy (1.26, which torch
# was built against) untouched; PLIP and OpenBabel need nothing else beyond
# lxml, installed here too.
#
# Then three checks, in order: the imports (torch, Biopython, PLIP, OpenBabel
# together in one interpreter), plip_labels.py's numpy self-test, and the
# end-to-end demo on fpocket's 1ATP sample (present once build_fpocket.sh has
# run): our merged complex next to PLIP on the original file. Expect H-bonds
# and salt bridges to agree closely (checked: 3 / 3 salt bridges, 8 of our 12
# H-bonds on the original's protein atoms - the original also binds ATP through
# Mn ions and waters our complex leaves out by design).
set -e
module load apptainer 2>/dev/null || true
cd $WORK
SB=$WORK/dmasif_sandbox
DIR=${PLIP_PKGS:-$WORK/envs/plip_pkgs}
mkdir -p $DIR
# Not --cleanenv: pip needs the login node's proxy settings, if it has any.
apptainer exec $SB python -m pip install --no-deps --upgrade --target $DIR \
    plip==3.0.1 openbabel==3.2.1 lxml==6.1.3
EX="apptainer exec --cleanenv --env PYTHONPATH=$DIR $SB python"
$EX -c "
import numpy, torch, Bio, plip
from openbabel import pybel
print('imports ok: numpy', numpy.__version__, '| torch', torch.__version__,
      '| plip', plip.__file__.split('/')[-2], '| openbabel', pybel.__file__)
"
$EX src/plip_labels.py selftest
SAMPLE=$WORK/tools/fpocket-src/data/sample/1ATP.pdb
if [ -f "$SAMPLE" ]; then
    $EX src/plip_labels.py demo --pdb $SAMPLE --ligand ATP
else
    echo "(no $SAMPLE: run src/slurm/build_fpocket.sh for the end-to-end demo)"
fi
echo "PLIP ready: PYTHONPATH=$DIR"
