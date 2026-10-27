#!/bin/bash
# Create the directories jobs write into. Run ONCE on the login node, and
# again after cloning $WORK somewhere new.
#
# Why this has to exist: SLURM does NOT create the directory in
# `#SBATCH --output=logs/...`. If `logs/` is missing the job is submitted and
# then fails with no output file to tell you why, which is a confusing ten
# minutes. Every sbatch in src/slurm/ writes to logs/ as of 2026-10-27.
#
#   bash src/slurm/setup_dirs.sh
set -e
cd ${WORK:-.}
for d in logs tmp reports; do
    mkdir -p "$d"
    echo "  $d/"
done
echo
echo "logs/    every sbatch's --output (was: ~70 .log files loose in \$WORK)"
echo "tmp/     selftest scratch (train, probe, inventory)"
echo "reports/ retrieval_report.py dumps and TSVs"
echo
echo "Old logs still at the root can be swept in now that nothing writes there:"
echo "  mkdir -p logs && mv *.log logs/ 2>/dev/null; ls logs | wc -l"
