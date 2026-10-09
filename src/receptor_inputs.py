"""Receptor input policies: which atoms of an interaction's structure go to dMaSIF.

A dataset row is one protein-ligand interaction. Before it can be embedded, it
needs a concrete structure input: a file, the chains to read from it, and
(eventually) any cropping. How to choose that input for a ligand in a large
assembly is an open question, so it is a named, swappable policy rather than a
rule baked into the dataset builder.

Every policy maps manifest rows to three columns:

    input_id      cache key. Rows with the same input_id share ONE surface, so
                  it must change whenever the atoms handed to dMaSIF change.
    input_path    structure file to read
    input_chains  chain spec for extract.py ("1.A,1.B"; "" = every chain)

A policy that needs more (a crop centre, a radius) adds its own columns.

Adding a policy: write `def my_policy(df, params, root) -> DataFrame` returning
those columns on df's index, decorate it with @policy("my_policy"), and name it
in a dataset spec under receptor_input.policy. Its params land in the spec and
the provenance record of every dataset that uses it.

Depends on: ids.py, pandas.
Imported by: build_dataset.py.
"""

from pathlib import Path

import pandas as pd

from ids import parse_chains, receptor_chains, surface_tag

POLICIES = {}


def policy(name):
    def register(fn):
        POLICIES[name] = fn
        return fn
    return register


def resolve(df, name, params, root):
    """Apply policy `name` to manifest rows -> DataFrame of input columns."""
    if name not in POLICIES:
        raise SystemExit(f"unknown receptor_input.policy {name!r}; "
                         f"known: {sorted(POLICIES)}")
    try:
        out = POLICIES[name](df, params or {}, Path(root))
    except NotImplementedError as e:
        raise SystemExit(f"receptor_input.policy {name!r} is not implemented yet. {e}")
    missing = {"input_id", "input_path", "input_chains"} - set(out.columns)
    if missing:
        raise RuntimeError(f"policy {name!r} did not produce {sorted(missing)}")
    return out


# ----------------------------------------------------------------------------
@policy("plinder_receptor")
def plinder_receptor(df, params, root):
    """PLINDER's receptor.cif as fetched into structures/, unchanged.

    The chains PLINDER assigned to the system, from the biological assembly,
    protein only. Systems on the same receptor chains share one input, so
    several ligands reuse one surface. input_id is the same tag extract.py
    gives the .npz, so a cache built from structures/extract_inputs.txt lines
    up with this index directly.

    Known gap: assembly chains PLINDER did not assign to the system are absent,
    which can expose interface surface that is buried in the full complex.

    params:
        structures   directory of <receptor_key>.cif (default "structures")
    """
    structures = root / params.get("structures", "structures")
    chains = df["system_id"].map(receptor_chains)
    return pd.DataFrame({
        "input_id": [surface_tag(k, parse_chains(c))
                     for k, c in zip(df["receptor_key"], chains)],
        "input_path": [str(structures / f"{k}.cif") for k in df["receptor_key"]],
        "input_chains": chains,
    }, index=df.index)


# ----------------------------------------------------------------------------
# Candidates for later. Registered so a spec naming them fails with the design
# notes below instead of "unknown policy".
@policy("full_assembly")
def full_assembly(df, params, root):
    raise NotImplementedError(
        "full_assembly: every chain of the biological assembly, so no interface "
        "is falsely exposed. Needs the assembly files (not in PLINDER's system "
        "zips) and a size cap for very large complexes. Shares one input per "
        "(pdb_id, biounit)."
    )


@policy("pocket_crop")
def pocket_crop(df, params, root):
    raise NotImplementedError(
        "pocket_crop: atoms within `radius` A of the ligand, from a source "
        "structure chosen by another policy. Cheap, but the cut creates surface "
        "that does not exist; harmless only if it lies beyond the conv layers' "
        "reach from the pocket. The crop depends on the ligand, so input_id is "
        "per system and nothing is shared across ligands of one receptor."
    )
