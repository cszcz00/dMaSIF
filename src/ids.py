"""Identity rules shared by everything that names a surface.

The tag is the ONLY link between a manifest row and a file in feats/. It is
reconstructed by rule rather than stored, so every consumer must derive it the
same way. Keeping the rule in two places has already failed once: extract.py
gained dotted-label support in parse_chains while labels.py kept the old test,
which agrees for single-digit model indices ("1.A" -> 1A either way) and
diverges at ten ("10.A" -> 10A vs 01A). The divergence is silent - the npz is
simply reported missing.

Deliberately free of torch, PyG and dmasif_compat. Importing extract.py for
these functions would pull the whole dMaSIF stack in to do string manipulation.

Depends on: nothing.
Imported by: extract.py, labels.py, pack.py.
"""


def parse_chains(spec):
    """Chain spec -> set of labels, or None for "every chain".

        "AB"        -> {"A", "B"}        single-character PDB chain IDs
        "1.A"       -> {"1.A"}           one dotted assembly label
        "1.A,2.A"   -> {"1.A", "2.A"}    several, comma separated

    A dot or a comma marks a label that must NOT be split per character. Without
    that test "1.A" becomes {"1", ".", "A"}, every chain fails the membership
    check in load_protein_atoms, and the structure is skipped with "No protein
    atoms found" - which main() catches and prints as [skip], so a run that
    dropped most of its input looks like a successful one.
    """
    if spec is None:
        return None
    spec = str(spec)
    if not spec.strip():
        return None
    if "," in spec or "." in spec:
        return {c.strip() for c in spec.split(",") if c.strip()}
    return set(spec)


def surface_tag(stem, chains):
    """(file stem, chain set) -> the name of its .npz, without the extension.

    Chains are sorted so the tag does not depend on iteration order, and dots
    are stripped because PLINDER keys already contain them and a second dotted
    run makes the name unreadable. A receptor embedded with no chain filter
    keeps the bare stem, which is why a directory run and a list-file run of the
    same structures produce DIFFERENT filenames.

        surface_tag("11ba__1__1.A_1.B", {"1.A", "1.B"}) -> "11ba__1__1.A_1.B_1A1B"
        surface_tag("1STP", None)                       -> "1STP"
    """
    if not chains:
        return stem
    return stem + "_" + "".join(sorted(chains)).replace(".", "")


def receptor_chains(system_id):
    """Receptor chain labels from a PLINDER system_id, comma separated.

    `pdbid__biounit__receptorchains__ligandchains`; field 3 joins its chains
    with underscores. Mirrors build_manifest.py, which writes column 2 of
    receptors.txt with this rule.
    """
    parts = system_id.split("__")
    return ",".join(parts[2].split("_")) if len(parts) > 2 else ""


def ligand_chains(system_id):
    """Ligand chain labels from a PLINDER system_id -> {"1.C", ...}."""
    parts = system_id.split("__")
    return set(parts[3].split("_")) if len(parts) > 3 else set()


def receptor_key(system_id):
    """Systems differing only in which ligand they describe share a receptor,
    and so share one extraction. Mirrors build_manifest.py."""
    return "__".join(system_id.split("__")[:3])


def read_receptors(path):
    """meta/receptors.txt -> {receptor_key: surface tag}."""
    from pathlib import Path

    out = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        parts = line.rstrip("\n").split("\t")
        key = parts[0]
        out[key] = surface_tag(key, parse_chains(parts[1] if len(parts) > 1 else ""))
    return out