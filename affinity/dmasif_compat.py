"""Make the 2021 dMaSIF repo importable in a modern PyTorch / PyG environment.

Import this module BEFORE importing anything from the dMaSIF repo. It never
touches the dMaSIF model itself; it only papers over imports that the
dMaSIF (keops) path does not use:

  * torch_cluster  - only used by the DGCNN / PointNet++ baselines
  * pyvtk          - only used to write .vtk files for visualization
  * PyG names that were renamed/removed since PyG 1.6 (baselines only)

Usage:
    import dmasif_compat
    dmasif_compat.add_repo_to_path("/path/to/dMaSIF")
    from model import dMaSIF
"""

import importlib.machinery
import sys
import types

import torch


class _Unavailable(torch.nn.Module):
    """Placeholder for baseline-only layers; fails loudly if actually used."""

    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "This layer is a compatibility placeholder (baseline models only). "
            "The dMaSIF embedding layer does not need it."
        )


def _unavailable_fn(*args, **kwargs):
    raise RuntimeError("Compatibility placeholder: baseline-only function called.")


# Import PyG FIRST: at import time it probes importlib.util.find_spec('torch_cluster')
# to decide which optional extensions exist. A stub in sys.modules before that
# probe makes find_spec raise (stubs have no __spec__) or report a fake extension.
# PyG names used by the baselines that may no longer exist in PyG 2.x.
import torch_geometric.data as _pyg_data  # noqa: E402
import torch_geometric.nn as _pyg_nn  # noqa: E402

for _name in ["EdgeConv", "Reshape", "DynamicEdgeConv", "PointConv", "XConv"]:
    if not hasattr(_pyg_nn, _name):
        setattr(_pyg_nn, _name, _Unavailable)
for _name in ["fps", "radius", "global_max_pool", "knn_interpolate"]:
    if not hasattr(_pyg_nn, _name):
        setattr(_pyg_nn, _name, _unavailable_fn)
if not hasattr(_pyg_data, "DataLoader"):
    from torch_geometric.loader import DataLoader as _DL

    _pyg_data.DataLoader = _DL


# torch_cluster: compiled PyG extension, not installed in the container.
if "torch_cluster" not in sys.modules:
    try:
        import torch_cluster  # noqa: F401
    except ImportError:
        stub = types.ModuleType("torch_cluster")
        stub.__spec__ = importlib.machinery.ModuleSpec("torch_cluster", None)
        stub.knn = _unavailable_fn
        sys.modules["torch_cluster"] = stub

# pyvtk: imported at module level by geometry_processing.py, used only by save_vtk.
if "pyvtk" not in sys.modules:
    try:
        import pyvtk  # noqa: F401
    except ImportError:
        stub = types.ModuleType("pyvtk")
        stub.__spec__ = importlib.machinery.ModuleSpec("pyvtk", None)
        for name in ["PolyData", "PointData", "CellData", "Scalars", "Vectors", "VtkData"]:
            setattr(stub, name, _unavailable_fn)
        sys.modules["pyvtk"] = stub


def add_repo_to_path(repo_dir):
    """Put the dMaSIF repo on sys.path so `from model import dMaSIF` works."""
    repo_dir = str(repo_dir)
    if repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)
