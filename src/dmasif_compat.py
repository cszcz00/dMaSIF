"""
Module for importing 2021 dMaSIF repo soundly in modern environments.

Utilize this module before importing from dMaSIF repository.
Mainly 

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
    """Placeholder for baseline-only layers; fails if used."""

    def __init__(self, *args, **kwargs):
        raise RuntimeError(
            "compatibility placeholder (baseline models only): "
            "The dMaSIF embedding layer does not need it."
        )


def _unavailable_fn(*args, **kwargs):
    raise RuntimeError("Compatibility placeholder: baseline-only function called.")


# Import PyG FIRST
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


# torch_cluster not installed in our container (Empire AI build-specific).
if "torch_cluster" not in sys.modules:
    try:
        import torch_cluster  # noqa: F401
    except ImportError:
        stub = types.ModuleType("torch_cluster")
        stub.__spec__ = importlib.machinery.ModuleSpec("torch_cluster", None)
        stub.knn = _unavailable_fn
        sys.modules["torch_cluster"] = stub

# imported at module level by geometry_processing.py, but not used in our work.
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
