"""Optional adapter for the GNN Ex/Ey surrogate folder with a space in its path."""

from __future__ import annotations

import sys
from pathlib import Path


def load_e_field_surrogate(checkpoint_path=None, normalizer_path=None):
    """Load `ElectricFieldSurrogate` without requiring package installation."""

    repo_root = Path(__file__).resolve().parents[1]
    surrogate_root = repo_root / "GNN_E_surrogate model"
    if str(surrogate_root) not in sys.path:
        sys.path.insert(0, str(surrogate_root))
    from gnn_e_surrogate import ElectricFieldSurrogate

    return ElectricFieldSurrogate(
        checkpoint_path=checkpoint_path,
        normalizer_path=normalizer_path,
    )

