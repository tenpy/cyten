"""The spaces, i.e. the legs of a tensor."""

# Copyright (C) TeNPy Developers, Apache license
from __future__ import annotations

from .._core import (  # noqa: F401
    AbelianLegPipe,
    DirectSumSpace,
    ElementarySpace,
    Leg,
    LegPipe,
    Space,
    TensorProduct,
    _flat_leg_permutation,
    _parse_inputs_drop_symmetry,
    _sort_sectors,
    _twist_gate_diag,
    _unique_sorted_sectors,
    swap_gate,
    twist_gate,
)
