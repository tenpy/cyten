"""Backend based on fusion trees.

C++ implementations via pybind11 (``cyten._core``).
"""

# Copyright (C) TeNPy Developers, Apache license
from __future__ import annotations

from .._core import (  # noqa: F401
    BendInstruction,
    BraidInstruction,
    FactorizedTreeMapping,
    FusionTreeBackend,
    FusionTreeData,
    PermuteLegsInstructionEngine,
    TreePairMapping,
    TwistInstruction,
)

# Typing alias matching the old ABC; concrete instruction types are the three dataclasses above.
Instruction = BraidInstruction | BendInstruction | TwistInstruction

__all__ = [
    'FusionTreeData',
    'FusionTreeBackend',
    'Instruction',
    'BraidInstruction',
    'BendInstruction',
    'TwistInstruction',
    'PermuteLegsInstructionEngine',
    'TreePairMapping',
    'FactorizedTreeMapping',
]
