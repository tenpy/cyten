"""Defines classes describing the local physical Hilbert spaces.

The :class:`DegreeOfFreedom` is the prototype, read its docstring.
All other classes are base classes from which sites are derived.
"""

# Copyright (C) TeNPy Developers, Apache license

from .._core import (  # noqa: F401
    ALL_SPECIES,
    AnyonDOF,
    BosonicDOF,
    ClockDOF,
    FermionicDOF,
    OccupationDOF,
    Site,
    SpinDOF,
)
