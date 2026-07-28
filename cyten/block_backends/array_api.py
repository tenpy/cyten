"""Implements a generic BlockBackend that works with any library which follows the Array API.

The API standard is documented at https://data-apis.org/array-api/latest/purpose_and_scope.html
"""

# Copyright (C) TeNPy Developers, Apache license

# implemented in C++
from .._core import ArrayApiBlockBackend  # noqa: F401
