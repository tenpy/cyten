r"""cyten library - tensor library for high-level tensor network algorithms.

Provides a tensor class with block-sparsity from symmetries with an exchangable GPU or CPU backend.

"""
# Copyright (C) TeNPy Developers, GNU GPLv3

from . import (dtypes, spaces, backends, symmetries, tensors, random_matrix, sparse, krylov_based,
               trees)
from .symmetries import *
from .trees import *
from .spaces import *
from .backends import *
from .tensors import *
from .random_matrix import *
from .sparse import *
from .krylov_based import *
from .dtypes import *

from ._core import *  # import pybind11 bindings from C++ code

from . import version

__all__ = [
    'symmetries',
    'spaces',
    'trees',
    'backends',
    'tensors',
    'random_matrix',
    'sparse',
    'krylov_based',
    'dtypes',
    *symmetries.__all__,
    *trees.__all__,
    *spaces.__all__,
    *backends.__all__,
    *tensors.__all__,
    *random_matrix.__all__,
    *sparse.__all__,
    *krylov_based.__all__,
    *dtypes.__all__,
]

#: hard-coded version string
__version__ = version.version

#: full version from git description, compile flags, and numpy/scipy/python versions
__full_version__ = version.full_version


def show_config():
    """Print information about the version of tenpy and used libraries.

    The information printed is :attr:`cyten.version.version_summary`.
    """
    print(version.version_summary)
