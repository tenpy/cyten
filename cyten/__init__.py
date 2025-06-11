r"""cyten library - tensor library for high-level tensor network algorithms.

Provides a tensor class with block-sparsity from symmetries with an exchangeable GPU or CPU backend.

"""
# Copyright (C) TeNPy Developers, Apache license

from . import (dtypes, spaces, backends, symmetries, tensors, random_matrix, sparse, krylov_based,
               trees, dummy_config, tools)
from . import testing  # should be pretty late
from . import version

# TODO do explicit imports instead of *
from ._core import *  # import pybind11 bindings from C++ code


# modules under cyten
from .dtypes import Dtype
# from .dummy_config import
# from .krylov_based import
# from .random_matrix import
from .spaces import ElementarySpace, TensorProduct
# from .sparse import
from .symmetries import (
    SymmetryError, FusionStyle, BraidingStyle, no_symmetry, z2_symmetry, z3_symmetry, z4_symmetry,
    z5_symmetry, z6_symmetry, z7_symmetry, z8_symmetry, z9_symmetry, u1_symmetry, su2_symmetry,
    fermion_parity, semion_category, toric_code_category, double_semion_category,
    fibonacci_anyon_category, ising_anyon_category
)
from .tensors import (
    SymmetricTensor, DiagonalTensor, Mask, ChargedTensor, add_trivial_leg, angle, almost_equal,
    apply_mask, bend_legs, combine_legs, combine_to_matrix, complex_conj, dagger, compose, eigh,
    enlarge_leg, entropy, exp, imag, inner, is_scalar, item, move_leg, norm, on_device, outer,
    partial_trace, permute_legs, pinv, qr, real, real_if_close, lq, scale_axis, split_legs, sqrt,
    squeeze_legs, stable_log, svd, tdot, trace, transpose, truncated_svd, zero_like
)
from .trees import FusionTree, fusion_trees
from .version import version as __version__
from .version import full_version as __full_version__


# subpackages
from .backends import get_backend
from .networks import (
    group_sites, SpinHalfSite, SpinSite, FermionSite, SpinHalfFermionSite, SpinHalfHoleSite,
    BosonSite, ClockSite
)
# from .testing import
# from .tools import


def show_config():
    """Print information about the version of tenpy and used libraries.

    The information printed is :attr:`cyten.version.version_summary`.
    """
    print(version.version_summary)
