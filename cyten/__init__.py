r"""cyten library - tensor library for high-level tensor network algorithms.

Provides a tensor class with block-sparsity from symmetries with an exchangeable GPU or CPU backend.

"""
# Copyright (C) TeNPy Developers, Apache license

from . import (dtypes, spaces, backends, symmetries, tensors, random_matrix, sparse, krylov_based,
               trees, models, dummy_config, tools)
from . import testing  # should be pretty late
from . import version

# import pybind11 bindings from C++ code
# TODO do explicit imports instead of *
from ._core import *  # type: ignore


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
    Tensor, SymmetricTensor, DiagonalTensor, Mask, ChargedTensor, add_trivial_leg, angle,
    almost_equal, apply_mask, bend_legs, combine_legs, combine_to_matrix, complex_conj, dagger,
    compose, eigh, enlarge_leg, entropy, exp, eye, imag, inner, is_scalar, item, move_leg, norm,
    on_device, outer, partial_trace, permute_legs, pinv, qr, real, real_if_close, lq, scale_axis,
    split_legs, sqrt, squeeze_legs, stable_log, svd, tdot, tensor, trace, transpose, truncated_svd,
    zero_like
)
from .trees import FusionTree, fusion_trees
from .version import version as __version__
from .version import full_version as __full_version__


# subpackages
from .backends import get_backend
from .models import (
    SpinSite, SpinlessBosonSite, SpinlessFermionSite, SpinHalfFermionSite, ClockSite,
    GeneralAnyonSite, AnyonSite, FibonacciAnyonSite, IsingAnyonSite, GoldenSite, SU2kSpin1Site,
    Coupling, spin_spin_coupling, spin_field_coupling, aklt_coupling, heisenberg_coupling,
    chiral_3spin_coupling, chemical_potential, onsite_interaction, long_range_interaction,
    nearest_neighbor_interaction, next_nearest_neighbor_interaction, long_range_quadratic_coupling,
    nearest_neighbor_hopping, next_nearest_neighbor_hopping, onsite_sc_pairing,
    nearest_neighbor_sc_pairing, clock_clock_coupling, clock_field_coupling, gold_coupling
)
# from .testing import
# from .tools import


def show_config():
    """Print information about the version of tenpy and used libraries.

    The information printed is :attr:`cyten.version.version_summary`.
    """
    print(version.version_summary)


# expose Dtypes directly
bool = Dtype.bool
float32 = Dtype.float32
complex64 = Dtype.complex64
float64 = Dtype.float64
complex128 = Dtype.complex128
