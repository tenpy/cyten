r"""cyten library - tensor library for high-level tensor network algorithms.

Provides a tensor class with block-sparsity from symmetries with an exchangeable GPU or CPU backend.

"""
# Copyright (C) TeNPy Developers, Apache license

# note: order matters!
from . import (
    backends,
    block_backends,
    dtypes,
    dummy_config,
    krylov_based,
    models,
    planar,
    random_matrix,
    spaces,
    sparse,
    symmetries,
    tensors,
    testing,
    tools,
    trees,
    version,
)

# import pybind11 bindings from C++ code
# TODO do explicit imports instead of *
from ._core import *  # type: ignore

# subpackages
from .backends import get_backend

# modules under cyten
from .dtypes import Dtype
from .models import Coupling, couplings, sites
from .planar import PlanarDiagram

# from .dummy_config import
# from .krylov_based import
# from .random_matrix import
from .spaces import ElementarySpace, TensorProduct

# from .sparse import
from .symmetries import (
    BraidingStyle,
    FusionStyle,
    SymmetryError,
    double_semion_category,
    fermion_parity,
    fibonacci_anyon_category,
    ising_anyon_category,
    no_symmetry,
    semion_category,
    su2_symmetry,
    toric_code_category,
    u1_symmetry,
    z2_symmetry,
    z3_symmetry,
    z4_symmetry,
    z5_symmetry,
    z6_symmetry,
    z7_symmetry,
    z8_symmetry,
    z9_symmetry,
)
from .tensors import (
    ChargedTensor,
    DiagonalTensor,
    Mask,
    SymmetricTensor,
    Tensor,
    add_trivial_leg,
    almost_equal,
    angle,
    apply_mask,
    bend_legs,
    combine_legs,
    combine_to_matrix,
    complex_conj,
    compose,
    dagger,
    eigh,
    enlarge_leg,
    entropy,
    exp,
    eye,
    imag,
    inner,
    is_scalar,
    item,
    lq,
    move_leg,
    norm,
    on_device,
    outer,
    partial_trace,
    permute_legs,
    pinv,
    qr,
    real,
    real_if_close,
    scale_axis,
    split_legs,
    sqrt,
    squeeze_legs,
    stable_log,
    svd,
    tdot,
    tensor,
    trace,
    transpose,
    truncated_svd,
    zero_like,
)
from .trees import FusionTree, fusion_trees
from .version import full_version as __full_version__
from .version import version as __version__

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
