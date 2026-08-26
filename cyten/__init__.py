r"""cyten library - tensor library for high-level tensor network algorithms.

Provides a tensor class with block-sparsity from symmetries with an exchangeable GPU or CPU backend.

"""
# Copyright (C) TeNPy Developers, Apache license

# do config load -- should be very early!
from ._core import get_config

get_config()  # initialize config


# note: order matters!
from . import (
    config,
    tools,
    block_backends,
    symmetries,
    backends,
    tensors,
    models,
)

# import pybind11 bindings from C++ code
from .backends import TensorBackend, get_backend
from .block_backends import Block, BlockBackend, Dtype, NumpyBlockBackend, TorchBlockBackend

# subpackages
from .config import get_config, set_options, temporary_options
from .models import Coupling, Site, couplings, sites
from .symmetries._symmetries import (
    SU2,
    SUN,
    U1,
    ZN,
    BraidChiralityUnspecifiedError,
    BraidingStyle,
    FermionNumber,
    FermionParity,
    FibonacciAnyonCategory,
    FusionStyle,
    IsingAnyonCategory,
    NoSymmetry,
    QuantumDoubleZNAnyonCategory,
    Sector,
    SectorArray,
    SU2_kAnyonCategory,
    SU3_3AnyonCategory,
    Symmetry,
    SymmetryError,
    ToricCodeCategory,
    ZNAnyonCategory,
    ZNAnyonCategory2,
    double_semion_category,
    semion_category,
)
from .symmetries.spaces import AbelianLegPipe, ElementarySpace, Leg, LegPipe, Space, TensorProduct
from .symmetries.trees import FusionTree, fusion_trees
from .tensors import (
    ChargedTensor,
    DiagonalTensor,
    Identity,
    Mask,
    PlanarDiagram,
    PlanarLinearOperator,
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
    cutoff_inverse,
    dagger,
    eig,
    eigh,
    eigvals,
    eigvalsh,
    enlarge_leg,
    entropy,
    exp,
    eye,
    horizontal_factorization,
    imag,
    inner,
    is_scalar,
    item,
    krylov_based,
    lq,
    move_leg,
    norm,
    on_device,
    outer,
    partial_trace,
    permute_legs,
    pinv,
    # planar
    planar,
    planar_contraction,
    planar_partial_trace,
    planar_permute_legs,
    qr,
    real,
    real_if_close,
    scalar_multiply,
    scale_axis,
    slice_leg,
    # sparse
    sparse,
    split_legs,
    sqrt,
    squeeze_legs,
    stable_log,
    svd,
    tdot,
    tensor,
    tensor_from_grid,
    trace,
    transpose,
    truncated_svd,
    zero_like,
)
from ._version import __version__, __version_tuple__, __commit_id__


def show_version():
    """Print information about the version of cyten and used libraries.

    The information printed is :attr:`cyten.version.version_summary`.
    """
    import sys
    import numpy
    import pytorch

    summary = (
        f'cyten {__version__!s} at git commit {__commit_id__!s} using\n'
        f'  python {sys.version!s}\n'
        f'  numpy {numpy.__version__!s}\n'
        f'  pytorch {pytorch.__version__!s}\n'
    )
    return summary


# expose Dtypes directly
bool = Dtype.bool
float32 = Dtype.float32
complex64 = Dtype.complex64
float64 = Dtype.float64
complex128 = Dtype.complex128
