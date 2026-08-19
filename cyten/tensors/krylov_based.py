"""Krylov-based algorithms for tensors"""

# Copyright (C) TeNPy Developers, Apache license

from .._core import (  # noqa: F401
    GMRES,
    Arnoldi,
    ArnoldiEvolution,
    KrylovBased,
    LanczosEvolution,
    LanczosGroundState,
    lanczos,
)
from .sparse import HermitianNumpyArrayLinearOperator


def lanczos_arpack(H, psi, options={}):
    """Use :func:`scipy.sparse.linalg.eigsh` to find the ground state of `H`.

    This function has the same call/return structure as :func:`lanczos`, but uses
    the ARPACK package through the functions :func:`~cyten.tools.math.speigsh` instead of the
    custom lanczos implementation in :class:`LanczosGroundState`.

    .. warning ::
        This function is mostly intended for debugging, since it requires to convert the vector
        from cyten :class:`~cyten.tensors.Tensor` to a numpy array and back during
        *each* `matvec`-operation!

    Parameters
    ----------
    H, psi, options :
        See :class:`LanczosGroundState`.
        `H` and `psi` should have/use labels.

    Returns
    -------
    E0 : float
        Ground state energy.
    psi0 : :class:`~cyten.tensors.Tensor`
        Ground state vector.

    """
    H_np, psi_np = HermitianNumpyArrayLinearOperator.from_matvec_and_vector(
        H.matvec, psi, dtype=H.dtype.to_numpy_dtype()
    )
    tol = options.get('P_tol', 1.0e-14)
    N_min = options.get('N_min', None)
    kwargs = dict(num_ev=1, which='SA', v0_np=psi_np, tol=tol)
    if N_min is not None:
        kwargs['ncv'] = N_min
    Es, Vs = H_np.eigenvectors(**kwargs)
    return Es[0], Vs[0]
