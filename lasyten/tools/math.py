"""Different math functions needed at some point in the library.

.. autodata:: LeviCivita3
"""
# Copyright (C) TeNPy Developers, Apache license

import warnings

import numpy as np
import scipy.linalg
import scipy.sparse.linalg

from .misc import argsort


def _as_dense_square(A, d):
    """Dense ``(d, d)`` matrix from an ndarray or a scipy LinearOperator."""
    if isinstance(A, np.ndarray):
        return A
    dtype = getattr(A, 'dtype', None)
    return np.asarray(A @ np.eye(d, dtype=dtype))


def speigs(A, k, *args, **kwargs):
    """Wrapper around :func:`scipy.sparse.linalg.eigs`, lifting the restriction ``k < rank(A)-1``.

    Parameters
    ----------
    A : MxM ndarray or like :class:`scipy.sparse.linalg.LinearOperator`
        the (square) linear operator for which the eigenvalues should be computed.
    k : int
        the number of eigenvalues to be computed.
    *args:
        Further arguments directly given to :func:`scipy.sparse.linalg.eigs`
    **kwargs :
        Further keyword arguments directly given to :func:`scipy.sparse.linalg.eigs`

    Returns
    -------
    w : ndarray
        array of min(`k`, A.shape[0]) eigenvalues
    v : ndarray
        array of min(`k`, A.shape[0]) eigenvectors, ``v[:, i]`` is the `i`-th eigenvector.
        Only returned if ``kwargs['return_eigenvectors'] == True``.

    """
    d = A.shape[0]
    if A.shape != (d, d):
        raise ValueError('A.shape not a square matrix: ' + str(A.shape))
    if k < d - 1:
        return scipy.sparse.linalg.eigs(A, k, *args, **kwargs)
    else:
        if k > d:
            warnings.warn('trimming speigs k to smaller matrix dimension d', stacklevel=2)
            k = d
        Amat = _as_dense_square(A, d)
        ret_eigv = kwargs.get('return_eigenvectors', args[7] if len(args) > 7 else True)
        which = kwargs.get('which', args[2] if len(args) > 2 else 'LM')
        if ret_eigv:
            W, V = np.linalg.eig(Amat)
            keep = argsort(W, which)[:k]
            return W[keep], V[:, keep]
        else:
            W = np.linalg.eigvals(Amat)
            keep = argsort(W, which)[:k]
            return W


def speigsh(A, k, *args, **kwargs):
    """Wrapper around :func:`scipy.sparse.linalg.eigsh`, lifting the restriction ``k < rank(A)-1``.

    Parameters
    ----------
    A : MxM ndarray or like :class:`scipy.sparse.linalg.LinearOperator`
        The (square) hermitian linear operator for which the eigenvalues should be computed.
    k : int
        The number of eigenvalues to be computed.
    *args
        Further arguments directly given to :func:`scipy.sparse.linalg.eigsh`.
    **kwargs :
        Further keyword arguments directly given to :func:`scipy.sparse.linalg.eigsh`.

    Returns
    -------
    w : ndarray
        Array of min(`k`, A.shape[0]) eigenvalues.
    v : ndarray
        Array of min(`k`, A.shape[0]) eigenvectors, ``v[:, i]`` is the `i`-th eigenvector.
        Only returned if ``kwargs['return_eigenvectors'] == True``.

    """
    d = A.shape[0]
    if A.shape != (d, d):
        raise ValueError('A.shape not a square matrix: ' + str(A.shape))
    if k < d - 1:
        return scipy.sparse.linalg.eigsh(A, k, *args, **kwargs)
    else:
        if k > d:
            warnings.warn('trimming speigsh k to smaller matrix dimension d', stacklevel=2)
            k = d
        Amat = _as_dense_square(A, d)
        ret_eigv = kwargs.get('return_eigenvectors', args[7] if len(args) > 7 else True)
        which = kwargs.get('which', args[2] if len(args) > 2 else 'LM')
        if ret_eigv:
            W, V = np.linalg.eigh(Amat)
            keep = argsort(W, which)[:k]
            return W[keep], V[:, keep]
        else:
            W = np.linalg.eigvalsh(Amat)
            keep = argsort(W, which)[:k]
            return W
