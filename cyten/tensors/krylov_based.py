"""Krylov-based algorithms for tensors"""

# Copyright (C) TeNPy Developers, Apache license
import logging
from abc import ABCMeta, abstractmethod

import numpy as np

from ..tools.misc import argsort  # TODO replace this?
from ._tensors import VectorLike, inner, norm, scalar_multiply
from .sparse import (
    HermitianNumpyArrayLinearOperator,
    LinearOperator,
    ProjectedLinearOperator,
    ShiftedLinearOperator,
)

logger = logging.getLogger(__name__)


def _to_number(value):
    """Convert a cyten Scalar (or number) to a numpy scalar."""
    to_numpy = getattr(value, 'to_numpy', None)
    if callable(to_numpy):
        return to_numpy()
    return value


def _abs_number(value):
    """Absolute value as a Python/numpy float."""
    return abs(_to_number(value))


class KrylovBased(metaclass=ABCMeta):
    r"""Base class for iterative algorithms building a Krylov basis with cyten tensors.

    Algorithms like :class:`LanczosGroundState` and `:class:`ArnoldiDiagonalize`
    are based on iteratively building an orthonormal basis of the Krylov space spanned by
    ``|psi0>, H|psi0>, H^2|psi0>, ... H^N |psi0>``, where `N` is the number of iterations
    performed so far, and ``|psi0>`` is an initial guess and starting vector.
    During that iteration, the projection of `H` into the Krylov space is built, where it can
    be solved effectively (with `H` being just a N by N matrix), yielding the "Ritz" eigenvalues/
    eigenvectors. Finally, the solution can be translated back into the original space using the
    basis.

    An important strategy is also to (implicitly) restart the algorithm after some number of steps.
    This is **not** done here: when we use these classes, we usually have an explicit outer loop
    performed until convergence, e.g., the "sweeps" in DMRG.

    Parameters
    ----------
    H : :class:`~cyten.sparse.LinearOperator`
        A hermitian linear operator.
        In order to use :class:`~cyten.tensors.Tensor`s or other
        :class:`~cyten.tensors.Tensor` types, see :class:`~cyten.sparse.TensorLinearOperator`.
        The operator must map tensors to tensors with the same legs.
    psi0 : :class:`~cyten.tensors.VectorLike`
        The starting vector defining the Krylov basis.
        For finding the ground state, this should be the best guess available.
        A :class:`~cyten.tensors.Tensor` of any rank, or a :class:`~cyten.tensors.DirectSum`
        of tensors, is allowed.
    options : dict
        Further optional parameters as described in :cfg:config:`Lanczos`.
        The algorithm stops if *both* criteria for `e_tol` and `p_tol` are met
        or if the maximum number of steps was reached.

    Options
    -------
    .. cfg:config :: KrylovBased

        N_min : int
            Minimum number of steps to perform.
        N_max : int
            Maximum number of steps to perform.
        P_tol : float
            Tolerance for the error estimate from the Ritz Residual,
            stop if ``(RitzRes/gap)**2 < P_tol``
        min_gap : float
            Lower cutoff for the gap estimate used in the P_tol criterion.
        cutoff : float
            Cutoff to abort if the norm of the new krylov vector is too small.
            This is necessary if the rank of `H` is smaller than `N_max`, but it's *not* the error
            tolerance for final values!
        E_shift : float
            Shift the energy (=eigenvalues) by that amount *during* the Lanczos run by using the
            :class:`~cyten.sparse.ShiftedLinearOperator`.
            The ground state energy `E0` returned by :meth:`run` is made independent of the shift.
            This option is useful if the :class:`~cyten.sparse.ProjectedLinearOperator`
            is used: the orthogonal vectors are *exact* eigenvectors with eigenvalue 0 independent
            of the shift, so you can use it to ensure that the energy is smaller than zero
            to avoid getting those.
        reortho : bool
            For poorly conditioned matrices, one can quickly loose orthogonality of the
            generated Krylov basis.
            If `reortho` is True, we re-orthogonalize against all the
            vectors kept in cache to avoid that problem.

    Attributes
    ----------
    options : dict_like
        Optional parameters.
    H : :class:`~cyten.sparse.LinearOperator`
        The linear operator used for building the Krylov space.
    psi0 : :class:`~cyten.tensors.VectorLike`
        The *normalized* starting vector.
    N_min, N_max, P_tol, min_gap, _cutoff, E_shift:
        Parameters as described in the options.
    Es : ndarray, shape(N_max, N_max)
        ``Es[n, :]`` contains the energies of ``_h_krylov[:n+1, :n+1]`` in step `n`.
    _h_krylov : ndarray, shape (N_max + 1, N_max +1)
        The matrix representing `H` projected onto the orthonormalized Krylov basis.
    _psi0_norm : float
        Initial norm of the `psi0` parameter. Note that ``self.psi0`` gets normalized.
    _cache : list of psi0-like vectors
        The ONB of the Krylov space generated during the iteration.
        FIFO (first in first out) cache of at most `N_cache` vectors.
    _result_krylov : ndarray
        Result in the ONB of the Krylov space, e.g. the ground state of `_h_krylov`.
        What exactly this is depends on the subclass.

    Notes
    -----
    The Ritz residual `RitzRes` is computed according to
    http://web.eecs.utk.edu/~dongarra/etemplates/node103.html#estimate_residual.
    Given the gap, the Ritz residual gives a bound on the error in the wavefunction,
    ``err < (RitzRes/gap)**2``. The gap is estimated from the full Lanczos spectrum.

    """

    _dtype_h_krylov = np.complex128
    _dtype_E = np.complex128

    def __init__(self, H: LinearOperator, psi0: VectorLike, options):
        self.H = H
        self.psi0 = psi0.copy()
        self._psi0_norm = None
        self.options = {} if options is None else options
        self.N_min = self.options.get('N_min', 2)
        self.N_max = self.options.get('N_max', 20)
        self.N_cache = self.N_max
        self.P_tol = self.options.get('P_tol', 1.0e-14)
        self.min_gap = self.options.get('min_gap', 1.0e-12)
        self.reortho = self.options.get('reortho', False)
        self.E_shift = self.options.get('E_shift', None)
        if self.N_min < 2:
            raise ValueError('Should perform at least 2 steps.')
        self._cutoff = self.options.get('cutoff', psi0.dtype.eps * 100)
        if self.E_shift is not None:
            if isinstance(self.H, ProjectedLinearOperator):
                self.H.original_operator = ShiftedLinearOperator(self.H.original_operator, self.E_shift)
            else:
                self.H = ShiftedLinearOperator(self.H, self.E_shift)
        self._cache = []
        self.Es = np.zeros([self.N_max, self.N_max], dtype=self._dtype_E)
        self._h_krylov = np.zeros([self.N_max + 1, self.N_max + 1], dtype=self._dtype_h_krylov)

    @abstractmethod
    def run(self): ...

    @abstractmethod
    def _build_krylov(self): ...

    @abstractmethod
    def _calc_result_krylov(self, k): ...

    def _reset_krylov_state(self):
        """Clear cached Krylov vectors and the projected Hessenberg matrix."""
        self._cache = []
        self._h_krylov[:] = 0.0
        self.Es[:] = 0.0

    def _calc_result_full(self, N: int) -> VectorLike:
        """Transform the :attr:`_result_krylov` from the Krylov ONB to the original basis.

        Construct the result ``psi_f = sum_k  _result_krylov[k] psi[k]``, where ``psi[k]``
        is the k-th vector of the ONB of the Krylov space generated during the iteration.
        """
        # this implementation assumes there is a single state
        vf = self._result_krylov
        assert N == len(vf) > 1
        psif = vf[0] * self.psi0  # the start vector psi0 has been normalized by now
        len_cache = len(self._cache)
        # and the len_cache vectors have been cached
        for k in range(1, min(len_cache + 1, N)):
            psif = psif + vf[N - k] * self._cache[-k]
        # other vectors are not cached, so we need to restart the Lanczos iteration.
        self._cache = []  # free memory: we need at least two more vectors

        psif = self._rebuild_krylov_for_result_full(psif, N - len_cache - 1)

        psif_norm = _abs_number(norm(psif))
        if abs(1.0 - psif_norm) > 1.0e-5:
            # One reason can be that `H` is not Hermitian
            # Otherwise, the matrix (even if small) might be ill conditioned.
            # If you get this warning, you can try to set the parameters
            # `reortho`=True and `N_cache` >= `N_max`
            logger.warning('poorly conditioned H matrix in KrylovBased! |psi_0| = %f', psif_norm)
        return scalar_multiply(1.0 / psif_norm, psif)

    def _to_cache(self, psi):
        """Add psi to cache, keep at most self.N_cache."""
        cache = self._cache
        cache.append(psi)
        if len(cache) > self.N_cache:
            cache.pop(0)  # remove *first* entry


class GMRES:
    """GMRES solver for ``A x = b`` with cyten tensors.

    Parameters
    ----------
    A : :class:`~cyten.sparse.LinearOperator`
        Linear operator. Must implement `matvec`.
    x : :class:`~cyten.tensors.VectorLike`
        Initial guess. Copied; the caller's vector is not modified.
    b : :class:`~cyten.tensors.VectorLike`
        Right-hand side.
    options : dict
        Solver options.

    Options
    -------
    N_min : int
        Minimum number of Arnoldi steps per restart cycle before checking convergence.
    N_max : int
        Maximum Krylov dimension per restart cycle.
    restart : int
        Maximum number of restart cycles.
    res : float
        Relative residual tolerance ``|A x - b| / |b|``.

    """

    def __init__(self, A, x, b, options):
        options = {} if options is None else options
        self.options = options
        self.N_min = options.get('N_min', 5)
        self.N_max = options.get('N_max', 20)
        self.restart = options.get('restart', 10)
        self.res = options.get('res', 1.0e-8)
        self.A = A
        self.b = b.copy()
        self.x = x.copy()
        r0 = self.b - self.A.matvec(self.x)
        self.rs = [r0]
        self.b_norm = _abs_number(norm(self.b))
        self.r_norm = _abs_number(norm(r0))
        denom = self.b_norm if self.b_norm != 0 else 1.0
        self.total_error = [[self.r_norm / denom]]
        self.total_iters = []
        if self.r_norm > 0:
            self.qs = [scalar_multiply(1.0 / self.r_norm, r0)]
        else:
            self.qs = [r0]
        self._init_hessenberg()

    def _init_hessenberg(self):
        self.sine = np.zeros(self.N_max, dtype=np.complex128)
        self.cosine = np.zeros(self.N_max, dtype=np.complex128)
        self.e1 = np.zeros(self.N_max + 1, dtype=np.complex128)
        self.e1[0] = self.r_norm
        self.H = np.zeros((self.N_max + 1, self.N_max), dtype=np.complex128)

    def run(self):
        if self.total_error[0][0] < self.res:
            return self.x, self.total_error[0][0], self.total_error, self.total_iters
        for _ in range(self.restart):
            converged = False
            for k in range(0, self.N_max):
                self.arnoldi(k)
                self.apply_givens_rotation(k)
                self.e1[k + 1] = -self.sine[k] * self.e1[k]
                self.e1[k] = self.cosine[k] * self.e1[k]
                # The residual is the last element of the beta vector (see Wikipedia).
                error = np.abs(self.e1[k + 1]) / (self.b_norm if self.b_norm != 0 else 1.0)
                self.total_error[-1].append(error)
                if error < self.res and k >= self.N_min:
                    converged = True
                    break
            self.total_iters.append(k + 1)
            self.backsolve(k + 1)
            for i in range(k + 1):
                self.x = self.x + self.y[i] * self.qs[i]
            if not converged:
                self.reset()
            else:
                break

        rel = _abs_number(norm(self.A.matvec(self.x) - self.b))
        if self.b_norm != 0:
            rel = rel / self.b_norm
        return self.x, rel, self.total_error, self.total_iters

    def arnoldi(self, k):
        # Iterative build orthogonal Krylov subspace and Hessenberg matrix.
        q = self.A.matvec(self.qs[-1])
        for i in range(k + 1):
            hik = _to_number(inner(self.qs[i], q))
            self.H[i, k] = hik
            q = q - hik * self.qs[i]
        self.H[k + 1, k] = _abs_number(norm(q))
        if self.H[k + 1, k] > 0:  # avoid warning if norm(q)==0, error=0 in that case
            q = scalar_multiply(1.0 / self.H[k + 1, k], q)
        self.qs.append(q)

    def apply_givens_rotation(self, k):
        # Apply rotation to H so that it becomes upper triangular.
        for i in range(k):
            temp = self.cosine[i] * self.H[i, k] + self.sine[i] * self.H[i + 1, k]
            self.H[i + 1, k] = -self.sine[i] * self.H[i, k] + self.cosine[i] * self.H[i + 1, k]
            self.H[i, k] = temp

        self.givens_rotation(k)
        self.H[k, k] = self.cosine[k] * self.H[k, k] + self.sine[k] * self.H[k + 1, k]
        self.H[k + 1, k] = 0

    def givens_rotation(self, k):
        # Find cosine and sine such that the element below the diagonal of kth column of H is removed.
        v1, v2 = self.H[k, k], self.H[k + 1, k]
        t = np.sqrt(v1**2 + v2**2)
        self.cosine[k] = v1 / t
        self.sine[k] = v2 / t

    def backsolve(self, k):
        # H is now upper triangular; backsolve to find y exactly.
        H = self.H[:k, :k]
        e2 = self.e1[:k]
        y = np.zeros(k, dtype=np.complex128)
        for i in range(k - 1, -1, -1):
            y[i] = e2[i]
            for j in range(i + 1, k):
                y[i] -= H[i, j] * y[j]
            y[i] /= H[i, i]
        self.y = y

    def reset(self):
        # Restart GMRES using current x as initial guess.
        r = self.b - self.A.matvec(self.x)
        self.rs.append(r)
        self.r_norm = _abs_number(norm(r))
        denom = self.b_norm if self.b_norm != 0 else 1.0
        self.total_error.append([self.r_norm / denom])
        if self.r_norm > 0:
            self.qs = [scalar_multiply(1.0 / self.r_norm, r)]
        else:
            self.qs = [r]
        self._init_hessenberg()


class Arnoldi(KrylovBased):
    """Arnoldi method for diagonalizing square, non-hermitian/symmetric matrices.

    Generalization of :class:`LanczosGroundState`, allowing general, square matrices.

    Options
    -------
    .. cfg:config :: Arnoldi
        :include: KrylovBased

        E_tol : float
            Stop if energy difference per step < `E_tol`
        which : ``'LM' | 'LR' | 'SR'``
            Determines which (extremal) eigenvalues to look for, name
            largest magnitude (in absolute value, ``'LM'``), or
            largest or smallest real part (``'LR'`` and ``'SR'``, respectively).
        num_ev : int
            Number of eigenvectors to look for/return in `run`.

    """

    def __init__(self, H, psi0, options):
        super().__init__(H, psi0, options)
        self.E_tol = self.options.get('E_tol', np.inf)
        self.which = self.options.get('which', 'LM')
        self.num_ev = self.options.get('num_ev', 1)  # number of desired eigenvectors

    def run(self):
        """Find the ground state of self.H.

        Returns
        -------
        E0s : numpy array
            Best eigenvalue estimates, :cfg:option:`Arnoldi.num_ev` entries,
            sorted according to :cfg:option:`Arnoldi.which`.
        psis : list of :class:`~cyten.tensors.Tensor`
            Corresponding best eigenvectors (estimates).
        N : int
            Used dimension of the Krylov space, i.e., how many iterations where performed.

        """
        assert self.N_cache >= self.N_max
        N = self._build_krylov()
        E0 = self.Es[N - 1, : self.num_ev]
        if self.E_shift is not None:
            E0 = E0 - self.E_shift
        if N == 1:
            return E0, [self.psi0], N  # no better estimate available
        return E0, self._calc_result_full(N), N

    def _build_krylov(self):
        """Build the Krylov space and the projection of H into it.

        Returns the number of steps performed.
        """
        h = self._h_krylov
        w = self.psi0  # initialize
        w_norm = _abs_number(norm(w))
        self.psi0 = w / w_norm
        for k in range(self.N_max):
            w = scalar_multiply(1.0 / w_norm, w)
            self._to_cache(w)
            w = self.H.matvec(w)
            for i, v_i in enumerate(self._cache):
                ov = inner(v_i, w)
                h[i, k] = _to_number(ov)
                w = w - ov * v_i
            h[k + 1, k] = w_norm = _abs_number(norm(w))
            self._calc_result_krylov(k)
            if w_norm < self._cutoff or (k + 1 >= self.N_min and self._converged(k)):
                break
        return k + 1

    def _calc_result_krylov(self, k):
        """Calculate ground state of _h_krylov[:k+1, :k+1]"""
        h = self._h_krylov
        if k == 0:
            self.Es[0, 0] = h[0, 0]
            self._result_krylov = np.ones([1, 1], self._dtype_h_krylov)
        else:
            # Diagonalize h
            E_kr, v_kr = np.linalg.eig(h[: k + 1, : k + 1])  # not hermitian!
            sort = argsort(E_kr, self.which)
            self.Es[k, : k + 1] = E_kr[sort]
            self._result_krylov = v_kr[:, sort]  # ground state of _h_krylov

    def _calc_result_full(self, N: int) -> list[VectorLike]:
        """Transform the :attr:`_result_krylov` from the Krylov ONB to the original basis.

        Construct the result ``psi_f = sum_k  _result_krylov[k] psi[k]``, where ``psi[k]``
        is the k-th vector of the ONB of the Krylov space generated during the iteration.
        """
        psis = []
        for i in range(min(N, self.num_ev)):
            vf = self._result_krylov[:, i]
            vf = np.real_if_close(vf)  # try to convert to real:
            # e.g. the dominant eigenvectors of the MPS transfermatrix should be equivalent to
            # the power method, which will be purely real for H.dtype=float, even if there might
            # be other eigenvectors which are complex
            assert N == len(vf) > 1
            krylov_basis = self._cache
            assert len(krylov_basis) >= N
            psi = vf[0] * krylov_basis[0]  # copy!
            # and the last len_cache vectors have been cached
            for k in range(1, N):
                psi = psi + vf[k] * krylov_basis[k]

            psi_norm = _abs_number(norm(psi))
            if abs(1.0 - psi_norm) > 1.0e-5:
                # One reason can be that `H` is not Hermitian
                # Otherwise, the matrix (even if small) might be ill conditioned.
                # If you get this warning, you can try to set the parameters
                # `reortho`=True and `N_cache` >= `N_max`
                logger.warning('poorly conditioned H matrix in Arnoldi! |psi| = %f', psi_norm)
            psis.append(scalar_multiply(1.0 / psi_norm, psi))
        return psis

    def _to_cache(self, psi):
        """Add psi to cache, keep at most self.N_cache."""
        cache = self._cache
        cache.append(psi)
        assert len(cache) <= self.N_cache

    def _converged(self, k):
        v0 = self._result_krylov[:, 0]
        E = self.Es[k, :]  # current energies
        RitzRes = abs(v0[k]) * self._h_krylov[k + 1, k]
        gap = max(min([np.min(np.abs(E[i + 1 :] - E[i])) for i in range(self.num_ev)]), self.min_gap)
        P_err = (RitzRes / gap) ** 2
        Delta_E0 = self.Es[k - 1, 0] - E[0]
        return P_err < self.P_tol and Delta_E0 < self.E_tol


class ArnoldiEvolution(Arnoldi):
    r"""Compute :math:`exp(\delta H) |\psi_0\rangle` using Arnoldi for non-Hermitian `H`.

    Drop-in replacement for :class:`LanczosEvolution` when `H` is not Hermitian.
    Builds an upper Hessenberg projection of `H` via full Gram-Schmidt orthogonalization
    (Arnoldi iteration), then computes the matrix exponential of the small projected matrix
    via eigendecomposition (``numpy.linalg.eig`` + pointwise scalar exponentials).

    Parameters
    ----------
    H, psi0, options :
        Same as :class:`Arnoldi`. Note that `H` need not be Hermitian.

    Options
    -------
    .. cfg:config :: ArnoldiEvolution
        :include: Arnoldi

        E_tol, which, num_ev :
            Inherited but ignored.

    Attributes
    ----------
    delta : float/complex or None
        Prefactor of H in the exponential.
    _result_norm : float
        Norm of the result vector.

    """

    def __init__(self, H, psi0, options):
        super().__init__(H, psi0, options)
        self._result_norm = 1.0
        self.delta = None
        # Arnoldi._build_krylov does not set _psi0_norm; do it here.
        self._psi0_norm = _abs_number(norm(psi0))

    def run(self, delta, normalize=None):
        """Compute ``expm(delta * H).dot(psi0)`` using Arnoldi.

        Parameters
        ----------
        delta : float/complex
            Prefactor of H in the exponential. Note that the complex ``i`` is *not* included.
        normalize : bool
            Whether to normalize the result. Defaults to ``False``.
            Unlike :class:`LanczosEvolution` (which defaults to ``np.real(delta) == 0``),
            non-Hermitian evolution does not in general preserve the norm, so normalization
            would strip physically meaningful decay or growth and is off by default.

        Returns
        -------
        psi_f : :class:`~cyten.tensors.Tensor`
            Best approximation for ``expm(delta * H).dot(psi0)``.
        N : int
            Krylov space dimension used.

        """
        assert self.N_cache >= self.N_max  # all basis vectors required for back-transform
        self.delta = delta
        # Arnoldi._to_cache does not pop old entries, so we must clear state between calls.
        self._reset_krylov_state()
        N = self._build_krylov()
        if N > 1:
            logger.debug('ArnoldiEvolution N=%d, |result[-1]|=%.3e', N, abs(self._result_krylov[N - 1, 0]))
        else:
            logger.debug('ArnoldiEvolution N=1, |h[0,0]|=%.3e', abs(self._h_krylov[0, 0]))
        if N == 1:
            result_full = self._result_krylov[0, 0] * self.psi0
        else:
            result_full = self._calc_result_full_evolution(N)
        if normalize is None:
            normalize = False
        if normalize:
            return result_full, N
        return (self._psi0_norm * self._result_norm) * result_full, N

    def _calc_result_krylov(self, k):
        """Compute ``exp(delta * h[:k+1, :k+1]) @ e0`` via eigendecomposition.

        For a general (non-Hermitian) matrix h with right eigenvectors V and eigenvalues E,
        h = V diag(E) V^{-1}, so exp(delta h) e0 = V diag(exp(delta E)) (V^{-1} e0).
        This mirrors :class:`LanczosEvolution` (which uses ``eigh`` and V^{-1}=V†),
        but uses ``eig`` and an explicit solve instead of conjugate-transpose.
        """
        h = self._h_krylov
        delta = self.delta
        if k == 0:
            exp_dE = np.exp(delta * h[0, 0])
            self._result_norm = np.abs(exp_dE)
            self._result_krylov = np.array([[exp_dE / self._result_norm]])
        else:
            E_kr, v_kr = np.linalg.eig(h[: k + 1, : k + 1])
            # V^{-1} e0 = first column of V^{-1}; use solve for numerical stability
            e0 = np.zeros(k + 1, dtype=complex)
            e0[0] = 1.0
            coeff = np.linalg.solve(v_kr, e0)
            exp_dH_e0 = np.dot(v_kr, np.exp(E_kr * delta) * coeff)
            self._result_norm = np.linalg.norm(exp_dH_e0)
            # Shape (k+1, 1) to be compatible with Arnoldi._converged reading [:, 0].
            self._result_krylov = (exp_dH_e0 / self._result_norm).reshape(-1, 1)

    def _converged(self, k):
        """Converged when the last coefficient of ``expm(delta*h)*e0`` is below `P_tol`."""
        return np.abs(self._result_krylov[k, 0]) < self.P_tol

    def _calc_result_full_evolution(self, N):
        """Back-transform Krylov coefficients to the original basis."""
        vf = self._result_krylov[:N, 0]  # 1-D coefficient array
        cache = self._cache
        assert len(cache) >= N
        psif = vf[0] * cache[0]
        for k in range(1, N):
            psif = psif + vf[k] * cache[k]
        psif_norm = _abs_number(norm(psif))
        if abs(1.0 - psif_norm) > 1.0e-5:
            logger.warning('poorly conditioned H in ArnoldiEvolution! |psi|=%f', psif_norm)
        return scalar_multiply(1.0 / psif_norm, psif)


class LanczosGroundState(KrylovBased):
    """Lanczos algorithm to find the ground state.

    **Assumes** that `H` is hermitian.


    Options
    -------
    .. cfg:config :: LanczosGroundState
        :include: KrylovBased

        E_tol : float
            Stop if energy difference per step < `E_tol`
        N_cache : int
            The maximum number of `psi` to keep in memory during the first iteration.
            By default, we keep all states (up to N_max).
            Set this to a number >= 2 if you are short on memory.
            The penalty is that one needs another Lanczos iteration to
            determine the ground state in the end, i.e., runtime is large.
    """

    _dtype_h_krylov = np.float64
    _dtype_E = np.float64

    def __init__(self, H, psi0: VectorLike, options):
        super().__init__(H=H, psi0=psi0, options=options)
        self.E_tol = self.options.get('E_tol', np.inf)
        self.N_cache = self.options.get('N_cache', self.N_max)
        if self.N_cache < 2:
            raise ValueError('Need to cache at least two vectors.')

    def run(self):
        """Find the ground state of H.

        Returns
        -------
        E0 : float
            Ground state energy (estimate).
        psi0 : :class:`~cyten.tensors.VectorLike`
            Ground state vector (estimate).
        N : int
            Used dimension of the Krylov space, i.e., how many iterations where performed.

        """
        N = self._build_krylov()
        E0 = self.Es[N - 1, 0]
        if N > 1:
            logger.debug(
                'Lanczos N=%d, gap=%.3e, DeltaE0=%.3e, _result_krylov[-1]=%.3e',
                N,
                self.Es[N - 1, 1] - E0,
                self.Es[N - 2, 0] - E0,
                self._result_krylov[-1],
            )
        else:
            logger.debug('Lanczos N=%d, first alpha=%.3e, beta=%.3e', N, self._h_krylov[0, 0], self._h_krylov[0, 1])
        if self.E_shift is not None:
            E0 -= self.E_shift
        if N == 1:
            return E0, self.psi0, N  # no better estimate available
        return E0, self._calc_result_full(N), N

    def _build_krylov(self):
        """Build the Krylov space and the projection of H into it.

        Returns the number of steps performed.
        """
        h = self._h_krylov
        w = self.psi0  # initialize
        beta = _abs_number(norm(w))
        if beta < self._cutoff:
            raise ValueError(f'Norm of self.psi0 too small: {beta}')
        self.psi0 = w / beta
        if self._psi0_norm is None:
            # this is only needed for normalization in LanczosEvolution
            self._psi0_norm = beta
        for k in range(self.N_max):
            w = scalar_multiply(1.0 / beta, w)
            self._to_cache(w)
            w = self.H.matvec(w)
            alpha = _to_number(inner(w, self._cache[-1]).real())
            h[k, k] = alpha
            self._calc_result_krylov(k)
            w = w - alpha * self._cache[-1]
            if self.reortho:
                for c in self._cache[:-1]:
                    w = w - inner(c, w) * c
            elif k > 0:
                w = w - beta * self._cache[-2]
            beta = _abs_number(norm(w))
            h[k, k + 1] = h[k + 1, k] = beta  # needed for the next step and convergence criteria
            if abs(beta) < self._cutoff or (k + 1 >= self.N_min and self._converged(k)):
                break
        return k + 1

    def _converged(self, k):
        v0 = self._result_krylov
        E = self.Es[k, :]  # current energies
        RitzRes = abs(v0[k]) * self._h_krylov[k, k + 1]
        gap = max(E[1] - E[0], self.min_gap)
        P_err = (RitzRes / gap) ** 2
        Delta_E0 = self.Es[k - 1, 0] - E[0]
        return P_err < self.P_tol and Delta_E0 < self.E_tol

    def _rebuild_krylov_for_result_full(self, psif, N_max):
        vf = self._result_krylov
        h = self._h_krylov
        w = self.psi0  # initialize
        beta = None
        for k in range(0, N_max):
            self._to_cache(w)
            w = self.H.matvec(w)
            alpha = h[k, k]
            w = w - alpha * self._cache[-1]
            if self.reortho:
                for c in self._cache[:-1]:
                    w = w - inner(c, w) * c
            elif k > 0:
                w = w - beta * self._cache[-2]
            beta = h[k, k + 1]  # = norm(w)
            w = scalar_multiply(1.0 / beta, w)
            psif = psif + vf[k + 1] * w
        return psif

    def _calc_result_krylov(self, k):
        """Calculate ground state of _h_krylov[:k+1, :k+1]"""
        h = self._h_krylov
        if k == 0:
            self.Es[0, 0] = h[0, 0]
            self._result_krylov = np.ones(1, np.float64)
        else:
            # Diagonalize h
            E_kr, v_kr = np.linalg.eigh(h[: k + 1, : k + 1])
            self.Es[k, : k + 1] = E_kr
            self._result_krylov = v_kr[:, 0]  # ground state of _h_krylov


class LanczosEvolution(LanczosGroundState):
    """Calculate :math:`exp(delta H) |psi0>` using Lanczos.

    It turns out that the Lanczos algorithm is also good for calculating the matrix exponential
    applied to the starting vector. Instead of diagonalizing the tri-diagonal `h` and taking the
    ground state, we now calculate ``exp(delta h) e_0`` in the Krylov ONB, where
    ``e_0 = (1, 0, 0, ...)`` corresponds to ``psi0`` in the original basis.

    Parameters
    ----------
    H, psi0, options :
        Hamiltonian, starting vector and parameters as defined in :class:`LanczosGroundState`.
        The option :cfg:option`LanczosEvolution.P_tol` defines when convergence is reached,
        see :meth:`_converged` for details.

    Options
    -------
    .. cfg:config :: LanczosEvolution
        :include: LanczosGroundState

        E_tol :
            Ignored.
        min_gap :
            Ignored.

    Attributes
    ----------
    delta : float/complex
        Prefactor of H in the exponential.
    _result_norm : float
        Norm of the resulting vector.

    """

    def __init__(self, H, psi0, options):
        super().__init__(H, psi0, options)
        self._result_norm = 1.0
        self.delta = None  # set in run()

    def run(self, delta, normalize=None):
        """Calculate ``expm(delta H).dot(psi0)`` using Lanczos.

        Parameters
        ----------
        delta : float/complex
            Time step by which we should evolve psi0: prefactor of H in the exponential.
            Note that the complex `i` is *not* included!
        normalize : bool
            Whether to normalize the resulting state.
            Defaults to ``np.real(delta) == 0``.

        Returns
        -------
        psi_f : :class:`~cyten.tensors.Tensor`
            Best approximation for ``expm(delta H).dot(psi0)``.
            If :cfg:option:`Lanczos.E_shift` is used, it's an approximation for
            ``expm(delta (H + E_shift)).dot(psi)``.
        N : int
            Krylov space dimension used.

        """
        self.delta = delta
        self._reset_krylov_state()
        N = self._build_krylov()
        if N > 1:
            logger.debug('Lanczos N=%d, |result_krylov[-1]|=%.3e', N, abs(self._result_krylov[-1]))
        else:
            logger.debug('Lanczos N=%d, first alpha=%.3e, beta=%.3e', N, self._h_krylov[0, 0], self._h_krylov[0, 1])
        if N == 1:
            result_full = self._result_krylov[0] * self.psi0  # _result_krylov[0] is only a phase
        else:
            result_full = self._calc_result_full(N)
        # result_full is normalized at this point
        if normalize is None:
            normalize = np.real(delta) == 0.0
        if normalize:
            return result_full, N
        # else:
        return (self._psi0_norm * self._result_norm) * result_full, N

    def _calc_result_krylov(self, k):
        """Calculate ``expm(delta h).dot(e0)`` for ``h = _h_krylov[:k+1, :k+1]``"""
        # self._result_krylov should be a normalized vector.
        h = self._h_krylov
        delta = self.delta
        if k == 0:
            E = h[0, 0]
            exp_dE = np.exp(delta * E)
            self._result_norm = np.abs(exp_dE)  # np.linalg.norm for individual element
            self._result_krylov = np.array([exp_dE / self._result_norm])
        else:
            #     e0 = np.zeros(k + 1, dtype=float)
            #     e0[0] = 1.
            #     exp_dH_e0 = expm(_h_krylov[:k + 1, :k + 1] * delta).dot(e0)
            # scipy.linalg.expm is using sparse tools; instead fully diagonalize
            # given that h is hermitian, this is easy:
            # H V = V diag(E)  -> H  = V E V^D
            # exp(H*delta) e_0 = V diag(exp(E*delta)) V^D e_0
            E_kr, v_kr = np.linalg.eigh(h[: k + 1, : k + 1])
            exp_dH_e0 = np.dot(v_kr, np.exp(E_kr * delta) * np.conj(v_kr[0, :]))

            self._result_norm = np.linalg.norm(exp_dH_e0)
            self._result_krylov = exp_dH_e0 / self._result_norm

    def _converged(self, k):
        return np.abs(self._result_krylov[k]) < self.P_tol


def lanczos(H, psi, options={}):
    """Simple wrapper calling ``LanczosGroundState(H, psi, options).run()``

    Parameters
    ----------
    H, psi, options:
        See :class:`LanczosGroundState`.

    Returns
    -------
    E0, psi0, N :
        See :meth:`LanczosGroundState.run`.

    """
    return LanczosGroundState(H, psi, options).run()


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
