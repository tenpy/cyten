"""Providing support for sparse algorithms (using matrix-vector products only).

Some linear algebra algorithms, e.g. Lanczos, do not require the full representations of a linear
operator, but only the action on a vector, i.e., a matrix-vector product `matvec`. Here we define
the structure of such a general operator, :class:`LinearOperator`, as it is used in our own
implementations of these algorithms (e.g., :mod:`~cyten.krylov_based`). Moreover, the
:class:`NumpyArrayLinearOperator` allows to use all the scipy sparse methods by providing
functionality to convert flat numpy arrays to and from cyten tensors.
"""
# Copyright (C) TeNPy Developers, Apache license

from __future__ import annotations

import warnings
from abc import ABCMeta, abstractmethod
from numbers import Number
from typing import Literal

import numpy as np
from scipy.sparse.linalg import ArpackNoConvergence
from scipy.sparse.linalg import LinearOperator as ScipyLinearOperator

from ..backends import TensorBackend
from ..block_backends import Dtype
from ..symmetries import Sector, Space, SymmetryError, TensorProduct
from ..tools.math import speigs, speigsh
from ..tools.misc import argsort
from ._tensors import (
    ChargedTensor,
    SymmetricTensor,
    Tensor,
    combine_legs,
    combine_to_matrix,
    inner,
    norm,
    outer,
    permute_legs,
    split_legs,
    tdot,
    zero_like,
)


def _same_legs(legs1, legs2) -> bool:
    return len(legs1) == len(legs2) and all(a == b for a, b in zip(legs1, legs2))


class LinearOperator(metaclass=ABCMeta):
    """Base class for a linear operator acting on cyten tensors.

    Attributes
    ----------
    vector_legs : list of Space
        The legs of tensors that this operator can act on.
    vector_labels : list of str or None
        Labels of the vectors that this operator can act on, or ``None``.
    dtype : Dtype
        The dtype of a full representation of the operator
    acts_on : list of str
        Labels of the state on which the operator can act. NB: Class attribute.

    """

    acts_on = None  # Derived classes should set this as a class attribute

    def __init__(self, vector_legs, dtype: Dtype, vector_labels=None):
        self.vector_legs = list(vector_legs)
        self.vector_labels = None if vector_labels is None else list(vector_labels)
        self.dtype = dtype

    @abstractmethod
    def matvec(self, vec: Tensor) -> Tensor:
        """Apply the linear operator to a "vector".

        We consider as vectors all tensors with :attr:`vector_legs`,
        and in particular allow multi-leg tensors as "vectors".
        The result of `matvec` must be a tensor of the same shape.
        """
        ...

    @abstractmethod
    def to_tensor(self, **kw) -> Tensor:
        """Compute a full tensor representation of the linear operator.

        Returns
        -------
        A tensor `t` with ``2 * N`` legs ``[a1, a2, ..., aN, aN*, ..., a2*, a1*]``, where
        ``[a1, a2, ..., aN]`` are the legs of the vectors this operator acts on.
        S.t. ``self.matvec(vec)`` is equivalent to ``tdot(t, vec, [N, ..., 2*N-1], [N-1,...,0])``.

        """
        ...

    def to_matrix(self, backend: TensorBackend = None) -> Tensor:
        """The tensor representation of self, reshaped to a matrix."""
        N = len(self.vector_legs)
        return combine_to_matrix(self.to_tensor(backend=backend), list(range(N)), list(range(N, 2 * N)))

    def adjoint(self) -> LinearOperator:
        """Return the hermitian conjugate operator.

        If `self` is hermitian, subclasses *can* choose to implement this to define
        the adjoint operator of `self` to be `self`.
        """
        raise NotImplementedError('No adjoint defined')


class TensorLinearOperator(LinearOperator):
    """Linear operator defined by a two-leg tensor with contractible legs.

    The matvec is defined by contracting one of the two legs of this tensor with the vector.
    This class is effectively a thin wrapper around tensors that allows them to be used as inputs
    for sparse linear algebra routines, such as lanczos.

    Parameters
    ----------
    tensor :
        The tensor that is contracted with the vector on matvec
    which_leg : int or str
        Which leg of `tensor` is to be contracted on matvec.

    """

    def __init__(self, tensor: SymmetricTensor, which_leg: int | str = -1):
        if tensor.num_legs != 2:
            raise ValueError('Expected a two-leg tensor')
        which_idcs = tensor.get_leg_idcs(which_leg)
        if len(which_idcs) != 1:
            raise ValueError('which_leg must refer to a single leg')
        self.which_leg = which_leg = which_idcs[0]
        self.other_leg = other_leg = 1 - which_leg
        if tensor.legs[which_leg] != tensor.legs[other_leg].dual:
            raise ValueError('Expected contractible legs')
        self.tensor = tensor
        other_label = tensor.labels[other_leg]
        super().__init__(
            vector_legs=[tensor.legs[other_leg]],
            dtype=tensor.dtype,
            vector_labels=None if other_label is None else [other_label],
        )

    def matvec(self, vec: Tensor) -> Tensor:
        assert vec.num_legs == 1
        return tdot(self.tensor, vec, [self.which_leg], [0])

    def to_tensor(self, **kw) -> Tensor:
        return permute_legs(self.tensor, codomain=[self.other_leg], domain=[self.which_leg])

    def adjoint(self) -> TensorLinearOperator:
        # dagger of an endomorphism ``[V, V.dual]`` keeps that leg order, so the contracted leg
        # stays at the same index.
        return TensorLinearOperator(tensor=self.tensor.dagger, which_leg=self.which_leg)


class LinearOperatorWrapper(LinearOperator):
    """Base class for wrapping around another :class:`LinearOperator`.

    The wrapped operator is stored as :attr:`original_operator`.
    Use :meth:`unwrapped` to recover the innermost operator.

    .. warning ::
        If there are multiple levels of wrapping operators, the order might be critical to get
        correct results; e.g. :class:`ProjectedLinearOperator` needs to be the outer-most
        wrapper to produce correct results and/or be efficient.

    Parameters
    ----------
    original_operator : :class:`LinearOperator`
        The original operator implementing the `matvec`.

    """

    def __init__(self, original_operator: LinearOperator):
        super().__init__(
            vector_legs=original_operator.vector_legs,
            dtype=original_operator.dtype,
            vector_labels=original_operator.vector_labels,
        )
        self.original_operator = original_operator

    def unwrapped(self, recursive: bool = True) -> LinearOperator:
        """Return the original :class:`LinearOperator`

        By default, unwrapping is done recursively, such that the result is *not* a `LinearOperatorWrapper`.
        """
        parent = self.original_operator
        if not recursive:
            return parent
        for _ in range(10000):
            try:
                parent = parent.unwrapped()
            except AttributeError:
                # parent has no :meth:`unwrapped`, so we can stop unwrapping
                return parent
        raise ValueError('maximum recursion depth for unwrapping reached')


class SumLinearOperator(LinearOperatorWrapper):
    """The sum of multiple operators"""

    def __init__(self, original_operator: LinearOperator, *more_operators: LinearOperator):
        super().__init__(original_operator=original_operator)
        assert all(_same_legs(op.vector_legs, original_operator.vector_legs) for op in more_operators)
        self.more_operators = more_operators
        self.dtype = Dtype.common(original_operator.dtype, *(op.dtype for op in more_operators))

    def matvec(self, vec: Tensor) -> Tensor:
        return sum((op.matvec(vec) for op in self.more_operators), self.original_operator.matvec(vec))

    def to_tensor(self, **kw) -> Tensor:
        return sum((op.to_tensor(**kw) for op in self.more_operators), self.original_operator.to_tensor(**kw))

    def adjoint(self) -> LinearOperator:
        return SumLinearOperator(self.original_operator.adjoint(), *(op.adjoint() for op in self.more_operators))


class ShiftedLinearOperator(LinearOperatorWrapper):
    """A shifted operator, i.e. ``original_operator + shift * identity``.

    This can be useful e.g. for better Lanczos convergence.
    """

    def __init__(self, original_operator: LinearOperator, shift: Number):
        if shift in [0, 0.0]:
            warnings.warn('shift=0: no need for ShiftedLinearOperator', stacklevel=2)
        super().__init__(original_operator=original_operator)
        self.shift = shift
        if np.iscomplexobj(shift):
            self.dtype = original_operator.dtype.to_complex

    def matvec(self, vec: Tensor) -> Tensor:
        return self.original_operator.matvec(vec) + self.shift * vec

    def to_tensor(self, **kw) -> Tensor:
        res = self.original_operator.to_tensor(**kw)
        identity = SymmetricTensor.from_eye(
            self.vector_legs, backend=res.backend, labels=self.vector_labels, dtype=res.dtype
        )
        return res + self.shift * identity

    def adjoint(self):
        return ShiftedLinearOperator(original_operator=self.original_operator.adjoint(), shift=np.conj(self.shift))


class ProjectedLinearOperator(LinearOperatorWrapper):
    """Projected version ``P H P + penalty * (1 - P)`` of an original operator ``H``.

    The projector ``P = 1 - sum_o |o> <o|`` is given in terms of a set :attr:`ortho_vecs` of vectors
    ``|o>``.

    The result is that all vectors from the subspace spanned by the :attr:`ortho_vecs` are eigenvectors
    with eigenvalue `penalty`, while the eigensystem in the "rest" (i.e. in the orthogonal complement
    to that subspace) remains unchanged.

    This can be used to exclude the :attr:`ortho_vecs` from extremal eigensolvers, i.e. to find
    the extremal eigenvectors among those that are orthogonal to the :attr:`ortho_vecs`.
    In previous versions of tenpy, this behavior was achieved by an argument called `orthogonal_to`.
    If this is done, at least for krylov-based eigensolvers such as lanczos, the penalty should be chosen
    such that the `ortho_vecs` are somewhere in the bulk of the spectrum.
    This is because lanczos has best convergence for the extremal eigenvalues and we want to converge
    the solutions well, not the `ortho_vecs`.
    E.g. for a typical Hamiltonian with a spectrum symmetric around zero, ``project_operator=True``
    and ``penalty=None`` shifts the `ortho_vecs` to eigenvalue zero, thus fulfilling this criterion.
    However, for operators with e.g. strictly positive spectrum, this prescription might fail.

    Parameters
    ----------
    original_operator : :class:`LinearOperator`-like
        The original operator, denoted ``H`` in the summary above.
    ortho_vecs : list of :class:`~cyten.tensors.Tensor`
        The list of vectors spanning the projected space.
        They need not be orthonormal, as Gram-Schmidt is performed on them explicitly.
    project_operator: bool
        If False (True per default), the projection of the operator ``H -> P H P`` is skipped
        and ``H + penalty * (1 - P)`` is represented instead.
    penalty : complex, optional
        See summary above. Defaults to ``None``, which is equivalent to ``0.``.

    """

    def __init__(
        self,
        original_operator: LinearOperator,
        ortho_vecs: list[Tensor],
        project_operator: bool = True,
        penalty: Number = None,
    ):
        if len(ortho_vecs) == 0:
            warnings.warn('empty ortho_vecs: no need for ProjectedLinearOperator', stacklevel=2)
        if not project_operator and penalty is None:
            warnings.warn('project_operator=False and penalty=None means ProjectedLinearOperator does not do anything')
        super().__init__(original_operator=original_operator)
        assert all(_same_legs(v.legs, original_operator.vector_legs) for v in ortho_vecs)
        self.ortho_vecs = gram_schmidt(ortho_vecs)
        self.project_operator = project_operator
        self.penalty = penalty

    def matvec(self, vec: Tensor) -> Tensor:
        res = vec
        # 1: res = P vec
        if self.project_operator:
            # form ``P vec`` and keep coefficients for later use in the penalty term
            coefficients = []
            for o in self.ortho_vecs:
                c = inner(o, res)
                coefficients.append(c)
                res = res - c * o
        else:
            coefficients = [inner(o, res) for o in self.ortho_vecs]
        # 2: res = H P vec
        res = self.original_operator.matvec(res)
        # 3: res = P H P vec
        if self.project_operator:
            for o in self.ortho_vecs:
                res = res - inner(o, res) * o
        # 4: res = P H P vec + (1 - P) vec
        if self.penalty is not None:
            for c, o in zip(coefficients, self.ortho_vecs):
                res = res + self.penalty * c * o
        # done
        return res

    def to_tensor(self, **kw) -> Tensor:
        res = self.original_operator.to_tensor(**kw)
        P_ortho = zero_like(res)
        for o in self.ortho_vecs:
            P_ortho = P_ortho + outer(o, o.dagger)
        if self.project_operator:
            identity = SymmetricTensor.from_eye(
                self.vector_legs, backend=res.backend, labels=self.vector_labels, dtype=res.dtype
            )
            P = identity - P_ortho
            N = len(self.vector_legs)
            first = list(range(N))
            last = list(range(N, 2 * N))
            rev_first = list(reversed(first))
            res = tdot(res, P, last, rev_first)
            res = tdot(P, res, last, rev_first)
        if self.penalty is not None:
            res = res + self.penalty * P_ortho
        return res

    def adjoint(self) -> LinearOperator:
        return ProjectedLinearOperator(
            original_operator=self.original_operator.adjoint(),
            ortho_vecs=self.ortho_vecs,  # hc(|o> <o|) = |o> <o|  ->  can use same ortho_vecs
            project_operator=self.project_operator,
            penalty=None if self.penalty is None else np.conj(self.penalty),
        )


class NumpyArrayLinearOperator(ScipyLinearOperator):
    """Square Linear operator acting on numpy arrays based on a matvec acting on cyten tensors.

    Note that this class represents a square linear operator.

    Parameters
    ----------
    cyten_matvec : callable
        Function with signature ``cyten_matvec(vec: Tensor) -> Tensor``.
        Has to return a tensor with the same legs and has to be linear.
        Unless `labels` are given, the leg order of the output must be the same as for the input.
    legs : list of :class:`~cyten.spaces.ElementarySpace`
        The legs of a Tensor that `cyten_matvec` can act on.
    backend : :class:`~cyten.backends.abstract_backend.Backend`
        The backend for self
    dtype
        The numpy dtype for this operator.
    labels : list of str, optional
        The labels for inputs to `cyten_matvec`.
    charge_sector : None | Sector | 'trivial'
        If given, only the specified charge sector is considered.
        Per default, or if the string ``'trivial'`` is given, the trivial sector of the symmetry is used.
        ``None`` stands for *all* sectors: the numpy vector is the full dense representation,
        converted via a :class:`~cyten.tensors.ChargedTensor` with a specified
        :attr:`~cyten.tensors.ChargedTensor.charged_state`.
        This requires a group-like symmetry with :attr:`~cyten.symmetries.Symmetry.can_be_dropped`.

    Attributes
    ----------
    cyten_matvec : callable
        Function with signature ``cyten_matvec(vec: Tensor) -> Tensor``.
    legs : list of :class:`~cyten.spaces.Space`
        The legs of a Tensor that `cyten_matvec` can act on.
    backend : :class:`~cyten.backends.abstract_backend.Backend`
        The backend for self
    dtype
        The numpy dtype for this operator.
    labels : list of str, optional
        The labels for inputs to `cyten_matvec`.
    charge_sector : None | Sector | 'trivial'
        If given, only the specified charge sector is considered.
        If ``'trivial'`` is given, the trivial sector of the symmetry is used.
        ``None`` stands for *all* sectors, represented via :class:`~cyten.tensors.ChargedTensor`
        with a specified :attr:`~cyten.tensors.ChargedTensor.charged_state`.
    matvec_count : int
        The number of times `cyten_matvec` was called.
    N : int
        The length of the numpy vectors that this operator acts on
    domain : :class:`~cyten.spaces.TensorProduct`
        The product of the :attr:`legs`. Self is an operator on either this entire space,
        or one of its sectors, as specified by :attr:`charge_sector`.
    pipe : LegPipe
        Combined pipe of :attr:`legs`, used for convertion to/from a 1-leg tensor.
    symmetry
        The symmetry of all involved spaces
    shape : (int, int)
        The shape of self as an operator on 1D numpy arrays

    """

    def __init__(
        self,
        cyten_matvec,
        legs: TensorProduct | list[Space],
        backend: TensorBackend,
        dtype,
        labels: list[str] = None,
        charge_sector: None | Sector | Literal['trivial'] = 'trivial',
    ):
        self.cyten_matvec = cyten_matvec
        self.backend = backend
        if not isinstance(legs, TensorProduct):
            self.legs = list(legs)
            self.domain = TensorProduct(self.legs)
            self.symmetry = self.legs[0].symmetry
        else:
            self.domain = legs
            self.legs = list(legs.factors)
            self.symmetry = legs.symmetry
        if len(self.legs) == 1:
            self.pipe = self.legs[0]
        else:
            self.pipe = backend.make_pipe(self.legs, is_dual=False)
        self.matvec_count = 0
        self.labels = labels

        self.shape = None  # set by charge_sector.setter
        self._charge_sector = None  # set by charge_sector.setter
        self.charge_sector = charge_sector  # uses setter with its input checks and conversions

        ScipyLinearOperator.__init__(self, dtype=dtype, shape=self.shape)

    @classmethod
    def from_Tensor(
        cls,
        tensor: SymmetricTensor,
        legs1: list[int | str],
        legs2: list[int | str],
        charge_sector: None | Sector | Literal['trivial'] = 'trivial',
    ) -> NumpyArrayLinearOperator:
        """Create a :class:`NumpyArrayLinearOperator` from a tensor that acts via contraction (`tdot`).

        The `cyten_matvec` acting on ``vec`` is given by ``tdot(tensor, vec, legs1, legs2)``.

        Parameters
        ----------
        tensor : Tensor
            A tensors whose legs specified by `legs1` are contractible with the remaining legs.
        legs1 : list of {int | str}
            Which legs of `tensor` should be contracted on matvec
        legs2 : list of {int | str}
            Which legs of the "vector" should be contracted on `matvec`
        charge_sector : None | Sector | 'trivial'
            If given, only the specified charge sector is considered.
            If ``'trivial'`` is given, the trivial sector of the symmetry is used.
            ``None`` stands for *all* sectors.

        """
        idcs1 = tensor.get_leg_idcs(legs1)
        tensor_contr_legs = [tensor.legs[idx] for idx in idcs1]
        res_legs = [tensor.legs[idx] for idx in range(tensor.num_legs) if idx not in idcs1]
        res_labels = [tensor.labels[idx] for idx in range(tensor.num_legs) if idx not in idcs1]
        if None in res_labels:
            res_labels = None
        vec_contr_legs = []
        for l in legs2:
            if isinstance(l, int):
                vec_contr_legs.append(res_legs[l])
            else:
                vec_contr_legs.append(tensor.get_leg(l))
        if not all(l_t == l_v.dual for l_t, l_v in zip(tensor_contr_legs, vec_contr_legs)):
            raise ValueError('Expected contractible legs')

        def cyten_matvec(vec):
            return tdot(tensor, vec, legs1, legs2)

        return cls(
            cyten_matvec,
            legs=vec_contr_legs,
            backend=tensor.backend,
            dtype=tensor.dtype.to_numpy_dtype(),
            labels=res_labels,
            charge_sector=charge_sector,
        )

    @classmethod
    def from_matvec_and_vector(
        cls, cyten_matvec, vector: Tensor, dtype=None
    ) -> tuple[NumpyArrayLinearOperator, np.ndarray]:
        """Create a :class:`NumpyArrayLinearOperator` from a matvec and a vector that it can act on.

        This is a convenience wrapper around the constructor where arguments are inferred
        from the example `vector` that is given.
        Additionally, the `vector` is converted via :meth:`tensor_to_flat_array`.
        The resulting `NumpyArrayLinearOperator` has a `charge_sector` set to be the sector of
        `vector`.

        Parameters
        ----------
        cyten_matvec : callable
            Function with signature ``cyten_matvec(vec: Tensor) -> Tensor``.
            Has to return a tensor with the same leg and has to be linear.
        vector : :class:`~cyten.tensors.Tensor` | :class:`~cyten.tensors.ChargedTensor`
            A tensor that `cyten_matvec` can act on.
            If a ChargedTensor with a single one-dimensional charge, that sector is used as
            :attr:`charge_sector`. If it has a specified :attr:`~cyten.tensors.ChargedTensor.charged_state`
            spanning multiple sectors, :attr:`charge_sector` is ``None``.
        dtype
            The *numpy* dtype of the operator. Per default, the dtype of `vector` is used.

        Returns
        -------
        op : :class:`NumpyArrayLinearOperator`
            The resulting operator
        vec_flat : 1D ndarray
            Flat numpy vector representing `vector` within its charge sector.

        """
        if isinstance(vector, ChargedTensor):
            charge = vector.charge_leg
            try:
                single_charge = charge.num_sectors == 1 and charge.multiplicities[0] == 1
            except AttributeError:
                # e.g. a LegPipe charge leg from the all-sector identity construction
                single_charge = False
            if single_charge:
                sector = charge.sector_decomposition[0]
            elif vector.charged_state is not None:
                sector = None
            else:
                raise ValueError('Cannot infer charge_sector from a ChargedTensor with unspecified charged_state')
        else:
            sector = 'trivial'
        if dtype is None:
            dtype = vector.dtype.to_numpy_dtype()
        labels = vector.labels
        if labels is not None and not any(l is not None for l in labels):
            labels = None
        op = cls(
            cyten_matvec,
            legs=vector.legs,
            backend=vector.backend,
            dtype=dtype,
            labels=labels,
            charge_sector=sector,
        )
        vec_flat = op.tensor_to_flat_array(vector)
        return op, vec_flat

    @property
    def charge_sector(self):
        return self._charge_sector

    @charge_sector.setter
    def charge_sector(self, value):
        if isinstance(value, str) and value == 'trivial':
            sector = self.symmetry.trivial_sector
        elif value is None:
            if not self.symmetry.can_be_dropped:
                raise SymmetryError(
                    'charge_sector=None uses ChargedTensor.charged_state and is only defined '
                    f'for symmetries that can be dropped. Got {self.symmetry}.'
                )
            sector = None
        else:
            assert self.symmetry.is_valid_sector(value)
            sector = value
        self._charge_sector = value
        if sector is None:
            size = int(self.domain.dim)
        else:
            sector_idx = self.domain.sector_decomposition_where(sector)
            if sector_idx is None:
                raise ValueError('Domain of linear operator does not have this sector')
            size = int(self.domain.block_size(sector))
        self.shape = (size, size)

    def _matvec(self, vec):
        """Matvec operation acting on a numpy ndarray of the selected charge sector.

        Parameters
        ----------
        vec : np.ndarray
            A length ``N`` vector (or ``N`` x 1 matrix) where ``N`` is the total dimension
            of the selected charge sector in the parent space, or the total dimension of the
            parent space if "all" charge sectors are selected.

        Returns
        -------
        matvec_vec : 1D ndarray
            The result of the linear operation as a length ``N`` vector

        """
        vec = np.asarray(vec)
        if vec.ndim != 1:  # convert Nx1 matrix to vector
            vec = np.squeeze(vec, axis=1)
            assert vec.ndim == 1
        tens = self.flat_array_to_tensor(vec)
        tens = self.cyten_matvec(tens)
        self.matvec_count += 1
        return self.tensor_to_flat_array(tens)

    def _combine_vector_legs(self, tens: Tensor) -> Tensor:
        if tens.num_legs == 1:
            return tens
        return combine_legs(tens, list(range(tens.num_legs)), pipes=[self.pipe])

    def _split_vector_legs(self, tens: Tensor) -> Tensor:
        if len(self.legs) == 1:
            return tens
        return split_legs(tens, 0)

    def flat_array_to_tensor(self, vec: np.ndarray) -> Tensor:
        """Convert flat numpy data to a tensor in the selected charge sector."""
        assert vec.shape == (self.shape[1],)
        block = self.backend.block_backend.block_from_numpy(vec)
        if self._charge_sector is None:
            # The numpy vector is the full dense state on the parent space. Represent it as
            # ChargedTensor(from_eye, charged_state=block): the identity invariant part maps the
            # charge leg onto the vector space, and charged_state holds the dense components.
            inv = SymmetricTensor.from_eye(
                [self.pipe],
                backend=self.backend,
                labels=[None, ChargedTensor._CHARGE_LEG_LABEL],
                dtype=Dtype.from_numpy_dtype(self.dtype),
            )
            tens = ChargedTensor(inv, charged_state=block)
        elif isinstance(self._charge_sector, str) and self._charge_sector == 'trivial':
            tens = SymmetricTensor.from_dense_block_trivial_sector(vector=block, space=self.pipe, backend=self.backend)
        else:
            tens = ChargedTensor.from_dense_block_single_sector(
                vector=block,
                space=self.pipe,
                sector=self._charge_sector,
                backend=self.backend,
            )
        res = self._split_vector_legs(tens)
        if self.labels is not None:
            res.set_labels(self.labels)
        return res

    def tensor_to_flat_array(self, tens: Tensor) -> np.ndarray:
        """Convert a tensor in the selected charge sector to a flat numpy array."""
        if (
            self.labels is not None
            and all(l is not None for l in self.labels)
            and all(l in tens.labels for l in self.labels)
        ):
            tens = permute_legs(tens, tens.get_leg_idcs(self.labels))
        tens = self._combine_vector_legs(tens)
        if self._charge_sector is None:
            res = tens.to_dense_block(understood_braiding=True)
        elif isinstance(self._charge_sector, str) and self._charge_sector == 'trivial':
            res = tens.to_dense_block_trivial_sector()
        else:
            res = tens.to_dense_block_single_sector()
        res = self.backend.block_backend.to_numpy(res)
        res = np.reshape(res, (self.shape[0],))
        return res

    def eigenvectors(
        self,
        num_ev: int = 1,
        max_num_ev: int = None,
        max_tol: float = 1.0e-12,
        which: str = 'LM',
        v0_np: np.ndarray = None,
        v0_tensor: Tensor = None,
        cutoff: float = 1.0e-10,
        hermitian: bool = False,
        **kwargs,
    ):
        """Find the (dominant) eigenvector(s) of self using :func:`scipy.sparse.linalg.eigs`.

        If a charge_sector was specified, these are the dominant eigenvectors *within that sector*.
        Otherwise, we look in all charge sectors.

        Parameters
        ----------
        num_ev : int
            Number of eigenvalues/vectors to look for.
        max_num_ev : int
            :func:`scipy.sparse.linalg.speigs` sometimes raises a NoConvergenceError for small
            `num_ev`, which might be avoided by increasing `num_ev`. As a work-around,
            we try it again in the case of an error, just with larger `num_ev` up to `max_num_ev`.
            ``None`` defaults to ``num_ev + 2``.
        max_tol : float
            After the first `NoConvergenceError` we increase the `tol` argument to that value.
        which : str
            Which eigenvalues to look for, see :func:`scipy.sparse.linalg.eigs`.
            More details also in :func:`~cyten.tools.misc.argsort`.
        v0_np : 1D ndarray
            Initial guess as a flat numpy array, i.e. a suitable input to :meth:`_matvec`.
        v0_tensor : :class:`~cyten.tensors.Tensor` | :class:`~cyten.tensors.ChargedTensor`
            Initial guess as a tensor, i.e. a suitable input to :meth:`tensor_to_flat_array`.
        cutoff : float
            Only used if ``self.charge_sector is None``; in that case it determines when entries in
            a given charge-block are considered nonzero, and what counts as degenerate.
        hermitian : bool
            If False (default), use :func:`scipy.sparse.linalg.eigs`
            If True, assume that self is hermitian and use :func:`scipy.sparse.linalg.eigsh`.
        **kwargs :
            Further keyword arguments given to :func:`scipy.sparse.linalg.eigsh` or
            :func:`scipy.sparse.linalg.eigs`, respectively.

        Returns
        -------
        eta : 1D ndarray
            The eigenvalues, sorted according to `which`.
        w : list of :class:`~cyten.tensors.Tensor` or :class:`~cyten.tensors.ChargedTensor`
            The corresponding eigenvectors as tensors.

        """
        if max_num_ev is None:
            max_num_ev = num_ev + 2
        if v0_tensor is not None:
            assert v0_np is None
            v0_np = self.tensor_to_flat_array(v0_tensor)
        if v0_np is not None:
            kwargs['v0'] = v0_np

        for k in range(num_ev, max_num_ev + 1):
            if k > num_ev:
                warnings.warn(f'Increasing `num_ev` to {k}')
            try:
                if hermitian:
                    eta, A = speigsh(self, k=k, which=which, **kwargs)
                else:
                    eta, A = speigs(self, k=k, which=which, **kwargs)
                break
            except ArpackNoConvergence:
                if k == max_num_ev:
                    raise
            kwargs['tol'] = max(max_tol, kwargs.get('tol', 0))
        cutoff = max(cutoff, 10 * kwargs.get('tol', 1.0e-16))
        A = np.real_if_close(A)

        vecs = [self.flat_array_to_tensor(A[:, j]) for j in range(A.shape[1])]

        perm = argsort(eta, which)
        return np.array(eta)[perm], [vecs[j] for j in perm]


class HermitianNumpyArrayLinearOperator(NumpyArrayLinearOperator):
    """Hermitian variant of :class:`NumpyArrayLinearOperator`.

    Note that we don't check hermicity of :meth:`matvec`.
    """

    def _adjoint(self):
        return self

    def eigenvectors(self, *args, **kwargs):
        kwargs['hermitian'] = True
        return NumpyArrayLinearOperator.eigenvectors(self, *args, **kwargs)


def gram_schmidt(vecs: list[Tensor], rcond=1.0e-14) -> list[Tensor]:
    """Gram-Schmidt orthonormalization of a list of tensors.

    Parameters
    ----------
    vecs : list of :class:`~cyten.tensors.Tensor`
        The list of vectors to be orthogonalized. All with the same legs.
    rcond : _type_, optional
        Vectors of ``norm < rcond`` (after projecting out previous vectors) are discarded.

    Returns
    -------
    list of :class:`~cyten.tensors.Tensor`
        A list of orthonormal vectors which span the same space as `vecs`.

    """
    res = []
    for vec in vecs:
        for other in res:
            ov = inner(other, vec)
            vec = vec - ov * other
        n = abs(norm(vec).to_numpy())
        if n > rcond:
            res.append(vec * (1.0 / float(n)))
    return res
