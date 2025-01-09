"""The spaces, i.e. the legs of a tensor.

TODO elaborate
"""
# Copyright (C) TeNPy Developers, Apache license
from __future__ import annotations
from abc import ABCMeta, abstractmethod
import numpy as np
from numpy import ndarray
import bisect
import itertools as it
from math import prod
from typing import TYPE_CHECKING, Sequence, Iterator, Literal
import warnings

from .dummy_config import printoptions
from .symmetries import (Sector, SectorArray, Symmetry, ProductSymmetry, no_symmetry, FusionStyle,
                         SymmetryError)
from .tools.misc import (inverse_permutation, rank_data, to_iterable, make_stride,
                         find_row_differences, unstridify, iter_common_sorted_arrays)
from .tools.string import format_like_list
from .trees import FusionTree, fusion_trees

if TYPE_CHECKING:
    from .backends.abstract_backend import Block

__all__ = ['Leg', 'Space', 'ElementarySpace']


class Leg(metaclass=ABCMeta):
    """Common base class for a single leg of a tensor.

    A single leg on a tensor can either be an :class:`ElementarySpace` or, e.g. as the result
    of combining legs, a :class:`LegPipe`.

    TODO do we need the following at this level?
     - drop_symmetry
     - repr
     - num_parameters
     - tools for displaying the leg, in Tensor.__repr__ or in Tensor.ascii_diagram

    Attributes
    ----------
    symmetry : Symmetry
        The symmetry associated with this leg.
    dim : int or float
        The (quantum-)dimension of this leg.
        Is integer if ``symmetry.can_be_dropped``, otherwise may be float.
    is_dual : bool
        A boolean flag that changes when the :attr:`dual` is taken. May or may not have additional
        meaning and implications, depending on the concrete subclass of :class:`Leg`.
    """
    
    def __init__(self, symmetry: Symmetry, dim: int | float, is_dual: bool):
        self.symmetry = symmetry
        self.dim = dim
        self.is_dual = is_dual

    def test_sanity(self):
        pass

    @abstractmethod
    def as_Space(self) -> Space:
        """Convert to (an appropriate subclass of) :class:`Space`."""
        ...

    def as_ElementarySpace(self, is_dual: bool = False) -> ElementarySpace:
        """Convert to an isomorphic :class:`ElementarySpace`"""
        # can be overridden for performance
        return self.as_Space().as_ElementarySpace(is_dual=is_dual)

    @property
    @abstractmethod
    def dual(self) -> Leg:
        """The dual leg, that is obtained when bending this leg."""
        ...


class LegPipe(Leg):
    """A group of legs, i.e. resulting from :func:`~cyten.tensors.combine_legs`.

    Note that the abelian backend defines a custom subclass.

    TODO elaborate

    Attributes
    ----------
    legs
        The legs that were grouped, and that this pipe can be split into.
    """

    def __init__(self, legs: Sequence[Leg], is_dual: bool = False):
        self.legs = legs[:]
        self.num_legs = num_legs = len(legs)
        assert num_legs > 0
        Leg.__init__(self, symmetry=legs[0].symmetry, dim=prod(l.dim for l in legs),
                     is_dual=is_dual)

    def test_sanity(self):
        assert all(l.symmetry == self.symmetry for l in self.legs)
        for l in self.legs:
            l.test_sanity()
        Leg.test_sanity(self)

    def as_Space(self):
        return TensorProduct([l.as_Space() for l in self.legs], symmetry=self.symmetry)

    @property
    def dual(self) -> LegPipe:
        return LegPipe([l.dual for l in reversed(self.legs)], is_dual=not self.is_dual)

    def __getitem__(self, idx):
        return self.legs[idx]

    def __iter__(self):
        return iter(self.legs)

    def __len__(self):
        return self.num_legs


class Space(metaclass=ABCMeta):
    r"""Base class for symmetry spaces, see :class:`ElementarySpace` for the standard case.

    A symmetry space is e.g. a vector space with a representation of a symmetry group.

    Each symmetry space is equivalent to a direct sum of sectors, that
    is :math:`V \cong \bigoplus_a \bigoplus_{\mu=1}{N_a} a`.
    This is e.g. because the representation of the symmetry group is equivalent to a direct sum of
    irreducible representations. From a different perspective, the vector space decomposes into
    different charge sectors of the conserved charge. The unique sectors :math:`a` that appear in
    the decomposition at least once, e.g. with `N_a > 0`, are stored in :attr:`sector_decomposition`
    in a canonical order, while their multiplicities :math:`N_a` are stored in :attr:`multiplicities`.

    TODO should this have sectors_of_basis here or is that an attribute for ElementarySpace only?

    Attributes
    ----------
    symmetry: Symmetry
        The symmetry associated with this space.
    sector_decomposition : 2D numpy array of int
        The unique sectors that appear in the sector decomposition. A 2D array of integers with
        axes [s, q] where s goes over different sectors and q over the (one or more) numbers needed
        to label a sector. The sectors (to be precise, the rows ``sector_decomposition[i, :]``) are
        unique. We use :attr:`multiplicities` to  account for duplicates.
    sector_order : 'sorted' | 'dual_sorted' | None
        Indicates if (and how) the :attr:`sector_decomposition` is sorted.
        If ``'sorted'``, indicates that they are sorted by sector, i.e. such that
        ``np.lexsort(sector_decomposition.T) == np.arange(num_sectors)``.
        If ``'dual_sorted'``, indicated that the duals are sorted, i.e. such that
        ``np.lexsort(dual_sectors(sector_decomposition).T) == np.arange(num_sectors)``.
        If ``None``, no particular order is guaranteed.
    multiplicities : 1D numpy array of int | None
        How often each of the sectors in :attr:`sector_decomposition` appears. A 1D array of positive
        integers with axis [s]. ``sector_decomposition[i, :]`` appears ``multiplicities[i]`` times.
        ``None`` is equivalent to a sequence of ``1`` of appropriate length.
    num_sectors : int
        The number of sectors in the :attr:`sector_decomposition`.
    sector_dims : 1D array of int | None
        If ``symmetry.can_be_dropped``, the integer dimension of each sector of the
        :attr:`sector_decomposition`. Otherwise, not defined and set to ``None``.
    sector_qdims : 1D array of float
        The (quantum) dimension of each of the sectors. Unlike :attr:`sector_dims` this is always
        defined, but may not always be integer.
    dim : int | float
        The total dimension. Is integer if ``symmetry.can_be_dropped``, otherwise may be float.
    slices : 2D numpy array of int | None
        TODO do we keep these?
        For every sector ``sector_decomposition[n]``, the start ``slices[n, 0]`` and stop
        ``slices[n, 1]`` of indices (in the *internal* basis order) that belong to this sector.
        Conversely, ``basis_perm[slices[n, 0]:slices[n, 1]]`` are the elements of the public
        basis that live in ``sector_decomposition[n]``. Only available if ``symmetry.can_be_dropped``.
    """

    def __init__(self, symmetry: Symmetry, sector_decomposition: SectorArray | Sequence[Sequence[int]],
                 multiplicities: Sequence[int] | None = None,
                 sector_order: Literal['sorted'] | Literal['dual_sorted'] | None = None):
        self.symmetry = symmetry
        self.sector_decomposition = sector_decomposition = np.asarray(sector_decomposition, dtype=int)
        self.sector_order = sector_order
        if sector_decomposition.ndim != 2 or sector_decomposition.shape[1] != symmetry.sector_ind_len:
            msg = (f'Wrong sectors.shape: Expected (*, {symmetry.sector_ind_len}), '
                   f'got {sector_decomposition.shape}.')
            raise ValueError(msg)
        assert sector_decomposition.ndim == 2 and sector_decomposition.shape[1] == symmetry.sector_ind_len
        self.num_sectors = num_sectors = len(sector_decomposition)
        if multiplicities is None:
            self.multiplicities = multiplicities = np.ones((num_sectors,), dtype=int)
        else:
            self.multiplicities = multiplicities = np.asarray(multiplicities, dtype=int)
        if symmetry.can_be_dropped:
            self.sector_dims = sector_dims = symmetry.batch_sector_dim(sector_decomposition)
            self.sector_qdims = sector_dims
            slices = np.zeros((len(sector_decomposition), 2), dtype=np.intp)
            slices[:, 1] = slice_ends = np.cumsum(multiplicities * sector_dims)
            slices[1:, 0] = slice_ends[:-1]  # slices[0, 0] remains 0, which is correct
            self.slices = slices
            self.dim = np.sum(sector_dims * multiplicities).item()
        else:
            self.sector_dims = None
            self.sector_qdims = sector_qdims = symmetry.batch_qdim(sector_decomposition)
            self.slices = None
            self.dim = np.sum(sector_qdims * multiplicities).item()

    def test_sanity(self):
        assert self.dim >= 0
        # sectors
        if self.sector_decomposition.shape != (self.num_sectors, self.symmetry.sector_ind_len):
            raise AssertionError('wrong sectors.shape')
        assert all(self.symmetry.is_valid_sector(s) for s in self.sector_decomposition), 'invalid sectors'
        assert len(np.unique(self.sector_decomposition, axis=0)) == self.num_sectors, 'duplicate sectors'
        if self.sector_order == 'sorted':
            assert np.all(np.lexsort(self.sector_decomposition.T) == np.arange(self.num_sectors)), 'wrong sector order'
        elif self.sector_order == 'dual_sorted':
            expect_sorted = self.symmetry.dual_sectors(self.sector_decomposition)
            assert np.all(np.lexsort(expect_sorted.T) == np.arange(self.num_sectors)), 'wrong sector order'
        elif self.sector_order is None:
            pass  # nothing to check
        else:
            raise AssertionError(f'Invalid sector_order: {self.sector_order}')
        # multiplicities
        assert np.all(self.multiplicities > 0)
        assert self.multiplicities.shape == (self.num_sectors,)
        if self.symmetry.can_be_dropped:
            # slices
            assert self.slices.shape == (self.num_sectors, 2)
            slice_diffs = self.slices[:, 1] - self.slices[:, 0]
            assert np.all(self.sector_dims == self.symmetry.batch_sector_dim(self.sector_decomposition))
            expect_diffs = self.sector_dims * self.multiplicities
            assert np.all(slice_diffs == expect_diffs)
            # slices should be consecutive
            if self.num_sectors > 0:
                assert self.slices[0, 0] == 0
                assert np.all(self.slices[1:, 0] == self.slices[:-1, 1])
                assert self.slices[-1, 1] == self.dim

    # ABSTRACT

    @property
    @abstractmethod
    def dual(self) -> Space:
        """The dual space of the same type.

        A dual space necessarily has a :attr:`sector_decomposition` which consists of the
        :meth:`Symmetry.dual_sectors` of the original (though )
        
        Strictly speaking, this only guarantees to give one possible choice for a dual space and
        might differ from *the* dual space by an irrelevant isomorphism.

        TODO discuss duality in all class docstrings
        """
        ...

    @property
    def is_trivial(self) -> bool:
        """If the space is trivial, i.e. isomorphic to the one-dimensional trivial sector.

        A trivial space is one-dimensional and transforms trivially under a symmetry group.
        In category speak, it is (isomorphic to) the monoidal unit.
        """
        if self.num_sectors > 1:
            return False
        if self.multiplicities[0] > 1:
            return False
        return np.all(self.sector_decomposition[0] == self.symmetry.trivial_sector)

    @abstractmethod
    def __eq__(self, other):
        msg = (f'{self.__class__.__name__} does not support "==" comparison. '
               f'Use `is_isomorphic_to` instead.')
        raise TypeError(msg)

    def is_isomorphic_to(self, other: Space) -> bool:
        """If the two spaces are isomorphic, i.e. have the same :attr:`sector_decomposition`."""
        if self.symmetry != other.symmetry:
            raise SymmetryError('Incompatible symmetries')
        if self.num_sectors != other.num_sectors:
            return False

        # find perm1 and perm2 such that ``self.sector_decomposition[perm1]`` and ``other.sector_decomposition[perm2]``
        # have the same sorting convention and can be directly compared
        if self.sector_order is None:
            if other.sector_order == 'sorted':
                perm1 = np.lexsort(self.sector_decomposition.T)
                perm2 = slice(None, None, None)
            elif other.sector_order == 'dual_sorted':
                perm1 = np.lexsort(self.symmetry.dual_sectors(self.sector_decomposition).T)
                perm2 = slice(None, None, None)
            else:
                perm1 = np.lexsort(self.sector_decomposition.T)
                perm2 = np.lexsort(other.sector_decomposition.T)
        elif other.sector_order is None:
            if self.sector_order == 'sorted':
                perm1 = slice(None, None, None)
                perm2 = np.lexsort(other.sector_decomposition.T)
            elif self.sector_order == 'dual_sorted':
                perm1 = slice(None, None, None)
                perm2 = np.lexsort(self.symmetry.dual_sectors(other.sector_decomposition).T)
            else:
                raise RuntimeError  # case should have been covered above
        elif self.sector_order == other.sector_order:
            perm1 = perm2 = slice(None, None, None)
        elif self.sector_order == 'sorted':
            perm1 = slice(None, None, None)
            perm2 = np.lexsort(other.sector_decomposition.T)
        elif other.sector_order == 'sorted':
            perm1 = np.lexsort(self.sector_decomposition.T)
            perm2 = slice(None, None, None)
        else:
            raise RuntimeError  # all cases should have been covered.

        if not np.all(self.multiplicities[perm1] == other.multiplicities[perm2]):
            return False
        return np.all(self.sector_decomposition[perm1] == other.sector_decomposition[perm2])

    def is_subspace_of(self, other: Space) -> bool:
        """Whether self is (isomorphic to) a subspace of other.

        Per convention, self is never a subspace of other, if the :attr:`symmetry` are different.

        See Also
        --------
        ElementarySpace.from_largest_common_subspace
        """
        if not self.symmetry.is_same_symmetry(other.symmetry):
            return False
        if self.num_sectors == 0:
            return True
        if self.sector_order == 'sorted' == other.sector_order:
            # sectors are sorted, so we can just iterate over both of them
            n_self = 0
            for other_sector, other_mult in zip(other.sector_decomposition, other.multiplicities):
                if np.all(self.sector_decomposition[n_self] == other_sector):
                    if self.multiplicities[n_self] > other_mult:
                        return False
                    n_self += 1
                if n_self == self.num_sectors:
                    # have checked all sectors of self
                    return True
            # reaching this line means self has sectors which other does not have
            return False

        # OPTIMIZE sort once instead of looking up each time
        num_sectors_checked = 0
        for sector, mult in zip(other.sector_decomposition, other.multiplicities):
            m = self.sector_multiplicity(sector)
            if m == 0:
                continue
            if m > mult:
                return False
            num_sectors_checked += 1
        if num_sectors_checked < self.num_sectors:
            # this means self has some sectors that other doesn't have
            return False
        return True

    def as_ElementarySpace(self, is_dual: bool = False) -> ElementarySpace:
        """Convert to an isomorphic :class:`ElementarySpace`."""
        if is_dual:
            defining_sectors = self.symmetry.dual_sectors(self.sector_decomposition)
            is_sorted = (self.sector_order == 'dual_sorted')
        else:
            defining_sectors = self.sector_decomposition
            is_sorted = (self.sector_order == 'sorted')

        if is_sorted:
            return ElementarySpace(symmetry=self.symmetry, defining_sectors=defining_sectors,
                                   multiplicities=self.multiplicities, is_dual=is_dual)
        return ElementarySpace.from_sectors(symmetry=self.symmetry, defining_sectors=defining_sectors,
                                            multiplicities=self.multiplicities, is_dual=is_dual,
                                            unique_sectors=True)

    @abstractmethod
    def change_symmetry(self, symmetry: Symmetry, sector_map: callable, injective: bool = False
                        ) -> ElementarySpace:
        """Change the symmetry by specifying how the sectors change.

        TODO this assumes by construction that the mapping is one-to-one.
             It does not fit well with e.g. relaxing SU(2) -> U(1)

        Parameters
        ----------
        symmetry : :class:`~cyten.groups.Symmetry`
            The symmetry of the new space
        sector_map : function (SectorArray,) -> (SectorArray,)
            A map of sectors (2D int arrays), such that ``new_sectors = sector_map(old_sectors)``.
            The map is assumed to cooperate with duality, i.e. we assume without checking that
            ``symmetry.dual_sectors(sector_map(old_sectors))`` is the same as
            ``sector_map(old_symmetry.dual_sectors(old_sectors))``.
            TODO do we need to assume more, i.e. compatibility with fusion?
        injective: bool
            If ``True``, the `sector_map` is assumed to be injective, i.e. produce a list of
            unique outputs, if the inputs are unique.

        Returns
        -------
        A space with the new symmetry. The order of the basis is preserved, but every
        basis element lives in a new sector, according to `sector_map`.
        """
        ...

    @abstractmethod
    def drop_symmetry(self, which: int | list[int] = None):
        """Drop some or all symmetries.

        Parameters
        ----------
        which : None | (list of) int
            If ``None`` (default) the entire symmetry is dropped and the result has ``no_symmetry``.
            An integer or list of integers assume that ``self.symmetry`` is a ``ProductSymmetry``
            and indicates which of its factors to drop.
        """
        ...

    @abstractmethod
    def _repr(self, show_symmetry: bool):
        ...

    # CONCRETE IMPLEMENTATIONS

    def __repr__(self):
        res = self._repr(show_symmetry=True)
        if res is None:
            return f'<{self.__class__.__name__}>'
        return res

    def sector_decomposition_where(self, sector: Sector) -> int | None:
        """Find the index of a given sector in the :attr:`sector_decomposition`.

        Returns
        -------
        idx : int | None
            If the `sector` is found the :attr:`sector_decomposition`, its index there such
            that ``sector_decomposition[idx] == sector``. Otherwise ``None``.
        """
        # OPTIMIZE : if sector_order allows it, use that sectors are sorted to speed up the lookup
        where = np.where(np.all(self.sector_decomposition == sector, axis=1))[0]
        if len(where) == 0:
            return None
        if len(where) == 1:
            return int(where[0])
        # sector_decomposition should be unique, so one of the above if statements should trigger.
        # If we get here, something is wrong / inconsistent.
        self.test_sanity()  # this should raise an informative error
        raise RuntimeError('This should not happen. Please report this bug on github.')

    def sector_multiplicity(self, sector: Sector) -> int:
        """The multiplicity of a given sector in the :attr:`sector_decomposition`."""
        idx = self.sector_decomposition_where(sector)
        if idx is None:
            return 0
        return self.multiplicities[idx]


class ElementarySpace(Space, Leg):
    r"""A :class:`Space` that is defined as (the dual of) a direct sum of sectors.

    While every :class:`Space` is isomorphic to a direct sum of sectors, an :class:`ElementarySpace`
    is by definition *equal* to such a direct sum, or to the dual of such a sum. We distinguish
    "ket" spaces :math:`V_k := a_1 \oplus a_2 \oplus \dots \plus a_N` with ``is_dual=False`` and
    "bra" spaces :math:`V_b := [b_1 \oplus b_2 \oplus \dots \plus b_N]^*` with ``is_dual=True``.
    The listed sectors, :math:`\{a_n\}` for the ket space :math:`V_k` and the :math:`\{b_n\}`
    for the bra space, are the :attr:`defining_sectors` of the space. For a ket space, they coincide
    with the :attr:`sector_decomposition`, while for a bra space they are mutually dual, since
    we have :math:`V_b \cong \bar{b}_1 \oplus \bar{b}_2 \oplus \dots \plus \bar{b}_N`.

    We impose a canonical order of sectors, such that the :attr:`defining_sectors` are sorted.
    This in turn means that the :attr:`sector_order` is ``'sorted'`` for ket spaces and
    ``'dual_sorted'`` for bra spaces.

    If the symmetry :attr:`Symmetry.can_be_dropped`, there is a notion of a basis for the
    spaces. We demand the basis to be compatible with the symmetry, i.e. each basis vector
    needs to lie in one of the sectors of the symmetry. The *internal* basis order that results
    from demanding that the sectors are contiguous and sorted may, however, not be the desired
    basis order, e.g. for matrix representations. For example, the standard basis of a spin-1
    degree of freedom with ``'Sz_parity'`` conservation has sectors ``[[1], [0], [1]]`` and is
    neither sorted by sector nor contiguous. We allow these different *public* basis orders
    and store the relevant permutations as :attr:`basis_perm` and :attr:`inverse_basis_perm`.
    See also :attr:`sectors_of_basis` and :meth:`from_basis`.

    Parameters
    ----------
    symmetry, sectors, multiplicities, is_dual, basis_perm
        Like attributes of the same name, except nested sequences are allowed in place of arrays.

    Attributes
    ----------
    is_dual: bool
    defining_sectors: 2D array of int
    """

    def __init__(self, symmetry: Symmetry, defining_sectors: SectorArray,
                 multiplicities: ndarray = None, is_dual: bool = False,
                 basis_perm: ndarray | None = None):
        defining_sectors = np.asarray(defining_sectors, dtype=int)
        if is_dual:
            sector_decomposition = symmetry.dual_sectors(defining_sectors)
            sector_order = 'dual_sorted'
        else:
            sector_decomposition = defining_sectors
            sector_order = 'sorted'
        Space.__init__(self, symmetry=symmetry, sector_decomposition=sector_decomposition,
                       multiplicities=multiplicities, sector_order=sector_order)
        Leg.__init__(self, symmetry=symmetry, dim=self.dim, is_dual=is_dual)
        self.defining_sectors = defining_sectors
        if basis_perm is None:
            self._basis_perm = self._inverse_basis_perm = None
        else:
            if not symmetry.can_be_dropped:
                msg = f'basis_perm is meaningless for {symmetry}.'
                raise SymmetryError(msg)
            self._basis_perm = basis_perm = np.asarray(basis_perm, dtype=int)
            self._inverse_basis_perm = inverse_permutation(basis_perm)

    def test_sanity(self):
        if not self.symmetry.can_be_dropped:
            assert self._basis_perm is None
        if self._basis_perm is None:
            assert self._inverse_basis_perm is None
        else:
            assert self._inverse_basis_perm is not None
            assert self._basis_perm.shape == self._inverse_basis_perm.shape == (self.dim,)
            assert len(np.unique(self._basis_perm)) == self.dim  # is a permutation
            assert len(np.unique(self._inverse_basis_perm)) == self.dim  # is a permutation
            assert np.all(self._basis_perm[self._inverse_basis_perm] == np.arange(self.dim))
        assert self.defining_sectors.shape == (self.num_sectors, self.symmetry.sector_ind_len)
        if self.is_dual:
            assert self.sector_order == 'dual_sorted'
        else:
            assert self.sector_order == 'sorted'
        Space.test_sanity(self)
        Leg.test_sanity(self)

    @classmethod
    def from_basis(cls, symmetry: Symmetry, sectors_of_basis: Sequence[Sequence[int]]
                   ) -> ElementarySpace:
        """Create an ElementarySpace by specifying the sector of every basis element.

        .. note ::
            Unlike :meth:`from_sectors`, this method expects the same sector to be listed
            multiple times, if the sector is multi-dimensional. The Hilbert Space of a spin-one-half
            D.O.F. can e.g. be created as ``ElementarySpace.from_basis(su2, [spin_half, spin_half])``
            or as ``ElementarySpace.from_sectors(su2, [spin_half])``. In the former case we need to
            list the same sector both for the spin up and spin down state.

        Parameters
        ----------
        symmetry: Symmetry
            The symmetry associated with this space.
        sectors_of_basis : iterable of iterable of int
            Specifies the basis. ``sectors_of_basis[n]`` is the sector of the ``n``-th basis element.
            In particular, for a ``d`` dimensional sector, we expect an integer multiple of ``d``
            occurrences. They need not be contiguous though. They will be grouped by order of
            appearance, such that they ``m``-th time a sector appears, that basis state is interpreted
            as the ``(m % d)``-th state of the multiplet.

        See Also
        --------
        :attr:`sectors_of_basis`
            Reproduces the `sectors_of_basis` parameter.
        """
        if not symmetry.can_be_dropped:
            msg = f'from_basis is meaningless for {symmetry}.'
            raise SymmetryError(msg)
        sectors_of_basis = np.asarray(sectors_of_basis, dtype=int)
        assert sectors_of_basis.shape[1] == symmetry.sector_ind_len
        # note: numpy.lexsort is stable, i.e. it preserves the order of equal keys.
        basis_perm = np.lexsort(sectors_of_basis.T)
        sectors = sectors_of_basis[basis_perm]
        diffs = find_row_differences(sectors, include_len=True)
        sectors = sectors[diffs[:-1]]  # [:-1] to exclude len
        dims = symmetry.batch_sector_dim(sectors)
        num_occurrences = diffs[1:] - diffs[:-1]  # how often each appears in the input sectors_of_basis
        multiplicities, remainders = np.divmod(num_occurrences, dims)
        if np.any(remainders > 0):
            msg = ('Sectors must appear in whole multiplets, i.e. a number of times that is an '
                   'integer multiple of their dimension.')
            raise ValueError(msg)
        return cls(symmetry=symmetry, defining_sectors=sectors, multiplicities=multiplicities,
                   is_dual=False, basis_perm=basis_perm)

    @classmethod
    def from_independent_symmetries(cls, independent_descriptions: list[ElementarySpace]
                                    ) -> ElementarySpace:
        """Create an ElementarySpace with multiple independent symmetries.

        TODO this interface is more general than it needs to be. The use case in GroupedSite
        would allow us to specialize, if that is easier. A given state is in the trivial sector
        for all but one of the independent_descriptions

        Parameters
        ----------
        independent_descriptions : list of :class:`ElementarySpace`
            Each entry describes the resulting :class:`ElementarySpace` in terms of *one* of
            the independent symmetries. Spaces with a :class:`NoSymmetry` are ignored.
        """
        # OPTIMIZE this can be implemented better. if many consecutive basis elements have the same
        #          resulting sector, we can skip over all of them.
        assert len(independent_descriptions) > 0
        dim = independent_descriptions[0].dim
        assert all(s.dim == dim for s in independent_descriptions)
        # ignore those with no_symmetry
        independent_descriptions = [s for s in independent_descriptions if s.symmetry != no_symmetry]
        if len(independent_descriptions) == 0:
            # all descriptions had no_symmetry
            return cls.from_trivial_sector(dim=dim)
        symmetry = ProductSymmetry.from_nested_factors(
            [s.symmetry for s in independent_descriptions]
        )
        if not symmetry.can_be_dropped:
            msg = f'from_independent_symmetries is not supported for {symmetry}.'
            # TODO is there a way to define this?
            #      the straight-forward picture works only if we have a vector space and can identify states.
            raise SymmetryError(msg)
        sectors_of_basis = np.concatenate([s.sectors_of_basis for s in independent_descriptions],
                                          axis=1)
        return cls.from_basis(symmetry, sectors_of_basis)

    @classmethod
    def from_largest_common_subspace(cls, *spaces: Space, is_dual: bool = False) -> ElementarySpace:
        """The largest common subspace of a list of spaces.

        The largest :class:`ElementarySpace` that :meth:`is_subspace_of` all of the `spaces`.
        I.e. the :attr:`sector_decomposition` is given by the "sector-wise minimum" of all
        multiplicities of the `spaces`.

        See Also
        --------
        is_subspace_of
        """
        if len(spaces) == 0:
            raise ValueError('Need at least one space')
        if len(spaces) == 1:
            return spaces[0].as_ElementarySpace(is_dual=is_dual)
        sp1, sp2, *more = spaces
        if more:
            # OPTIMIZE directly implement for many
            sp = ElementarySpace.from_largest_common_subspace(sp1, sp2)
            return ElementarySpace.from_largest_common_subspace(sp, *more, is_dual=is_dual)
        sectors = []
        mults = []
        if sp1.sector_order == 'sorted' == sp2.sector_order:
            for i, j in iter_common_sorted_arrays(sp1.sector_decomposition, sp2.sector_decomposition):
                sectors.append(sp1.sector_decomposition[i])
                mults.append(min(sp1.multiplicities[i], sp2.multiplicities[j]))
        else:
            # OPTIMIZE implementation for mixed orders? or just override this in ElementarySpace?
            for i, sector in enumerate(sp1.sector_decomposition):
                j = sp2.sector_decomposition_where(sector)
                if j is None:
                    continue
                sectors.append(sector)
                mults.append(min(sp1.multiplicities[i], sp2.multiplicities[j]))

        # TODO implement convenience function ElementarySpace.from_sector_decomposition??
        return ElementarySpace(sp1.symmetry, sectors, mults).with_is_dual(is_dual)

    @classmethod
    def from_null_space(cls, symmetry: Symmetry, is_dual: bool = False) -> ElementarySpace:
        """The zero-dimensional space, i.e. the span of the empty set."""
        return cls(symmetry=symmetry, defining_sectors=symmetry.empty_sector_array,
                   multiplicities=np.zeros(0, int), is_dual=is_dual)

    @classmethod
    def from_sectors(cls, symmetry: Symmetry, defining_sectors: SectorArray,
                     multiplicities: Sequence[int] = None, is_dual: bool = False,
                     basis_perm: ndarray = None, unique_sectors: bool = False,
                     return_sorting_perm: bool = False
                     ) -> ElementarySpace | tuple[ElementarySpace, ndarray]:
        """Similar to the constructor, but with fewer requirements.

        .. note ::
            Unlike :meth:`from_basis`, this method expects a multi-dimensional sector to be listed
            only once to mean its entire multiplet of basis states. The Hilbert Space of a spin-1/2
            D.O.F. can e.g. be created as ``ElementarySpace.from_basis(su2, [spin_half, spin_half])``
            or as ``ElementarySpace.from_sectors(su2, [spin_half])``. In the former case we need to
            list the same sector both for the spin up and spin down state.

        Parameters
        ----------
        symmetry: Symmetry
            The symmetry associated with this space.
        defining_sectors: 2D array_like of int
            Like the :attr:`defining_sectors` attribute, but can be in any order and may contain
            duplicates (see `unique_sectors`).
        multiplicities: 1D array_like of int, optional
            How often each of the `defining_sectors` appears. A 1D array of positive integers with
            axis [s]. ``defining_sectors[i_s, :]`` appears ``multiplicities[i_s]`` times.
            If not given, a multiplicity ``1`` is assumed for all `defining_sectors`.
        is_dual: bool
            If the result is a bra- or a ket space, like the attribute :attr:`is_dual`.
            Note that this changes the meaning of the `defining_sectors`.
        basis_perm: ndarray, optional
            The permutation from the desired public basis to the basis described by
            `defining_sectors` and `multiplicities`.
        unique_sectors: bool
            If ``True``, the `sectors` are assumed to be duplicate-free.
        return_sorting_perm: bool
            If ``True``, the permutation ``np.lexsort(sectors.T)`` is returned too.

        Returns
        -------
        space: ElementarySpace
        sector_sort: 1D array, optional
            Only ``if return_sorting_perm``. The permutation that sorts the `defining_sectors`.
        """
        defining_sectors = np.asarray(defining_sectors, dtype=int)
        assert defining_sectors.ndim == 2 and defining_sectors.shape[1] == symmetry.sector_ind_len
        if multiplicities is None:
            multiplicities = np.ones((len(defining_sectors),), dtype=int)
        else:
            multiplicities = np.asarray(multiplicities, dtype=int)
            assert multiplicities.shape == ((len(defining_sectors),))

        # sort sectors
        if symmetry.can_be_dropped:
            num_states = symmetry.batch_sector_dim(defining_sectors) * multiplicities
            basis_slices = np.concatenate([[0], np.cumsum(num_states)], axis=0)
            defining_sectors, multiplicities, sort = _sort_sectors(defining_sectors, multiplicities)
            if len(defining_sectors) == 0:
                basis_perm = np.zeros(0, int)
            else:
                if basis_perm is None:
                    basis_perm = np.arange(np.sum(num_states))
                basis_perm = np.concatenate([basis_perm[basis_slices[i]: basis_slices[i + 1]]
                                            for i in sort])
        else:
            defining_sectors, multiplicities, sort = _sort_sectors(defining_sectors, multiplicities)
            assert basis_perm is None
        # combine duplicate sectors (does not affect basis_perm)
        if not unique_sectors:
            mult_slices = np.concatenate([[0], np.cumsum(multiplicities)], axis=0)
            diffs = find_row_differences(defining_sectors, include_len=True)
            multiplicities = mult_slices[diffs[1:]] - mult_slices[diffs[:-1]]
            defining_sectors = defining_sectors[diffs[:-1]]  # [:-1] to exclude len
        res = cls(symmetry=symmetry, defining_sectors=defining_sectors,
                  multiplicities=multiplicities, is_dual=is_dual, basis_perm=basis_perm)
        if return_sorting_perm:
            return res, sort
        return res

    @classmethod
    def from_trivial_sector(cls, dim: int = 1, symmetry: Symmetry = no_symmetry,
                            is_dual: bool = False, basis_perm: ndarray = None) -> ElementarySpace:
        """Create an ElementarySpace that lives in the trivial sector (i.e. it is symmetric).

        Parameters
        ----------
        dim : int
            The dimension of the space.
        symmetry : :class:`~cyten.groups.Symmetry`
            The symmetry of the space.
        is_dual : bool
            If the space should be bra or a ket space.
        """
        if dim == 0:
            return cls.from_null_space(symmetry=symmetry, is_dual=is_dual)
        return cls(symmetry=symmetry, defining_sectors=symmetry.trivial_sector[None, :],
                   multiplicities=[dim], is_dual=is_dual, basis_perm=basis_perm)

    @property
    def basis_perm(self) -> ndarray:
        """Permutation that translates between public and internal basis order.

        For the inverse permutation, see :attr:`inverse_basis_perm`.

        The tensor manipulations of ``cyten`` benefit from choosing a canonical order for the
        basis of vector spaces. This attribute translates between the "public" order of the basis,
        in which e.g. the inputs to :meth:`from_dense_block` are interpreted to this internal order,
        such that ``public_basis[basis_perm] == internal_basis``.
        The internal order is such that the basis vectors are grouped and sorted by sector.
        We can translate indices as ``public_idx == basis_perm[internal_idx]``.
        Only available if ``symmetry.can_be_dropped``, as otherwise there is no well-defined
        notion of a basis.

        ``_basis_perm`` is the internal version which may be ``None`` if the permutation is trivial.
        """
        if not self.symmetry.can_be_dropped:
            msg = f'basis_perm is meaningless for {self.symmetry}.'
            raise SymmetryError(msg)
        if self._basis_perm is None:
            return np.arange(self.dim)
        return self._basis_perm

    @property
    def inverse_basis_perm(self) -> ndarray:
        """Inverse permutation of :attr:`basis_perm`."""
        if not self.symmetry.can_be_dropped:
            msg = f'basis_perm is meaningless for {self.symmetry}.'
            raise SymmetryError(msg)
        if self._inverse_basis_perm is None:
            return np.arange(self.dim)
        return self._inverse_basis_perm

    @property
    def sectors_of_basis(self):
        """The sector (from the :attr:`sector_decomposition`) of each basis vector."""
        if not self.symmetry.can_be_dropped:
            msg = f'sectors_of_basis is meaningless for {self.symmetry}.'
            raise SymmetryError(msg)
        # build in internal basis, then permute
        res = np.zeros((self.dim, self.symmetry.sector_ind_len), dtype=int)
        for sect, slc in zip(self.sector_decomposition, self.slices):
            res[slice(*slc), :] = sect[None, :]
        if self._inverse_basis_perm is not None:
            res = res[self._inverse_basis_perm]
        return res

    def _repr(self, show_symmetry: bool):
        # used by Space.__repr__
        indent = printoptions.indent * ' '
        # 1) Try showing all data
        if 3 * self.defining_sectors.size < printoptions.linewidth:
            # otherwise there is no chance to print all sectors in one line anyway
            if self._basis_perm is None:
                basis_perm = 'None'
            else:
                basis_perm = format_like_list(self._basis_perm)
            elements = []
            if show_symmetry:
                elements.append(f'{self.symmetry!r}')
            elements.extend([
                f'defining_sectors={format_like_list(self.symmetry.sector_str(a) for a in self.defining_sectors)}',
                f'multiplicities={format_like_list(self.multiplicities)}',
                f'basis_perm={basis_perm}',
                f'is_dual={self.is_dual}'
            ])
            one_line = f'ElementarySpace(' + ', '.join(elements) + ')'
            if len(one_line) <= printoptions.linewidth:
                return one_line
            line_lengths_ok = all(len(l) <= printoptions.linewidth for l in elements)
            num_lines_ok = (len(elements) + 2) <= printoptions.maxlines_spaces
            if line_lengths_ok and num_lines_ok:
                elements = [f'{indent}{line},' for line in elements]
                return f'ElementarySpace(\n' + '\n'.join(elements) + '\n)'
        # 2) Try showing summarized data
        elements = [f'<ElementarySpace:']
        if show_symmetry:
            elements.append(f'{self.symmetry!s}')
        elements.extend([
            f'{self.num_sectors} sectors',
            f'basis_perm={"None" if self._basis_perm is None else "[...]"}',
            f'is_dual={self.is_dual}',
            '>',
        ])
        one_line = ' '.join(elements)
        if len(one_line) < printoptions.linewidth:
            return one_line
        if all(len(l) <= printoptions.linewidth for l in elements) and len(elements) <= printoptions.maxlines_spaces:
            elements[1:-1] = [f'{indent}{line},' for line in elements[1:-1]]
            return '\n'.join(elements)
        # 3) Try showing only symmetry
        if show_symmetry:
            elements[2:-1] = []
            one_line = ' '.join(elements)
            if len(one_line) < printoptions.linewidth:
                return one_line
            line_lengths_ok = all(len(l) <= printoptions.linewidth for l in elements)
            num_lines_ok = len(elements) <= printoptions.maxlines_spaces
            if line_lengths_ok and num_lines_ok:
                elements[1:-1] = [f'{indent}{line},' for line in elements[1:-1]]
                return '\n'.join(elements)
        # 4) Show no data at all
        return None

    def __eq__(self, other):
        if not isinstance(other, ElementarySpace):
            return NotImplemented
        if self.is_dual != other.is_dual:
            return False
        if self.symmetry != other.symmetry:
            return False
        if self.num_sectors != other.num_sectors:  # check this first to safely compare later
            return False
        if not np.all(self.multiplicities == other.multiplicities):
            return False
        if not np.all(self.defining_sectors == other.defining_sectors):
            return False
        if (self._basis_perm is not None) or (other._basis_perm is not None):
            if not np.all(self.basis_perm == other.basis_perm):
                return False
        else:
            pass  # both permutations are trivial, thus equal
        return True

    def as_Space(self):
        return self

    def as_ElementarySpace(self, is_dual: bool = False) -> ElementarySpace:
        if bool(is_dual) == self.is_dual:
            return self
        return self.with_opposite_duality()

    def as_ket_space(self):
        """The ket space (``is_dual=False``) isomorphic or equal to self."""
        if not self.is_dual:
            return self
        return self.with_opposite_duality()

    def as_bra_space(self):
        """The bra space (``is_dual=False``) isomorphic or equal to self."""
        if self.is_dual:
            return self
        return self.with_opposite_duality()

    def change_symmetry(self, symmetry: Symmetry, sector_map: callable, injective: bool = False
                        ) -> ElementarySpace:
        return ElementarySpace.from_sectors(
            symmetry=symmetry, defining_sectors=sector_map(self.defining_sectors),
            multiplicities=self.multiplicities, is_dual=self.is_dual, basis_perm=self._basis_perm,
            unique_sectors=injective
        )

    def direct_sum(self, *others: ElementarySpace) -> ElementarySpace:
        """Form the direct sum (i.e. stacking).

        The basis of the new space results from concatenating the individual bases.

        Spaces must have the same symmetry and is_dual.
        The result is a space with the same symmetry and is_dual, whose sectors are those
        that appear in any of the spaces and multiplicities are the sum of the multiplicities
        in each of the spaces.
        """
        if not others:
            return self
        assert all(o.symmetry == self.symmetry for o in others)
        assert all(o.is_dual == self.is_dual for o in others)
        if self.symmetry.can_be_dropped:
            offsets = np.cumsum([self.dim, *(o.dim for o in others)])
            basis_perm = np.concatenate(
                [self.basis_perm] + [o.basis_perm + n for o, n in zip(others, offsets)]
            )
        else:
            basis_perm = None
        return ElementarySpace.from_sectors(
            symmetry=self.symmetry,
            defining_sectors=np.concatenate([self.defining_sectors, *(o.defining_sectors for o in others)]),
            multiplicities=np.concatenate([self.multiplicities, *(o.multiplicities for o in others)]),
            is_dual=self.is_dual, basis_perm=basis_perm
        )

    def drop_symmetry(self, which: int | list[int] = None):
        which, remaining_symmetry = _parse_inputs_drop_symmetry(which, self.symmetry)
        if which is None:
            return ElementarySpace.from_trivial_sector(
                dim=self.dim, symmetry=remaining_symmetry, is_dual=self.is_dual,
                basis_perm=self._basis_perm
            )
        mask = np.ones((self.symmetry.sector_ind_len,), dtype=bool)
        for i in which:
            start, stop = self.symmetry.sector_slices[i:i + 2]
            mask[start:stop] = False
        return self.change_symmetry(symmetry=remaining_symmetry,
                                    sector_map=lambda sectors: sectors[:, mask])

    @property
    def dual(self) -> ElementarySpace:
        return ElementarySpace(
            self.symmetry, defining_sectors=self.defining_sectors,
            multiplicities=self.multiplicities, is_dual=not self.is_dual,
            basis_perm=self._basis_perm
        )

    def parse_index(self, idx: int) -> tuple[int, int]:
        """Utility function to translate an index.

        Parameters
        ----------
        idx : int
            An index of the leg, labelling an element of the public computational basis of self.

        Returns
        -------
        sector_idx : int
            The index of the corresponding sector,
            indicating that the `idx`-th basis element lives in ``self.sector_decomposition[sector_idx]``.
        multiplicity_idx : int
            The index "within the sector", in ``range(sector_dim * self.multiplicities[sector_index])``.
        """
        if not self.symmetry.can_be_dropped:
            msg = f'parse_index is meaningless for {self.symmetry}.'
            raise SymmetryError(msg)
        if self._inverse_basis_perm is not None:
            idx = self._inverse_basis_perm[idx]
        sector_idx = bisect.bisect(self.slices[:, 0], idx) - 1
        multiplicity_idx = idx - self.slices[sector_idx, 0]
        return sector_idx, multiplicity_idx

    def idx_to_sector(self, idx: int) -> Sector:
        sector_idx, _ = self.parse_index(idx)
        return self.sector_decomposition[sector_idx]

    def take_slice(self, blockmask: Block) -> ElementarySpace:
        """Take a "slice" of the leg, keeping only some of the basis states.

        Parameters
        ----------
        blockmask : 1D array-like of bool
            For every basis state of self, in the public basis order,
            if it should be kept (``True``) or discarded (``False``).
        """
        if not self.symmetry.can_be_dropped:
            msg = f'take_slice is meaningless for {self.symmetry}.'
            raise SymmetryError(msg)
        blockmask = np.asarray(blockmask, dtype=bool)
        if self._basis_perm is not None:
            blockmask = blockmask[self._basis_perm]
        #
        sectors = []
        mults = []
        for a, d_a, slc in zip(self.defining_sectors, self.sector_dims, self.slices):
            sector_mask = blockmask[slice(*slc)]
            per_basis_state = np.reshape(sector_mask, (-1, d_a))
            if not np.all(per_basis_state == per_basis_state[:, 0, None]):
                msg = 'Multiplets need to be kept or discarded as a whole.'
                raise ValueError(msg)
            num_kept = np.sum(sector_mask)
            assert num_kept % d_a == 0  # should be guaranteed by check above already, but to be sure...
            mult = num_kept // d_a
            if mult > 0:
                sectors.append(a)
                mults.append(mult)
        if len(sectors) == 0:
            sectors = self.symmetry.empty_sector_array
            mults = np.zeros(0, int)
        # build basis_perm for small leg.
        # it is determined by demanding
        #    a) that the following diagram commutes
        #
        #        (self, public) ---- self.basis_perm ---->  (self, internal)
        #         |                                           |
        #         v public_blockmask                          v projection_internal
        #         |                                           |
        #        (res, public) ----- small_leg_perm ----->  (res, internal)
        #
        #    b) that projection_internal is also just a mask (i.e it preserves ordering)
        #       which is given by public_blockmask[self.basis_perm]
        #
        # this allows us to internally (e.g. in the abelian backend) store only 1D boolean masks
        # as blocks.
        #
        # note blockmask is in the private basis order.
        basis_perm = rank_data(self.basis_perm[blockmask])
        return ElementarySpace(symmetry=self.symmetry, defining_sectors=sectors, multiplicities=mults,
                               is_dual=self.is_dual, basis_perm=basis_perm)

    def with_opposite_duality(self):
        """A space isomorphic to self with opposite ``is_dual`` attribute."""
        if self.is_dual:
            # already have the self.symmetry.dual_sectors(self.defining_sectors)
            dual_defining_sectors = self.sector_decomposition
        else:
            dual_defining_sectors = self.symmetry.dual_sectors(self.defining_sectors)
        # note: dual_defining_sectors are not sorted, but they are unique.
        return ElementarySpace.from_sectors(
            symmetry=self.symmetry, defining_sectors=dual_defining_sectors,
            multiplicities=self.multiplicities, is_dual=not self.is_dual,
            basis_perm=self._basis_perm, unique_sectors=True
        )

    def with_is_dual(self, is_dual: bool) -> ElementarySpace:
        """A space isomorphic to self with given ``is_dual`` attribute."""
        if is_dual == self.is_dual:
            return self  # TODO copy?
        return self.with_opposite_duality()


class TensorProduct(Space):
    """Represents a tensor product of :class:`Spaces`s, e.g. the (co-)domain of a tensor.

    TODO discuss / review how this relates to :class:`LegPipe`.
         Mostly, this is not a leg pipe, because it does not have an ``is_dual`` attr??
         And LegPipes are not TensorProducts because they dont have a ``sector_decomposition``
    """

    def __init__(self, spaces: list[Space], symmetry: Symmetry = None):
        self.num_spaces = num_spaces = len(spaces)
        if symmetry is None:
            if num_spaces == 0:
                raise ValueError('If spaces is empty, the symmetry arg is required.')
            symmetry = spaces[0].symmetry
        if not all(sp.symmetry == symmetry for sp in spaces):
            raise SymmetryError('Incompatible symmetries.')
        self.symmetry = symmetry
        self.spaces = spaces[:]
        sectors, multiplicities = self._calc_sectors(spaces)
        Space.__init__(self, symmetry=symmetry, sector_decomposition=sectors, multiplicities=multiplicities,
                       sector_order='sorted')

    def test_sanity(self):
        assert len(self.spaces) == self.num_spaces
        for sp in self.spaces:
            sp.test_sanity()
        Space.test_sanity(self)

    # CLASSMETHODS

    @classmethod
    def from_partial_products(cls, *factors: TensorProduct) -> TensorProduct:
        r"""Form the :class:`TensorProduct` of all :attr:`spaces` from partial products.

        The result has as :attr:`spaces` all those spaces that appear on the `factors`.
        I.e. we form :math:`V_1 \otimes V_2 \otimes W_1 \otimes W_2 \dots` from
        :math:`V_1 \otimes V_2` and :math:`W_1 \otimes W_2 \dots`.
        """
        spaces = factors[0].spaces[:]
        symmetry = factors[0].symmetry
        for f in factors[1:]:
            spaces.extend(f.spaces)
            assert f.symmetry == symmetry, 'Mismatched symmetries'
        # TODO faster computation of sectors etc
        return TensorProduct(spaces=spaces, symmetry=symmetry)

    # PROPERTIES

    @property
    def dual(self):
        # TODO is this needed ...?
        return TensorProduct([sp.dual for sp in reversed(self.spaces)], symmetry=self.symmetry)

    # METHODS

    def block_size(self, coupled: Sector | int) -> int:
        """The size of a block.

        Parameters
        ----------
        coupled : Sector or int
            Specify the coupled sector, either directly as a sector or as an integer, which
            is interpreted as an index, i.e. is equivalent to the sector
            ``self.sector_decomposition[coupled]``.
        """
        if isinstance(coupled, int):
            return self.multiplicities[coupled]
        return self.sector_multiplicity(coupled)

    def change_symmetry(self, symmetry, sector_map, injective=False):
        # TODO can we avoid recomputation of fusion?
        return TensorProduct(
            [space.change_symmetry(symmetry, sector_map, injective)
             for space in self.spaces],
            symmetry=self.symmetry
        )

    def drop_symmetry(self, which=None):
        # TODO can we avoid recomputation of fusion?
        return TensorProduct(
            [space.drop_symmetry(which) for space in self.spaces],
            symmetry=self.symmetry
        )

    def forest_block_size(self, uncoupled: tuple[Sector], coupled: Sector) -> int:
        """The size of a forest-block"""
        # OPTIMIZE ?
        num_trees = len(fusion_trees(self.symmetry, uncoupled, coupled))
        return num_trees * self.tree_block_size(uncoupled)

    def forest_block_slice(self, uncoupled: tuple[Sector], coupled: Sector) -> slice:
        """The range of indices of a forest-block within its block, as a slice."""
        # OPTIMIZE ?
        offset = 0
        for _unc in self.iter_uncoupled():
            if all(np.all(a == b) for a, b in zip(_unc, uncoupled)):
                break
            offset += self.forest_block_size(_unc, coupled)
        else:  # no break occurred
            raise ValueError('Uncoupled sectors incompatible')
        size = self.forest_block_size(uncoupled, coupled)
        return slice(offset, offset + size)

    def insert_multiply(self, other: Space, pos: int) -> TensorProduct:
        """Insert a new space into the product at position `pos`."""
        # TODO optimize (can compute sectors etc more efficiently)
        return TensorProduct(self.spaces[:pos] + [other] + self.spaces[pos:], symmetry=self.symmetry)

    def iter_uncoupled(self) -> Iterator[tuple[Sector]]:
        """Iterate over all combinations of sectors"""
        return it.product(*(s.sector_decomposition for s in self.spaces))
    
    def left_multiply(self, other: Space) -> TensorProduct:
        """Add a new factor at the left / beginning of the spaces"""
        return self.insert_multiply(other, 0)

    def right_multiply(self, other: Space) -> TensorProduct:
        """Add a new factor at the right / end of the spaces"""
        return self.insert_multiply(other, -1)

    def tree_block_size(space: TensorProduct, uncoupled: tuple[Sector]) -> int:
        """The size of a tree-block"""
        # OPTIMIZE ?
        return prod(s.sector_multiplicity(a) for s, a in zip(space.spaces, uncoupled))

    def tree_block_slice(self, tree: FusionTree) -> slice:
        """The range of indices of a tree-block within its block, as a slice."""
        # OPTIMIZE ?
        offset = 0
        for _unc in self.iter_uncoupled():
            if all(np.all(a == b) for a, b in zip(_unc, tree.uncoupled)):
                break
            offset += self.forest_block_size(_unc, tree.coupled)
        else:  # no break occurred
            raise ValueError('Uncoupled sectors incompatible')
        tree_block_sizes = self.tree_block_size(tree.uncoupled)
        tree_idx = fusion_trees(self.symmetry, tree.uncoupled, tree.coupled, tree.are_dual).index(tree)
        offset += tree_block_sizes * tree_idx
        size = tree_block_sizes
        return slice(offset, offset + size)

    # DUNDERS AND INTERNAL HELPERS

    def __eq__(self, other):
        if not isinstance(other, TensorProduct):
            return NotImplemented
        if self.num_spaces != other.num_spaces:
            return False
        if self.symmetry != other.symmetry:
            return False
        return all(s1 == s2 for s1, s2 in zip(self.spaces, other.spaces, strict=True))

    def __getitem__(self, idx):
        return self.spaces[idx]

    def __iter__(self):
        return iter(self.spaces)

    def __len__(self):
        return self.num_spaces

    def _repr(self, show_symmetry):
        raise NotImplementedError  # TODO rm this from Space class??

    def _calc_sectors(self, spaces: list[Space]) -> tuple[SectorArray, ndarray]:
        """Helper function for :meth:`__init__`"""
        if len(spaces) == 0:
            return self.symmetry.trivial_sector[None, :], np.ones([1], int)

        if len(spaces) == 1:
            sectors = spaces[0].sector_decomposition
            mults = spaces[0].multiplicities
            if spaces[0].sector_order == 'sorted':
                return sectors, mults
            perm = np.lexsort(sectors.T)
            return sectors[perm], mults[perm]

        if self.symmetry.is_abelian:
            grid = np.indices(tuple(space.num_sectors for space in spaces), np.intp)
            grid = grid.T.reshape(-1, len(spaces))
            sectors = self.symmetry.multiple_fusion_broadcast(
                *(sp.sector_decomposition[gr] for sp, gr in zip(spaces, grid.T))
            )
            multiplicities = np.prod(
                [space.multiplicities[gr] for space, gr in zip(spaces, grid.T)],
                axis=0
            )
            sectors, multiplicities, fusion_outcomes_sort = _unique_sorted_sectors(sectors, multiplicities)
            return sectors, multiplicities

        # define recursively
        sectors, mults = self._calc_sectors(spaces[:-1])
        sector_arrays = []
        mult_arrays = []
        for s2, m2 in zip(spaces[-1].sector_decomposition, spaces[-1].multiplicities):
            for s1, m1 in zip(sectors, mults):
                new_sects = self.symmetry.fusion_outcomes(s1, s2)
                sector_arrays.append(new_sects)
                if self.symmetry.fusion_style <= FusionStyle.multiple_unique:
                    new_mults = m1 * m2 * np.ones(len(new_sects), dtype=int)
                else:
                    # OPTIMIZE support batched N symbol?
                    new_mults = m1 * m2 * np.array([self.symmetry._n_symbol(s1, s2, c) for c in new_sects], dtype=int)
                mult_arrays.append(new_mults)
        sectors, multiplicities, _ = _unique_sorted_sectors(
            np.concatenate(sector_arrays, axis=0),
            np.concatenate(mult_arrays, axis=0)
        )
        return sectors, multiplicities


class AbelianLegPipe(LegPipe, ElementarySpace):
    r"""Special case of a :class:`LegPipe` for abelian symmetries.

    This class essentially exists to allow specialized handling of combined legs in the
    :class:`AbelianBackend`. For this backend, we want to treat combined legs, i.e. pipes, exactly
    the same as regular legs. This is why this class also inherits from :class:`ElementarySpace`,
    which are the "uncombined" legs. Crucially, this allows the pipe to have
    :attr:`defining_sectors` for the :attr:`cyten.backends.abelian.AbelianBackendData.block_inds`
    to point to, a well behaved :attr:`is_dual` attribute and to have a :attr:`basis_perm`,
    which can account for the basis permutation that is induced by going from sectors of the
    individual legs to a sorted list of coupled sectors on the pipe.

    Attributes
    ----------
    legs:
        The individual legs that form this pipe, and that the pipe can be split into.
        In particular, these are such that the pipe, as an :class:`ElementarySpace`, is isomorphic
        to their tensor product ``TensorProduct(legs)``, i.e. has the same :attr:`sector_decomposition`.
        TODO make this a test
    sector_strides : 1D numpy array of int
        F-style strides for the shape ``[leg.num_sectors for leg in self.legs]``. This allows
        one-to-one mapping between multi-indices (one block_ind per space) to a single index.
        Used in :meth:`AbelianBackend.combine_legs`.
    fusion_outcomes_sort : 1D numpy array of int
        The permutation that sorts the list of fusion outcomes.
        To calculate the :attr:`sector_decomposition` of the pipe, we go through all combinations
        of sectors from the :attr:`legs` in C-style order, i.e. varying sectors from the last leg
        the fastest. For each combination of sectors, we perform their fusion, which yields a single
        sector in the abelian case assumed here. The resulting list of fused sectors is in general
        neither sorted nor unique. This permutation (stable) sorts the resulting list.
    block_ind_map_slices : 1D numpy array of int
        Slices for embedding the unique fused sectors in the sorted list of all fusion outcomes.
        Shape is ``(K,)`` where ``K == pipe.num_sectors + 1``.
        Fusing all sectors from the :attr:`sector_decomposition` of all legs and sorting the
        outcomes gives a list which contains (in general) duplicates.
        The slice ``block_ind_map_slices[n]:block_ind_map_slices[n + 1]`` within this sorted list
        contains the same entry, namely ``pipe.sector_decomposition[n]``.
        Used in :math:`AbelianBackend.split_legs`.
    block_ind_map : 2D numpy array of int
        Map for the embedding of uncoupled to coupled indices, see notes below.
        Shape is ``(M, N)`` where ``M`` is the number of combinations of sectors,
        i.e. ``M == prod(s.num_sectors for s in spaces)`` and ``N == 3 + len(spaces)``.

    Notes
    -----
    TODO review these old notes. do they still apply? should they live somewhere else?
         this references the outdated ``ProductSpace``
    
    For ``np.reshape``, taking, for example,  :math:`i,j,... \rightarrow k` amounted to
    :math:`k = s_1*i + s_2*j + ...` for appropriate strides :math:`s_1,s_2`.

    In the charged case, however, we want to block :math:`k` by charge, so we must
    implicitly permute as well.  This reordering is encoded in `_block_ind_map` as follows.

    Each block index combination :math:`(i_1, ..., i_{nlegs})` of the `nlegs=len(spaces)`
    input `Space`s will end up getting placed in some slice :math:`a_j:a_{j+1}` of the
    resulting `ProductSpace`. Within this slice, the data is simply reshaped in usual row-major
    fashion ('C'-order), i.e., with strides :math:`s_1 > s_2 > ...` given by the block size.

    It will be a subslice of a new total block in the `ProductSpace` labelled by block index
    :math:`J`. We fuse charges according to the rule::

        ProductSpace.sector_decomposition[J] = fusion_outcomes(*[lsector_decomposition[i_l]
            for l, i_l, l in zip(incoming_block_inds, spaces)])

    Since many charge combinations can fuse to the same total charge,
    in general there will be many tuples :math:`(i_1, ..., i_{nlegs})` belonging to the same
    charge block :math:`J` in the `ProductSpace`.

    The rows of `_block_ind_map` are precisely the collections of
    ``[b_{J,k}, b_{J,k+1}, i_1, . . . , i_{nlegs}, J]``.
    Here, :math:`b_k:b_{k+1}` denotes the slice of this block index combination *within*
    the total block `J`, i.e., ``b_{J,k} = a_j - self.slices[J]``.

    The rows of `_block_ind_map` are lex-sorted first by ``J``, then the ``i``.
    Each ``J`` will have multiple rows, and the order in which they are stored in `block_inds`
    is the order the data is stored in the actual tensor.
    Thus, ``_block_ind_map`` might look like ::

        [ ...,
        [ b_{J,k},   b_{J,k+1},  i_1,    ..., i_{nlegs}   , J,   ],
        [ b_{J,k+1}, b_{J,k+2},  i'_1,   ..., i'_{nlegs}  , J,   ],
        [ 0,         b_{J+1,1},  i''_1,  ..., i''_{nlegs} , J + 1],
        [ b_{J+1,1}, b_{J+1,2},  i'''_1, ..., i'''_{nlegs}, J + 1],
        ...]

    """

    def __init__(self, legs: Sequence[ElementarySpace], is_dual: bool = False):
        LegPipe.__init__(self, legs=legs, is_dual=is_dual)
        sectors, mults = self._calc_sectors()  # also sets some attributes
        basis_perm = self._calc_basis_perm()
        ElementarySpace.__init__(self, symmetry=self.symmetry, defining_sectors=sectors,
                                 multiplicities=mults, is_dual=is_dual, basis_perm=basis_perm)

    def test_sanity(self):
        for l in self.legs:
            assert isinstance(l, ElementarySpace)
            if isinstance(l, LegPipe):
                assert isinstance(l, AbelianLegPipe)
            l.test_sanity()
        # check self.sector_strides
        assert self.sector_strides.shape == (self.num_legs,)
        expect = 1
        for i, num in enumerate(l.num_sectors for l in self.legs):
            assert self.sector_strides[i] == expect
            expect *= num
        # check block_ind_map_slices
        # note: we do not check for full correctness, just for consistency as slices
        assert self.block_ind_map_slices.shape == (self.num_sectors + 1,)
        assert self.block_ind_map_slices[0] == 0
        assert self.block_ind_map_slices[-1] == np.prod([l.num_sectors for l in self.legs])
        assert np.all(self.block_ind_map_slices[1:] >= self.block_ind_map_slices[:-1])
        # check block_ind_map
        assert self.block_ind_map.shape[0] <= np.prod([l.num_sectors for l in self.legs])
        assert self.block_ind_map.shape[1] == 3 + self.num_legs
        for i, (b1, b2, *idcs, J) in enumerate(self.block_ind_map):
            if i > 0 and J == self.block_ind_map[i - 1][-1]:
                assert b1 == self.block_ind_map[i - 1][1]
            else:
                assert b1 == 0
            charges = (leg.sector_decomposition[i] for i, leg in zip(idcs, self.legs))
            fused = self.symmetry.multiple_fusion(*charges)
            assert np.all(fused == self.sector_decomposition[J])
        # call to super class(es)
        LegPipe.test_sanity(self)
        ElementarySpace.test_sanity(self)

    def as_Space(self):
        return self

    def as_ElementarySpace(self, is_dual: bool = False):
        return self.with_is_dual(is_dual=is_dual)

    @property
    def dual(self) -> AbelianLegPipe:
        # TODO can we avoid recomputation of _calc_sectors and/or _calc_basis_perm??
        return AbelianLegPipe([l.dual for l in reversed(self.legs)], is_dual=not self.is_dual)

    @classmethod
    def from_basis(cls, *a, **kw):
        raise TypeError('from_basis is not supported for AbelianLegPipe')

    @classmethod
    def from_independent_symmetries(cls, independent_descriptions):
        assert all(isinstance(i, AbelianLegPipe) for i in independent_descriptions)
        is_dual = independent_descriptions[0].is_dual
        assert all(i.is_dual == is_dual for i in independent_descriptions[1:])
        legs = [
            i_legs[0].from_independent_symmetries(i_legs)
            for i_legs in zip(*(i.legs for i in independent_descriptions), strict=True)
        ]
        return cls(legs, is_dual=is_dual)

    @classmethod
    def from_null_space(cls, symmetry, is_dual=False):
        raise TypeError('from_null_space is not supported for AbelianLegPipe')

    @classmethod
    def from_sectors(cls, *a, **kw):
        raise TypeError('from_sectors is not supported for AbelianLegPipe')

    @classmethod
    def from_trivial_sector(cls, *a, **kw):
        raise TypeError('from_trivial_sector is not supported for AbelianLegPipe')

    def change_symmetry(self, symmetry, sector_map, injective=False):
        # TODO can we avoid some recomputation of _calc_sectors and _basis_perm?
        legs = [l.change_symmetry(symmetry, sector_map, injective) for l in self.legs]
        return AbelianLegPipe(legs, is_dual=self.is_dual)

    def drop_symmetry(self, which: int | list[int] = None):
        # TODO can we avoid some recomputation of _calc_sectors and _basis_perm?
        legs = [l.drop_symmetry(which) for l in self.legs]
        return AbelianLegPipe(legs, is_dual=self.is_dual)

    def take_slice(self, blockmask):
        msg = (
            'Using `AbelianLegPipe.take_slice` loses the product (pipe) structure and results in '
            'a plain ElementarySpace. Explicitly convert using `as_ElementarySpace` to suppress '
            'this warning.'
        )
        warnings.warn(msg, stacklevel=2)
        return self.as_ElementarySpace(is_dual=self.is_dual).take_slice(blockmask)

    def with_opposite_duality(self):
        # TODO can we avoid some recomputation here?
        return AbelianLegPipe(legs=self.legs, is_dual=not self.is_dual)

    def __eq__(self, other):
        if not isinstance(other, AbelianLegPipe):
            return NotImplemented
        if self.is_dual != other.is_dual:
            return False
        if self.num_legs != other.num_legs:
            return False
        return all(l1 == l2 for l1, l2 in zip(self.legs, other.legs))

    def _calc_sectors(self):
        """Helper function for :meth:`__init__`. Assumes ``LegPipe.__init__`` was called.
        
        Returns the defining_sectors and related multiplicities. Also sets the some attributes.
        """
        legs_num_sectors = tuple(l.num_sectors for l in self.legs)
        self.sector_strides = make_stride(legs_num_sectors, cstyle=False)

        # create a grid to select the multi-index sector
        grid = np.indices(legs_num_sectors, np.intp)
        # grid is an array with shape ``(num_legs, *legs_num_sectors)``,
        # with grid[li, ...] = {np.arange(space_block_numbers[li]) increasing in li-th direction}

        # collapse the different directions into one.
        grid = grid.T.reshape(-1, self.num_legs)  # *this* is the actual `reshaping`
        # *rows* of grid are now all possible combinations of block_inds.
        # transpose before reshape ensures that grid.T is np.lexsort()-ed

        nblocks = grid.shape[0]  # number of blocks in pipe = np.product(spaces_num_sectors)
        # this is different from num_sectors

        # determine block_ind_map -- it's essentially the grid.
        block_ind_map = np.zeros((nblocks, 3 + self.num_legs), dtype=np.intp)
        block_ind_map[:, 2:-1] = grid  # possible combinations of indices
        # block_ind_map[:, :2] and [:, -1] are set later.

        # the multiplicity for given (i1, i2, ...) is the product of ``multiplicities[il]``
        # advanced indexing:
        # ``grid.T[li]`` is a 1D array containing the block_indices `b_li` of leg ``li`` for all blocks
        multiplicities = np.prod([space.multiplicities[gr] for space, gr in zip(self.legs, grid.T)],
                                 axis=0)

        # calculate new defining_sectors
        sectors = self.symmetry.multiple_fusion_broadcast(
            *(s.sector_decomposition[gr] for s, gr in zip(self.legs, grid.T))
        )
        if self.is_dual:
            # the above are the future self.sector_decomposition
            # but we want to compute (and in particular sort according to) the defining_sectors
            sectors = self.symmetry.dual_sectors(sectors)

        # sort (non-dual) charge sectors.
        self.fusion_outcomes_sort = fusion_outcomes_sort = np.lexsort(sectors.T)
        block_ind_map = block_ind_map[fusion_outcomes_sort]
        sectors = sectors[fusion_outcomes_sort]
        multiplicities = multiplicities[fusion_outcomes_sort]

        slices = np.concatenate([[0], np.cumsum(multiplicities)], axis=0)
        block_ind_map[:, 0] = slices[:-1]  # start with 0
        block_ind_map[:, 1] = slices[1:]

        # bunch sectors with equal charges together
        diffs = find_row_differences(sectors, include_len=True)
        self.block_ind_map_slices = diffs
        slices = slices[diffs]
        multiplicities = slices[1:] - slices[:-1]
        diffs = diffs[:-1]

        sectors = sectors[diffs]

        new_block_ind = np.zeros(len(block_ind_map), dtype=np.intp)  # = J
        new_block_ind[diffs[1:]] = 1  # not for the first entry => np.cumsum starts with 0
        block_ind_map[:, -1] = new_block_ind = np.cumsum(new_block_ind)
        # calculate the slices within blocks: subtract the start of each block
        block_ind_map[:, :2] -= slices[new_block_ind][:, np.newaxis]
        self.block_ind_map = block_ind_map

        return sectors, multiplicities

    def _calc_basis_perm(self):
        """Helper function for :meth:`__init__`.

        Assumes ``LegPipe.__init__`` and ``_calc_sectors` were called. Returns the basis_perm.
        """
        # OPTIMIZE (JU) could make this a (cached) property and only compute when needed
        # TODO triple check and test this! -> implications on to_numpy after combine (adjust test!)
        # C-style for compatibility with e.g. numpy.reshape
        strides = make_stride(shape=[space.dim for space in self.spaces], cstyle=True)
        order = unstridify(self._get_fusion_outcomes_perm(), strides).T  # indices of the internal bases
        return sum(stride * space.inverse_basis_perm[p]
                   for stride, space, p in zip(strides, self.spaces, order))

    def _get_fusion_outcomes_perm(self):
        r"""Get the permutation introduced by the fusion.

        This permutation arises as follows:
        For each of the :attr:`legs` consider all sectors by order of appearance in the internal
        order, i.e. in :attr:`ElementarySpace.sector_decomposition``. Take all combinations of
        sectors from all the legs in C-style order, i.e. varying those from the last space the
        fastest. For each combination, perform the fusion (for abelian symmetries this yields
        a single sector each). This yields a list of sectors.
        The target permutation np.lexsort( .T)s this list of sectors.
        """
        # OPTIMIZE (JU) this is probably not the most efficient way to do this, but it hurts my brain
        #               and i need to get this to work, if only in an ugly way...
        fusion_outcomes_inverse_sort = inverse_permutation(self.fusion_outcomes_sort)
        # j : multi-index into the uncoupled private basis, i.e. into the C-style product of
        #     internal bases of the legs
        # i : index of self.legs
        # s : index of the list of all fusion outcomes / fusion channels
        dim_strides = make_stride([sp.dim for sp in self.legs])  # (num_legs,)
        sector_strides = make_stride([sp.num_sectors for sp in self.legs])  # (num_legs,)
        num_sector_combinations = np.prod([space.num_sectors for space in self.legs])
        # [i, j] :: position of the part of j in legs[i] within its private basis
        idcs = unstridify(np.arange(self.dim), dim_strides).T
        # [i, j] :: sector of the part of j in legs[i] is legs[i].sectors[sector_idcs[i, j]]
        #           sector_idcs[i, j] = bisect.bisect(legs[i].slices[:, 0], idcs[i, j]) - 1
        sector_idcs = np.array(
            [[bisect.bisect(sp.slices[:, 0], idx) - 1 for idx in idx_col]
             for sp, idx_col in zip(self.legs, idcs)]
        )  # OPTIMIZE can bisect.bisect be broadcast somehow? is there a numpy alternative?
        # [i, j] :: the part of j in legs[i] is the degeneracy_idcs[i, j]-th state within that sector
        #           degeneracy_idcs[i, j] = idcs[i, j] - legs[i].slices[sector_idcs[i, j], 0]
        degeneracy_idcs = idcs - np.stack(
            [sp.slices[si_col, 0] for sp, si_col in zip(self.legs, sector_idcs)]
        )
        # [i, j] :: strides for combining degeneracy indices.
        #           degeneracy_strides[:, j] = make_stride([... mults with sector_idcs[:, j]])
        degeneracy_strides = np.array(
            [make_stride([sp.multiplicities[si] for sp, si in zip(self.legs, si_row)])
             for si_row in sector_idcs.T]
        ).T  # OPTIMIZE make make_stride broadcast?
        # [j] :: position of j in the unsorted list of fusion outcomes
        fusion_outcome = np.sum(sector_idcs * sector_strides[:, None], axis=0)
        # [i, s] :: sector combination s has legs[i].sectors[all_sector_idcs[i, s]]
        all_sector_idcs = unstridify(np.arange(num_sector_combinations), sector_strides).T
        # [i, s] :: all_mults[i, s] = legs[i].multiplicities[all_sector_idcs[i, s]]
        all_mults = np.array([sp.multiplicities[comb] for sp, comb in zip(self.legs, all_sector_idcs)])
        # [s] : total multiplicity of the fusion channel
        fusion_outcome_multiplicities = np.prod(all_mults, axis=0)
        # [s] : !!shape == (L_s + 1,)!!  ; starts ([s]) and stops ([s + 1]) of fusion channels in the sorted list
        fusion_outcome_slices = np.concatenate(
            [[0], np.cumsum(fusion_outcome_multiplicities[self.fusion_outcomes_sort])]
        )
        # [j] : position of fusion channel after sorting
        sorted_pos = fusion_outcomes_inverse_sort[fusion_outcome]
        # [j] :: contribution from the sector, i.e. start of all the js of the same fusion channel
        sector_part = fusion_outcome_slices[sorted_pos]
        # [j] :: contribution from the multiplicities, i.e. position with all js of the same fusion channel
        degeneracy_part = np.sum(degeneracy_idcs * degeneracy_strides, axis=0)
        return inverse_permutation(sector_part + degeneracy_part)


def _unique_sorted_sectors(unsorted_sectors: SectorArray, unsorted_multiplicities: np.ndarray):
    """Sort sectors and merge duplicates.

    Given unsorted sectors which may contain duplicates,
    return a sorted list of unique sectors and corresponding *aggregate* multiplicities

    Returns
    -------
    sectors
        The unique entries of the `unsorted_sectors`, sorted according to ``np.lexsort( .T)``.
    multiplicities
        The corresponding aggregate multiplicities, i.e. the sum of all entries in
        `unsorted_multiplicities` which correspond to the given sector
    perm
        The permutation that sorts the input, i.e. ``np.lexsort(unsorted_sectors.T)``.
    """
    sectors, multiplicities, perm = _sort_sectors(unsorted_sectors, unsorted_multiplicities)
    slices = np.concatenate([[0], np.cumsum(multiplicities)], axis=0)
    diffs = find_row_differences(sectors, include_len=True)
    slices = slices[diffs]
    multiplicities = slices[1:] - slices[:-1]
    sectors = sectors[diffs[:-1]]
    return sectors, multiplicities, perm


def _sort_sectors(sectors: SectorArray, multiplicities: np.ndarray):
    perm = np.lexsort(sectors.T)
    return sectors[perm], multiplicities[perm], perm


def _parse_inputs_drop_symmetry(which: int | list[int] | None, symmetry: Symmetry
                                ) -> tuple[list[int] | None, Symmetry]:
    """Input parsing for :meth:`Space.drop_symmetry`.

    Returns
    -------
    which : None | list of int
        Which symmetries to drop, as integers in ``range(len(symmetries.factors))``.
        ``None`` indicates to drop all.
    remaining_symmetry : Symmetry
        The symmetry that remains.
    """
    if which is None or which == []:
        pass
    elif isinstance(symmetry, ProductSymmetry):
        which = to_iterable(which)
        num_factors = len(symmetry.factors)
        # normalize negative indices to be in range(num_factors)
        for i, w in enumerate(which):
            if not -num_factors <= w < num_factors:
                raise ValueError(f'which entry {w} out of bounds for {num_factors} symmetries.')
            if w < 0:
                which[i] += num_factors
        if len(which) == num_factors:
            which = None
    elif which == 0 or which == [0]:
        which = None
    else:
        msg = f'Can not drop which={which} for a single (non-ProductSymmetry) symmetry.'
        raise ValueError(msg)

    if which is None:
        remaining_symmetry = no_symmetry
    else:
        factors = [f for i, f in enumerate(symmetry.factors) if i not in which]
        if len(factors) == 1:
            remaining_symmetry = factors[0]
        else:
            remaining_symmetry = ProductSymmetry(factors)

    return which, remaining_symmetry
