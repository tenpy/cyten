"""Couplings are the building blocks of Hamiltonians for lattice models.

This module defines a base class for couplings, which are given in a MPO-like factorized form,
as well as functions that create common couplings such as e.g. a Heisenberg couplings between
two sites that have a spin degree of freedom.
"""

# Copyright (C) TeNPy Developers, Apache license
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..backends import get_same_backend
from ..block_backends import Block, Dtype
from ..symmetries import (
    BraidChiralityUnspecifiedError,
    FermionParity,
    FibonacciAnyonCategory,
    IsingAnyonCategory,
    NoSymmetry,
    SU2,
    SU2_kAnyonCategory,
    Sector,
    SymmetryError,
    U1,
    ZN,
    fibonacci_anyon_category,
)
from ..tensors import (
    SymmetricTensor, add_trivial_leg, almost_equal, compose, horizontal_factorization, permute_legs,
    squeeze_legs,
)
from .degrees_of_freedom import ALL_SPECIES, BosonicDOF, ClockDOF, FermionicDOF, Site, SpinDOF
from .sites import (
    ClockSite, FibonacciAnyonSite, GoldenSite, IsingAnyonSite, SpinHalfFermionSite, SpinSite,
    SpinlessBosonSite, SpinlessFermionSite, SU2kSpin1Site,
)


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

def _symmetry_factor_to_dict(factor) -> dict:
    """Serialize a single SymmetryFactor to a dict."""
    dn = factor.descriptive_name  # included for all types; Symmetry.__eq__ checks this
    if isinstance(factor, NoSymmetry):
        return {'type': 'NoSymmetry'}
    if isinstance(factor, FermionParity):
        return {'type': 'FermionParity', 'descriptive_name': dn}
    if isinstance(factor, U1):
        return {'type': 'U1', 'descriptive_name': dn}
    if isinstance(factor, SU2):
        return {'type': 'SU2', 'descriptive_name': dn}
    if isinstance(factor, ZN):
        return {'type': 'ZN', 'N': int(factor.N), 'descriptive_name': dn}
    if isinstance(factor, FibonacciAnyonCategory):
        return {'type': 'FibonacciAnyonCategory', 'handedness': factor.handedness}
    if isinstance(factor, IsingAnyonCategory):
        return {'type': 'IsingAnyonCategory', 'nu': int(factor.nu)}
    if isinstance(factor, SU2_kAnyonCategory):
        return {'type': 'SU2_kAnyonCategory', 'k': int(factor.k), 'handedness': factor.handedness}
    raise NotImplementedError(
        f'Cannot serialize symmetry factor {type(factor).__name__!r}. '
        f'Add a branch to _symmetry_factor_to_dict.'
    )


def _symmetry_to_dict(sym) -> dict:
    """Serialize a (possibly product) Symmetry to a dict."""
    return {'factors': [_symmetry_factor_to_dict(f) for f in sym.factors]}


def _space_to_dict(space) -> dict:
    """Serialize a single ElementarySpace to a  dict."""
    sectors = space.defining_sectors
    if hasattr(sectors, 'tolist'):
        sectors = sectors.tolist()
    else:
        sectors = [[(s.item() if hasattr(s, 'item') else int(s)) for s in row] for row in sectors]
    bp = space._basis_perm
    return {
        'symmetry': _symmetry_to_dict(space.symmetry),
        'sectors': sectors,
        'multiplicities': [(m.item() if hasattr(m, 'item') else int(m)) for m in space.multiplicities],
        'is_dual': bool(space.is_dual),
        'basis_perm': bp.tolist() if bp is not None else None,
    }


def _site_to_dict(site) -> dict:
    """Serialize a Site to a dict of {type, constructor-params}.

    Only the parameters needed to reconstruct the site are stored — the physical
    leg and operators are derived by the constructor, so they are not duplicated.
    """
    name = type(site).__name__
    if isinstance(site, SpinSite):
        return {'type': name, 'S': float(site.S), 'conserve': site.conserve}
    if isinstance(site, SpinHalfFermionSite):
        return {
            'type': name,
            'conserve_N': site.conserve_N,
            'conserve_S': site.conserve_S,
            'filling': site.filling,
        }
    if isinstance(site, SpinlessBosonSite):
        Nmax = site.Nmax
        if hasattr(Nmax, 'tolist'):
            Nmax = Nmax.tolist()
        elif hasattr(Nmax, '__iter__') and not isinstance(Nmax, (str, int)):
            Nmax = list(Nmax)
        return {'type': name, 'Nmax': Nmax, 'conserve': site.conserve, 'filling': site.filling}
    if isinstance(site, SpinlessFermionSite):
        return {
            'type': name,
            'num_species': site.num_species,
            'conserve': site.conserve,
            'filling': site.filling,
        }
    if isinstance(site, FibonacciAnyonSite):
        return {'type': name, 'handedness': site.symmetry.handedness}
    if isinstance(site, IsingAnyonSite):
        return {'type': name, 'nu': int(site.symmetry.nu)}
    if isinstance(site, GoldenSite):
        return {'type': name, 'handedness': site.symmetry.handedness}
    if isinstance(site, SU2kSpin1Site):
        return {'type': name, 'k': int(site.symmetry.k), 'handedness': site.symmetry.handedness}
    if isinstance(site, ClockSite):
        return {'type': name, 'q': int(site.q), 'conserve': site.conserve}
    raise NotImplementedError(
        f'Cannot serialize site type {name!r}. '
        f'Add a branch to _site_to_dict.'
    )


def _adjacent_transpositions(permutation: Sequence[int]) -> list[int]:
    """Decompose a permutation into a sequence of adjacent position swaps.

    Parameters
    ----------
    permutation : list of int
        A permutation of ``range(len(permutation))``; ``permutation[k]`` is the value that ends
        up at position `k`.

    Returns
    -------
    swap_positions : list of int
        Positions `pos` such that applying the swaps ``(pos, pos + 1)`` in order to
        ``list(range(len(permutation)))`` produces `permutation`. Realizes `permutation` with the
        minimal number of adjacent transpositions, i.e. its number of inversions.

    """
    n = len(permutation)
    working = list(range(n))
    swap_positions = []
    for target_pos in range(n):
        value = permutation[target_pos]
        cur = working.index(value, target_pos)
        while cur > target_pos:
            swap_positions.append(cur - 1)
            working[cur - 1], working[cur] = working[cur], working[cur - 1]
            cur -= 1
    assert working == list(permutation)
    return swap_positions


def freeze(obj):
    """Recursively turn the output of the ``_*_to_dict`` helpers into hashable nested tuples."""
    if isinstance(obj, dict):
        return tuple((k, freeze(v)) for k, v in sorted(obj.items()))
    if isinstance(obj, (list, tuple)):
        return tuple(freeze(v) for v in obj)
    return obj


class Coupling:
    """A coupling is an operator on a few :class:`Site` s, factorized as one tensor per site.

    A coupling represents an operator of the following form::

        |        p0   p1   ..   pN
        |        │    │    │    │
        |       ┏┷━━━━┷━━━━┷━━━━┷┓
        |       ┃       h        ┃
        |       ┗┯━━━━┯━━━━┯━━━━┯┛
        |        │    │    │    │
        |        p0*  p1*  ..  pN*

    The intended use case is to build tensor network representations (e.g. MPOs) of Hamiltonians.

    Attributes
    ----------
    sites : list of :class:`Site`
        The sites that the operators act on.
    factorization : list of :class:`SymmetricTensor`
        A list of tensors that, if contracted, give the operator that is represented.
        Each tensor ``factorization[i]`` has legs ``[wL, p, wR, p*]``, where ``p`` and ``p*``
        are the physical :attr:`Site.leg` of the corresponding ``sites[i]``, and where contracting
        the ``wL`` and ``wR`` legs in an MPO-like geometry gives the multi-site operator.
    name : str, optional
        A descriptive name that can be used when pretty-printing, to identify the coupling.
        For example, a Heisenberg coupling is usually initialized with name ``'S.S'``.

    """

    def __init__(
        self, sites: list[Site], factorization: list[SymmetricTensor], name: str = None, skip_sanity: bool = False
    ):
        self.sites = sites
        assert len(factorization) == len(sites) #or len(factorization) == len(sites) + 1
        self.factorization = factorization
        self.name = name
        self._levels: list[int] = list(range(1, len(sites) + 1))
        # cache of previously computed permutations of this instance, filled by :meth:`permute`.
        self._permuted: list[tuple[tuple[int, ...], Coupling]] = []
        if not skip_sanity:
            self.test_sanity()

    def test_sanity(self):
        """Perform sanity checks."""
        backend = get_same_backend(*self.sites)
        site_idx = 0
        for W in self.factorization:
            W.test_sanity()
            assert W.backend == backend
            assert W.num_codomain_legs == 2
            assert W.num_domain_legs == 2
            assert W.labels == ['wL', 'p', 'wR', 'p*']
            if site_idx < len(self.sites):
                s = self.sites[site_idx]
                assert W.get_leg_co_domain('p') == s.leg
                assert W.get_leg_co_domain('p*') == s.leg
                site_idx += 1
        assert self.factorization[0].get_leg('wL').is_trivial
        for W1, W2 in zip(self.factorization[:-1], self.factorization[1:]):
            assert W1.get_leg_co_domain('wR') == W2.get_leg_co_domain('wL')
        assert self.factorization[-1].get_leg('wR').is_trivial

    def _key(self):
        """Structural identity used by :meth:`__hash__`/:meth:`__eq__`.

        Contains only hashable metadata, no floating-point tensor
        data, which is instead compared numerically (via :func:`~cyten.tensors.almost_equal`) in
        :meth:`__eq__`. This is what determines whether two (distinct) `Coupling` instances may
        share the same key in an :class:`~tenpy.networks.mpo.MPOGraph`.

        Note that `Coupling` is currently mutable, so this key/``__hash__``/``__eq__`` scheme is
        only correct as long as a coupling already used as a dict key isn't mutated afterwards;
        enforcing immutability is left for a future translation of this code to C++.
        """
        return (
            self.name,
            tuple(freeze(_site_to_dict(s)) for s in self.sites),
            tuple(
                (
                    tuple(t.shape),
                    tuple(t.labels),
                    t.dtype.name,
                    tuple(freeze(_space_to_dict(f)) for f in t.codomain.factors),
                    tuple(freeze(_space_to_dict(f)) for f in t.domain.factors),
                )
                for t in self.factorization
            ),
        )

    def __hash__(self):
        # recomputed on every call (not cached): see the mutability note in :meth:`_key`.
        return hash(self._key())

    def __eq__(self, other):
        if self is other:
            return True
        if not isinstance(other, Coupling):
            return NotImplemented
        if self._key() != other._key():
            return False
        # `_key()` already guarantees matching legs for corresponding tensors, so `almost_equal`
        # (which requires identical legs) cannot raise here.
        for t1, t2 in zip(self.factorization, other.factorization):
            if t1 is t2:
                continue
            if not almost_equal(t1, t2):
                return False
        return True

    def __repr__(self):
        site_names = [type(s).__name__ for s in self.sites]
        shapes = [tuple(t.shape) for t in self.factorization]
        return f'Coupling(name={self.name!r}, sites={site_names}, shapes={shapes})'

    @classmethod
    def from_dense_block(
        cls,
        operator: Block,
        sites: list[Site],
        name: str = None,
        dtype: Dtype = None,
        understood_braiding: bool = False,
        cutoff_singular_values: float = None,
    ) -> Coupling:
        """Convert a dense block to a :class:`Coupling`.

        Parameters
        ----------
        operator : Block
            The data to be converted to a Coupling as a backend-specific block or some data that
            can be converted using :meth:`BlockBackend.as_block`. The order of axes must match the
            `sites`, that is, the axes correspond to ``[p0, p1, ..., p1*, p0*]`` (codomain legs
            ascending, domain legs descending), where ``pi`` corresponds to site ``sites[i]``.
            The block should be given in the "public" basis order of the sites, i.e.,
            according to ``sites[i].sectors_of_basis``.
        sites : list of :class:`Site`
            The sites that the operators act on.
        name : str, optional
            A descriptive name that can be used when pretty-printing, to identify the coupling.
        dtype : :class:`Dtype`, optional
            If given, the block is converted to that dtype and the resulting tensors in the
            factorization will have that dtype. By default, we detect the dtype from the block.
        cutoff_singular_values : float, optional
            If given, truncate singular values (see :func:`cyten.horizontal_factorization`)
            below this threshold.

        """
        backend = get_same_backend(*sites)
        device = sites[0].default_device
        assert all(s.default_device == device for s in sites[1:])
        co_domain = [s.leg for s in sites]
        p_labels = [f'p{i}' for i in range(len(sites))]
        labels = [*p_labels, *[f'{pi}*' for pi in p_labels][::-1]]
        op = SymmetricTensor.from_dense_block(
            operator,
            co_domain,
            co_domain,
            backend=backend,
            labels=labels,
            dtype=dtype,
            device=device,
            understood_braiding=understood_braiding,
        )
        return cls.from_tensor(op, sites=sites, name=name, cutoff_singular_values=cutoff_singular_values)

    @classmethod
    def from_tensor(
        cls,
        operator: SymmetricTensor,
        sites: list[Site],
        name: str = None,
        cutoff_singular_values: float = None,
    ) -> Coupling:
        """Convert an operator / tensor to a :class:Coupling.

        Decomposes an operator into factors using :func:`cyten.horizontal_factorization` to
        obtain the :attr:`factorization` of the coupling.

        Parameters
        ----------
        operator : :class:`SymmetricTensor`
            Operator to be converted to a coupling. The legs should be ordered as
            ``[p0, p1, ..., p1*, p0*]``, where ``pi`` and ``pi*`` correspond to the legs associated
            with site ``sites[i]``.
        sites : list of :class:`Site`
            The sites that the operator acts on.
        name : str, optional
            A descriptive name that can be used when pretty-printing, to identify the coupling.
            For example, a Heisenberg coupling is usually initialized with name ``'S.S'``.
        cutoff_singular_values : float, optional
            If given, truncate singular values (see :func:`cyten.horizontal_factorization`)
            below this threshold.

        """
        assert operator.backend == get_same_backend(*sites)
        assert operator.codomain.factors == [site.leg for site in sites]
        assert operator.domain.factors == operator.codomain.factors
        p_labels = [f'p{i}' for i in range(len(sites))]
        assert operator.labels == [*p_labels, *[f'{pi}*' for pi in p_labels][::-1]]

        if len(sites) == 1:
            W = add_trivial_leg(operator, codomain_pos=0, label='wL')
            W = add_trivial_leg(W, domain_pos=1, label='wR')
            W.relabel({'p0': 'p', 'p0*': 'p*'})
            factorization = [W]
        else:
            W, rest = horizontal_factorization(
                operator, 1, 1, new_labels=['wR', 'wL'], cutoff_singular_values=cutoff_singular_values
            )
            W.relabel({'p0': 'p', 'p0*': 'p*'})
            factorization = [add_trivial_leg(W, codomain_pos=0, label='wL')]
            for i in range(1, len(sites) - 1):
                W, rest = horizontal_factorization(
                    rest, 2, 1, new_labels=['wR', 'wL'], cutoff_singular_values=cutoff_singular_values
                )
                W.relabel({f'p{i}': 'p', f'p{i}*': 'p*'})
                factorization.append(W)
            assert (rest.num_codomain_legs, rest.num_domain_legs) == (2, 1)
            rest.relabel({f'p{len(sites) - 1}': 'p', f'p{len(sites) - 1}*': 'p*'})
            factorization.append(add_trivial_leg(rest, domain_pos=1, label='wR'))
        return Coupling(sites=sites, factorization=factorization, name=name)

    @property
    def num_sites(self):
        return len(self.sites)

    def to_tensor(self) -> SymmetricTensor:
        """Convert to a tensor."""
        # TODO : this would be a great use case for a planar diagram as well...
        res = squeeze_legs(self.factorization[0], 'wL')
        res = permute_legs(res, [-1, 0], [1], bend_right=False)
        res.relabel({'p': 'p0', 'p*': 'p0*'})
        for i in range(1, self.num_sites):
            W = permute_legs(self.factorization[i], ['wL'], ['p*', 'wR', 'p'], bend_right=True)
            res = compose(res, W, relabel2={'p': f'p{i}', 'p*': f'p{i}*'})
            res = permute_legs(res, [-1, *range(2 * i), 2 * i], [-2], bend_right={-1: False, -3: True})
        res = squeeze_legs(res, 'wR')
        codom_labels = [f'p{i}' for i in range(len(self.sites))]
        dom_labels = [l + '*' for l in codom_labels]
        res = permute_legs(res, codom_labels, dom_labels, bend_right=False)
        return res

    def to_numpy(
        self, leg_order: list[int | str] = None, numpy_dtype=None, understood_braiding: bool = False
    ) -> np.ndarray:
        """Convert to a numpy array."""
        return self.to_tensor().to_numpy(leg_order, numpy_dtype, understood_braiding)

    def insert_identity_between_sites(self, position: int) -> Coupling:
        """Insert identity tensor between sites at given position."""
        if position <= 0 or position >= len(self.sites):
            raise ValueError(f'Position must be between 1 and {len(self.sites) - 1}, got {position}')

        site_left = self.sites[position - 1]
        site_right = self.sites[position]

        left_block = self.factorization[position - 1]
        right_block = self.factorization[position]

        wR_space = left_block.domain.factors[-1]
        wL_space = right_block.codomain.factors[0]

        if isinstance(wR_space, list) or isinstance(wL_space, list):
            raise NotImplementedError('Multi-bond insertions not yet supported')

        if site_left.leg != site_right.leg:
            raise ValueError('Sites must have same physical leg.')

        # delegates to Site.identity_tensor, which also checks wR_space == wL_space
        identity = site_left.identity_tensor(wR_space, wL_space)

        new_sites = self.sites[:position] + [site_left] + self.sites[position:]
        new_factorization = (
            self.factorization[:position]
            + [identity]
            + self.factorization[position:]
        )

        return Coupling(sites=new_sites, factorization=new_factorization, name=self.name, skip_sanity=True)

    def permute(
        self, permutation: Sequence[int], levels: Sequence[int | None], over_braid: Sequence[bool | None]
    ) -> Coupling:
        """Permute the sites of this coupling, braiding through the (possibly anyonic) legs.

        Contracts `self` to a single tensor (:meth:`to_tensor`), realizes `permutation` as a
        sequence of elementary adjacent-site transpositions (each one braiding the full ``(p, p*)``
        leg pair of one site past that of its neighbour, as a single unit -- the two legs of one
        site never cross each other), and re-factorizes the result (:meth:`from_tensor`) with the
        sites reordered accordingly. This is analogous to how
        :class:`~cyten.backends.fusion_tree_backend.PermuteLegsInstructionEngine` realizes a leg
        permutation as a sequence of elementary swaps, tracking a `levels` list that is itself
        reordered as legs move.

        Results are cached on `self` (not shared with `self`'s other permutations, or with the
        result of this call): calling ``self.permute(permutation, ...)`` twice with the same
        `permutation` returns the same (cached) result, using `levels`/`over_braid` only the first
        time; a different `permutation` triggers a new computation.

        Parameters
        ----------
        permutation : list of int
            A permutation of ``range(len(self.sites))``. ``permutation[k]`` is the index (in
            `self`'s current order) of the site that ends up at new position `k`.
        levels : list of int | None
            One entry per site of `self` (in `self`'s current order): its "height", used to
            derive the braid chirality (over/under) for any elementary transposition whose
            `over_braid` entry is ``None``, the same way :attr:`~cyten.symmetries.Symmetry` legs
            with a higher level braid over those with a lower one. Only needed for symmetries
            without a symmetric braid (see :attr:`~cyten.symmetries.Symmetry.braiding_style`);
            ignored otherwise.
        over_braid : list of bool | None
            One entry per elementary adjacent-site transposition needed to realize `permutation`
            (i.e. NOT one entry per site -- the number of transpositions depends on
            `permutation`, e.g. via the number of its inversions). Explicitly fixes the braid
            chirality for that transposition (``True`` = the site moving from the lower position
            over the one moving from the higher position); ``None`` derives it from `levels`.

        Returns
        -------
        Coupling
            A new coupling with :attr:`sites` (and the represented operator) reordered according
            to `permutation`.

        """
        n = len(self.sites)
        permutation = list(permutation)
        if sorted(permutation) != list(range(n)):
            raise ValueError(f'`permutation` must be a permutation of range({n}), got {permutation}')
        if len(levels) != n:
            raise ValueError(f'need {n} `levels`, one per site, got {len(levels)}')

        key = tuple(permutation)
        for cached_key, cached_coupling in self._permuted:
            if cached_key == key:
                return cached_coupling

        swap_positions = _adjacent_transpositions(permutation)
        if len(over_braid) != len(swap_positions):
            raise ValueError(
                f'need {len(swap_positions)} entries in `over_braid` (one per elementary '
                f'adjacent transposition realizing this permutation), got {len(over_braid)}'
            )

        tensor = self.to_tensor()
        sites = list(self.sites)
        levels_state = list(levels)
        # current label (p{original_idx} / p{original_idx}*) at each position; labels are
        # intrinsic to a leg and travel with it, only their position changes.
        codomain_labels = [f'p{i}' for i in range(n)]
        domain_labels = [f'p{i}*' for i in range(n)]

        for step, pos in enumerate(swap_positions):
            over = over_braid[step]
            if over is None:
                level_1, level_2 = levels_state[pos], levels_state[pos + 1]
                if level_1 is None or level_2 is None:
                    raise BraidChiralityUnspecifiedError('Sites that braid must have specified levels.')
                if level_1 == level_2:
                    raise BraidChiralityUnspecifiedError('Sites that braid can not have the same level.')
                over = level_1 > level_2
            new_codomain = list(codomain_labels)
            new_codomain[pos], new_codomain[pos + 1] = new_codomain[pos + 1], new_codomain[pos]
            new_domain = list(domain_labels)
            new_domain[pos], new_domain[pos + 1] = new_domain[pos + 1], new_domain[pos]
            # ket and bra of the same site have the same level: they move together and never
            # cross each other.
            level_dict = {
                codomain_labels[pos]: 1 if over else 0,
                domain_labels[pos]: 1 if over else 0,
                codomain_labels[pos + 1]: 0 if over else 1,
                domain_labels[pos + 1]: 0 if over else 1,
            }
            tensor = permute_legs(tensor, codomain=new_codomain, domain=new_domain, levels=level_dict)
            codomain_labels, domain_labels = new_codomain, new_domain
            sites[pos], sites[pos + 1] = sites[pos + 1], sites[pos]
            levels_state[pos], levels_state[pos + 1] = levels_state[pos + 1], levels_state[pos]

        relabelling = {}
        for new_pos, old_idx in enumerate(permutation):
            relabelling[f'p{old_idx}'] = f'p{new_pos}'
            relabelling[f'p{old_idx}*'] = f'p{new_pos}*'
        tensor = tensor.relabel(relabelling)

        result = Coupling.from_tensor(tensor, sites=sites, name=self.name)
        result._levels = [self._levels[i] for i in permutation]
        self._permuted.append((key, result))
        return result

# SPIN COUPLINGS


def spin_spin_coupling(
    sites: list[SpinDOF], Jx: float = 0, Jy: float = 0, Jz: float = 0, name: str = 'spin-spin'
) -> Coupling:
    r"""Two-site coupling between spins.

    .. math ::
        h_{ij} = \mathtt{Jx} S_i^x S_j^x + \mathtt{Jy} S_i^y S_j^y + \mathtt{Jz} S_i^z S_j^z

    Parameters
    ----------
    sites: list of Site
        The sites that the coupling acts on. Note that the order matters for the final leg order.
    Jx, Jy, Jz: float
        Prefactor, as given above. By default, all prefactors vanish.

    """
    if len(sites) != 2:
        raise ValueError(f'Invalid number of sites. Expected 2, got {len(sites)}')
    s1 = sites[0].spin_vector
    s2 = sites[1].spin_vector
    h = 0  # build in leg order [p0, p0*, p1, p1*] and transpose only once before returning
    h += Jx * np.tensordot(s1[:, :, 0], s2[:, :, 0], axes=0)
    h += Jy * np.tensordot(s1[:, :, 1], s2[:, :, 1], axes=0)
    h += Jz * np.tensordot(s1[:, :, 2], s2[:, :, 2], axes=0)
    h = np.transpose(h, [0, 2, 3, 1])
    return Coupling.from_dense_block(h, sites, name=name, understood_braiding=True)


def spin_field_coupling(
    sites: list[SpinDOF], hx: float = 0, hy: float = 0, hz: float = 0, name: str = 'spin-field'
) -> Coupling:
    r"""Single-site coupling of a spin to an external field.

    .. math ::
        h_i = \mathtt{hx} S_i^x + \mathtt{hy} S_i^y + \mathtt{hz} S_i^z

    Parameters
    ----------
    sites: list of Site
        The sites that the coupling acts on. Note that the order matters for the final leg order.
    hx, hy, hz: float
        Prefactor, as given above. By default, all prefactors vanish.

    """
    if len(sites) != 1:
        raise ValueError(f'Invalid number of sites. Expected 1, got {len(sites)}')
    s = sites[0].spin_vector
    h = hx * s[:, :, 0] + hy * s[:, :, 1] + hz * s[:, :, 2]
    return Coupling.from_dense_block(h, sites, name=name, understood_braiding=True)


def aklt_coupling(sites: list[SpinDOF], J: float = 1, name: str = 'AKLT') -> Coupling:
    r"""Two-site AKLT coupling between spins.

    .. math ::
        h_{ij} = \mathtt{J} [\vec{S}_i \cdot \vec{S}_j + \frac{1}{3} (\vec{S}_i \cdot \vec{S}_j)^2]

    This is the coupling originally defined by Affleck, Kennedy, Lieb, Tasaki
    in :cite:`affleck1987`, except we drop the constant part of 1/3 per bond and rescale with a
    factor of 2, i.e. :math:`h_{ij} = 2 P^{S=2}_{i, j} + const.`.

    It was defined for spin-1 degrees of freedom in the original work, but we allow any site
    with a spin DOF. Note that the coupling simplifies to a Heisenberg coupling for spin-1/2.

    Parameters
    ----------
    sites: list of Site
        The sites that the coupling acts on. Note that the order matters for the final leg order.
    J: float
        Prefactor, as given above. By default use ``1``.

    """
    if len(sites) != 2:
        raise ValueError(f'Invalid number of sites. Expected 2, got {len(sites)}')
    s1 = sites[0].spin_vector
    s2 = sites[1].spin_vector
    S_dot_S = np.tensordot(s1, s2, axes=[2, 2])
    S_dot_S = np.transpose(S_dot_S, [0, 2, 3, 1])
    S_dot_S_square = np.tensordot(S_dot_S, S_dot_S, axes=[[3, 2], [0, 1]])
    h = J * (S_dot_S + S_dot_S_square / 3.0)
    return Coupling.from_dense_block(h, sites, name=name, understood_braiding=True)


def heisenberg_coupling(sites: list[SpinDOF], J: float = 1, name: str = 'S.S') -> Coupling:
    r"""Two-site Heisenberg coupling between spins.

    .. math ::
        h_{ij} = \mathtt{J} \vec{S}_i \cdot \vec{S}_j

    Parameters
    ----------
    sites: list of Site
        The sites that the coupling acts on. Note that the order matters for the final leg order.
    J: float
        Prefactor, as given above. By default use ``1``, i.e. an anti-ferromagnetic coupling.

    """
    return spin_spin_coupling(sites=sites, Jx=J, Jy=J, Jz=J, name=name)


def chiral_3spin_coupling(sites: list[SpinDOF], chi: float = 1, name: str = 'S.SxS') -> Coupling:
    r"""Chiral coupling of three spins.

    .. math ::
        h_{ijk} = \mathtt{chi} \vec{S}_i \cdot ( \vec{S}_j \times \vec{S}_k )

    Parameters
    ----------
    sites: list of Site
        The sites that the coupling acts on. Note that the order matters for the final leg order.
    chi: float
        Prefactor, as given above. By default use ``1``.

    """
    if len(sites) != 3:
        raise ValueError(f'Invalid number of sites. Expected 3, got {len(sites)}')
    SxS = np.cross(
        sites[1].spin_vector[:, None, None, :, :], sites[2].spin_vector[None, :, :, None, :], axis=4
    )  # [p1, p2, p2*, p1*, i]
    h = chi * np.tensordot(sites[0].spin_vector, SxS, (-1, -1))  # [p0, p0*, p1, p2, p2*, p1*]
    h = np.transpose(h, [0, 2, 3, 4, 5, 1])
    return Coupling.from_dense_block(h, sites, name=name, understood_braiding=True)


# BOSON AND FERMION COUPLINGS


def chemical_potential(
    sites: list[BosonicDOF] | list[FermionicDOF],
    mu: float,
    species: int | str | list[int | str] = ALL_SPECIES,
    name: str = 'chem. pot.',
) -> Coupling:
    r"""Chemical potential for bosons or fermions. Single-site coupling.

    .. math ::
        h_i = -\mathtt{mu} \sum_{k \in \mathtt{species} n_{i, k}

    where :math:`n_{i, k}` is the occupation number of species :math:`k` on site :math:`i`.

    Parameters
    ----------
    sites: list of Site
        The sites that the coupling acts on. Note that the order matters for the final leg order.
    mu: float
        Chemical potential, as defined above.
    species: (list of) int | str, optional
        If given, the chemical potential only couples to the occupation of this species.
        By default, it couples to the total occupation of all species.

    """
    if len(sites) != 1:
        raise ValueError(f'Invalid number of sites. Expected 1, got {len(sites)}')
    h = -mu * sites[0].get_occupation_numpy(species=species)
    return Coupling.from_dense_block(h, sites=sites, name=name, understood_braiding=True)


def onsite_interaction(
    sites: list[BosonicDOF] | list[FermionicDOF],
    U: float = 1,
    species: int | str = ALL_SPECIES,
    name: str = 'onsite interaction',
) -> Coupling:
    r"""Onsite interaction for bosons or fermions. Single-site coupling.

    .. math ::
        h_i = \frac{U}{2} n_i^2

    where :math:`n_i` is the total occupation number, or the occupation of a single `species`.

    Parameters
    ----------
    sites: list of Site
        The sites that the coupling acts on. Note that the order matters for the final leg order.
    U: float
        Prefactor, as defined above. By default, use ``1``, i.e. a repulsive interaction.
    species: int | str, optional
        If given, we use only the occupation of this one species as the density :math:`n_i`.
        By default, we use the total occupation of all species.

    """
    if len(sites) != 1:
        raise ValueError(f'Invalid number of sites. Expected 1, got {len(sites)}')
    n_i = sites[0].get_occupation_numpy(species=species)
    h = 0.5 * U * n_i @ n_i
    return Coupling.from_dense_block(h, sites=sites, name=name, understood_braiding=True)


def density_density_interaction(
    sites: list[BosonicDOF] | list[FermionicDOF],
    V: float = 1,
    species_i: int | str = ALL_SPECIES,
    species_j: int | str = ALL_SPECIES,
    name: str = 'density-density',
) -> Coupling:
    r"""Density-density interaction. Two-site coupling.

    .. math ::
        h_{ij} = \mathtt{V} n_i n_j

    where :math:`n_i` is the total occupation number.

    Parameters
    ----------
    sites: list of Site
        The sites that the coupling acts on. Note that the order matters for the final leg order.
    V: float
        Prefactor, as defined above. By default, use ``1``, i.e. a repulsive interaction.
    species_i, species_j: int | str, optional
        If given, we use only the occupation of this one species as the density :math:`n_{i/j}`.
        By default, we use the total occupation of all species.
        Note that if the two species are different, this coupling alone is not hermitian!

    """
    if len(sites) != 2:
        raise ValueError(f'Invalid number of sites. Expected 2, got {len(sites)}')
    is_bosonic = [isinstance(site, BosonicDOF) for site in sites]
    if all(is_bosonic) != any(is_bosonic):
        msg = 'Bosonic and fermionic sites are incompatible and cannot be combined for constructing couplings.'
        raise SymmetryError(msg)
    n_i = sites[0].get_occupation_numpy(species=species_i)
    n_j = sites[1].get_occupation_numpy(species=species_j)
    h = V * n_i[:, None, None, :] * n_j[None, :, :, None]  # [p0, p1, p1*, p0*]
    return Coupling.from_dense_block(h, sites, name=name, understood_braiding=True)


def _quadratic_coupling_numpy(sites: list[BosonicDOF] | list[FermionicDOF], is_pairing: bool, species) -> np.ndarray:
    """Create the numpy representation for both :func:`hopping` and :func:`pairing`."""
    if len(sites) != 2:
        raise ValueError(f'Invalid number of sites. Expected 2, got {len(sites)}')
    is_bosonic = [isinstance(site, BosonicDOF) for site in sites]
    if all(is_bosonic) != any(is_bosonic):
        msg = 'Bosonic and fermionic sites are incompatible and cannot be combined for constructing couplings.'
        raise SymmetryError(msg)
    site_i, site_j = sites
    species_i, species_j = species
    if species_i is ALL_SPECIES:
        species_i = [*range(site_i.num_species)]
    if species_j is ALL_SPECIES:
        species_j = [*range(site_j.num_species)]
    if len(species_i) == 0 or len(species_j) == 0:
        return np.zeros([site_i.dim, site_j.dim, site_j.dim, site_i.dim])
    h = 0
    for k_i, k_j in zip(species_i, species_j, strict=True):
        # since we work with numpy representations here, we need to consider JW strings.
        # visually (where columns represent different species)
        # |  site i   |  site j  |       |  site i   |  site j  |
        # | J J J O   |          |   =   |  op_i     |          |
        # | J J J J J | J J J O  |       |  JW_i     |  op_j    |
        op_i = site_i.get_creator_numpy(species=k_i, include_JW=True)

        # OPTIMIZE rm check?
        sign = -1 if isinstance(site_i, FermionicDOF) else +1
        assert np.allclose(op_i @ site_i._JW, sign * site_i._JW @ op_i)

        if is_pairing:
            op_j = site_i.get_creator_numpy(species=k_j, include_JW=True)
        else:
            op_j = site_i.get_annihilator_numpy(species=k_j, include_JW=True)
        h += (op_i @ site_i._JW)[:, None, None, :] * op_j[None, :, :, None]  # [p0, p1, p1*, p0*]
    return h + np.transpose(h.conj(), [3, 2, 1, 0])


def hopping(
    sites: list[BosonicDOF] | list[FermionicDOF],
    t: float = 1,
    species: tuple[list[int | str], list[int | str]] = (ALL_SPECIES, ALL_SPECIES),
    name: str = 'hopping',
) -> Coupling:
    r"""Hopping of fermions or bosons. Two-site coupling.

    .. math ::
        h_{ij} = -\mathtt{t} \sum_{k \in \mathtt{species}} a_{i, k_i}^\dagger a_{j, k_j} + h.c.

    Parameters
    ----------
    sites: list of Site
        The sites that the coupling acts on. Note that the order matters for the final leg order.
    t : float
        Prefactor, as given above. By default ``1``.
    species : tuple of list of (int | str), optional
        Which species should participate (the sum above goes over ``k_i, k_j in zip(*species)``).
        By default, we let :math:`k_i = k_j` go over all species, i.e. include all
        "species preserving" hoppings.

    """
    h = -t * _quadratic_coupling_numpy(sites, is_pairing=False, species=species)
    return Coupling.from_dense_block(h, sites=sites, name=name, understood_braiding=True)


def pairing(
    sites: list[BosonicDOF] | list[FermionicDOF],
    Delta: float = 1.0,
    species: tuple[list[int | str], list[int | str]] = (ALL_SPECIES, ALL_SPECIES),
    name: str = 'pairing',
) -> Coupling:
    r"""Superconducting pairing of fermions or bosons. Two-site coupling.

    .. math ::
        h_{ij} = \mathtt{Delta} \sum_{k\in\mathtt{species}} a_{i, k_i}^\dagger a_{j, k_j}^\dagger + h.c.

    .. note ::
        This coupling assumes distinct sites :math:`i \neq j`.
        Use :func:`onsite_pairing` for :math:`i = j`.

    Parameters
    ----------
    sites: list of Site
        The sites that the coupling acts on. Note that the order matters for the final leg order.
    Delta : float
        Prefactor, as given above. By default ``1``.
    species : tuple of list of (int | str), optional
        Which species should participate (the sum above goes over ``k_i, k_j in zip(*species)``).
        By default, we let :math:`k_i = k_j` go over all species, i.e. include all "same-species"
        pairings.

    See Also
    --------
    onsite_pairing

    """
    h = Delta * _quadratic_coupling_numpy(sites, is_pairing=True, species=species)
    return Coupling.from_dense_block(h, sites=sites, name=name, understood_braiding=True)


def onsite_pairing(
    sites: list[BosonicDOF] | list[FermionicDOF],
    Delta: float = 1.0,
    species: tuple[list[int | str], list[int | str]] = (ALL_SPECIES, ALL_SPECIES),
    name: str = 'onsite pairing',
) -> Coupling:
    r"""Superconducting pairing of fermions or bosons. Single-site coupling.

    .. math ::
        h_i = \mathtt{Delta} \sum_{k\in\mathtt{species}} a_{i, k_1}^\dagger a_{i, k_2}^\dagger + h.c.

    Parameters
    ----------
    sites: list of Site
        The sites that the coupling acts on. Note that the order matters for the final leg order.
    Delta : float
        Prefactor, as given above. By default ``1``.
    species : tuple of list of (int | str), optional
        Which species should participate (the sum above goes over ``k_1, k_2 in zip(*species)``).
        By default, we let :math:`k_1 = k_2` go over all species, i.e. include all "same-species"
        pairings.

    See Also
    --------
    pairing

    """
    if len(sites) != 1:
        raise ValueError(f'Invalid number of sites. Expected 1, got {len(sites)}')
    (site,) = sites
    species_1, species_2 = species
    if species_1 is ALL_SPECIES:
        species_1 = [*range(site.num_species)]
    if species_2 is ALL_SPECIES:
        species_2 = [*range(site.num_species)]
    h = 0
    for k_1, k_2 in zip(species_1, species_2, strict=True):
        a_i_hc = site.get_creator_numpy(species=k_1, include_JW=True)
        a_j_hc = site.get_creator_numpy(species=k_2, include_JW=True)
        h += Delta * a_i_hc @ a_j_hc
    h += np.transpose(h.conj())
    return Coupling.from_dense_block(h, sites=sites, name=name, understood_braiding=True)


# CLOCK COUPLINGS


def clock_clock_coupling(sites: list[ClockDOF], Jx: float = 0, Jz: float = 0, name: str = 'clock-clock') -> Coupling:
    r"""Two-site coupling between quantum clocks.

    .. math ::
        h_{ij} = \mathtt{Jx} X_i X_j^\dagger + \mathtt{Jz} Z_i Z_j^\dagger + h.c.

    Parameters
    ----------
    sites: list of Site
        The sites that the coupling acts on. Note that the order matters for the final leg order.
    Jx, Jz: float
        Prefactor, as given above. By default, all prefactors vanish.

    """
    if len(sites) != 2:
        raise ValueError(f'Invalid number of sites. Expected 2, got {len(sites)}')
    X_i = sites[0].clock_operators[:, :, 0]
    Z_i = sites[0].clock_operators[:, :, 1]
    X_j = sites[1].clock_operators[:, :, 0]
    Z_j = sites[1].clock_operators[:, :, 1]
    h = Jx * X_i[:, None, None, :] * X_j.T.conj()[None, :, :, None]  # [p0, p1, p1*, p0*]
    h += Jz * Z_i[:, None, None, :] * Z_j.T.conj()[None, :, :, None]
    h = h + np.transpose(h.conj(), [3, 2, 1, 0])
    return Coupling.from_dense_block(h, sites, name=name)


def clock_field_coupling(
    sites: list[ClockDOF], hx: float = None, hz: float = None, name: str = 'clock-field'
) -> Coupling:
    r"""Single-site coupling of a quantum clock to an external field.

    .. math ::
        h_i = \mathtt{hx} X_i + \mathtt{hz} Z_i + h.c.

    Parameters
    ----------
    sites: list of Site
        The sites that the coupling acts on. Note that the order matters for the final leg order.
    hx, hz: float
        Prefactor, as given above. By default, all prefactors vanish.

    """
    if len(sites) != 1:
        raise ValueError(f'Invalid number of sites. Expected 1, got {len(sites)}')
    X = sites[0].clock_operators[:, :, 0]
    Z = sites[0].clock_operators[:, :, 1]
    h = hx * (X + X.T.conj()) + hz * (Z + Z.T.conj())
    return Coupling.from_dense_block(h, sites, name=name)


# ANYONIC COUPLINGS


def sector_projection_coupling(sites: list[Site], J: float, sector: Sector, name: str) -> Coupling:
    """Coupling that is given by the projector onto a single sector

    The number of sites is arbitrary and the operator :math:`h_{ij...}` is given
    by :meth:`cyten.SymmetricTensor.from_sector_projection`, with prefactor `J`.
    Note that positive `J` mean that states that fuse to the given `sector` are energetically
    *disfavored*.
    """
    backend = get_same_backend(*sites)
    device = sites[0].default_device
    assert all(s.default_device == device for s in sites[1:])
    labels = [f'p{i}' for i in range(len(sites))]
    labels = [*labels, *[f'{l}*' for l in reversed(labels)]]
    projector = SymmetricTensor.from_sector_projection(
        [s.leg for s in sites], sector=sector, backend=backend, labels=labels, device=device
    )
    return Coupling.from_tensor(J * projector, sites=sites, name=name)


def gold_coupling(sites: list[GoldenSite], J: float = 1, name: str = 'gold') -> Coupling:
    r"""Two-site coupling of Fibonacci anyons that energy splits fusion to vacuum or tau.

    .. math ::
        h_{ij} = -J P^\text{vac}_{i, j}

    Parameters
    ----------
    sites: list of Site
        The sites that the coupling acts on. Note that the order matters for the final leg order.
    J: float
        Prefactor, as given above. By default ``1``. Positive `J` energetically favor the
        trivial fusion channel, i.e. they are the "antiferromagnetic" analog.

    """
    if len(sites) != 2:
        raise ValueError(f'Invalid number of sites. Expected 2, got {len(sites)}')
    for site in sites:
        assert site.symmetry.is_equivalent_to(fibonacci_anyon_category)
        assert site.leg.sector_decomposition_where(FibonacciAnyonCategory.tau) is not None
    return sector_projection_coupling(sites, J=-J, sector=FibonacciAnyonCategory.vacuum, name=name)
