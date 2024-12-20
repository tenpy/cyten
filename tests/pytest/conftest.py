"""Provide test configuration for backends etc.

Fixtures
--------

The following table summarizes the available fixtures.
There are three groups; miscellaneous independent fixtures, unconstrained fixtures
(with ``any`` in their name) and constrained fixtures (with ``compatible`` in their name).
The latter two groups are similar in terms of signature.
The unconstrained fixtures are intended if a test should be parametrized over possible
symmetry backends *or* over possible symmetries, but not both.
The constrained ("compatible") fixtures are intended if a test should be parametrized over
possible combinations of a symmetry backend *and* a symmetry it is compatible with.
They should not be mixed in any single test, as that would generate unnecessarily many tests.
Whenever applicable, the unconstrained fixtures should be preferred, since e.g. most symmetries
appear multiple times as values of ``compatible_symmetry`` (same argument for ``compatible_backend``).


=============================  ======================  ===========================================
Fixture                        Depends on / # cases    Description
=============================  ======================  ===========================================
np_random                      -                       A numpy random Generator. Use this for
                                                       reproducibility.
-----------------------------  ----------------------  -------------------------------------------
block_backend                  Generates ~2 cases      Goes over all block backends, as str
                                                       descriptions, valid for ``get_backend``.
=============================  ======================  ===========================================
any_symmetry_backend           Generates 3 cases       Goes over all symmetry backends, as str
                                                       descriptions, valid for ``get_backend``.
-----------------------------  ----------------------  -------------------------------------------
any_backend                    block_backend           Goes over all backends.
                               any_symmetry_backend
-----------------------------  ----------------------  -------------------------------------------
any_symmetry                   Generates ~10 cases     Goes over some representative symmetries.
-----------------------------  ----------------------  -------------------------------------------
make_any_sectors               any_symmetry            RNG for sectors with ``any_symmetry``.
                                                       Note that fewer than ``num`` may result.
                                                       ``make(num, sort=False)``
-----------------------------  ----------------------  -------------------------------------------
make_any_space                 any_symmetry            RNG for spaces with ``any_symmetry``.
                                                       ``make(max_sectors=5, max_mult=5, is_dual=None)``
-----------------------------  ----------------------  -------------------------------------------
make_any_block                 any_backend             RNG for blocks of ``any_backend``.
                                                       ``make(size, real=False)``
=============================  ======================  ===========================================
compatible_pairs               Generates ~20 cases     Not a public fixture, only generates
                                                       the cases. Compatible pairs are built like
                                                       combinations of ``any_symmetry_backend``
                                                       and ``any_symmetry``, constrained by
                                                       compatibility.
-----------------------------  ----------------------  -------------------------------------------
compatible_symmetry_backend    compatible_pairs        The symmetry backend of a compatible pair.
-----------------------------  ----------------------  -------------------------------------------
compatible_symmetry            compatible_pairs        The symmetry of a compatible pair.
-----------------------------  ----------------------  -------------------------------------------
compatible_backend             compatible_pairs        A backend that is compatible with
                               block_backend           ``compatible_symmetry``.
-----------------------------  ----------------------  -------------------------------------------
make_compatible_sectors        compatible_pairs        RNG for sectors with ``compatible_symmetry``.
                                                       Note that fewer than ``num`` may result.
                                                       ``make(num, sort=False)``
-----------------------------  ----------------------  -------------------------------------------
make_compatible_space          compatible_pairs        RNG for spaces with ``compatible_symmetry``.
                                                       ``make(max_sectors=5, max_mult=5, is_dual=None)``
-----------------------------  ----------------------  -------------------------------------------
make_compatible_block          compatible_backend      RNG for blocks with ``compatible_backend``.
                                                       ``make(size, real=False)``
-----------------------------  ----------------------  -------------------------------------------
make_compatible_tensor         compatible_backend      RNG for tensors with ``compatible_backend``.
                                                       Signature see below.
=============================  ======================  ===========================================

The function returned by the fixture ``make_compatible_tensor`` has the following inputs::

    codomain, domain:
        Both the domain and the codomain can be specified in the following four ways:
        1) ``None``.
            For the codomain, this means a random space if cls is DiagonalTensor or Mask.
            For SymmetricTensor or ChargedTensor, codomain may not be None
            For the domain, this means "the same as codomain" for DiagonalTensor, a random space
            that contains the codomain for Mask and an empty domain for Symmetric/Charged Tensor.
        2) an integer
            The respective number of legs is randomly generated.
        3) a list
            Each entry specifies a leg. It can already be a space. A str specifies the label
            for that leg and is otherwise equivalent to None. None means to generate a random leg
        4) a ProductSpace
            The finished (co)domain
        For Symmetric/Charged Tensor, if any legs are generated, this is done in such a way
        as to guarantee that the resulting tensor allows some blocks.
    labels: list[str | None] (default: all None)
        The labels for the resulting tensor. Note that labels can also be specified via (co)domain.
    dtype: Dtype
        The dtype for the tensor.
    device: str, optional
        The device for the tensor. Per default, use the default device of the backend.
    *
    like: Tensor, optional
        If given, the codomain, domain, labels, dtype and cls are taken to be the
        same as for `like` and th explicit arguments are ignored.
        For ChargedTensors, the same charge leg and state are initialized.
    max_blocks: int (default 5)
        The maximum number of blocks for the resulting tensor
    max_block_size: int (default 5)
        The maximum multiplicity of any sector.
    empty_ok: bool (default False)
        If an empty tensor (with no allowed blocks) is ok, or should raise.
    all_blocks: bool (default False)
        If all allowed blocks should be filled, or if some should be dropped randomly
    cls: Tensor subtype
        The type of tensor to create: SymmetricTensor, DiagonalTensor, Mask or ChargedTensor


Marks
-----

Deselecting invalid ChargedTensor cases
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
There is a custom mark ``deselect_invalid_ChargedTensor_cases``.
It can be used as a decorator ``@pytest.mark.deselect_invalid_ChargedTensor_cases`` for test cases.
The intended use case is in a situation where a test is parametrized over multiple symmetries and
over multiple tensor types.
Some symmetries will then be incompatible with the ``ChargedTensor`` type.
Those cases should be deselected.

The decorator takes two optional keyword arguments.
``@pytest.mark.deselect_invalid_ChargedTensor_cases(get_cls: callable, get_sym: callable)``.
Both are functions, and during setup of pytest they are called as e.g. ``get_cls(kwargs)``
where kwargs are the explicit keyword arguments (e.g. parametrize keyword) of the test function.
They should return the tensor type, e.g. ``ChargedTensor`` and the symmetry instance respectively.
The default values are ``get_cls = lambda kw: kw['cls']``
and ``get_sym = lambda kw: kw['_compatible_backend_symm_pairs'][1]``.
As such, the default decorator with no arguments works in the most common design pattern for tests,
where the symmetry is determined by the fixtures (e.g. because ``make_compatible_tensor`` is used)
and the tensor cls comes from a parametrize with argname ``cls``.

"""
# Copyright (C) TeNPy Developers, Apache license
from __future__ import annotations
import numpy as np
import pytest

from cyten import backends, spaces, symmetries, tensors, Dtype


# OVERRIDE pytest routines
def pytest_collection_modifyitems(config, items):

    # deselection logic:
    removed = []
    kept = []
    for item in items:
        m = item.get_closest_marker('deselect_invalid_ChargedTensor_cases')
        if m:
            get_cls = m.kwargs.get('get_cls', lambda kw: kw['cls'])
            get_sym = m.kwargs.get('get_sym', lambda kw: kw['_compatible_backend_symm_pairs'][1])
            cls = get_cls(item.callspec.params)
            sym = get_sym(item.callspec.params)
            if cls is tensors.ChargedTensor and not tensors.ChargedTensor.supports_symmetry(sym):
                removed.append(item)
                continue
        kept.append(item)
    if removed:
        config.hook.pytest_deselected(items=removed)
        items[:] = kept


# QUICK CONFIGURATION

_block_backends = ['numpy', 'torch']
_symmetries = {
    # groups:
    'NoSymm': symmetries.no_symmetry,
    'U1': symmetries.u1_symmetry,
    'Z4_named': symmetries.ZNSymmetry(4, "My_Z4_symmetry"),
    'U1xZ3': symmetries.ProductSymmetry([symmetries.u1_symmetry, symmetries.z3_symmetry]),
    'SU2': symmetries.SU2Symmetry(),
    # anyons:
    'fermion': symmetries.fermion_parity,
    'FibonacciAnyon': symmetries.fibonacci_anyon_category,
    'IsingAnyon': symmetries.ising_anyon_category,
    'Fib_U1': symmetries.fibonacci_anyon_category * symmetries.u1_symmetry,
}


# "UNCONSTRAINED" FIXTURES  ->  independent (mostly) of the other features. no compatibility guarantees.

@pytest.fixture
def np_random() -> np.random.Generator:
    return np.random.default_rng(seed=12345)


@pytest.fixture(params=_block_backends)
def block_backend(request) -> str:
    if request.param == 'torch':
        torch = pytest.importorskip('torch', reason='torch not installed')
    return request.param


@pytest.fixture(params=['no_symmetry', 'abelian', pytest.param('fusion_tree', marks=pytest.mark.FusionTree)])
def any_symmetry_backend(request) -> str:
    return request.param


@pytest.fixture
def any_backend(block_backend, any_symmetry_backend) -> backends.TensorBackend:
    return backends.backend_factory.get_backend(any_symmetry_backend, block_backend)


@pytest.fixture(params=list(_symmetries.values()), ids=list(_symmetries.keys()))
def any_symmetry(request) -> symmetries.Symmetry:
    return request.param


@pytest.fixture
def make_any_sectors(any_symmetry, np_random):
    # if the symmetry does not have enough sectors, we return fewer!
    def make(num: int, sort: bool = False) -> symmetries.SectorArray:
        # return SectorArray
        return random_symmetry_sectors(any_symmetry, num, sort, np_random=np_random)
    return make


@pytest.fixture
def make_any_space(any_symmetry, np_random):
    def make(max_sectors: int = 5, max_mult: int = 5, is_dual: bool = None) -> spaces.ElementarySpace:
        # return ElementarySpace
        return random_vector_space(any_symmetry, max_sectors, max_mult, is_dual, np_random=np_random)
    return make


@pytest.fixture
def make_any_block(any_backend, np_random):
    def make(size: tuple[int, ...], real=False) -> backends.Block:
        # return Block
        return random_block(any_backend.block_backend, size, real=real, np_random=np_random)
    return make


# "COMPATIBLE" FIXTURES  ->  only go over those pairings of backend and symmetry that are compatible

# build the compatible pairs
_compatible_pairs = {'NoSymmetry': ('no_symmetry', symmetries.no_symmetry)}  # {id: param}
for _sym_name, _sym in _symmetries.items():
    if isinstance(_sym, symmetries.AbelianGroup):
        _compatible_pairs[f'AbelianBackend-{_sym_name}'] = ('abelian', _sym)
    _compatible_pairs[f'FusionTreeBackend-{_sym_name}'] = pytest.param(
        ('fusion_tree', _sym), marks=pytest.mark.FusionTree
    )


@pytest.fixture(params=list(_compatible_pairs.values()), ids=list(_compatible_pairs.keys()))
def _compatible_backend_symm_pairs(request) -> tuple[str, symmetries.Symmetry]:
    """Helper fixture that allows us to generate the *compatible* fixtures.

    Values are pairs (symmetry_backend: str, symmetry: Symmetry)
    """
    return request.param


@pytest.fixture
def compatible_symmetry_backend(_compatible_backend_symm_pairs) -> str:
    symmetry_backend, symmetry = _compatible_backend_symm_pairs
    return symmetry_backend


@pytest.fixture
def compatible_backend(compatible_symmetry_backend, block_backend) -> backends.TensorBackend:
    return backends.backend_factory.get_backend(compatible_symmetry_backend, block_backend)


@pytest.fixture
def compatible_symmetry(_compatible_backend_symm_pairs) -> symmetries.Symmetry:
    symmetry_backend, symmetry = _compatible_backend_symm_pairs
    return symmetry


@pytest.fixture
def make_compatible_sectors(compatible_symmetry, np_random):
    # if the symmetry does not have enough sectors, we return fewer!
    def make(num: int, sort: bool = False) -> symmetries.SectorArray:
        # returns SectorArray
        return random_symmetry_sectors(compatible_symmetry, num, sort, np_random=np_random)
    return make


@pytest.fixture
def make_compatible_space(compatible_symmetry, np_random):
    def make(max_sectors: int = 5, max_mult: int = 5, is_dual: bool = None,
             allow_basis_perm: bool = True) -> spaces.ElementarySpace:
        # returns ElementarySpace
        return random_vector_space(compatible_symmetry, max_sectors, max_mult, is_dual,
                                   allow_basis_perm=allow_basis_perm, np_random=np_random)
    return make


@pytest.fixture
def make_compatible_block(compatible_backend, np_random):
    def make(size: tuple[int, ...], real: bool = False) -> backends.Block:
        # returns Block
        return random_block(compatible_backend.block_backend, size, real=real, np_random=np_random)
    return make


@pytest.fixture
def make_compatible_tensor(compatible_backend, compatible_symmetry, np_random):
    """Tensor RNG."""
    def make(codomain: list[spaces.Space | str | None] | spaces.ProductSpace | int = None,
             domain: list[spaces.Space | str | None] | spaces.ProductSpace | int = None,
             labels: list[str | None] = None, dtype: Dtype = None, device: str = None,
             *,
             like: tensors.Tensor = None, max_blocks=5, max_block_size=5, empty_ok=False,
             all_blocks=False, cls=tensors.SymmetricTensor, allow_basis_perm: bool = True):
        return random_tensor(
            symmetry=compatible_symmetry, codomain=codomain, domain=domain, labels=labels,
            dtype=dtype, backend=compatible_backend, device=device, like=like, max_blocks=max_blocks,
            max_block_size=max_block_size, empty_ok=empty_ok, all_blocks=all_blocks, cls=cls,
            allow_basis_perm=allow_basis_perm, np_random=np_random
        )
    return make


# RANDOM GENERATION

def random_block(block_backend, size, real=False, np_random=np.random.default_rng(0)):
    block = np_random.normal(size=size)
    if not real:
        block = block + 1.j * np_random.normal(size=size)
    return block_backend.block_from_numpy(block)


def random_symmetry_sectors(symmetry: symmetries.Symmetry, num: int, sort: bool = False,
                            np_random=np.random.default_rng()) -> symmetries.SectorArray:
    """random unique symmetry sectors, optionally sorted"""
    if isinstance(symmetry, symmetries.SU2Symmetry):
        res = np_random.choice(int(1.3 * num), replace=False, size=(num, 1))
    elif isinstance(symmetry, symmetries.U1Symmetry):
        vals = list(range(-num, num)) + [123]
        res = np_random.choice(vals, replace=False, size=(num, 1))
    elif symmetry.num_sectors < np.inf:
        if symmetry.num_sectors <= num:
            res = np_random.permutation(symmetry.all_sectors())
        else:
            which = np_random.choice(symmetry.num_sectors, replace=False, size=num)
            res = symmetry.all_sectors()[which, :]
    elif isinstance(symmetry, symmetries.ProductSymmetry):
        factor_len = max(3, num // len(symmetry.factors))
        factor_sectors = [random_symmetry_sectors(factor, factor_len, np_random=np_random)
                          for factor in symmetry.factors]
        combs = np.indices([len(s) for s in factor_sectors]).T.reshape((-1, len(factor_sectors)))
        if len(combs) > num:
            combs = np_random.choice(combs, replace=False, size=num)
        res = np.hstack([fs[i] for fs, i in zip(factor_sectors, combs.T)])
    else:
        pytest.skip("don't know how to get symmetry sectors")  # raises Skipped
    if sort:
        order = np.lexsort(res.T)
        res = res[order]
    return res


def random_vector_space(symmetry, max_num_blocks=5, max_block_size=5, is_dual=None,
                        allow_basis_perm=True, np_random=None):
    if np_random is None:
        np_random = np.random.default_rng()
    num_sectors = np_random.integers(1, max_num_blocks, endpoint=True)
    sectors = random_symmetry_sectors(symmetry, num_sectors, sort=True, np_random=np_random)
    # if there are very few sectors, e.g. for symmetry==NoSymmetry(), dont let them be one-dimensional
    min_mult = min(max_block_size, max(4 - len(sectors), 1))
    mults = np_random.integers(min_mult, max_block_size, size=(len(sectors),), endpoint=True)
    if symmetry.can_be_dropped and allow_basis_perm:
        dim = np.sum(symmetry.batch_sector_dim(sectors) * mults)
        basis_perm = np_random.permutation(dim) if np_random.random() < 0.7 else None
    else:
        basis_perm = None
    if is_dual is None:
        is_dual = np_random.random() < 0.5
    res = spaces.ElementarySpace(
        symmetry, sectors, mults, basis_perm=basis_perm, is_dual=is_dual
    )
    res.test_sanity()
    return res


def randomly_drop_blocks(res: tensors.SymmetricTensor | tensors.DiagonalTensor,
                         max_blocks: int | None, empty_ok: bool, np_random=np.random.default_rng()):

    if isinstance(res.backend, backends.NoSymmetryBackend):
        # nothing to do
        return res
    if not isinstance(res.backend, (backends.AbelianBackend, backends.FusionTreeBackend)):
        raise NotImplementedError

    num_blocks = len(res.data.blocks)
    min_blocks = 0 if empty_ok else 1
    if max_blocks is None:
        max_blocks = num_blocks
    else:
        max_blocks = min(num_blocks, max_blocks)
    if max_blocks < min_blocks:
        return res

    if np_random.uniform() < 0.5:
        # with 50% chance, keep maximum number
        num_keep = max_blocks
    else:
        num_keep = np_random.integers(min_blocks, max_blocks, endpoint=True)
    if num_keep == num_blocks:
        return res
    which = np_random.choice(num_blocks, size=num_keep, replace=False, shuffle=False)
    which = np.sort(which)

    if isinstance(res.backend, backends.AbelianBackend):
        res.data = backends.AbelianBackendData(
            dtype=res.dtype,
            blocks=[res.data.blocks[n] for n in which],
            block_inds=res.data.block_inds[which],
            is_sorted=True,
            device=res.data.device
        )
        return res

    if isinstance(res.backend, backends.FusionTreeBackend):
        res.data = backends.FusionTreeData(
            block_inds=res.data.block_inds[which, :],
            blocks=[res.data.blocks[n] for n in which],
            dtype=res.data.dtype,
            device=res.data.device
        )
        return res

    raise ValueError('Backend not recognized')


def find_last_leg(same: spaces.ProductSpace, opposite: spaces.ProductSpace,
                  max_sectors: int, max_mult: int,
                  extra_sectors=None, np_random=np.random.default_rng()):
    """Find a leg such that the resulting tensor allows some non-zero blocks

    Parameters
    ----------
    same, opposite
        The domain and codomain of the resulting tensor, up the missing leg.
        Same is the one of the two that the resulting leg should be added to.
    max_sectors, max_mult
        Upper bounds for the number of sectors and the multiplicities, resp.
    extra_sectors
        If given, extra sectors to mix in
    """
    assert same.num_sectors > 0
    assert opposite.num_sectors > 0
    prod = spaces.ProductSpace.from_partial_products(same.dual, opposite)
    sectors = prod.sectors
    mults = prod.multiplicities
    if len(sectors) > max_sectors:
        which = np_random.choice(len(sectors), size=max_sectors, replace=False, shuffle=False)
        sectors = sectors[which, :]
        mults = mults[which]
    mults = np.minimum(mults, max_mult)
    if extra_sectors is not None:
        # replace some sectors by extra_sectors
        duplicates = np.any(np.all(extra_sectors[None, :, :] == sectors[:, None, :], axis=2), axis=0)
        extra_sectors = extra_sectors[np.logical_not(duplicates)]
        # replace some sectors
        min_replace = max(1, int(.2 * len(sectors)))
        max_replace = min(int(.5 * len(sectors)), len(extra_sectors))
        if max_replace >= min_replace:
            num_replace = np_random.integers(min_replace, max_replace, endpoint=True)
            which = np_random.choice(len(sectors), size=num_replace, replace=False)
            sectors[which, :] = extra_sectors[:num_replace, :]
    # guarantee sorting
    order = np.lexsort(sectors.T)
    sectors = sectors[order]
    mults = mults[order]
    #
    res = spaces.ElementarySpace(prod.symmetry, sectors=sectors, multiplicities=mults)
    #
    # check that it actually worked
    # OPTIMIZE remove?
    parent_space = spaces.ProductSpace.from_partial_products(same.left_multiply(res), opposite.dual)
    assert parent_space.sector_multiplicity(same.symmetry.trivial_sector) > 0
    res.test_sanity()

    return res


def random_tensor(symmetry: symmetries.Symmetry,
                  codomain: list[spaces.Space | str | None] | spaces.ProductSpace | int = None,
                  domain: list[spaces.Space | str | None] | spaces.ProductSpace | int = None,
                  labels: list[str | None] = None, dtype: Dtype = None,
                  backend: backends.TensorBackend = None, device: str = None,
                  like: tensors.Tensor = None, max_blocks=5, max_block_size=5, empty_ok=False,
                  all_blocks=False, cls=tensors.SymmetricTensor, allow_basis_perm: bool = True,
                  np_random=np.random.default_rng()):
    if backend is None:
        backend = backends.get_backend()
    assert isinstance(backend, backends.TensorBackend)
    
    if like is not None:
        assert like.backend is backend
        assert like.symmetry is symmetry
        if isinstance(like, tensors.ChargedTensor):
            inv_part = random_tensor(symmetry=symmetry, backend=backend, like=like.invariant_part)
            return tensors.ChargedTensor(inv_part, like.charged_state)
        elif isinstance(like, tensors.Tensor):
            return random_tensor(symmetry=symmetry, codomain=like.codomain, domain=like.domain,
                                 labels=like.labels, backend=backend, dtype=like.dtype,
                                 device=like.device, max_blocks=max_blocks,
                                 max_block_size=max_block_size, cls=type(like), np_random=np_random)
        else:
            raise TypeError(f'like must be a Tensor. Got {type(like)}')
    
    if isinstance(codomain, list):
        codomain = codomain[:]  # we do inplace operations below.
    if isinstance(domain, list):
        domain = domain[:]  # we do inplace operations below.

    # 0) default for codomain
    # ======================================================================================
    if codomain is None:
        if cls in [tensors.SymmetricTensor, tensors. ChargedTensor]:
            codomain = 2
            if domain is None:
                domain = 2
        elif cls in [tensors.DiagonalTensor, tensors.Mask]:
            codomain = [None]
        else:
            raise ValueError

    # 1) deal with strings in codomain / domain.
    # ======================================================================================
    if isinstance(codomain, spaces.ProductSpace):
        assert codomain.symmetry == symmetry
        num_codomain = codomain.num_spaces
        codomain_complete = True
        codomain_labels = [None] * len(codomain)
    else:
        if isinstance(codomain, int):
            codomain = [None] * codomain
        num_codomain = len(codomain)
        codomain_labels = [None] * len(codomain)
        for n, sp in enumerate(codomain):
            if isinstance(sp, str):
                codomain_labels[n] = sp
                codomain[n] = None
        codomain_complete = (None not in codomain)
    #
    if domain is None:
        if cls in [tensors.SymmetricTensor, tensors.ChargedTensor]:
            domain = []
        if cls in [tensors.DiagonalTensor, tensors.Mask]:
            domain = [None]
    if isinstance(domain, spaces.ProductSpace):
        assert domain.symmetry == symmetry
        num_domain = domain.num_spaces
        domain_labels = [None] * len(domain)
        domain_complete = True
    else:
        if isinstance(domain, int):
            domain = [None] * domain
        num_domain = len(domain)
        domain_labels = [None] * len(domain)
        for n, sp in enumerate(domain):
            if isinstance(sp, str):
                domain_labels[n] = sp
                domain[n] = None
        domain_complete = (None not in domain)
    #
    num_legs = num_codomain + num_domain
    if labels is None:
        labels = [None] * num_legs
    for n, l in enumerate(codomain_labels):
        if l is None:
            continue
        assert labels[n] is None
        labels[n] = l
    for n, l in enumerate(domain_labels):
        if l is None:
            continue
        assert labels[-1-n] is None
        labels[-1-n] = l
    #
    # 2) Deal with other tensor types
    # ======================================================================================
    if cls is tensors.ChargedTensor:
        charge_leg = random_vector_space(symmetry=symmetry, max_num_blocks=1, max_block_size=1,
                                         is_dual=False, allow_basis_perm=allow_basis_perm,
                                         np_random=np_random)
        if isinstance(domain, spaces.ProductSpace):
            inv_domain = domain.left_multiply(charge_leg, backend=backend)
        else:
            inv_domain = [charge_leg, *domain]
        inv_labels = [*labels, tensors.ChargedTensor._CHARGE_LEG_LABEL]
        inv_part = random_tensor(
            symmetry=symmetry, codomain=codomain, domain=inv_domain, labels=inv_labels, dtype=dtype,
            backend=backend, device=device, max_blocks=max_blocks, max_block_size=max_block_size,
            empty_ok=empty_ok, all_blocks=all_blocks, cls=tensors.SymmetricTensor,
            allow_basis_perm=allow_basis_perm, np_random=np_random
        )

        charged_state = [1] if inv_part.symmetry.can_be_dropped else None
        res = tensors.ChargedTensor(inv_part, charged_state=charged_state)
        res.test_sanity()
        return res
    #
    if cls is tensors.DiagonalTensor:
        # fill in legs.
        if isinstance(codomain, spaces.ProductSpace):
            assert codomain.num_spaces == 1
            leg = codomain.spaces[0]
            if isinstance(domain, spaces.ProductSpace):
                assert domain == codomain
            else:
                assert len(domain) == 1
                assert domain[0] is None or domain[0] == leg
        else:
            assert len(codomain) == 1
            if isinstance(domain, spaces.ProductSpace):
                assert domain.num_spaces == 1
                leg = domain.spaces[0]
                assert codomain[0] is None or codomain[0] == leg
            else:
                assert len(domain) == 1
                if domain[0] is None and codomain[0] is None:
                    leg = random_vector_space(symmetry=symmetry, max_num_blocks=max_blocks,
                                              max_block_size=max_block_size,
                                              allow_basis_perm=allow_basis_perm, np_random=np_random)
                elif domain[0] is None:
                    leg = codomain[0]
                elif codomain[0] is None:
                    leg = domain[0]
                else:
                    leg = codomain[0]
                    assert domain[0] == leg
        #
        real = False if dtype is None else dtype.is_real
        res = tensors.DiagonalTensor.from_block_func(
            lambda size: random_block(block_backend=backend.block_backend, size=size, real=real, np_random=np_random),
            leg=leg, backend=backend, labels=labels, dtype=dtype, device=device
        )
        if not all_blocks:
            res = randomly_drop_blocks(res, max_blocks=max_blocks, empty_ok=empty_ok,
                                        np_random=np_random)
        res.test_sanity()
        return res
    #
    if cls is tensors.Mask:
        assert dtype in [None, Dtype.bool]
        if isinstance(codomain, spaces.ProductSpace):
            assert codomain.num_spaces == 1
            small_leg = codomain.spaces[0]
        elif codomain is None:
            small_leg is None
        else:
            assert len(codomain) == 1
            small_leg = codomain[0]
        if isinstance(domain, spaces.ProductSpace):
            assert domain.num_spaces == 1
            large_leg = domain.spaces[0]
        elif domain is None:
            large_leg = None
        else:
            assert len(domain) == 1
            large_leg = domain[0]
        #
        if large_leg is None:
            if small_leg is None:
                large_leg = random_vector_space(
                    symmetry=symmetry, max_num_blocks=max_blocks, max_block_size=max_block_size,
                    allow_basis_perm=allow_basis_perm, np_random=np_random
                )
            else:
                # TODO looks like this generates a basis_perm incompatible with the mask!
                raise NotImplementedError('Mask generation broken')
                extra = random_vector_space(symmetry=symmetry, max_num_blocks=max_blocks,
                                            max_block_size=max_block_size, is_dual=small_leg.is_dual,
                                            allow_basis_perm=allow_basis_perm, np_random=np_random)
                large_leg = small_leg.direct_sum(extra)

        if isinstance(backend, backends.FusionTreeBackend):
            with pytest.raises(NotImplementedError, match='diagonal_to_mask'):
                _ = tensors.Mask.from_random(large_leg=large_leg, small_leg=small_leg,
                                             backend=backend, p_keep=.6,
                                             labels=labels, np_random=np_random)
            pytest.xfail()

        if small_leg is not None and small_leg.dim > large_leg.dim:
            res = tensors.Mask.from_random(large_leg=small_leg, small_leg=large_leg,
                                           backend=backend, p_keep=.6, min_keep=1,
                                           labels=labels, device=device, np_random=np_random)
            res = tensors.dagger(res)
        else:
            res = tensors.Mask.from_random(large_leg=large_leg, small_leg=small_leg,
                                           backend=backend, p_keep=.6, min_keep=1,
                                           labels=labels, device=device, np_random=np_random)
        assert res.small_leg.num_sectors > 0
        res.test_sanity()
        return res
    #
    # 3) Fill in missing legs
    # ======================================================================================
    if (not codomain_complete) and (not domain_complete):
        # can just fill up the codomain with random legs.
        for n, sp in enumerate(codomain):
            if sp is None:
                codomain[n] = random_vector_space(symmetry=symmetry, max_num_blocks=max_blocks,
                                                  max_block_size=max_block_size,
                                                  allow_basis_perm=allow_basis_perm)
        codomain = spaces.ProductSpace(codomain, symmetry=symmetry, backend=backend)
        codomain_complete = True
    if not codomain_complete:
        # can assume that domain is complete
        if not isinstance(domain, spaces.ProductSpace):
            domain = spaces.ProductSpace(domain, symmetry=symmetry, backend=backend)
        missing = [n for n, sp in enumerate(codomain) if sp is None]
        for n in missing[:-1]:
            codomain[n] = random_vector_space(symmetry=symmetry, max_num_blocks=max_blocks,
                                              max_block_size=max_block_size,
                                              allow_basis_perm=allow_basis_perm)
        last = missing[-1]
        partial_codomain = spaces.ProductSpace(codomain[:last] + codomain[last + 1:],
                                                symmetry=symmetry,
                                                backend=backend)
        leg = find_last_leg(same=partial_codomain, opposite=domain, max_sectors=max_blocks,
                            max_mult=max_block_size)
        codomain = partial_codomain.insert_multiply(leg, last, backend=backend)
    elif not domain_complete:
        # can assume codomain is complete
        if not isinstance(codomain, spaces.ProductSpace):
            codomain = spaces.ProductSpace(codomain, symmetry=symmetry, backend=backend)
        missing = [n for n, sp in enumerate(domain) if sp is None]
        for n in missing[:-1]:
            domain[n] = random_vector_space(symmetry=symmetry, max_num_blocks=max_blocks,
                                            max_block_size=max_block_size,
                                            allow_basis_perm=allow_basis_perm)
        last = missing[-1]
        partial_domain = spaces.ProductSpace(domain[:last] + domain[last + 1:],
                                                symmetry=symmetry,
                                                backend=backend)
        leg = find_last_leg(same=partial_domain, opposite=codomain, max_sectors=max_blocks,
                            max_mult=max_block_size)
        domain = partial_domain.insert_multiply(leg, last, backend=backend)
    else:
        if not isinstance(codomain, spaces.ProductSpace):
            codomain = spaces.ProductSpace(codomain, symmetry=symmetry, backend=backend)
        if not isinstance(domain, spaces.ProductSpace):
            domain = spaces.ProductSpace(domain, symmetry=symmetry, backend=backend)
    #
    # 3) Finish up
    # ======================================================================================
    if cls is not tensors.SymmetricTensor:
        raise ValueError(f'Unknown tensor cls: {cls}')

    real = False if dtype is None else dtype.is_real
    res = tensors.SymmetricTensor.from_block_func(
        lambda size: random_block(block_backend=backend.block_backend, size=size, real=real, np_random=np_random),
        codomain=codomain, domain=domain, backend=backend, labels=labels,
        dtype=dtype, device=device
    )
    if not all_blocks:
        res = randomly_drop_blocks(res, max_blocks=max_blocks, empty_ok=empty_ok,
                                    np_random=np_random)
    res.test_sanity()
    return res
