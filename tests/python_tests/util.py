"""Utility functions for testing"""
# Copyright (C) TeNPy Developers, Apache license
from __future__ import annotations

import numpy as np
import pytest

from cyten import symmetries, spaces, tensors, backends, Dtype


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


def random_ElementarySpace(symmetry, max_sectors=5, max_multiplicity=5, is_dual=None,
                           allow_basis_perm=True, np_random=None):
    if np_random is None:
        np_random = np.random.default_rng()
    num_sectors = np_random.integers(1, max_sectors, endpoint=True)
    sectors = random_symmetry_sectors(symmetry, num_sectors, sort=True, np_random=np_random)
    # if there are very few sectors, e.g. for symmetry==NoSymmetry(), dont let them be one-dimensional
    min_mult = min(max_multiplicity, max(4 - len(sectors), 1))
    mults = np_random.integers(min_mult, max_multiplicity, size=(len(sectors),), endpoint=True)
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


def find_last_leg(same: spaces.TensorProduct, opposite: spaces.TensorProduct,
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
    prod = spaces.TensorProduct.from_partial_products(same.dual, opposite)
    sectors = prod.sector_decomposition
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
    res = spaces.ElementarySpace(prod.symmetry, defining_sectors=sectors, multiplicities=mults)
    #
    # check that it actually worked
    # OPTIMIZE remove?
    parent_space = spaces.TensorProduct.from_partial_products(same.left_multiply(res), opposite.dual)
    assert parent_space.sector_multiplicity(same.symmetry.trivial_sector) > 0
    res.test_sanity()

    return res


def random_tensor(symmetry: symmetries.Symmetry,
                  codomain: list[spaces.Space | str | None] | spaces.TensorProduct | int = None,
                  domain: list[spaces.Space | str | None] | spaces.TensorProduct | int = None,
                  labels: list[str | None] = None, dtype: Dtype = None,
                  backend: backends.TensorBackend = None, device: str = None,
                  like: tensors.Tensor = None, max_blocks=5, max_multiplicity=5,
                  empty_ok=False, all_blocks=False, cls=tensors.SymmetricTensor,
                  allow_basis_perm: bool = True, np_random=np.random.default_rng()):
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
                                 max_multiplicity=max_multiplicity, cls=type(like), np_random=np_random)
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
    if isinstance(codomain, spaces.TensorProduct):
        assert codomain.symmetry == symmetry
        num_codomain = codomain.num_factors
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
    if isinstance(domain, spaces.TensorProduct):
        assert domain.symmetry == symmetry
        num_domain = domain.num_factors
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
        charge_leg = random_ElementarySpace(symmetry=symmetry, max_sectors=1, max_multiplicity=1,
                                            is_dual=False, allow_basis_perm=allow_basis_perm,
                                            np_random=np_random)
        if isinstance(domain, spaces.TensorProduct):
            inv_domain = domain.left_multiply(charge_leg)
        else:
            inv_domain = [charge_leg, *domain]
        inv_labels = [*labels, tensors.ChargedTensor._CHARGE_LEG_LABEL]
        inv_part = random_tensor(
            symmetry=symmetry, codomain=codomain, domain=inv_domain, labels=inv_labels, dtype=dtype,
            backend=backend, device=device, max_blocks=max_blocks, max_multiplicity=max_multiplicity,
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
        if isinstance(codomain, spaces.TensorProduct):
            assert codomain.num_factors == 1
            leg = codomain.factors[0]
            if isinstance(domain, spaces.TensorProduct):
                assert domain == codomain
            else:
                assert len(domain) == 1
                assert domain[0] is None or domain[0] == leg
        else:
            assert len(codomain) == 1
            if isinstance(domain, spaces.TensorProduct):
                assert domain.num_factors == 1
                leg = domain.factors[0]
                assert codomain[0] is None or codomain[0] == leg
            else:
                assert len(domain) == 1
                if domain[0] is None and codomain[0] is None:
                    leg = random_ElementarySpace(
                        symmetry=symmetry, max_sectors=max_blocks, max_multiplicity=max_multiplicity,
                        allow_basis_perm=allow_basis_perm, np_random=np_random
                    )
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
        if isinstance(codomain, spaces.TensorProduct):
            assert codomain.num_factors == 1
            small_leg = codomain.factors[0]
        elif codomain is None:
            small_leg is None
        else:
            assert len(codomain) == 1
            small_leg = codomain[0]
        if isinstance(domain, spaces.TensorProduct):
            assert domain.num_factors == 1
            large_leg = domain.factors[0]
        elif domain is None:
            large_leg = None
        else:
            assert len(domain) == 1
            large_leg = domain[0]
        #
        if large_leg is None:
            if small_leg is None:
                large_leg = random_ElementarySpace(
                    symmetry=symmetry, max_sectors=max_blocks, max_multiplicity=max_multiplicity,
                    allow_basis_perm=allow_basis_perm, np_random=np_random
                )
            else:
                # TODO looks like this generates a basis_perm incompatible with the mask!
                raise NotImplementedError('Mask generation broken')
                extra = random_ElementarySpace(symmetry=symmetry, max_sectors=max_blocks,
                                            max_multiplicity=max_multiplicity, is_dual=small_leg.is_dual,
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
                codomain[n] = random_ElementarySpace(symmetry=symmetry, max_sectors=max_blocks,
                                                  max_multiplicity=max_multiplicity,
                                                  allow_basis_perm=allow_basis_perm)
        codomain = spaces.TensorProduct(codomain, symmetry=symmetry)
        codomain_complete = True
    if not codomain_complete:
        # can assume that domain is complete
        if not isinstance(domain, spaces.TensorProduct):
            domain = spaces.TensorProduct(domain, symmetry=symmetry)
        missing = [n for n, sp in enumerate(codomain) if sp is None]
        for n in missing[:-1]:
            codomain[n] = random_ElementarySpace(symmetry=symmetry, max_sectors=max_blocks,
                                              max_multiplicity=max_multiplicity,
                                              allow_basis_perm=allow_basis_perm)
        last = missing[-1]
        partial_codomain = spaces.TensorProduct(codomain[:last] + codomain[last + 1:],
                                                symmetry=symmetry)
        leg = find_last_leg(same=partial_codomain, opposite=domain, max_sectors=max_blocks,
                            max_mult=max_multiplicity)
        codomain = partial_codomain.insert_multiply(leg, last)
    elif not domain_complete:
        # can assume codomain is complete
        if not isinstance(codomain, spaces.TensorProduct):
            codomain = spaces.TensorProduct(codomain, symmetry=symmetry)
        missing = [n for n, sp in enumerate(domain) if sp is None]
        for n in missing[:-1]:
            domain[n] = random_ElementarySpace(symmetry=symmetry, max_sectors=max_blocks,
                                            max_multiplicity=max_multiplicity,
                                            allow_basis_perm=allow_basis_perm)
        last = missing[-1]
        partial_domain = spaces.TensorProduct(domain[:last] + domain[last + 1:], symmetry=symmetry)
        leg = find_last_leg(same=partial_domain, opposite=codomain, max_sectors=max_blocks,
                            max_mult=max_multiplicity)
        domain = partial_domain.insert_multiply(leg, last)
    else:
        if not isinstance(codomain, spaces.TensorProduct):
            codomain = spaces.TensorProduct(codomain, symmetry=symmetry)
        if not isinstance(domain, spaces.TensorProduct):
            domain = spaces.TensorProduct(domain, symmetry=symmetry)
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
