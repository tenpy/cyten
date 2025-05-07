"""A collection of tests for cyten.trees."""
# Copyright (C) TeNPy Developers, Apache license

import numpy as np
import pytest

from cyten import trees
from cyten.symmetries import Symmetry, SymmetryError
from cyten.spaces import ElementarySpace, TensorProduct
from cyten.dtypes import Dtype
from cyten.backends.backend_factory import get_backend
from cyten.backends.abstract_backend import Block


@pytest.mark.xfail(reason='Test not implemented yet')
def test_FusionTree_class():
    # TODO test hash, eq, str, repr, copy
    raise NotImplementedError


def test_FusionTree_manipulations(compatible_symmetry, compatible_backend, make_compatible_sectors, np_random):
    # TODO add a symmetry that detects the difference between conjugating the F symbols
    # and not conjugating them. SU(3)_3, SU(2) and SU(2)_k are not suitable for this.
    sym = compatible_symmetry
    backend = compatible_backend

    # test insert and split
    num_uncoupled = np_random.integers(4, 8)
    # generate uncoupled sectors like this to allow identical sectors in trees
    uncoupled = np.vstack([np_random.choice(make_compatible_sectors(5)) for _ in range(num_uncoupled)])
    are_dual = np_random.choice([True, False], size=num_uncoupled)
    all_trees = random_trees_from_uncoupled(sym, uncoupled, np_random)
    random_trees = np_random.choice(all_trees, size=10)
    for tree in random_trees:
        n_split = np_random.integers(0, num_uncoupled + 1)

        # test errors
        if n_split == num_uncoupled or n_split < 2:
            if n_split == num_uncoupled:
                msg = r'Right tree has no vertices \(n >= num_uncoupled\)'
            else:
                msg = r'Left tree has no vertices \(n < 2\)'
            with pytest.raises(ValueError, match=msg):
                _ = tree.split(n_split)
            continue

        left_tree, right_tree = tree.split(n_split)
        split_sector = tree.inner_sectors[n_split - 2]

        # test left tree
        assert np.all(left_tree.uncoupled == tree.uncoupled[:n_split])
        assert np.all(left_tree.are_dual == tree.are_dual[:n_split])
        assert np.all(left_tree.inner_sectors == tree.inner_sectors[:n_split - 2])
        assert np.all(left_tree.coupled == split_sector)
        assert np.all(left_tree.multiplicities == tree.multiplicities[:n_split - 1])

        # test right tree
        assert np.all(right_tree.uncoupled == np.vstack((split_sector, tree.uncoupled[n_split:])))
        assert np.all(right_tree.are_dual == np.append([False], tree.are_dual[n_split:]))
        assert np.all(right_tree.inner_sectors == tree.inner_sectors[n_split - 1:])
        assert np.all(right_tree.coupled == tree.coupled)
        assert np.all(right_tree.multiplicities == tree.multiplicities[n_split - 1:])

        # test insert
        assert tree == right_tree.insert(left_tree)

    # test insert_at
    num_uncoupled = np_random.integers(3, 6, size=2)
    uncoupled1 = np.vstack([np_random.choice(make_compatible_sectors(5)) for _ in range(num_uncoupled[0])])
    # no dual sectors here to possibly enable insert_at for every uncoupled sector
    all_trees1 = random_trees_from_uncoupled(sym, uncoupled1, np_random)
    random_trees1 = np_random.choice(all_trees1, size=5)

    uncoupled2 = np.vstack([np_random.choice(make_compatible_sectors(5)) for _ in range(num_uncoupled[1])])
    are_dual = np_random.choice([True, False], size=num_uncoupled[1])
    all_trees2 = random_trees_from_uncoupled(sym, uncoupled2, np_random, are_dual=are_dual)
    coupled2 = all_trees2[0].coupled
    random_trees2 = np_random.choice(all_trees2, size=5)

    for i in range(num_uncoupled[0]):
        if not all(coupled2 == uncoupled1[i]):
            continue  # make sure inserting is possible
        perm = list(range(i))
        perm.extend(list(range(num_uncoupled[0], sum(num_uncoupled))))
        perm.extend(list(range(i, num_uncoupled[0])))
        for tree1 in random_trees1:
            for tree2 in random_trees2:
                # do this check for all symmetries (also checks the sectors in the trees)
                check_insert_at_via_f_symbols(tree1, tree2, i)
                # check with as_block
                if sym.can_be_dropped:
                    block1 = tree1.as_block(backend)
                    block2 = tree2.as_block(backend)
                    expect = backend.block_backend.tdot(block1, block2, [i], [-1])
                    expect = backend.block_backend.permute_axes(expect, perm)
                    combined_tree = tree1.insert_at(i, tree2)
                    combined_block = tree_superposition_as_block(combined_tree, backend)
                    assert backend.block_backend.allclose(combined_block, expect, rtol=1e-8, atol=1e-5)


def check_insert_at_via_f_symbols(tree1: trees.FusionTree, tree2: trees.FusionTree, i: int):
    """Check correct amplitudes, normalization (sum of amplitudes), uncoupled sectors,
    inner sectors, coupled sectors and multilcities.
    """
    combined_tree = tree1.insert_at(i, tree2)
    uncoupled = np.vstack((tree1.uncoupled[:i], tree2.uncoupled, tree1.uncoupled[i + 1:]))
    are_dual = np.concatenate([tree1.are_dual[:i], tree2.are_dual, tree1.are_dual[i + 1:]])
    coupled = tree1.coupled
    norm = 0
    for tree, amp in combined_tree.items():
        tree.test_sanity()
        assert np.all(tree.uncoupled == uncoupled)
        assert np.all(tree.are_dual == are_dual)
        assert np.all(tree.coupled == coupled)
        assert np.all(tree.multiplicities[i + tree2.num_vertices:] == tree1.multiplicities[i:])
        if i > 0:
            assert np.all(tree.inner_sectors[i - 1 + tree2.num_vertices:] == tree1.inner_sectors[i - 1:])
            assert np.all(tree.inner_sectors[:i - 1] == tree1.inner_sectors[:i - 1])
            assert np.all(tree.multiplicities[:i - 1] == tree1.multiplicities[:i - 1])

        if i == 0 or tree2.num_uncoupled == 1:
            fs = 1  # no F symbols to apply
        else:
            f_symbols = []
            a = tree.uncoupled[0] if i == 1 else tree.inner_sectors[i - 2]
            for j in range(tree2.num_uncoupled - 1):
                b = tree.uncoupled[i] if j == 0 else tree2.inner_sectors[j - 1]
                c = tree.uncoupled[i + j + 1]
                d = tree.coupled if i + j + 1 == tree.num_uncoupled - 1 else tree.inner_sectors[i + j]
                e = tree2.coupled if j == tree2.num_inner_edges else tree2.inner_sectors[j]
                f = tree.inner_sectors[i + j - 1]
                f_symbols.append(np.conj(tree1.symmetry.f_symbol(a, b, c, d, e, f)))

            # deal with multiplicities
            kap = tree.multiplicities[i - 1]
            lam = tree.multiplicities[i]
            mu = tree2.multiplicities[0]
            nu = tree1.multiplicities[i - 1]
            fs = f_symbols[0][mu, :, kap, lam]
            for j, f in enumerate(f_symbols[1:]):
                lam = tree.multiplicities[i + j + 1]
                mu = tree2.multiplicities[j + 1]
                fs = np.tensordot(fs, f[mu, :, :, lam], [0, 1])
            fs = fs[nu]
        assert np.isclose(fs, amp)
        norm += amp * np.conj(amp)
    assert np.isclose(norm, 1)


def random_trees_from_uncoupled(symmetry, uncoupled, np_random, are_dual=None
                                ) -> list[trees.FusionTree]:
    """Choose a random coupled sector consistent with the given uncoupled sectors and
    return all fusion trees with consistent inner sectors and multiplicities as list.
    """
    spaces = [ElementarySpace(symmetry, [a]) for a in uncoupled]
    domain = TensorProduct(spaces)
    coupled = np_random.choice(domain.sector_decomposition)
    return list(trees.fusion_trees(symmetry, uncoupled, coupled, are_dual=are_dual))


def tree_superposition_as_block(superposition, backend, dtype=None) -> Block:
    for i, (tree, amp) in enumerate(superposition.items()):
        if i == 0:
            res = amp * tree.as_block(backend, dtype)
        else:
            res += amp * tree.as_block(backend, dtype)
    return res


def check_fusion_trees(it: trees.fusion_trees, expect_len: int = None):
    if expect_len is None:
        expect_len = len(it)
    else:
        assert len(it) == expect_len

    # make sure they run.
    _ = str(it)
    _ = repr(it)

    num_trees = 0
    for tree in it:
        assert np.all(tree.are_dual == it.are_dual)
        tree.test_sanity()
        assert it.index(tree) == num_trees
        num_trees += 1
    assert num_trees == expect_len
        
    
def test_fusion_trees(any_symmetry: Symmetry, make_any_sectors, np_random):
    """test the ``fusion_trees`` iterator"""
    some_sectors = make_any_sectors(20)  # generates unique sectors
    non_trivial_sectors = some_sectors[np.any(some_sectors != any_symmetry.trivial_sector[None, :], axis=1)]
    i = any_symmetry.trivial_sector

    print('consistent fusion: [] -> i')
    check_fusion_trees(trees.fusion_trees(any_symmetry, [], i), expect_len=1)
    
    print('consistent fusion: i -> i')
    check_fusion_trees(trees.fusion_trees(any_symmetry, [i], i, [False]), expect_len=1)
    check_fusion_trees(trees.fusion_trees(any_symmetry, [i], i, [True]), expect_len=1)

    print('large consistent fusion')
    uncoupled = some_sectors[:5]
    are_dual = np_random.choice([True, False], size=len(uncoupled), replace=True)
    # find the allowed coupled sectors
    allowed = TensorProduct([ElementarySpace(any_symmetry, [a]) for a in uncoupled]).sector_decomposition
    some_allowed = np_random.choice(allowed, axis=0)
    print(f'  uncoupled={", ".join(map(str, uncoupled))}   coupled={some_allowed}')
    it = trees.fusion_trees(any_symmetry, uncoupled, some_allowed, are_dual=are_dual)
    assert len(it) > 0
    check_fusion_trees(it)

    print('large inconsistent fusion')
    # find a forbidden coupled sector
    are_allowed = np.any(np.all(some_sectors[:, None, :] == allowed[None, :, :], axis=2), axis=1)
    forbidden_idcs = np.where(np.logical_not(are_allowed))[0]
    if len(forbidden_idcs) > 0:
        forbidden = some_sectors[np_random.choice(forbidden_idcs)]
        it = trees.fusion_trees(any_symmetry, uncoupled, forbidden, are_dual=are_dual)
        check_fusion_trees(it, expect_len=0)

    # rest of the checks assume we have access to at least one non-trivial sector
    if len(non_trivial_sectors) == 0:
        return
    c = non_trivial_sectors[0]
    c_dual = any_symmetry.dual_sector(c)

    print(f'consistent fusion: c -> c')
    check_fusion_trees(trees.fusion_trees(any_symmetry, [c], c, [True]), expect_len=1)
    check_fusion_trees(trees.fusion_trees(any_symmetry, [c], c, [False]), expect_len=1)

    print(f'consistent fusion: [c, dual(c)] -> i')
    check_fusion_trees(trees.fusion_trees(any_symmetry, [c, c_dual], i, [False, False]), expect_len=1)
    check_fusion_trees(trees.fusion_trees(any_symmetry, [c, c_dual], i, [False, True]), expect_len=1)

    # rest of the checks assume we have access to at least two non-trivial sector
    if len(non_trivial_sectors) == 1:
        return
    d = non_trivial_sectors[1]

    print(f'inconsistent fusion: c -> d')
    check_fusion_trees(trees.fusion_trees(any_symmetry, [c], d, [True]), expect_len=0)
    check_fusion_trees(trees.fusion_trees(any_symmetry, [c], d, [False]), expect_len=0)

    print('consistent fusion: [c, d] -> ?')
    e = any_symmetry.fusion_outcomes(c, d)[0]
    N = any_symmetry.n_symbol(c, d, e)
    check_fusion_trees(trees.fusion_trees(any_symmetry, [c, d], e, [False, False]), expect_len=N)
    check_fusion_trees(trees.fusion_trees(any_symmetry, [c, d], e, [False, True]), expect_len=N)


def check_to_block(symmetry, backend, uncoupled, np_random, dtype):
    """Common implementation for test_to_block and test_to_block_no_backend"""
    all_trees = random_trees_from_uncoupled(symmetry, uncoupled, np_random)

    if not symmetry.can_be_dropped:
        with pytest.raises(SymmetryError, match='Can not convert to block for symmetry .*'):
            _ = all_trees[0].as_block(backend, dtype)
        return
    
    coupled_dim = symmetry.sector_dim(all_trees[0].coupled)
    uncoupled_dims = symmetry.batch_sector_dim(uncoupled)
    all_blocks = [t.as_block(backend, dtype) for t in all_trees]
    axes = list(range(len(uncoupled)))
    if symmetry.fusion_tensor_dtype.is_complex:
        expect_dtype = dtype.to_complex()
    else:
        expect_dtype = dtype
        
    if backend is None:
        backend = get_backend()
    coupled_eye = backend.block_backend.eye_block([coupled_dim], dtype)
    coupled_zero = backend.block_backend.zeros([coupled_dim, coupled_dim], dtype)
    for i, X in enumerate(all_blocks):
        assert backend.block_backend.get_shape(X) == (*uncoupled_dims, coupled_dim)
        assert backend.block_backend.get_dtype(X) == expect_dtype
        for j, Y in enumerate(all_blocks):
            if i < j:
                continue  # redundant with (i, j) <-> (j, i)
            X_Y = backend.block_backend.tdot(backend.block_backend.conj(X), Y, axes, axes)
            expect = coupled_eye if i == j else coupled_zero
            assert backend.block_backend.allclose(X_Y, expect, rtol=1e-8, atol=1e-5)


@pytest.mark.parametrize('dtype', [Dtype.float64, Dtype.complex128])
def test_to_block(compatible_symmetry, compatible_backend, make_compatible_sectors, np_random, dtype):
    # need two test_* functions to generate the cases, implement actual test in check_to_block...
    uncoupled = make_compatible_sectors(4)
    check_to_block(compatible_symmetry, compatible_backend, uncoupled, np_random, dtype)


@pytest.mark.parametrize('dtype', [Dtype.float64, Dtype.complex128])
def test_to_block_no_backend(any_symmetry, make_any_sectors, np_random, dtype):
    # need two test_* functions to generate the cases, implement actual test in check_to_block
    coupled = make_any_sectors(4)
    check_to_block(any_symmetry, None, coupled, np_random, dtype)
