"""A collection of tests for cyten.backends.fusion_tree_backend"""
# Copyright (C) TeNPy Developers, Apache license
from __future__ import annotations
from typing import Callable
import pytest
import numpy as np
from math import prod

from cyten.backends import fusion_tree_backend, get_backend
from cyten.trees import FusionTree
from cyten.spaces import ElementarySpace, TensorProduct
from cyten import backends
from cyten.tensors import DiagonalTensor, SymmetricTensor, move_leg
from cyten.symmetries import (
    ProductSymmetry, Symmetry, fibonacci_anyon_category, SU2Symmetry, SU3_3AnyonCategory,
    ising_anyon_category, SU2_kAnyonCategory, z5_symmetry, u1_symmetry
)
from cyten.dtypes import Dtype
from cyten.testing import assert_tensors_almost_equal
from ..util import random_tensor, random_ElementarySpace


def test_c_symbol_fibonacci_anyons(block_backend: str, np_random: np.random.Generator):
    move_leg_or_permute_leg = np_random.choice(['move_leg', 'permute_leg'])
    print('use ' + move_leg_or_permute_leg)
    backend = get_backend('fusion_tree', block_backend)
    funcs = [cross_check_single_c_symbol_tree_blocks,
             cross_check_single_c_symbol_tree_cols,
             apply_single_c_symbol]
    zero_block = backend.block_backend.zeros
    eps = 1.e-14
    sym = fibonacci_anyon_category
    s1 = ElementarySpace(sym, [[1]], [1])  # only tau
    s2 = ElementarySpace(sym, [[0], [1]], [1, 1])  # 1 and tau
    codomain = TensorProduct([s2, s1, s2, s2])
    domain = TensorProduct([s2, s1, s2])

    block_inds = np.array([[0,0], [1,1]])
    blocks = [backend.block_backend.random_uniform((8, 3), Dtype.complex128),
              backend.block_backend.random_uniform((13, 5), Dtype.complex128)]
    data = backends.FusionTreeData(block_inds, blocks, Dtype.complex128,
                                   device=backend.block_backend.default_device)
    tens = SymmetricTensor(data, codomain, domain, backend=backend)

    levels = list(range(tens.num_legs))[::-1]  # for the exchanges

    # exchange legs 0 and 1 (in codomain)
    r1 = np.exp(-4j*np.pi/5)  # R symbols
    rtau = np.exp(3j*np.pi/5)
    expect = [zero_block([8, 3], Dtype.complex128), zero_block([13, 5], Dtype.complex128)]

    expect[0][0, :] = blocks[0][0, :]
    expect[0][1, :] = blocks[0][1, :]
    expect[0][2, :] = blocks[0][2, :]
    expect[0][3, :] = blocks[0][3, :] * r1
    expect[0][4, :] = blocks[0][4, :] * rtau
    expect[0][5, :] = blocks[0][5, :] * rtau
    expect[0][6, :] = blocks[0][6, :] * r1
    expect[0][7, :] = blocks[0][7, :] * rtau

    expect[1][0, :] = blocks[1][0, :]
    expect[1][1, :] = blocks[1][1, :]
    expect[1][2, :] = blocks[1][2, :]
    expect[1][3, :] = blocks[1][3, :]
    expect[1][4, :] = blocks[1][4, :]
    expect[1][5, :] = blocks[1][5, :] * rtau
    expect[1][6, :] = blocks[1][6, :] * r1
    expect[1][7, :] = blocks[1][7, :] * rtau
    expect[1][8, :] = blocks[1][8, :] * r1
    expect[1][9, :] = blocks[1][9, :] * rtau
    expect[1][10, :] = blocks[1][10, :] * r1
    expect[1][11, :] = blocks[1][11, :] * rtau
    expect[1][12, :] = blocks[1][12, :] * rtau

    expect_data = backends.FusionTreeData(block_inds, expect, Dtype.complex128,
                                          device=backend.block_backend.default_device)
    expect_codomain = TensorProduct([s1, s2, s2, s2])
    expect_tens = SymmetricTensor(expect_data, expect_codomain, domain, backend=backend)

    # do this without permute_legs for the different implementations
    for func in funcs:
        new_data, new_codomain, new_domain = func(tens, leg=0, levels=levels)
        new_tens = SymmetricTensor(new_data, new_codomain, new_domain, backend=backend)
        assert_tensors_almost_equal(new_tens, expect_tens, eps, eps)

    if move_leg_or_permute_leg == 'permute_leg':
        new_data, new_codomain, new_domain = backend.permute_legs(tens, [1, 0, 2, 3], [6, 5, 4], levels)
        new_tens = SymmetricTensor(new_data, new_codomain, new_domain, backend=backend)
        assert_tensors_almost_equal(new_tens, expect_tens, eps, eps)
    elif move_leg_or_permute_leg == 'move_leg':
        new_tens = move_leg(tens, 0, codomain_pos=1, levels=levels)
        assert_tensors_almost_equal(new_tens, expect_tens, eps, eps)

    # exchange legs 5 and 6 (in domain)
    expect = [zero_block([8, 3], Dtype.complex128), zero_block([13, 5], Dtype.complex128)]

    expect[0][:, 0] = blocks[0][:, 0]
    expect[0][:, 1] = blocks[0][:, 1] * r1
    expect[0][:, 2] = blocks[0][:, 2] * rtau

    expect[1][:, 0] = blocks[1][:, 0]
    expect[1][:, 1] = blocks[1][:, 1]
    expect[1][:, 2] = blocks[1][:, 2] * rtau
    expect[1][:, 3] = blocks[1][:, 3] * r1
    expect[1][:, 4] = blocks[1][:, 4] * rtau

    expect_data = backends.FusionTreeData(block_inds, expect, Dtype.complex128,
                                          device=backend.block_backend.default_device)
    expect_domain = TensorProduct([s1, s2, s2])
    expect_tens = SymmetricTensor(expect_data, codomain, expect_domain, backend=backend)

    for func in funcs:
        new_data, new_codomain, new_domain = func(tens, leg=5, levels=levels)
        new_tens = SymmetricTensor(new_data, new_codomain, new_domain, backend=backend)
        assert_tensors_almost_equal(new_tens, expect_tens, eps, eps)

    if move_leg_or_permute_leg == 'permute_leg':
        new_data, new_codomain, new_domain = backend.permute_legs(tens, [0, 1, 2, 3], [5, 6, 4], levels)
        new_tens = SymmetricTensor(new_data, new_codomain, new_domain, backend=backend)
        assert_tensors_almost_equal(new_tens, expect_tens, eps, eps)
    elif move_leg_or_permute_leg == 'move_leg':
        new_tens = move_leg(tens, 6, domain_pos=1, levels=levels)
        assert_tensors_almost_equal(new_tens, expect_tens, eps, eps)

    # exchange legs 2 and 3 (in codomain)
    phi = (1 + 5**0.5) / 2
    ctttt11 = phi**-1 * r1.conj()  # C symbols
    cttttt1 = phi**-0.5 * rtau * r1.conj()
    ctttt1t = phi**-0.5 * rtau.conj()
    ctttttt = -1*phi**-1
    expect = [zero_block([8, 3], Dtype.complex128), zero_block([13, 5], Dtype.complex128)]

    expect[0][0, :] = blocks[0][1, :]
    expect[0][1, :] = blocks[0][0, :]
    expect[0][2, :] = blocks[0][2, :] * rtau
    expect[0][3, :] = blocks[0][3, :]
    expect[0][4, :] = blocks[0][5, :]
    expect[0][5, :] = blocks[0][4, :]
    expect[0][6, :] = blocks[0][6, :] * r1
    expect[0][7, :] = blocks[0][7, :] * rtau

    expect[1][0, :] = blocks[1][0, :]
    expect[1][1, :] = blocks[1][2, :]
    expect[1][2, :] = blocks[1][1, :]
    expect[1][3, :] = blocks[1][3, :] * ctttt11 + blocks[1][4, :] * cttttt1
    expect[1][4, :] = blocks[1][3, :] * ctttt1t + blocks[1][4, :] * ctttttt
    expect[1][5, :] = blocks[1][5, :]
    expect[1][6, :] = blocks[1][8, :]
    expect[1][7, :] = blocks[1][9, :]
    expect[1][8, :] = blocks[1][6, :]
    expect[1][9, :] = blocks[1][7, :]
    expect[1][10, :] = blocks[1][10, :] * rtau
    expect[1][11, :] = blocks[1][11, :] * ctttt11 + blocks[1][12, :] * cttttt1
    expect[1][12, :] = blocks[1][11, :] * ctttt1t + blocks[1][12, :] * ctttttt

    expect_data = backends.FusionTreeData(block_inds, expect, Dtype.complex128,
                                          device=backend.block_backend.default_device)
    expect_tens = SymmetricTensor(expect_data, codomain, domain, backend=backend)

    for func in funcs:
        new_data, new_codomain, new_domain = func(tens, leg=2, levels=levels)
        new_tens = SymmetricTensor(new_data, new_codomain, new_domain, backend=backend)
        assert_tensors_almost_equal(new_tens, expect_tens, eps, eps)

    if move_leg_or_permute_leg == 'permute_leg':
        new_data, new_codomain, new_domain = backend.permute_legs(tens, [0, 1, 3, 2], [6, 5, 4], levels)
        new_tens = SymmetricTensor(new_data, new_codomain, new_domain, backend=backend)
        assert_tensors_almost_equal(new_tens, expect_tens, eps, eps)
    elif move_leg_or_permute_leg == 'move_leg':
        new_tens = move_leg(tens, 3, codomain_pos=2, levels=levels)
        assert_tensors_almost_equal(new_tens, expect_tens, eps, eps)

    # exchange legs 4 and 5 (in domain)
    expect = [zero_block([8, 3], Dtype.complex128), zero_block([13, 5], Dtype.complex128)]

    expect[0][:, 0] = blocks[0][:, 0] * r1
    expect[0][:, 1] = blocks[0][:, 1]
    expect[0][:, 2] = blocks[0][:, 2] * rtau

    expect[1][:, 0] = blocks[1][:, 0]
    expect[1][:, 1] = blocks[1][:, 1] * rtau
    expect[1][:, 2] = blocks[1][:, 2]
    expect[1][:, 3] = blocks[1][:, 3] * ctttt11 + blocks[1][:, 4] * cttttt1
    expect[1][:, 4] = blocks[1][:, 3] * ctttt1t + blocks[1][:, 4] * ctttttt

    expect_data = backends.FusionTreeData(block_inds, expect, Dtype.complex128,
                                          device=backend.block_backend.default_device)
    expect_domain = TensorProduct([s2, s2, s1])
    expect_tens = SymmetricTensor(expect_data, codomain, expect_domain, backend=backend)

    for func in funcs:
        new_data, new_codomain, new_domain = func(tens, leg=4, levels=levels)
        new_tens = SymmetricTensor(new_data, new_codomain, new_domain, backend=backend)
        assert_tensors_almost_equal(new_tens, expect_tens, eps, eps)

    if move_leg_or_permute_leg == 'permute_leg':
        new_data, new_codomain, new_domain = backend.permute_legs(tens, [0, 1, 2, 3], [6, 4, 5], levels)
        new_tens = SymmetricTensor(new_data, new_codomain, new_domain, backend=backend)
        assert_tensors_almost_equal(new_tens, expect_tens, eps, eps)
    elif move_leg_or_permute_leg == 'move_leg':
        new_tens = move_leg(tens, 4, domain_pos=1, levels=levels)
        assert_tensors_almost_equal(new_tens, expect_tens, eps, eps)

    # braid 10 times == trivial
    assert_repeated_braids_trivial(tens, funcs, levels, repeat=10, eps=eps)

    # braid clockwise and then counter-clockwise == trivial
    assert_clockwise_counterclockwise_trivial(tens, funcs, levels, eps=eps)

    # rescaling axes and then braiding == braiding and then rescaling axes
    assert_braiding_and_scale_axis_commutation(tens, funcs, levels, eps=eps)

    # do and undo sequence of braids == trivial (may include b symbols)
    for _ in range(2):
        assert_clockwise_counterclockwise_trivial_long_range(tens, move_leg_or_permute_leg, eps, np_random)


@pytest.mark.slow  # TODO can we speed it up?
def test_c_symbol_product_sym(block_backend: str, np_random: np.random.Generator):
    move_leg_or_permute_leg = np_random.choice(['move_leg', 'permute_leg'])
    print('use ' + move_leg_or_permute_leg)
    backend = get_backend('fusion_tree', block_backend)
    funcs = [cross_check_single_c_symbol_tree_blocks,
             cross_check_single_c_symbol_tree_cols,
             apply_single_c_symbol]
    zero_block = backend.block_backend.zeros
    eps = 1.e-14
    sym = ProductSymmetry([fibonacci_anyon_category, SU2Symmetry()])
    s1 = ElementarySpace(sym, [[1, 1]], [2])  # only (tau, spin-1/2)
    s2 = ElementarySpace(sym, [[0, 0], [1, 1]], [1, 2])  # (1, spin-0) and (tau, spin-1/2)
    codomain = TensorProduct([s2, s2, s2])
    domain = TensorProduct([s2, s1, s2])

    # block charges: 0: [0, 0], 1: [1, 0], 2: [0, 1], 3: [1, 1]
    #                4: [0, 2], 5: [1, 2], 6: [0, 3], 7: [1, 3]
    block_inds = np.array([[i, i] for i in range(8)])
    shapes = [(13, 8), (12, 8), (16, 16), (38, 34), (12, 8), (12, 8), (8, 8), (16, 16)]
    blocks = [backend.block_backend.random_uniform(shp, Dtype.complex128) for shp in shapes]
    data = backends.FusionTreeData(block_inds, blocks, Dtype.complex128,
                                   device=backend.block_backend.default_device)
    tens = SymmetricTensor(data, codomain, domain, backend=backend)

    levels = list(range(tens.num_legs))[::-1]  # for the exchanges

    # exchange legs 0 and 1 (in codomain)
    r1 = np.exp(-4j*np.pi/5)  # Fib R symbols
    rtau = np.exp(3j*np.pi/5)
    exc = [0, 2, 1, 3]
    exc2 = [4, 5, 6, 7, 0, 1, 2, 3]
    exc3 = [0, 1, 4, 5, 2, 3, 6, 7]

    expect = [zero_block(shp, Dtype.complex128) for shp in shapes]

    expect[0][:9, :] = blocks[0][[0] + [1 + i for i in exc2], :]
    expect[0][9:, :] = blocks[0][[9 + i for i in exc], :] * r1 * -1

    expect[1][:8, :] = blocks[1][exc2, :]
    expect[1][8:, :] = blocks[1][[8 + i for i in exc], :] * rtau * -1

    expect[2][:8, :] = blocks[2][exc3, :] * rtau * -1
    expect[2][8:, :] = blocks[2][[8 + i for i in exc3], :] * rtau

    expect[3][:6, :] = blocks[3][[0, 1, 4, 5, 2, 3], :]
    expect[3][6:14, :] = blocks[3][[6 + i for i in exc3], :] * r1 * -1
    expect[3][14:22, :] = blocks[3][[14 + i for i in exc3], :] * r1
    expect[3][22:30, :] = blocks[3][[22 + i for i in exc3], :] * rtau * -1
    expect[3][30:, :] = blocks[3][[30 + i for i in exc3], :] * rtau

    expect[4][:8, :] = blocks[4][exc2, :]
    expect[4][8:, :] = blocks[4][[8 + i for i in exc], :] * r1

    expect[5][:8, :] = blocks[5][exc2, :]
    expect[5][8:, :] = blocks[5][[8 + i for i in exc], :] * rtau

    expect[6][:, :] = blocks[6][exc3, :] * rtau

    expect[7][:8, :] = blocks[7][exc3, :] * r1
    expect[7][8:, :] = blocks[7][[8 + i for i in exc3], :] * rtau

    expect_data = backends.FusionTreeData(block_inds, expect, Dtype.complex128,
                                          device=backend.block_backend.default_device)
    expect_tens = SymmetricTensor(expect_data, codomain, domain, backend=backend)

    for func in funcs:
        new_data, new_codomain, new_domain = func(tens, leg=0, levels=levels)
        new_tens = SymmetricTensor(new_data, new_codomain, new_domain, backend=backend)
        assert_tensors_almost_equal(new_tens, expect_tens, eps, eps)

    if move_leg_or_permute_leg == 'permute_leg':
        new_data, new_codomain, new_domain = backend.permute_legs(tens, [1, 0, 2], [5, 4, 3], levels)
        new_tens = SymmetricTensor(new_data, new_codomain, new_domain, backend=backend)
        assert_tensors_almost_equal(new_tens, expect_tens, eps, eps)
    elif move_leg_or_permute_leg == 'move_leg':
        new_tens = move_leg(tens, 1, codomain_pos=0, levels=levels)
        assert_tensors_almost_equal(new_tens, expect_tens, eps, eps)

    # exchange legs 4 and 5 (in domain)
    expect = [zero_block(shp, Dtype.complex128) for shp in shapes]

    expect[0][:, :4] = blocks[0][:, :4]
    expect[0][:, 4:] = blocks[0][:, [4 + i for i in exc]] * r1 * -1

    expect[1][:, :4] = blocks[1][:, :4]
    expect[1][:, 4:] = blocks[1][:, [4 + i for i in exc]] * rtau * -1

    expect[2][:, :8] = blocks[2][:, exc3] * rtau * -1
    expect[2][:, 8:] = blocks[2][:, [8 + i for i in exc3]] * rtau

    expect[3][:, :2] = blocks[3][:, :2]
    expect[3][:, 2:10] = blocks[3][:, [2 + i for i in exc3]] * r1 * -1
    expect[3][:, 10:18] = blocks[3][:, [10 + i for i in exc3]] * r1
    expect[3][:, 18:26] = blocks[3][:, [18 + i for i in exc3]] * rtau * -1
    expect[3][:, 26:34] = blocks[3][:, [26 + i for i in exc3]] * rtau

    expect[4][:, :4] = blocks[4][:, :4]
    expect[4][:, 4:] = blocks[4][:, [4 + i for i in exc]] * r1

    expect[5][:, :4] = blocks[5][:, :4]
    expect[5][:, 4:] = blocks[5][:, [4 + i for i in exc]] * rtau

    expect[6][:, :] = blocks[6][:, exc3] * rtau

    expect[7][:, :8] = blocks[7][:, exc3] * r1
    expect[7][:, 8:] = blocks[7][:, [8 + i for i in exc3]] * rtau

    expect_data = backends.FusionTreeData(block_inds, expect, Dtype.complex128,
                                          device=backend.block_backend.default_device)
    expect_domain = TensorProduct([s1, s2, s2])
    expect_tens = SymmetricTensor(expect_data, codomain, expect_domain, backend=backend)

    for func in funcs:
        new_data, new_codomain, new_domain = func(tens, leg=4, levels=levels)
        new_tens = SymmetricTensor(new_data, new_codomain, new_domain, backend=backend)
        assert_tensors_almost_equal(new_tens, expect_tens, eps, eps)

    if move_leg_or_permute_leg == 'permute_leg':
        new_data, new_codomain, new_domain = backend.permute_legs(tens, [0, 1, 2], [4, 5, 3], levels)
        new_tens = SymmetricTensor(new_data, new_codomain, new_domain, backend=backend)
        assert_tensors_almost_equal(new_tens, expect_tens, eps, eps)
    elif move_leg_or_permute_leg == 'move_leg':
        new_tens = move_leg(tens, 5, domain_pos=1, levels=levels)
        assert_tensors_almost_equal(new_tens, expect_tens, eps, eps)

    # exchange legs 3 and 4 (in domain)
    phi = (1 + 5**0.5) / 2
    ctttt11 = phi**-1 * r1.conj()  # C symbols
    cttttt1 = phi**-0.5 * rtau * r1.conj()
    ctttt1t = phi**-0.5 * rtau.conj()
    ctttttt = -1*phi**-1
    exc4 = [0, 2, 1, 3, 4, 6, 5, 7]
    expect = [zero_block(shp, Dtype.complex128) for shp in shapes]

    expect[0][:, :4] = blocks[0][:, exc] * r1 * -1
    expect[0][:, 4:] = blocks[0][:, 4:]

    expect[1][:, :4] = blocks[1][:, exc] * rtau * -1
    expect[1][:, 4:] = blocks[1][:, 4:]

    # f-symbols for su(2) [e -> f]: 0 -> 0: -1/2, 2 -> 2: 1/2, 0 -> 2 and 2 -> 0: 3**.5/2
    expect[2][:, :8] = (blocks[2][:, exc4] * rtau * (-1/4 + 3/4)
                       + blocks[2][:, [8 + i for i in exc4]] * rtau * (3**0.5/4 + 3**0.5/4))
    expect[2][:, 8:] = (blocks[2][:, exc4] * rtau * (3**0.5/4 + 3**0.5/4)
                       + blocks[2][:, [8 + i for i in exc4]] * rtau * (1/4 - 3/4))

    expect[3][:, :2] = blocks[3][:, :2]
    expect[3][:, 2:10] = (blocks[3][:, [2 + i for i in exc4]] * ctttt11 * (-1/4 + 3/4)
                         + blocks[3][:, [10 + i for i in exc4]] * ctttt11 * (3**0.5/4 + 3**0.5/4)
                         + blocks[3][:, [18 + i for i in exc4]] * cttttt1 * (-1/4 + 3/4)
                         + blocks[3][:, [26 + i for i in exc4]] * cttttt1 * (3**0.5/4 + 3**0.5/4))
    expect[3][:, 10:18] = (blocks[3][:, [2 + i for i in exc4]] * ctttt11 * (3**0.5/4 + 3**0.5/4)
                          + blocks[3][:, [10 + i for i in exc4]] * ctttt11 * (1/4 - 3/4)
                          + blocks[3][:, [18 + i for i in exc4]] * cttttt1 * (3**0.5/4 + 3**0.5/4)
                          + blocks[3][:, [26 + i for i in exc4]] * cttttt1 * (1/4 - 3/4))
    expect[3][:, 18:26] = (blocks[3][:, [2 + i for i in exc4]] * ctttt1t * (-1/4 + 3/4)
                          + blocks[3][:, [10 + i for i in exc4]] * ctttt1t * (3**0.5/4 + 3**0.5/4)
                          + blocks[3][:, [18 + i for i in exc4]] * ctttttt * (-1/4 + 3/4)
                          + blocks[3][:, [26 + i for i in exc4]] * ctttttt * (3**0.5/4 + 3**0.5/4))
    expect[3][:, 26:34] = (blocks[3][:, [2 + i for i in exc4]] * ctttt1t * (3**0.5/4 + 3**0.5/4)
                          + blocks[3][:, [10 + i for i in exc4]] * ctttt1t * (1/4 - 3/4)
                          + blocks[3][:, [18 + i for i in exc4]] * ctttttt * (3**0.5/4 + 3**0.5/4)
                          + blocks[3][:, [26 + i for i in exc4]] * ctttttt * (1/4 - 3/4))

    expect[4][:, :4] = blocks[4][:, exc] * r1
    expect[4][:, 4:] = blocks[4][:, 4:]

    expect[5][:, :4] = blocks[5][:, exc] * rtau
    expect[5][:, 4:] = blocks[5][:, 4:]

    expect[6][:, :] = blocks[6][:, exc4] * rtau

    expect[7][:, :8] = blocks[7][:, exc4] * ctttt11 + blocks[7][:, [8 + i for i in exc4]] * cttttt1
    expect[7][:, 8:] = blocks[7][:, exc4] * ctttt1t + blocks[7][:, [8 + i for i in exc4]] * ctttttt

    expect_data = backends.FusionTreeData(block_inds, expect, Dtype.complex128,
                                          device=backend.block_backend.default_device)
    expect_domain = TensorProduct([s2, s2, s1])
    expect_tens = SymmetricTensor(expect_data, codomain, expect_domain, backend=backend)

    for func in funcs:
        new_data, new_codomain, new_domain = func(tens, leg=3, levels=levels)
        new_tens = SymmetricTensor(new_data, new_codomain, new_domain, backend=backend)
        assert_tensors_almost_equal(new_tens, expect_tens, eps, eps)

    if move_leg_or_permute_leg == 'permute_leg':
        new_data, new_codomain, new_domain = backend.permute_legs(tens, [0, 1, 2], [5, 3, 4], levels)
        new_tens = SymmetricTensor(new_data, new_codomain, new_domain, backend=backend)
        assert_tensors_almost_equal(new_tens, expect_tens, eps, eps)
    elif move_leg_or_permute_leg == 'move_leg':
        new_tens = move_leg(tens, 4, domain_pos=2, levels=levels)
        assert_tensors_almost_equal(new_tens, expect_tens, eps, eps)

    # braid 10 times == trivial
    assert_repeated_braids_trivial(tens, funcs, levels, repeat=10, eps=eps)

    # braid clockwise and then counter-clockwise == trivial
    assert_clockwise_counterclockwise_trivial(tens, funcs, levels, eps=eps)

    # rescaling axes and then braiding == braiding and then rescaling axes
    assert_braiding_and_scale_axis_commutation(tens, funcs, levels, eps=eps)

    # do and undo sequence of braids == trivial (may include b symbols)
    for _ in range(2):
        assert_clockwise_counterclockwise_trivial_long_range(tens, move_leg_or_permute_leg, eps, np_random)


@pytest.mark.slow  # TODO can we speed it up?
def test_c_symbol_su3_3(block_backend: str, np_random: np.random.Generator):
    move_leg_or_permute_leg = np_random.choice(['move_leg', 'permute_leg'])
    print('use ' + move_leg_or_permute_leg)
    backend = get_backend('fusion_tree', block_backend)
    funcs = [cross_check_single_c_symbol_tree_blocks,
             cross_check_single_c_symbol_tree_cols,
             apply_single_c_symbol]
    zero_block = backend.block_backend.zeros
    eps = 1.e-14
    sym = SU3_3AnyonCategory()
    s1 = ElementarySpace(sym, [[1], [2]], [1, 1])  # 8 and 10
    s2 = ElementarySpace(sym, [[1]], [2])  # 8 with multiplicity 2
    [c0, c1, c2, c3] = [np.array([i]) for i in range(4)]  # charges
    codomain = TensorProduct([s1, s1, s1])
    domain = TensorProduct([s1, s2, s2])

    block_inds = np.array([[i, i] for i in range(4)])
    shapes = [(6, 12), (16, 36), (5, 12), (5, 12)]
    blocks = [backend.block_backend.random_uniform(shp, Dtype.complex128) for shp in shapes]
    data = backends.FusionTreeData(block_inds, blocks, Dtype.complex128,
                                   device=backend.block_backend.default_device)
    tens = SymmetricTensor(data, codomain, domain, backend=backend)

    levels = list(range(tens.num_legs))[::-1]  # for the exchanges

    # exchange legs 0 and 1 (in codomain)
    # SU(3)_3 R symbols
    # exchanging two 8s gives -1 except if they fuse to 8, then
    r8 = [-1j, 1j]  # for the two multiplicities
    # all other R symbols are trivial
    expect = [zero_block(shp, Dtype.complex128) for shp in shapes]

    for i in [0, 2, 3]:
        expect[i][0, :] = blocks[i][0, :] * r8[0]
        expect[i][1, :] = blocks[i][1, :] * r8[1]
        expect[i][2, :] = blocks[i][2, :] * -1
        expect[i][[3, 4], :] = blocks[i][[4, 3], :]
    expect[0][5, :] = blocks[0][5, :]

    expect[1][[0, 5, 6], :] = blocks[1][[0, 5, 6], :] * -1
    expect[1][[1, 3, 7], :] = blocks[1][[1, 3, 7], :] * r8[0]
    expect[1][[2, 4, 8], :] = blocks[1][[2, 4, 8], :] * r8[1]
    expect[1][9:, :] = blocks[1][[12, 13, 14, 9, 10, 11, 15], :]

    expect_data = backends.FusionTreeData(block_inds, expect, Dtype.complex128,
                                          device=backend.block_backend.default_device)
    expect_tens = SymmetricTensor(expect_data, codomain, domain, backend=backend)

    for func in funcs:
        new_data, new_codomain, new_domain = func(tens, leg=0, levels=levels)
        new_tens = SymmetricTensor(new_data, new_codomain, new_domain, backend=backend)
        assert_tensors_almost_equal(new_tens, expect_tens, eps, eps)

    if move_leg_or_permute_leg == 'permute_leg':
        new_data, new_codomain, new_domain = backend.permute_legs(tens, [1, 0, 2], [5, 4, 3], levels)
        new_tens = SymmetricTensor(new_data, new_codomain, new_domain, backend=backend)
        assert_tensors_almost_equal(new_tens, expect_tens, eps, eps)
    elif move_leg_or_permute_leg == 'move_leg':
        new_tens = move_leg(tens, 1, codomain_pos=0, levels=levels)
        assert_tensors_almost_equal(new_tens, expect_tens, eps, eps)

    # exchange legs 4 and 5 (in domain)
    expect = [zero_block(shp, Dtype.complex128) for shp in shapes]

    for i in [0, 2, 3]:
        expect[i][:, :4] = blocks[i][:, :4] * r8[0]
        expect[i][:, 4:8] = blocks[i][:, 4:8] * r8[1]
        expect[i][:, 8:] = blocks[i][:, 8:]

    expect[1][:, :4] = blocks[1][:, :4] * -1
    expect[1][:, 4:8] = blocks[1][:, 4:8] * r8[0]
    expect[1][:, 8:12] = blocks[1][:, 8:12] * r8[1]
    expect[1][:, 12:16] = blocks[1][:, 12:16] * r8[0]
    expect[1][:, 16:20] = blocks[1][:, 16:20] * r8[1]
    expect[1][:, 20:28] = blocks[1][:, 20:28] * -1
    expect[1][:, 28:] = blocks[1][:, 28:]

    expect_data = backends.FusionTreeData(block_inds, expect, Dtype.complex128,
                                          device=backend.block_backend.default_device)
    expect_domain = TensorProduct([s2, s1, s2])
    expect_tens = SymmetricTensor(expect_data, codomain, expect_domain, backend=backend)

    for func in funcs:
        new_data, new_codomain, new_domain = func(tens, leg=4, levels=levels)
        new_tens = SymmetricTensor(new_data, new_codomain, new_domain, backend=backend)
        assert_tensors_almost_equal(new_tens, expect_tens, eps, eps)

    if move_leg_or_permute_leg == 'permute_leg':
        new_data, new_codomain, new_domain = backend.permute_legs(tens, [0, 1, 2], [4, 5, 3], levels)
        new_tens = SymmetricTensor(new_data, new_codomain, new_domain, backend=backend)
        assert_tensors_almost_equal(new_tens, expect_tens, eps, eps)
    elif move_leg_or_permute_leg == 'move_leg':
        new_tens = move_leg(tens, 5, domain_pos=1, levels=levels)
        assert_tensors_almost_equal(new_tens, expect_tens, eps, eps)

    # exchange legs 1 and 2 (in codomain)
    # we usually use the convention that in the codomain, the two final indices are f, e
    # the F symbols f2 and f1 are chosen such that we use the indices e, f
    # e.g. f2[e, f] = _f_symbol(10, 8, 8, 8, f, e)
    #      f1[e, f] = _f_symbol(8, 10, 8, 8, f, e)
    f2 = np.array([[-.5, -3**.5/2], [3**.5/2, -.5]])
    f1 = f2.T
    csym = sym._c_symbol
    expect = [zero_block(shp, Dtype.complex128) for shp in shapes]

    expect[0][0, :] = blocks[0][0, :] * r8[0]
    expect[0][1, :] = blocks[0][1, :] * r8[1]
    expect[0][[2, 3], :] = blocks[0][[3, 2], :]
    expect[0][4, :] = blocks[0][4, :] * -1
    expect[0][5, :] = blocks[0][5, :]

    v = [blocks[1][i, :] for i in range(7)]
    charges = [c0] + [c1]*4 + [c2, c3]
    mul1 = [0] * 7
    mul2 = [0] * 7
    mul1[2], mul1[4] = 1, 1
    mul2[3], mul2[4] = 1, 1

    for i in range(7):
        w = [csym(c1, c1, c1, c1, charges[i], charges[j])[mul1[i], mul2[i], mul1[j], mul2[j]] for j in range(7)]
        amplitudes = zero_block([7, backend.block_backend.get_shape(expect[1])[1]], Dtype.complex128)
        for j in range(7):
            amplitudes[j, :] = v[j] * w[j]
        expect[1][i, :] = backend.block_backend.sum(amplitudes, ax=0)

    expect[1][7, :] = (blocks[1][9, :] * (f2[0,0]*f2[0,0] + f2[0,1]*f2[1,0])
                       + blocks[1][10, :] * (f2[1,0]*f2[0,0] + f2[1,1]*f2[1,0]))
    expect[1][8, :] = (blocks[1][9, :] * (f2[0,0]*f2[0,1] + f2[0,1]*f2[1,1])
                       + blocks[1][10, :] * (f2[1,0]*f2[0,1] + f2[1,1]*f2[1,1]))
    expect[1][9, :] = (blocks[1][7, :] * (f1[0,0]*f1[0,0] + f1[0,1]*f1[1,0])
                       + blocks[1][8, :] * (f1[1,0]*f1[0,0] + f1[1,1]*f1[1,0]))
    expect[1][10, :] = (blocks[1][7, :] * (f1[0,0]*f1[0,1] + f1[0,1]*f1[1,1])
                        + blocks[1][8, :] * (f1[1,0]*f1[0,1] + f1[1,1]*f1[1,1]))
    expect[1][11, :] = blocks[1][11, :]
    expect[1][12, :] = (blocks[1][12, :] * (f1[0,0]*r8[0]*f2[0,0] + f1[0,1]*r8[1]*f2[1,0])
                        + blocks[1][13, :] * (f1[1,0]*r8[0]*f2[0,0] + f1[1,1]*r8[1]*f2[1,0]))
    expect[1][13, :] = (blocks[1][12, :] * (f1[0,0]*r8[0]*f2[0,1] + f1[0,1]*r8[1]*f2[1,1])
                        + blocks[1][13, :] * (f1[1,0]*r8[0]*f2[0,1] + f1[1,1]*r8[1]*f2[1,1]))
    expect[1][[14, 15], :] = blocks[1][[15, 14], :] * -1

    expect[2][0, :] = (blocks[2][0, :] * (f1[0,0]*r8[0]*f2[0,0] + f1[0,1]*r8[1]*f2[1,0])
                       + blocks[2][1, :] * (f1[1,0]*r8[0]*f2[0,0] + f1[1,1]*r8[1]*f2[1,0]))
    expect[2][1, :] = (blocks[2][0, :] * (f1[0,0]*r8[0]*f2[0,1] + f1[0,1]*r8[1]*f2[1,1])
                       + blocks[2][1, :] * (f1[1,0]*r8[0]*f2[0,1] + f1[1,1]*r8[1]*f2[1,1]))
    expect[2][[2, 3], :] = blocks[2][[3, 2], :] * -1
    expect[2][4, :] = blocks[2][4, :] * -1

    expect[3][0, :] = (blocks[3][0, :] * (f2[0,0]*r8[0]*f1[0,0] + f2[0,1]*r8[1]*f1[1,0])
                       + blocks[3][1, :] * (f2[1,0]*r8[0]*f1[0,0] + f2[1,1]*r8[1]*f1[1,0]))
    expect[3][1, :] = (blocks[3][0, :] * (f2[0,0]*r8[0]*f1[0,1] + f2[0,1]*r8[1]*f1[1,1])
                       + blocks[3][1, :] * (f2[1,0]*r8[0]*f1[0,1] + f2[1,1]*r8[1]*f1[1,1]))
    expect[3][[2, 3], :] = blocks[3][[3, 2], :] * -1
    expect[3][4, :] = blocks[3][4, :] * -1

    expect_data = backends.FusionTreeData(block_inds, expect, Dtype.complex128,
                                          device=backend.block_backend.default_device)
    expect_tens = SymmetricTensor(expect_data, codomain, domain, backend=backend)

    for func in funcs:
        new_data, new_codomain, new_domain = func(tens, leg=1, levels=levels)
        new_tens = SymmetricTensor(new_data, new_codomain, new_domain, backend=backend)
        assert_tensors_almost_equal(new_tens, expect_tens, eps, eps)

    if move_leg_or_permute_leg == 'permute_leg':
        new_data, new_codomain, new_domain = backend.permute_legs(tens, [0, 2, 1], [5, 4, 3], levels)
        new_tens = SymmetricTensor(new_data, new_codomain, new_domain, backend=backend)
        assert_tensors_almost_equal(new_tens, expect_tens, eps, eps)
    elif move_leg_or_permute_leg == 'move_leg':
        new_tens = move_leg(tens, 1, codomain_pos=2, levels=levels)
        assert_tensors_almost_equal(new_tens, expect_tens, eps, eps)

    # exchange legs 3 and 4 (in domain)
    exc = [0, 2, 1, 3]
    exc4, exc8 = [4 + i for i in exc], [8 + i for i in exc]
    exc28, exc32 = [28 + i for i in exc], [32 + i for i in exc]
    expect = [zero_block(shp, Dtype.complex128) for shp in shapes]

    expect[0][:, :4] = blocks[0][:, exc] * r8[0]
    expect[0][:, 4:8] = blocks[0][:, exc4] * r8[1]
    expect[0][:, 8:] = blocks[0][:, exc8] * -1

    v = [blocks[1][:, [4*i + j for j in exc]] for i in range(7)]
    for i in range(7):
        w = [csym(c1, c1, c1, c1, charges[i], charges[j])[mul1[i], mul2[i], mul1[j], mul2[j]] for j in range(7)]
        amplitudes = zero_block([backend.block_backend.get_shape(expect[1])[0], 4], Dtype.complex128)
        for j in range(7):
            amplitudes += v[j] * w[j]
        expect[1][:, 4*i:4*(i+1)] = amplitudes

    expect[1][:, 28:32] = (blocks[1][:, exc28] * (f1[0,0]*r8[0]*f2[0,0] + f1[0,1]*r8[1]*f2[1,0])
                          + blocks[1][:, exc32] * (f1[1,0]*r8[0]*f2[0,0] + f1[1,1]*r8[1]*f2[1,0]))
    expect[1][:, 32:] = (blocks[1][:, exc28] * (f1[0,0]*r8[0]*f2[0,1] + f1[0,1]*r8[1]*f2[1,1])
                        + blocks[1][:, exc32] * (f1[1,0]*r8[0]*f2[0,1] + f1[1,1]*r8[1]*f2[1,1]))

    expect[2][:, :4] = (blocks[2][:, exc] * (f1[0,0]*r8[0]*f2[0,0] + f1[0,1]*r8[1]*f2[1,0])
                       + blocks[2][:, exc4] * (f1[1,0]*r8[0]*f2[0,0] + f1[1,1]*r8[1]*f2[1,0]))
    expect[2][:, 4:8] = (blocks[2][:, exc] * (f1[0,0]*r8[0]*f2[0,1] + f1[0,1]*r8[1]*f2[1,1])
                        + blocks[2][:, exc4] * (f1[1,0]*r8[0]*f2[0,1] + f1[1,1]*r8[1]*f2[1,1]))
    expect[2][:, 8:] = blocks[2][:, exc8] * -1

    expect[3][:, :4] = (blocks[3][:, exc] * (f2[0,0]*r8[0]*f1[0,0] + f2[0,1]*r8[1]*f1[1,0])
                       + blocks[3][:, exc4] * (f2[1,0]*r8[0]*f1[0,0] + f2[1,1]*r8[1]*f1[1,0]))
    expect[3][:, 4:8] = (blocks[3][:, exc] * (f2[0,0]*r8[0]*f1[0,1] + f2[0,1]*r8[1]*f1[1,1])
                        + blocks[3][:, exc4] * (f2[1,0]*r8[0]*f1[0,1] + f2[1,1]*r8[1]*f1[1,1]))
    expect[3][:, 8:] = blocks[3][:, exc8] * -1

    expect_data = backends.FusionTreeData(block_inds, expect, Dtype.complex128,
                                          device=backend.block_backend.default_device)
    expect_tens = SymmetricTensor(expect_data, codomain, domain, backend=backend)

    for func in funcs:
        new_data, new_codomain, new_domain = func(tens, leg=3, levels=levels)
        new_tens = SymmetricTensor(new_data, new_codomain, new_domain, backend=backend)
        assert_tensors_almost_equal(new_tens, expect_tens, eps, eps)

    if move_leg_or_permute_leg == 'permute_leg':
        new_data, new_codomain, new_domain = backend.permute_legs(tens, [0, 1, 2], [5, 3, 4], levels)
        new_tens = SymmetricTensor(new_data, new_codomain, new_domain, backend=backend)
        assert_tensors_almost_equal(new_tens, expect_tens, eps, eps)
    elif move_leg_or_permute_leg == 'move_leg':
        new_tens = move_leg(tens, 4, domain_pos=2, levels=levels)
        assert_tensors_almost_equal(new_tens, expect_tens, eps, eps)

    # braid 4 times == trivial
    assert_repeated_braids_trivial(tens, funcs, levels, repeat=4, eps=eps)

    # braid clockwise and then counter-clockwise == trivial
    assert_clockwise_counterclockwise_trivial(tens, funcs, levels, eps=eps)

    # rescaling axes and then braiding == braiding and then rescaling axes
    assert_braiding_and_scale_axis_commutation(tens, funcs, levels, eps=eps)

    # do and undo sequence of braids == trivial (may include b symbols)
    for _ in range(2):
        assert_clockwise_counterclockwise_trivial_long_range(tens, move_leg_or_permute_leg, eps, np_random)


@pytest.mark.slow  # TODO can we speed it up?
def test_b_symbol_fibonacci_anyons(block_backend: str, np_random: np.random.Generator):
    move_leg_or_permute_leg = np_random.choice(['move_leg', 'permute_leg'])
    print('use ' + move_leg_or_permute_leg)
    multiple = np_random.choice([True, False])
    backend = get_backend('fusion_tree', block_backend)
    funcs = [cross_check_single_b_symbol, apply_single_b_symbol]
    zero_block = backend.block_backend.zeros
    eps = 1.e-14
    sym = fibonacci_anyon_category
    s1 = ElementarySpace(sym, [[1]], [1])  # only tau
    s2 = ElementarySpace(sym, [[0], [1]], [1, 1])  # 1 and tau
    s3 = ElementarySpace(sym, [[0], [1]], [2, 3])  # 1 and tau

    # tensor with single leg in codomain; bend down
    codomain = TensorProduct([s2])
    domain = TensorProduct([], symmetry=sym)

    block_inds = np.array([[0, 0]])
    blocks = [backend.block_backend.random_uniform((1, 1), Dtype.complex128)]
    data = backends.FusionTreeData(block_inds, blocks, Dtype.complex128,
                                   device=backend.block_backend.default_device)
    tens = SymmetricTensor(data, codomain, domain, backend=backend)

    expect_codomain = TensorProduct([], symmetry=sym)
    expect_domain = TensorProduct([s2.dual])
    expect_tens = SymmetricTensor(data, expect_codomain, expect_domain, backend=backend)

    for func in funcs:
        new_data, new_codomain, new_domain = func(tens, False)
        new_tens = SymmetricTensor(new_data, new_codomain, new_domain, backend=backend)
        assert_tensors_almost_equal(new_tens, expect_tens, eps, eps)

    if move_leg_or_permute_leg == 'permute_leg':
        new_data, new_codomain, new_domain = backend.permute_legs(tens, [], [0], None)
        new_tens = SymmetricTensor(new_data, new_codomain, new_domain, backend=backend)
        assert_tensors_almost_equal(new_tens, expect_tens, eps, eps)
    elif move_leg_or_permute_leg == 'move_leg':
        new_tens = move_leg(tens, 0, domain_pos=0, levels=None)
        assert_tensors_almost_equal(new_tens, expect_tens, eps, eps)

    # tensor with single leg in domain; bend up
    codomain = TensorProduct([], symmetry=sym)
    domain = TensorProduct([s3])

    block_inds = np.array([[0,0]])
    blocks = [backend.block_backend.random_uniform((1, 2), Dtype.complex128)]
    data = backends.FusionTreeData(block_inds, blocks, Dtype.complex128,
                                   device=backend.block_backend.default_device)
    tens = SymmetricTensor(data, codomain, domain, backend=backend)

    expect = [backend.block_backend.reshape(blocks[0], (2, 1))]
    expect_data = backends.FusionTreeData(block_inds, expect, Dtype.complex128,
                                          device=backend.block_backend.default_device)
    expect_codomain = TensorProduct([s3.dual])
    expect_domain = TensorProduct([], symmetry=sym)
    expect_tens = SymmetricTensor(expect_data, expect_codomain, expect_domain, backend=backend)

    for func in funcs:
        new_data, new_codomain, new_domain = func(tens, True)
        new_tens = SymmetricTensor(new_data, new_codomain, new_domain, backend=backend)
        assert_tensors_almost_equal(new_tens, expect_tens, eps, eps)

    if move_leg_or_permute_leg == 'permute_leg':
        new_data, new_codomain, new_domain = backend.permute_legs(tens, [0], [], None)
        new_tens = SymmetricTensor(new_data, new_codomain, new_domain, backend=backend)
        assert_tensors_almost_equal(new_tens, expect_tens, eps, eps)
    elif move_leg_or_permute_leg == 'move_leg':
        new_tens = move_leg(tens, 0, codomain_pos=0, levels=None)
        assert_tensors_almost_equal(new_tens, expect_tens, eps, eps)

    # more complicated tensor
    codomain = TensorProduct([s2, s1, s1])
    domain = TensorProduct([s2, s1, s2])

    block_inds = np.array([[0,0], [1,1]])
    blocks = [backend.block_backend.random_uniform((2, 3), Dtype.complex128),
              backend.block_backend.random_uniform((3, 5), Dtype.complex128)]
    data = backends.FusionTreeData(block_inds, blocks, Dtype.complex128,
                                   device=backend.block_backend.default_device)
    tens = SymmetricTensor(data, codomain, domain, backend=backend)

    # bend up
    phi = (1 + 5**0.5) / 2
    expect = [zero_block([5, 1], Dtype.complex128), zero_block([8, 2], Dtype.complex128)]

    expect[0][0, 0] = blocks[0][0, 1] # (0, 0, 0) = (a, b, c) as in _b_symbol(a, b, c)
    expect[0][1, 0] = blocks[1][0, 3] * phi**0.5 # (0, 1, 1)
    expect[0][2, 0] = blocks[0][1, 1] # (0, 0, 0)
    expect[0][3, 0] = blocks[1][1, 3] * phi**0.5 # (0, 1, 1)
    expect[0][4, 0] = blocks[1][2, 3] * phi**0.5 # (0, 1, 1)

    expect[1][0, 0] = blocks[1][0, 0] # (1, 0, 1)
    expect[1][1, 0] = blocks[0][0, 0] * phi**-0.5 # (1, 1, 0)
    expect[1][2, 0] = blocks[1][0, 1] # (1, 1, 1)
    expect[1][3, 0] = blocks[1][1, 0] # (1, 0, 1)
    expect[1][4, 0] = blocks[1][2, 0] # (1, 0, 1)
    expect[1][5, 0] = blocks[1][1, 1] # (1, 1, 1)
    expect[1][6, 0] = blocks[0][1, 0] * phi**-0.5 # (1, 1, 0)
    expect[1][7, 0] = blocks[1][2, 1] # (1, 1, 1)

    expect[1][0, 1] = blocks[1][0, 2] # (1, 0, 1)
    expect[1][1, 1] = blocks[0][0, 2] * phi**-0.5 # (1, 1, 0)
    expect[1][2, 1] = blocks[1][0, 4] # (1, 1, 1)
    expect[1][3, 1] = blocks[1][1, 2] # (1, 0, 1)
    expect[1][4, 1] = blocks[1][2, 2] # (1, 0, 1)
    expect[1][5, 1] = blocks[1][1, 4] # (1, 1, 1)
    expect[1][6, 1] = blocks[0][1, 2] * phi**-0.5 # (1, 1, 0)
    expect[1][7, 1] = blocks[1][2, 4] # (1, 1, 1)

    expect_data = backends.FusionTreeData(block_inds, expect, Dtype.complex128,
                                          device=backend.block_backend.default_device)
    expect_codomain = TensorProduct([s2, s1, s1, s2.dual])
    expect_domain = TensorProduct([s2, s1])
    expect_tens = SymmetricTensor(expect_data, expect_codomain, expect_domain, backend=backend)

    for func in funcs:
        new_data, new_codomain, new_domain = func(tens, True)
        new_tens = SymmetricTensor(new_data, new_codomain, new_domain, backend=backend)
        assert_tensors_almost_equal(new_tens, expect_tens, eps, eps)

    if move_leg_or_permute_leg == 'permute_leg':
        new_data, new_codomain, new_domain = backend.permute_legs(tens, [0, 1, 2, 3], [5, 4], None)
        new_tens = SymmetricTensor(new_data, new_codomain, new_domain, backend=backend)
        assert_tensors_almost_equal(new_tens, expect_tens, eps, eps)
    elif move_leg_or_permute_leg == 'move_leg':
        new_tens = move_leg(tens, 3, codomain_pos=3, levels=None)
        assert_tensors_almost_equal(new_tens, expect_tens, eps, eps)

    # bend down
    expect = [zero_block([1, 5], Dtype.complex128), zero_block([2, 8], Dtype.complex128)]

    expect[0][0, 0] = blocks[1][1, 0] * phi**0.5 # (0, 1, 1)
    expect[0][0, 1] = blocks[1][1, 1] * phi**0.5 # (0, 1, 1)
    expect[0][0, 2] = blocks[1][1, 2] * phi**0.5 # (0, 1, 1)
    expect[0][0, 3] = blocks[1][1, 3] * phi**0.5 # (0, 1, 1)
    expect[0][0, 4] = blocks[1][1, 4] * phi**0.5 # (0, 1, 1)

    expect[1][0, 0] = blocks[1][0, 0] # (1, 1, 1)
    expect[1][0, 1] = blocks[0][0, 0] * phi**-0.5 # (1, 1, 0)
    expect[1][0, 2] = blocks[1][0, 1] # (1, 1, 1)
    expect[1][0, 3] = blocks[0][0, 1] * phi**-0.5 # (1, 1, 0)
    expect[1][0, 4] = blocks[1][0, 2] # (1, 1, 1)
    expect[1][0, 5] = blocks[1][0, 3] # (1, 1, 1)
    expect[1][0, 6] = blocks[0][0, 2] * phi**-0.5 # (1, 1, 0)
    expect[1][0, 7] = blocks[1][0, 4] # (1, 1, 1)

    expect[1][1, 0] = blocks[1][2, 0] # (1, 1, 1)
    expect[1][1, 1] = blocks[0][1, 0] * phi**-0.5 # (1, 1, 0)
    expect[1][1, 2] = blocks[1][2, 1] # (1, 1, 1)
    expect[1][1, 3] = blocks[0][1, 1] * phi**-0.5 # (1, 1, 0)
    expect[1][1, 4] = blocks[1][2, 2] # (1, 1, 1)
    expect[1][1, 5] = blocks[1][2, 3] # (1, 1, 1)
    expect[1][1, 6] = blocks[0][1, 2] * phi**-0.5 # (1, 1, 0)
    expect[1][1, 7] = blocks[1][2, 4] # (1, 1, 1)

    expect_data = backends.FusionTreeData(block_inds, expect, Dtype.complex128,
                                          device=backend.block_backend.default_device)
    expect_codomain = TensorProduct([s2, s1])
    expect_domain = TensorProduct([s2, s1, s2, s1.dual])
    expect_tens = SymmetricTensor(expect_data, expect_codomain, expect_domain, backend=backend)

    for func in funcs:
        new_data, new_codomain, new_domain = func(tens, False)
        new_tens = SymmetricTensor(new_data, new_codomain, new_domain, backend=backend)
        assert_tensors_almost_equal(new_tens, expect_tens, eps, eps)

    if move_leg_or_permute_leg == 'permute_leg':
        new_data, new_codomain, new_domain = backend.permute_legs(tens, [0, 1], [5, 4, 3, 2], None)
        new_tens = SymmetricTensor(new_data, new_codomain, new_domain, backend=backend)
        assert_tensors_almost_equal(new_tens, expect_tens, eps, eps)
    elif move_leg_or_permute_leg == 'move_leg':
        new_tens = move_leg(tens, 2, domain_pos=3, levels=None)
        assert_tensors_almost_equal(new_tens, expect_tens, eps, eps)

    spaces = [TensorProduct([], symmetry=sym), TensorProduct([s2]), TensorProduct([s3]),
              TensorProduct([s1, s3]), TensorProduct([s2, s3]), TensorProduct([s3, s1, s3, s2])]
    # bend up and down again (and vice versa) == trivial
    assert_bending_up_and_down_trivial(spaces, spaces, funcs, backend, multiple=multiple, eps=eps)

    # rescaling axis and then bending == bending and then rescaling axis
    assert_bending_and_scale_axis_commutation(tens, funcs, eps)


@pytest.mark.slow  # TODO can we speed it up?
def test_b_symbol_product_sym(block_backend: str, np_random: np.random.Generator):
    move_leg_or_permute_leg = np_random.choice(['move_leg', 'permute_leg'])
    print('use ' + move_leg_or_permute_leg)
    multiple = np_random.choice([True, False])
    backend = get_backend('fusion_tree', block_backend)
    funcs = [cross_check_single_b_symbol, apply_single_b_symbol]
    perm_axes = backend.block_backend.permute_axes
    reshape = backend.block_backend.reshape
    zero_block = backend.block_backend.zeros
    eps = 1.e-14
    sym = ProductSymmetry([fibonacci_anyon_category, SU2Symmetry()])
    s1 = ElementarySpace(sym, [[1, 1]], [1])  # only (tau, spin-1/2)
    s2 = ElementarySpace(sym, [[0, 0], [1, 1]], [1, 2])  # (1, spin-0) and (tau, spin-1/2)
    s3 = ElementarySpace(sym, [[0, 0], [1, 1], [1, 2]], [1, 2, 2])  # (1, spin-0), (tau, spin-1/2) and (tau, spin-1)

    # tensor with two legs in domain; bend up
    codomain = TensorProduct([], symmetry=sym)
    domain = TensorProduct([s2, s3])

    block_inds = np.array([[0, 0]])
    blocks = [backend.block_backend.random_uniform((1, 5), Dtype.complex128)]
    data = backends.FusionTreeData(block_inds, blocks, Dtype.complex128,
                                   device=backend.block_backend.default_device)
    tens = SymmetricTensor(data, codomain, domain, backend=backend)

    expect_block_inds = np.array([[0, 0], [1, 1]])
    expect = [zero_block([1, 1], Dtype.complex128), zero_block([2, 2], Dtype.complex128)]

    expect[0][0, 0] = blocks[0][0, 0]  # ([0, 0], [0, 0], [0, 0]) = (a, b, c) as in _b_symbol(a, b, c)
    expect[1][:, :] = perm_axes(reshape(blocks[0][0, 1:], (2, 2)), [1, 0])
    expect[1][:, :] *= sym.inv_sqrt_qdim(np.array([1, 1])) * -1  # ([1, 1], [1, 1], [0, 0])

    expect_data = backends.FusionTreeData(expect_block_inds, expect, Dtype.complex128,
                                          device=backend.block_backend.default_device)
    expect_codomain = TensorProduct([s3.dual])
    expect_domain = TensorProduct([s2])
    expect_tens = SymmetricTensor(expect_data, expect_codomain, expect_domain, backend=backend)

    for func in funcs:
        new_data, new_codomain, new_domain = func(tens, True)
        new_tens = SymmetricTensor(new_data, new_codomain, new_domain, backend=backend)
        assert_tensors_almost_equal(new_tens, expect_tens, eps, eps)

    if move_leg_or_permute_leg == 'permute_leg':
        new_data, new_codomain, new_domain = backend.permute_legs(tens, [0], [1], None)
        new_tens = SymmetricTensor(new_data, new_codomain, new_domain, backend=backend)
        assert_tensors_almost_equal(new_tens, expect_tens, eps, eps)
    elif move_leg_or_permute_leg == 'move_leg':
        new_tens = move_leg(tens, 0, codomain_pos=0, levels=None)
        assert_tensors_almost_equal(new_tens, expect_tens, eps, eps)

    # tensor with two legs in codomain, two leg in domain; bend down
    codomain = TensorProduct([s1, s3])
    domain = TensorProduct([s2, s3])

    # charges [0, 0], [1, 0], [0, 1], [1, 1], [0, 2], [1, 2], [0, 3], [1, 3]
    block_inds = np.array([[i, i] for i in range(8)])
    shapes = [(2, 5), (2, 4), (2, 4), (3, 8), (2, 4), (2, 6), (2, 4), (2, 4)]
    blocks = [backend.block_backend.random_uniform(shp, Dtype.complex128) for shp in shapes]
    data = backends.FusionTreeData(block_inds, blocks, Dtype.complex128,
                                   device=backend.block_backend.default_device)
    tens = SymmetricTensor(data, codomain, domain, backend=backend)

    expect_block_inds = np.array([[0, 2]])
    expect = [zero_block([1, 86], Dtype.complex128)]

    expect[0][0, :2] = blocks[0][:, 0]
    expect[0][0, 18:26] = reshape(perm_axes(blocks[0][:, 1:], [1, 0]), (1, 8))
    expect[0][0, np.r_[:2, 18:26]] *= sym.inv_sqrt_qdim(np.array([1, 1])) * -1
    # ([1, 1], [1, 1], [0, 0])

    expect[0][0, 34:42] = reshape(perm_axes(blocks[1][:, :], [1, 0]), (1, 8))
    expect[0][0, 34:42] *= sym.inv_sqrt_qdim(np.array([1, 1])) * sym.sqrt_qdim(np.array([1, 0])) * -1
    # ([1, 1], [1, 1], [1, 0])

    expect[0][0, 54:62] = reshape(perm_axes(blocks[2][:, :], [1, 0]), (1, 8))
    expect[0][0, 54:62] *= sym.inv_sqrt_qdim(np.array([1, 1])) * sym.sqrt_qdim(np.array([0, 1])) * -1
    # ([1, 1], [1, 2], [0, 1])

    expect[0][0, 2:4] = blocks[3][0, :2]
    expect[0][0, 12:14] = blocks[3][0, 2:4]
    expect[0][0, 50:54] = blocks[3][0, 4:]
    # ([1, 1], [0, 0], [1, 1])

    expect[0][0, 4:8] = reshape(perm_axes(blocks[3][1:, :2], [1, 0]), (1, 4)) * -1
    expect[0][0, 14:18] = reshape(perm_axes(blocks[3][1:, 2:4], [1, 0]), (1, 4)) * -1
    expect[0][0, 70:78] = reshape(perm_axes(blocks[3][1:, 4:], [1, 0]), (1, 8)) * -1
    # ([1, 1], [1, 2], [1, 1])

    expect[0][0, 26:34] = reshape(perm_axes(blocks[4][:, :], [1, 0]), (1, 8))
    expect[0][0, 26:34] *= sym.inv_sqrt_qdim(np.array([1, 1])) * sym.sqrt_qdim(np.array([0, 2]))
    # ([1, 1], [1, 1], [0, 2])

    expect[0][0, 8:12] = reshape(perm_axes(blocks[5][:, :2], [1, 0]), (1, 4))
    expect[0][0, 42:50] = reshape(perm_axes(blocks[5][:, 2:], [1, 0]), (1, 8))
    expect[0][0, np.r_[8:12, 42:50]] *= sym.inv_sqrt_qdim(np.array([1, 1])) * sym.sqrt_qdim(np.array([1, 2]))
    # ([1, 1], [1, 1], [1, 2])

    expect[0][0, 62:70] = reshape(perm_axes(blocks[6][:, :], [1, 0]), (1, 8))
    expect[0][0, 62:70] *= sym.inv_sqrt_qdim(np.array([1, 1])) * sym.sqrt_qdim(np.array([0, 3]))
    # ([1, 1], [1, 2], [0, 3])

    expect[0][0, 78:] = reshape(perm_axes(blocks[7][:, :], [1, 0]), (1, 8))
    expect[0][0, 78:] *= sym.inv_sqrt_qdim(np.array([1, 1])) * sym.sqrt_qdim(np.array([1, 3]))
    # ([1, 1], [1, 2], [1, 3])

    expect_data = backends.FusionTreeData(expect_block_inds, expect, Dtype.complex128,
                                          device=backend.block_backend.default_device)
    expect_codomain = TensorProduct([s1])
    expect_domain = TensorProduct([s2, s3, s3.dual])
    expect_tens = SymmetricTensor(expect_data, expect_codomain, expect_domain, backend=backend)

    for func in funcs:
        new_data, new_codomain, new_domain = func(tens, False)
        new_tens = SymmetricTensor(new_data, new_codomain, new_domain, backend=backend)
        assert_tensors_almost_equal(new_tens, expect_tens, eps, eps)

    if move_leg_or_permute_leg == 'permute_leg':
        new_data, new_codomain, new_domain = backend.permute_legs(tens, [0], [3, 2, 1], None)
        new_tens = SymmetricTensor(new_data, new_codomain, new_domain, backend=backend)
        assert_tensors_almost_equal(new_tens, expect_tens, eps, eps)
    elif move_leg_or_permute_leg == 'move_leg':
        new_tens = move_leg(tens, 1, domain_pos=2, levels=None)
        assert_tensors_almost_equal(new_tens, expect_tens, eps, eps)

    # similar tensor, replace one sector with its dual (Frobenius-Schur is now relevant); bend up
    codomain = TensorProduct([s1, s3])
    domain = TensorProduct([s2, s3.dual])

    blocks = [backend.block_backend.random_uniform(shp, Dtype.complex128) for shp in shapes]
    data = backends.FusionTreeData(block_inds, blocks, Dtype.complex128,
                                   device=backend.block_backend.default_device)
    tens = SymmetricTensor(data, codomain, domain, backend=backend)

    expect_block_inds = np.array([[0, 0], [3, 1]])
    expect = [zero_block([12, 1], Dtype.complex128), zero_block([37, 2], Dtype.complex128)]

    expect[0][2:4, 0] = blocks[0][:, 0] # ([0, 0], [0, 0], [0, 0])

    expect[1][3:7, :] = reshape(perm_axes(reshape(blocks[0][:, 1:], (2, 2, 2)), [0, 2, 1]), (4, 2))
    expect[1][3:7, :] *= sym.inv_sqrt_qdim(np.array([1, 1]))  # ([1, 1], [1, 1], [0, 0])

    expect[1][11:15, :] = reshape(perm_axes(reshape(blocks[1][:, :], (2, 2, 2)), [0, 2, 1]), (4, 2))
    expect[1][11:15, :] *= sym.inv_sqrt_qdim(np.array([1, 1])) * sym.sqrt_qdim(np.array([1, 0]))
    # ([1, 1], [1, 1], [1, 0])

    expect[1][21:25, :] = reshape(perm_axes(reshape(blocks[2][:, :], (2, 2, 2)), [0, 2, 1]), (4, 2))
    expect[1][21:25, :] *= sym.inv_sqrt_qdim(np.array([1, 1])) * sym.sqrt_qdim(np.array([0, 1])) * -1
    # ([1, 1], [1, 2], [0, 1])

    expect[0][:2, 0] = blocks[3][0, :2]
    expect[0][8:, :] = reshape(blocks[3][1:, :2], (4, 1))
    expect[0][np.r_[:2, 8:12], 0] *= sym.sqrt_qdim(np.array([1, 1])) * -1
    # ([0, 0], [1, 1], [1, 1])

    expect[1][0, :] = blocks[3][0, 2:4]
    expect[1][19:21, :] = blocks[3][1:, 2:4]
    # ([1, 1], [0, 0], [1, 1])

    expect[1][1:3, :] = perm_axes(reshape(blocks[3][0, 4:], (2, 2)), [1, 0]) * -1
    expect[1][29:33, :] = reshape(perm_axes(reshape(blocks[3][1:, 4:], (2, 2, 2)), [0, 2, 1]), (4, 2)) * -1
    # ([1, 1], [1, 2], [1, 1])

    expect[1][7:11, :] = reshape(perm_axes(reshape(blocks[4][:, :], (2, 2, 2)), [0, 2, 1]), (4, 2))
    expect[1][7:11, :] *= sym.inv_sqrt_qdim(np.array([1, 1])) * sym.sqrt_qdim(np.array([0, 2])) * -1
    # ([1, 1], [1, 1], [0, 2])

    expect[0][4:8, :] = reshape(blocks[5][:, :2], (4, 1))
    expect[0][4:8, :] *= sym.sqrt_qdim(np.array([1, 2]))
    # ([0, 0], [1, 2], [1, 2])

    expect[1][15:19, :] = reshape(perm_axes(reshape(blocks[5][:, 2:], (2, 2, 2)), [0, 2, 1]), (4, 2))
    expect[1][15:19, :] *= sym.inv_sqrt_qdim(np.array([1, 1])) * sym.sqrt_qdim(np.array([1, 2])) * -1
    # ([1, 1], [1, 1], [1, 2])

    expect[1][25:29, :] = reshape(perm_axes(reshape(blocks[6][:, :], (2, 2, 2)), [0, 2, 1]), (4, 2))
    expect[1][25:29, :] *= sym.inv_sqrt_qdim(np.array([1, 1])) * sym.sqrt_qdim(np.array([0, 3]))
    # ([1, 1], [1, 2], [0, 3])

    expect[1][33:, :] = reshape(perm_axes(reshape(blocks[7][:, :], (2, 2, 2)), [0, 2, 1]), (4, 2))
    expect[1][33:, :] *= sym.inv_sqrt_qdim(np.array([1, 1])) * sym.sqrt_qdim(np.array([1, 3]))
    # ([1, 1], [1, 2], [1, 3])

    expect_data = backends.FusionTreeData(expect_block_inds, expect, Dtype.complex128,
                                          device=backend.block_backend.default_device)
    expect_codomain = TensorProduct([s1, s3, s3])
    expect_domain = TensorProduct([s2])
    expect_tens = SymmetricTensor(expect_data, expect_codomain, expect_domain, backend=backend)

    for func in funcs:
        new_data, new_codomain, new_domain = func(tens, True)
        new_tens = SymmetricTensor(new_data, new_codomain, new_domain, backend=backend)
        assert_tensors_almost_equal(new_tens, expect_tens, eps, eps)

    if move_leg_or_permute_leg == 'permute_leg':
        new_data, new_codomain, new_domain = backend.permute_legs(tens, [0, 1, 2], [3], None)
        new_tens = SymmetricTensor(new_data, new_codomain, new_domain, backend=backend)
        assert_tensors_almost_equal(new_tens, expect_tens, eps, eps)
    elif move_leg_or_permute_leg == 'move_leg':
        new_tens = move_leg(tens, 2, codomain_pos=2, levels=None)
        assert_tensors_almost_equal(new_tens, expect_tens, eps, eps)

    spaces = [TensorProduct([], symmetry=sym), TensorProduct([s2]), TensorProduct([s3.dual]),
              TensorProduct([s1, s3]), TensorProduct([s2, s3.dual]), TensorProduct([s1, s3, s2.dual])]
    # bend up and down again (and vice versa) == trivial
    assert_bending_up_and_down_trivial(spaces, spaces, funcs, backend, multiple=multiple, eps=eps)

    # rescaling axis and then bending == bending and then rescaling axis
    assert_bending_and_scale_axis_commutation(tens, funcs, eps)


@pytest.mark.slow  # TODO can we speed it up?
def test_b_symbol_su3_3(block_backend: str, np_random: np.random.Generator):
    move_leg_or_permute_leg = np_random.choice(['move_leg', 'permute_leg'])
    print('use ' + move_leg_or_permute_leg)
    multiple = np_random.choice([True, False])
    backend = get_backend('fusion_tree', block_backend)
    funcs = [cross_check_single_b_symbol, apply_single_b_symbol]
    perm_axes = backend.block_backend.permute_axes
    reshape = backend.block_backend.reshape
    zero_block = backend.block_backend.zeros
    eps = 1.e-14
    sym = SU3_3AnyonCategory()
    s1 = ElementarySpace(sym, [[1], [2]], [1, 1])  # 8 and 10
    s2 = ElementarySpace(sym, [[1]], [2])  # 8 with multiplicity 2
    s3 = ElementarySpace(sym, [[0], [1], [3]], [1, 2, 3])  # 1, 8, 10-
    qdim8 = sym.sqrt_qdim(np.array([1]))  # sqrt of qdim of charge 8
    # when multiplying with qdims (from the b symbols), only 8 is relevant since all other qdim are 1
    # the b symbols are diagonal in the multiplicity index

    # tensor with two legs in codomain; bend down
    codomain = TensorProduct([s1, s3])
    domain = TensorProduct([], symmetry=sym)

    block_inds = np.array([[0, 0]])
    blocks = [backend.block_backend.random_uniform((5, 1), Dtype.complex128)]
    data = backends.FusionTreeData(block_inds, blocks, Dtype.complex128,
                                   device=backend.block_backend.default_device)
    tens = SymmetricTensor(data, codomain, domain, backend=backend)

    expect_block_inds = np.array([[0, 1], [1, 2]])
    expect = [zero_block([1, 2], Dtype.complex128), zero_block([1, 3], Dtype.complex128)]

    expect[0][0, :] = blocks[0][:2, 0] / qdim8  # (8, 8, 1) = (a, b, c) as in _b_symbol(a, b, c)
    expect[1][0, :] = blocks[0][2:, 0]  # (10, 10-, 1)

    expect_data = backends.FusionTreeData(expect_block_inds, expect, Dtype.complex128,
                                          device=backend.block_backend.default_device)
    expect_codomain = TensorProduct([s1])
    expect_domain = TensorProduct([s3.dual])
    expect_tens = SymmetricTensor(expect_data, expect_codomain, expect_domain, backend=backend)

    for func in funcs:
        new_data, new_codomain, new_domain = func(tens, False)
        new_tens = SymmetricTensor(new_data, new_codomain, new_domain, backend=backend)
        assert_tensors_almost_equal(new_tens, expect_tens, eps, eps)

    if move_leg_or_permute_leg == 'permute_leg':
        new_data, new_codomain, new_domain = backend.permute_legs(tens, [0], [1], None)
        new_tens = SymmetricTensor(new_data, new_codomain, new_domain, backend=backend)
        assert_tensors_almost_equal(new_tens, expect_tens, eps, eps)
    elif move_leg_or_permute_leg == 'move_leg':
        new_tens = move_leg(tens, 1, domain_pos=0, levels=None)
        assert_tensors_almost_equal(new_tens, expect_tens, eps, eps)

    # tensor with two legs in codomain, one leg in domain; bend down
    codomain = TensorProduct([s1, s3])
    domain = TensorProduct([s2])

    block_inds = np.array([[1, 0]])
    blocks = [backend.block_backend.random_uniform((10, 2), Dtype.complex128)]
    data = backends.FusionTreeData(block_inds, blocks, Dtype.complex128,
                                   device=backend.block_backend.default_device)
    tens = SymmetricTensor(data, codomain, domain, backend=backend)

    expect_block_inds = np.array([[0, 1], [1, 2]])
    expect = [zero_block([1, 16], Dtype.complex128), zero_block([1, 4], Dtype.complex128)]

    expect[0][0, :2] = blocks[0][0, :]  # (8, 1, 8)
    expect[0][0, 2:6] = reshape(perm_axes(blocks[0][1:3, :], [1, 0]), (1, 4))  # (8, 8, 8)
    expect[0][0, 6:10] = reshape(perm_axes(blocks[0][3:5, :], [1, 0]), (1, 4))  # (8, 8, 8)
    expect[0][0, 10:] = reshape(perm_axes(blocks[0][5:8, :], [1, 0]), (1, 6)) * -1 # (8, 10-, 8)

    expect[1][0, :] = reshape(perm_axes(blocks[0][8:, :], [1, 0]), (1, 4)) * qdim8 * -1  # (10, 8, 8)

    expect_data = backends.FusionTreeData(expect_block_inds, expect, Dtype.complex128,
                                          device=backend.block_backend.default_device)
    expect_codomain = TensorProduct([s1])
    expect_domain = TensorProduct([s2, s3.dual])
    expect_tens = SymmetricTensor(expect_data, expect_codomain, expect_domain, backend=backend)

    for func in funcs:
        new_data, new_codomain, new_domain = func(tens, False)
        new_tens = SymmetricTensor(new_data, new_codomain, new_domain, backend=backend)
        assert_tensors_almost_equal(new_tens, expect_tens, eps, eps)

    if move_leg_or_permute_leg == 'permute_leg':
        new_data, new_codomain, new_domain = backend.permute_legs(tens, [0], [2, 1], None)
        new_tens = SymmetricTensor(new_data, new_codomain, new_domain, backend=backend)
        assert_tensors_almost_equal(new_tens, expect_tens, eps, eps)
    elif move_leg_or_permute_leg == 'move_leg':
        new_tens = move_leg(tens, 1, domain_pos=1, levels=None)
        assert_tensors_almost_equal(new_tens, expect_tens, eps, eps)

    # same tensor, bend up
    expect_block_inds = np.array([[0, 0]])
    expect = [zero_block([20, 1], Dtype.complex128)]

    expect[0][:2, 0] = blocks[0][0, :] * qdim8  # (1, 8, 8)
    expect[0][2:6, :] = reshape(blocks[0][1:3, :], (4, 1)) * qdim8  # (1, 8, 8)
    expect[0][6:10, :] = reshape(blocks[0][3:5, :], (4, 1)) * qdim8  # (1, 8, 8)
    expect[0][10:16, :] = reshape(blocks[0][5:8, :], (6, 1)) * qdim8  # (1, 8, 8)
    expect[0][16:, :] = reshape(blocks[0][8:, :], (4, 1)) * qdim8  # (1, 8, 8)

    expect_data = backends.FusionTreeData(expect_block_inds, expect, Dtype.complex128,
                                          device=backend.block_backend.default_device)
    expect_codomain = TensorProduct([s1, s3, s2.dual])
    expect_domain = TensorProduct([], symmetry=sym)
    expect_tens = SymmetricTensor(expect_data, expect_codomain, expect_domain, backend=backend)

    for func in funcs:
        new_data, new_codomain, new_domain = func(tens, True)
        new_tens = SymmetricTensor(new_data, new_codomain, new_domain, backend=backend)
        assert_tensors_almost_equal(new_tens, expect_tens, eps, eps)

    if move_leg_or_permute_leg == 'permute_leg':
        new_data, new_codomain, new_domain = backend.permute_legs(tens, [0, 1, 2], [], None)
        new_tens = SymmetricTensor(new_data, new_codomain, new_domain, backend=backend)
        assert_tensors_almost_equal(new_tens, expect_tens, eps, eps)
    elif move_leg_or_permute_leg == 'move_leg':
        new_tens = move_leg(tens, 2, codomain_pos=2, levels=None)
        assert_tensors_almost_equal(new_tens, expect_tens, eps, eps)

    # more complicated tensor, bend down
    codomain = TensorProduct([s1, s2, s2])
    domain = TensorProduct([s2, s3])

    block_inds = np.array([[i, i] for i in range(4)])
    shapes = [(12, 4), (36, 16), (12, 4), (12, 4)]
    blocks = [backend.block_backend.random_uniform(shp, Dtype.complex128) for shp in shapes]
    data = backends.FusionTreeData(block_inds, blocks, Dtype.complex128,
                                   device=backend.block_backend.default_device)
    tens = SymmetricTensor(data, codomain, domain, backend=backend)

    expect_shapes = [(2, 32), (6, 88), (2, 32), (2, 32)]
    expect = [zero_block(shp, Dtype.complex128) for shp in expect_shapes]

    expect[0][:, :4] = reshape(perm_axes(reshape(blocks[1][:4, :2], (2, 2, 2)), [0, 2, 1]), (2, 4))
    expect[0][:, 4:12] = reshape(perm_axes(reshape(blocks[1][:4, 2:6], (2, 2, 4)), [0, 2, 1]), (2, 8))
    expect[0][:, 12:20] = reshape(perm_axes(reshape(blocks[1][:4, 6:10], (2, 2, 4)), [0, 2, 1]), (2, 8))
    expect[0][:, 20:32] = reshape(perm_axes(reshape(blocks[1][:4, 10:16], (2, 2, 6)), [0, 2, 1]), (2, 12))
    expect[0][:, :] *= qdim8  # (1, 8, 8)

    expect[1][:2, :4] = reshape(perm_axes(reshape(blocks[1][4:8, :2], (2, 2, 2)), [0, 2, 1]), (2, 4))
    expect[1][:2, 4:8] = reshape(perm_axes(reshape(blocks[1][12:16, :2], (2, 2, 2)), [0, 2, 1]), (2, 4))
    expect[1][2:4, :4] = reshape(perm_axes(reshape(blocks[1][8:12, :2], (2, 2, 2)), [0, 2, 1]), (2, 4))
    expect[1][2:4, 4:8] = reshape(perm_axes(reshape(blocks[1][16:20, :2], (2, 2, 2)), [0, 2, 1]), (2, 4))
    expect[1][4:, :4] = reshape(perm_axes(reshape(blocks[1][28:32, :2], (2, 2, 2)), [0, 2, 1]), (2, 4))
    expect[1][4:, 4:8] = reshape(perm_axes(reshape(blocks[1][32:, :2], (2, 2, 2)), [0, 2, 1]), (2, 4))
    # (8, 8, 8)

    expect[1][:2, 8:16] = reshape(perm_axes(reshape(blocks[0][:4, :], (2, 2, 4)), [0, 2, 1]), (2, 8))
    expect[1][2:4, 8:16] = reshape(perm_axes(reshape(blocks[0][4:8, :], (2, 2, 4)), [0, 2, 1]), (2, 8))
    expect[1][4:, 8:16] = reshape(perm_axes(reshape(blocks[0][8:, :], (2, 2, 4)), [0, 2, 1]), (2, 8))
    expect[1][:, 8:16] /= qdim8  # (1, 8, 1)

    expect[1][:2, 16:24] = reshape(perm_axes(reshape(blocks[1][4:8, 2:6], (2, 2, 4)), [0, 2, 1]), (2, 8))
    expect[1][2:4, 16:24] = reshape(perm_axes(reshape(blocks[1][8:12, 2:6], (2, 2, 4)), [0, 2, 1]), (2, 8))
    expect[1][4:, 16:24] = reshape(perm_axes(reshape(blocks[1][28:32, 2:6], (2, 2, 4)), [0, 2, 1]), (2, 8))
    # (8, 8, 8)

    expect[1][:2, 24:32] = reshape(perm_axes(reshape(blocks[1][4:8, 6:10], (2, 2, 4)), [0, 2, 1]), (2, 8))
    expect[1][2:4, 24:32] = reshape(perm_axes(reshape(blocks[1][8:12, 6:10], (2, 2, 4)), [0, 2, 1]), (2, 8))
    expect[1][4:, 24:32] = reshape(perm_axes(reshape(blocks[1][28:32, 6:10], (2, 2, 4)), [0, 2, 1]), (2, 8))
    # (8, 8, 8)

    expect[1][:2, 32:40] = reshape(perm_axes(reshape(blocks[1][12:16, 2:6], (2, 2, 4)), [0, 2, 1]), (2, 8))
    expect[1][2:4, 32:40] = reshape(perm_axes(reshape(blocks[1][16:20, 2:6], (2, 2, 4)), [0, 2, 1]), (2, 8))
    expect[1][4:, 32:40] = reshape(perm_axes(reshape(blocks[1][32:, 2:6], (2, 2, 4)), [0, 2, 1]), (2, 8))
    # (8, 8, 8)

    expect[1][:2, 40:48] = reshape(perm_axes(reshape(blocks[1][12:16, 6:10], (2, 2, 4)), [0, 2, 1]), (2, 8))
    expect[1][2:4, 40:48] = reshape(perm_axes(reshape(blocks[1][16:20, 6:10], (2, 2, 4)), [0, 2, 1]), (2, 8))
    expect[1][4:, 40:48] = reshape(perm_axes(reshape(blocks[1][32:, 6:10], (2, 2, 4)), [0, 2, 1]), (2, 8))
    # (8, 8, 8)

    expect[1][:2, 48:56] = reshape(perm_axes(reshape(blocks[2][:4, :], (2, 2, 4)), [0, 2, 1]), (2, 8))
    expect[1][2:4, 48:56] = reshape(perm_axes(reshape(blocks[2][4:8, :], (2, 2, 4)), [0, 2, 1]), (2, 8))
    expect[1][4:, 48:56] = reshape(perm_axes(reshape(blocks[2][8:, :], (2, 2, 4)), [0, 2, 1]), (2, 8))
    expect[1][:, 48:56] *= -1 / qdim8  # (8, 8, 10)

    expect[1][:2, 56:64] = reshape(perm_axes(reshape(blocks[3][:4, :], (2, 2, 4)), [0, 2, 1]), (2, 8))
    expect[1][2:4, 56:64] = reshape(perm_axes(reshape(blocks[3][4:8, :], (2, 2, 4)), [0, 2, 1]), (2, 8))
    expect[1][4:, 56:64] = reshape(perm_axes(reshape(blocks[3][8:, :], (2, 2, 4)), [0, 2, 1]), (2, 8))
    expect[1][:, 56:64] *= -1 / qdim8  # (8, 8, 10-)

    expect[1][:2, 64:76] = reshape(perm_axes(reshape(blocks[1][4:8, 10:], (2, 2, 6)), [0, 2, 1]), (2, 12))
    expect[1][:2, 76:] = reshape(perm_axes(reshape(blocks[1][12:16, 10:], (2, 2, 6)), [0, 2, 1]), (2, 12))
    expect[1][2:4, 64:76] = reshape(perm_axes(reshape(blocks[1][8:12, 10:], (2, 2, 6)), [0, 2, 1]), (2, 12))
    expect[1][2:4, 76:] = reshape(perm_axes(reshape(blocks[1][16:20, 10:], (2, 2, 6)), [0, 2, 1]), (2, 12))
    expect[1][4:, 64:76] = reshape(perm_axes(reshape(blocks[1][28:32, 10:], (2, 2, 6)), [0, 2, 1]), (2, 12))
    expect[1][4:, 76:] = reshape(perm_axes(reshape(blocks[1][32:, 10:], (2, 2, 6)), [0, 2, 1]), (2, 12))
    # (8, 8, 8)

    expect[2][:, :4] = reshape(perm_axes(reshape(blocks[1][20:24, :2], (2, 2, 2)), [0, 2, 1]), (2, 4))
    expect[2][:, 4:12] = reshape(perm_axes(reshape(blocks[1][20:24, 2:6], (2, 2, 4)), [0, 2, 1]), (2, 8))
    expect[2][:, 12:20] = reshape(perm_axes(reshape(blocks[1][20:24, 6:10], (2, 2, 4)), [0, 2, 1]), (2, 8))
    expect[2][:, 20:32] = reshape(perm_axes(reshape(blocks[1][20:24, 10:16], (2, 2, 6)), [0, 2, 1]), (2, 12))
    expect[2][:, :] *= qdim8 * -1  # (10, 8, 8)

    expect[3][:, :4] = reshape(perm_axes(reshape(blocks[1][24:28, :2], (2, 2, 2)), [0, 2, 1]), (2, 4))
    expect[3][:, 4:12] = reshape(perm_axes(reshape(blocks[1][24:28, 2:6], (2, 2, 4)), [0, 2, 1]), (2, 8))
    expect[3][:, 12:20] = reshape(perm_axes(reshape(blocks[1][24:28, 6:10], (2, 2, 4)), [0, 2, 1]), (2, 8))
    expect[3][:, 20:32] = reshape(perm_axes(reshape(blocks[1][24:28, 10:16], (2, 2, 6)), [0, 2, 1]), (2, 12))
    expect[3][:, :] *= qdim8 * -1  # (10-, 8, 8)

    expect_data = backends.FusionTreeData(block_inds, expect, Dtype.complex128,
                                          device=backend.block_backend.default_device)
    expect_codomain = TensorProduct([s1, s2])
    expect_domain = TensorProduct([s2, s3, s2.dual])
    expect_tens = SymmetricTensor(expect_data, expect_codomain, expect_domain, backend=backend)

    for func in funcs:
        new_data, new_codomain, new_domain = func(tens, False)
        new_tens = SymmetricTensor(new_data, new_codomain, new_domain, backend=backend)
        assert_tensors_almost_equal(new_tens, expect_tens, eps, eps)

    if move_leg_or_permute_leg == 'permute_leg':
        new_data, new_codomain, new_domain = backend.permute_legs(tens, [0, 1], [4, 3, 2], None)
        new_tens = SymmetricTensor(new_data, new_codomain, new_domain, backend=backend)
        assert_tensors_almost_equal(new_tens, expect_tens, eps, eps)
    elif move_leg_or_permute_leg == 'move_leg':
        new_tens = move_leg(tens, 2, domain_pos=2, levels=None)
        assert_tensors_almost_equal(new_tens, expect_tens, eps, eps)

    spaces = [TensorProduct([], symmetry=sym), TensorProduct([s2]), TensorProduct([s3.dual]),
              TensorProduct([s1, s3]), TensorProduct([s2, s3.dual]), TensorProduct([s1, s3, s2.dual])]
    # bend up and down again (and vice versa) == trivial
    assert_bending_up_and_down_trivial(spaces, spaces, funcs, backend, multiple=multiple, eps=eps)

    # rescaling axis and then bending == bending and then rescaling axis
    assert_bending_and_scale_axis_commutation(tens, funcs, eps)


@pytest.mark.parametrize('symmetry',
    [fibonacci_anyon_category, ising_anyon_category, SU2_kAnyonCategory(4),
     SU2_kAnyonCategory(5) * u1_symmetry, SU2Symmetry() * ising_anyon_category,
     SU3_3AnyonCategory() * u1_symmetry, fibonacci_anyon_category * z5_symmetry]
)
def test_nonabelian_transpose(symmetry: Symmetry, block_backend: str,
                              np_random: np.random.Generator):
    backend = get_backend('fusion_tree', block_backend)
    num_codom_legs, num_dom_legs = np_random.integers(low=2, high=4, size=2)
    tens = random_tensor(
        symmetry=symmetry, codomain=int(num_codom_legs), domain=int(num_dom_legs),
        backend=backend, max_multiplicity=3, cls=SymmetricTensor, np_random=np_random
    )

    data1, codom1, dom1 = tens.backend.transpose(tens)
    data2, codom2, dom2 = cross_check_transpose(tens)
    tens1 = SymmetricTensor(data1, codom1, dom1, backend=tens.backend)
    tens2 = SymmetricTensor(data2, codom2, dom2, backend=tens.backend)
    assert_tensors_almost_equal(tens1, tens2, rtol=1e-13, atol=1e-13)


# TODO add symmetry with off-diagonal entries in b symbols to test below
# TODO make sure that there is a case that fails when not complex conjugating
# the b symbols for domain trees
# (all tests currently seem to pass irrespective of complex conjugation)
@pytest.mark.parametrize('symmetry',
    [fibonacci_anyon_category, ising_anyon_category, SU2_kAnyonCategory(4),
     SU2_kAnyonCategory(5) * u1_symmetry, SU2Symmetry() * ising_anyon_category,
     SU3_3AnyonCategory() * u1_symmetry, fibonacci_anyon_category * z5_symmetry]
)
def test_nonabelian_partial_trace(symmetry: Symmetry, block_backend: str,
                                  np_random: np.random.Generator):
    backend = get_backend('fusion_tree', block_backend)
    num_codom_legs, num_dom_legs = np_random.integers(low=2, high=4, size=2)
    num_legs = num_codom_legs + num_dom_legs
    pairs = [[[0, 1]], [[0, 3]], [[1, 3]], [[0, 2]], [[0, 2], [1, 3]], [[0, 3], [1, 2]]]
    if num_legs >= 5:
        pairs.extend([[[0, 4]], [[1, 4], [0, 2]], [[0, 4], [1, 3]]])
    pairs = pairs[np_random.choice(len(pairs))]

    spaces = []
    idcs1 = [p[0] for p in pairs]
    idcs2 = [p[1] for p in pairs]
    for i in range(num_legs):
        if i in idcs2:
            idx = idcs2.index(i)
            if i >= num_codom_legs and idcs1[idx] < num_codom_legs:
                spaces.append(spaces[idcs1[idx]])
            else:
                spaces.append(spaces[idcs1[idx]].dual)
        else:
            spaces.append(random_ElementarySpace(symmetry=symmetry, np_random=np_random))

    levels = list(np_random.permutation(num_legs - len(pairs)))
    for idx in np.argsort(idcs2):
        level = levels[idcs1[idx]]
        levels = [l + 1 if l > level else l for l in levels]
        levels.insert(idcs2[idx], level + 1)

    codomain = spaces[:num_codom_legs]
    domain = spaces[num_codom_legs:][::-1]
    tens = random_tensor(
        symmetry=symmetry, codomain=codomain, domain=domain,
        backend=backend, max_multiplicity=3, cls=SymmetricTensor, np_random=np_random
    )

    data1, codom1, dom1 = tens.backend.partial_trace(tens, pairs=pairs, levels=levels)
    data2, codom2, dom2 = cross_check_partial_trace(tens, pairs=pairs, levels=levels)
    tens1 = SymmetricTensor(data1, codom1, dom1, backend=tens.backend)
    tens2 = SymmetricTensor(data2, codom2, dom2, backend=tens.backend)
    assert_tensors_almost_equal(tens1, tens2, 1e-13, 1e-13)


# HELPER FUNCTIONS FOR THE TESTS

def apply_single_b_symbol(ten: SymmetricTensor, bend_up: bool
                          ) -> tuple[fusion_tree_backend.FusionTreeData, TensorProduct, TensorProduct]:
    """Use the implementation of b symbols using `TreeMappingDicts` to return the action
    of a single b symbol in the same format (input and output) as the cross check
    implementation. This is of course inefficient usage of this implementation but a
    necessity in order to use the structure of the already implemented tests.
    """
    func = fusion_tree_backend.TreeMappingDict.from_b_or_c_symbol
    index = ten.num_codomain_legs - 1
    coupled = [ten.domain.sector_decomposition[ind[1]] for ind in ten.data.block_inds]

    if bend_up:
        axes_perm = list(range(ten.num_codomain_legs)) + [ten.num_legs - 1]
        axes_perm += [ten.num_codomain_legs + i for i in range(ten.num_domain_legs - 1)]
    else:
        axes_perm = list(range(ten.num_codomain_legs - 1))
        axes_perm += [ten.num_codomain_legs + i for i in range(ten.num_domain_legs)]
        axes_perm += [ten.num_codomain_legs - 1]

    mapp, new_codomain, new_domain, _ = func(ten.codomain, ten.domain, index, coupled,
                                             None, bend_up, ten.backend)
    new_data = mapp.apply_to_tensor(ten, new_codomain, new_domain, axes_perm, in_domain=None)
    return new_data, new_codomain, new_domain


def apply_single_c_symbol(ten: SymmetricTensor, leg: int | str, levels: list[int]
                          ) -> tuple[fusion_tree_backend.FusionTreeData, TensorProduct, TensorProduct]:
    """Use the implementation of c symbols using `TreeMappingDicts` to return the action
    of a single c symbol in the same format (input and output) as the cross check
    implementations. This is of course inefficient usage of this implementation but a
    necessity in order to use the structure of the already implemented tests.
    """
    func = fusion_tree_backend.TreeMappingDict.from_b_or_c_symbol
    index = ten.get_leg_idcs(leg)[0]
    in_domain = index > ten.num_codomain_legs - 1
    overbraid = levels[index] > levels[index + 1]
    coupled = [ten.domain.sector_decomposition[ind[1]] for ind in ten.data.block_inds]

    if not in_domain:
        axes_perm = list(range(ten.num_codomain_legs))
        index_ = index
    else:
        axes_perm = list(range(ten.num_domain_legs))
        index_ = ten.num_legs - 1 - (index + 1)
    axes_perm[index_:index_ + 2] = axes_perm[index_:index_ + 2][::-1]

    mapp, new_codomain, new_domain, _ = func(ten.codomain, ten.domain, index, coupled,
                                             overbraid, None, ten.backend)
    new_data = mapp.apply_to_tensor(ten, new_codomain, new_domain, axes_perm, in_domain)
    return new_data, new_codomain, new_domain


def assert_bending_and_scale_axis_commutation(a: SymmetricTensor, funcs: list[Callable], eps: float):
    """Check that when rescaling and bending legs, it does not matter whether one first
    performs the rescaling and then the bending process or vice versa. This is tested using
    `scale_axis` in `FusionTreeBackend`, i.e., not the funtion directly acting on tensors;
    this function is tested elsewhere.
    """
    bends = [True, False]
    for bend_up in bends:
        if a.num_codomain_legs == 0 and not bend_up:
            continue
        elif a.num_domain_legs == 0 and bend_up:
            continue

        for func in funcs:
            if bend_up:
                num_leg = a.num_codomain_legs - 1
                leg = a.legs[num_leg]
            else:
                num_leg = a.num_codomain_legs
                leg = a.legs[num_leg].dual

            diag = DiagonalTensor.from_random_uniform(leg, backend=a.backend, dtype=a.dtype)
            new_a = a.copy()
            new_a2 = a.copy()

            # apply scale_axis first
            new_a.data = new_a.backend.scale_axis(new_a, diag, num_leg)
            new_data, new_codomain, new_domain = func(new_a, bend_up)
            new_a = SymmetricTensor(new_data, new_codomain, new_domain, backend=new_a.backend)

            # bend first
            new_data, new_codomain, new_domain = func(new_a2, bend_up)
            new_a2 = SymmetricTensor(new_data, new_codomain, new_domain, backend=new_a2.backend)
            new_a2.data = new_a2.backend.scale_axis(new_a2, diag, num_leg)

            assert_tensors_almost_equal(new_a, new_a2, eps, eps)


def assert_bending_up_and_down_trivial(codomains: list[TensorProduct], domains: list[TensorProduct],
                                       funcs: list[Callable], backend: backends.TensorBackend,
                                       multiple: bool, eps: float):
    """Check that bending a leg up and down (or down and up) is trivial. All given codomains are combined with all
    given domains to construct random tensors for which the identities are checked. All codomains and domains must
    have the same symmetry; this is not explicitly checked.
    If `multiple == True`, the identity is also checked for bending multiple (i.e. more than one) legs up and down
    up to the maximum number possible. Otherwise, the identity is only checked for a single leg.
    """
    for codomain in codomains:
        for domain in domains:
            tens = SymmetricTensor.from_random_uniform(codomain, domain, backend=backend)
            if len(tens.data.blocks) == 0:  # trivial tensors
                continue

            bending = []
            for i in range(tens.num_domain_legs):
                bends = [True] * (i+1) + [False] * (i+1)
                bending.append(bends)
                if not multiple:
                    break
            for i in range(tens.num_codomain_legs):
                bends = [False] * (i+1) + [True] * (i+1)
                bending.append(bends)
                if not multiple:
                    break

            for func in funcs:
                for bends in bending:
                    new_tens = tens.copy()
                    for bend in bends:
                        new_data, new_codomain, new_domain = func(new_tens, bend)
                        new_tens = SymmetricTensor(new_data, new_codomain, new_domain, backend=backend)

                    assert_tensors_almost_equal(new_tens, tens, eps, eps)


def assert_braiding_and_scale_axis_commutation(a: SymmetricTensor, funcs: list[Callable],
                                               levels: list[int], eps: float):
    """Check that when rescaling and exchanging legs, it does not matter whether one first
    performs the rescaling and then the exchange process or vice versa. This is tested using
    `scale_axis` in `FusionTreeBackend`, i.e., not the funtion directly acting on tensors;
    this function is tested elsewhere.
    """
    for func in funcs:
        for leg in range(a.num_legs-1):
            if leg == a.num_codomain_legs - 1:
                continue

            legs = [a.legs[leg], a.legs[leg + 1]]
            if leg > a.num_codomain_legs - 1:  # in domain
                legs = [leg_.dual for leg_ in legs]
            diag_left = DiagonalTensor.from_random_uniform(legs[0], backend=a.backend, dtype=a.dtype)
            diag_right = DiagonalTensor.from_random_uniform(legs[1], backend=a.backend, dtype=a.dtype)
            new_a = a.copy()
            new_a2 = a.copy()

            # apply scale_axis first
            new_a.data = new_a.backend.scale_axis(new_a, diag_left, leg)
            new_a.data = new_a.backend.scale_axis(new_a, diag_right, leg + 1)
            new_data, new_codomain, new_domain = func(new_a, leg=leg, levels=levels)
            new_a = SymmetricTensor(new_data, new_codomain, new_domain, backend=new_a.backend)

            # exchange first
            new_data, new_codomain, new_domain = func(new_a2, leg=leg, levels=levels)
            new_a2 = SymmetricTensor(new_data, new_codomain, new_domain, backend=new_a2.backend)
            new_a2.data = new_a2.backend.scale_axis(new_a2, diag_left, leg + 1)
            new_a2.data = new_a2.backend.scale_axis(new_a2, diag_right, leg)

            assert_tensors_almost_equal(new_a, new_a2, eps, eps)
            

def assert_clockwise_counterclockwise_trivial(a: SymmetricTensor, funcs: list[Callable],
                                              levels: list[int], eps: float):
    """Check that braiding a pair of neighboring legs clockwise and then anti-clockwise
    (or vice versa) leaves the tensor invariant. `levels` specifies whether the first
    exchange is clockwise or anti-clockwise (the second is the opposite).
    This identity is checked for each neighboring pair of legs in the codomain and domain.
    """
    for func in funcs:
        for leg in range(a.num_legs-1):
            if leg == a.num_codomain_legs - 1:
                continue
            new_a = a.copy()
            new_levels = levels[:]
            for _ in range(2):
                new_data, new_codomain, new_domain = func(new_a, leg=leg, levels=new_levels)
                new_a = SymmetricTensor(new_data, new_codomain, new_domain, backend=a.backend)
                new_levels[leg:leg+2] = new_levels[leg:leg+2][::-1]

            assert_tensors_almost_equal(new_a, a, eps, eps)


def assert_clockwise_counterclockwise_trivial_long_range(a: SymmetricTensor, move_leg_or_permute_leg: str,
                                                         eps: float, np_random: np.random.Generator):
    """Same as `assert_clockwise_counterclockwise_trivial` with the difference that a random
    sequence of exchanges and bends is chosen. The identity is checked using `permute_legs`
    of the tensor backend or using `move_leg` depending on `move_leg_or_permute_leg`.
    """
    levels = list(np_random.permutation(a.num_legs))
    if move_leg_or_permute_leg == 'permute_leg':
        # more general case; needs more input
        permutation = list(np_random.permutation(a.num_legs))
        inv_permutation = [permutation.index(i) for i in range(a.num_legs)]
        inv_levels = [levels[i] for i in permutation]
        num_codomain = np.random.randint(a.num_legs + 1)

        new_data, new_codomain, new_domain = a.backend.permute_legs(a, permutation[:num_codomain],
                                                                    permutation[num_codomain:][::-1], levels)
        new_a = SymmetricTensor(new_data, new_codomain, new_domain, backend=a.backend)
        new_data, new_codomain, new_domain = a.backend.permute_legs(new_a, inv_permutation[:a.num_codomain_legs],
                                                                    inv_permutation[a.num_codomain_legs:][::-1], inv_levels)
        new_a = SymmetricTensor(new_data, new_codomain, new_domain, backend=a.backend)

    elif move_leg_or_permute_leg == 'move_leg':
        leg = np_random.integers(a.num_legs)
        co_dom_pos = np_random.integers(a.num_legs)

        if co_dom_pos >= a.num_codomain_legs:
            new_a = move_leg(a, leg, domain_pos=a.num_legs - 1 - co_dom_pos, levels=levels)
        else:
            new_a = move_leg(a, leg, codomain_pos=co_dom_pos, levels=levels)

        tmp = levels[leg]
        levels = [levels[i] for i in range(a.num_legs) if i != leg]
        levels.insert(co_dom_pos, tmp)
        if leg >= a.num_codomain_legs:
            new_a = move_leg(new_a, co_dom_pos, domain_pos=a.num_legs - 1 - leg, levels=levels)
        else:
            new_a = move_leg(new_a, co_dom_pos, codomain_pos=leg, levels=levels)

    assert_tensors_almost_equal(new_a, a, eps, eps)


def assert_repeated_braids_trivial(a: SymmetricTensor, funcs: list[Callable], levels: list[int],
                                   repeat: int, eps: float):
    """Check that repeatedly braiding two neighboring legs often enough leaves the tensor
    invariant. The number of required repetitions must be chosen to be even (legs must be
    at their initial positions again) and such that all (relevant) r symbols to the power
    of the repetitions is identity. `levels` specifies whether the repeated exchange is
    always clockwise or anti-clockwise.
    This identity is checked for each neighboring pair of legs in the codomain and domain.
    """
    for func in funcs:
        for leg in range(a.num_legs-1):
            if leg == a.num_codomain_legs - 1:
                continue
            new_a = a.copy()
            for _ in range(repeat):
                new_data, new_codomain, new_domain = func(new_a, leg=leg, levels=levels)
                new_a = SymmetricTensor(new_data, new_codomain, new_domain, backend=a.backend)

            assert_tensors_almost_equal(new_a, a, eps, eps)


# FUNCTIONS FOR CROSS CHECKING THE COMPUTATION OF THE ACTION OF B AND C SYMBOLS

def cross_check_partial_trace(ten: SymmetricTensor, pairs: list[tuple[int, int]], levels: list[int]):
    """There are different ways to compute partial traces. One particular
    choice is implemented in `partial_trace` in the `FusionTreeBackend` by
    choosing a certain way of braiding the paired legs such that they are
    adjacent to each other.
    
    Here we choose a different way to achieve this adjacency before performing
    the partial trace itself (which is again done by calling this very method
    `partial_trace`).
    """
    # we do not need to check that the levels are consistent since the result
    # of this function is compared to the result obtained using partial_trace
    # in the fusion_tree_backend, where the levels are checked.
    pairs = np.asarray([pair if pair[0] < pair[1] else (pair[1], pair[0]) for pair in pairs],
                       dtype=int)
    # sort w.r.t. the second entry first
    pairs = pairs[np.lexsort(pairs.T)]
    idcs1 = []
    idcs2 = []
    for i1, i2 in pairs:
        idcs1.append(i1)
        idcs2.append(i2)
    remaining = [n for n in range(ten.num_legs) if n not in idcs1 and n not in idcs2]

    insert_idcs = [np.searchsorted(remaining, pair[1]) + 2 * i for i, pair in enumerate(pairs)]
    # permute legs such that the ones with the larger index do not move
    num_codom_legs = ten.num_codomain_legs
    idcs = remaining[:]
    for idx, pair in zip(insert_idcs, pairs):
        idcs[idx: idx] = list(pair)
        if pair[0] < ten.num_codomain_legs and pair[1] >= ten.num_codomain_legs:
            num_codom_legs -= 1  # leg at pair[0] is bent down

    data, codom, dom = ten.backend.permute_legs(ten, codomain_idcs=idcs[:num_codom_legs],
                                                domain_idcs=idcs[num_codom_legs:][::-1],
                                                levels=levels)
    tr_idcs1 = [i for i, idx in enumerate(idcs) if idx in idcs1]
    tr_idcs2 = [i for i, idx in enumerate(idcs) if idx in idcs2]
    new_pairs = list(zip(tr_idcs1, tr_idcs2))

    ten = SymmetricTensor(data=data, codomain=codom, domain=dom, backend=ten.backend)
    return ten.backend.partial_trace(ten, pairs=new_pairs, levels=None)


def cross_check_single_c_symbol_tree_blocks(ten: SymmetricTensor, leg: int | str, levels: list[int]
                                            ) -> tuple[fusion_tree_backend.FusionTreeData,
                                                       TensorProduct, TensorProduct]:
    """Naive implementation of a single C symbol for test purposes on the level
    of the tree blocks. `ten.legs[leg]` is exchanged with `ten.legs[leg + 1]`.

    NOTE this function may be deleted at a later stage
    """
    # NOTE the case of braiding in domain and codomain are treated separately despite being similar
    # This way, it is easier to understand the more compact and more efficient function
    ftb = fusion_tree_backend
    index = ten.get_leg_idcs(leg)[0]
    backend = ten.backend
    block_backend = ten.backend.block_backend
    symmetry = ten.symmetry

    assert index != ten.num_codomain_legs - 1, 'Cannot apply C symbol without applying B symbol first.'
    assert index < ten.num_legs - 1

    in_domain = index > ten.num_codomain_legs - 1

    if in_domain:
        domain_index = ten.num_legs - 1 - (index + 1)  # + 1 because it braids with the leg left of it
        spaces = ten.domain[:domain_index] + [ten.domain[domain_index + 1]] + \
            [ten.domain[domain_index]] + ten.domain[domain_index + 2:]
        new_domain = TensorProduct(spaces, symmetry=symmetry)
        # TODO can re-use: _sectors=ten.domain.sector_decomposition, _multiplicities=ten.domain.multiplicities)
        new_codomain = ten.codomain
    else:
        spaces = ten.codomain[:index] + [ten.codomain[index + 1]] + [ten.codomain[index]] + ten.codomain[index + 2:]
        new_codomain = TensorProduct(spaces,  symmetry=symmetry)
        # TODO can re-use: ten.codomain.sector_decomposition, ten.codomain.multiplicities
        new_domain = ten.domain

    zero_blocks = [block_backend.zeros(block_backend.get_shape(block), dtype=Dtype.complex128)
                   for block in ten.data.blocks]
    new_data = ftb.FusionTreeData(ten.data.block_inds, zero_blocks, ten.data.dtype,
                                  device=block_backend.default_device)
    shape_perm = np.arange(ten.num_legs)  # for permuting the shape of the tree blocks
    shifted_index = ten.num_codomain_legs + domain_index if in_domain else index
    shape_perm[shifted_index:shifted_index+2] = shape_perm[shifted_index:shifted_index+2][::-1]
    shape_perm = list(shape_perm)  # torch does not like np.arrays

    for alpha_tree, beta_tree, tree_block in ftb._tree_block_iter(ten):
        block_charge = ten.domain.sector_decomposition_where(alpha_tree.coupled)
        block_charge = ten.data.block_ind_from_domain_sector_ind(block_charge)

        initial_shape = block_backend.get_shape(tree_block)
        modified_shape = [ten.codomain[i].sector_multiplicity(sec)
                          for i, sec in enumerate(alpha_tree.uncoupled)]
        modified_shape += [ten.domain[i].sector_multiplicity(sec)
                           for i, sec in enumerate(beta_tree.uncoupled)]

        tree_block = block_backend.reshape(tree_block, tuple(modified_shape))
        tree_block = block_backend.permute_axes(tree_block, shape_perm)
        tree_block = block_backend.reshape(tree_block, initial_shape)

        if index == 0 or index == ten.num_legs - 2:
            if in_domain:
                alpha_slice = new_codomain.tree_block_slice(alpha_tree)
                b = beta_tree.copy(True)
                b_unc, b_in, b_mul = b.uncoupled, b.inner_sectors, b.multiplicities
                f = b.coupled if len(b_in) == 0 else b_in[0]
                if symmetry.braiding_style.value >= 20 and levels[index] > levels[index + 1]:
                    r = symmetry.r_symbol(b_unc[0], b_unc[1], f)[b_mul[0]]
                else:
                    r = symmetry.r_symbol(b_unc[1], b_unc[0], f)[b_mul[0]].conj()
                b_unc[domain_index:domain_index+2] = b_unc[domain_index:domain_index+2][::-1]
                beta_slice = new_domain.tree_block_slice(b)
            else:
                beta_slice = new_domain.tree_block_slice(beta_tree)
                a = alpha_tree.copy(True)
                a_unc, a_in, a_mul = a.uncoupled, a.inner_sectors, a.multiplicities
                f = a.coupled if len(a_in) == 0 else a_in[0]
                if symmetry.braiding_style.value >= 20 and levels[index] > levels[index + 1]:
                    r = symmetry.r_symbol(a_unc[1], a_unc[0], f)[a_mul[0]]
                else:
                    r = symmetry.r_symbol(a_unc[0], a_unc[1], f)[a_mul[0]].conj()
                a_unc[index:index+2] = a_unc[index:index+2][::-1]
                alpha_slice = new_codomain.tree_block_slice(a)

            new_data.blocks[block_charge][alpha_slice, beta_slice] += r * tree_block
        else:
            if in_domain:
                alpha_slice = new_codomain.tree_block_slice(alpha_tree)

                beta_unc, beta_in, beta_mul = (beta_tree.uncoupled, beta_tree.inner_sectors,
                                               beta_tree.multiplicities)

                if domain_index == 1:
                    left_charge = beta_unc[domain_index-1]
                else:
                    left_charge = beta_in[domain_index-2]
                if domain_index == ten.num_domain_legs - 2:
                    right_charge = beta_tree.coupled
                else:
                    right_charge = beta_in[domain_index]

                for f in symmetry.fusion_outcomes(left_charge, beta_unc[domain_index+1]):
                    if not symmetry.can_fuse_to(f, beta_unc[domain_index], right_charge):
                        continue

                    if symmetry.braiding_style.value >= 20 and levels[index] > levels[index + 1]:
                        cs = symmetry.c_symbol(left_charge, beta_unc[domain_index], beta_unc[domain_index+1],
                                               right_charge, beta_in[domain_index-1], f)[
                                                beta_mul[domain_index-1], beta_mul[domain_index], :, :]
                    else:
                        cs = symmetry.c_symbol(left_charge, beta_unc[domain_index+1], beta_unc[domain_index],
                                               right_charge, f, beta_in[domain_index-1])[:, :,
                                               beta_mul[domain_index-1], beta_mul[domain_index]].conj()

                    b = beta_tree.copy(True)
                    b_unc, b_in, b_mul = b.uncoupled, b.inner_sectors, b.multiplicities

                    b_unc[domain_index:domain_index+2] = b_unc[domain_index:domain_index+2][::-1]
                    b_in[domain_index-1] = f
                    for (kap, lam), c in np.ndenumerate(cs):
                        if abs(c) < backend.eps:
                            continue
                        b_mul[domain_index-1] = kap
                        b_mul[domain_index] = lam

                        beta_slice = new_domain.tree_block_slice(b)
                        new_data.blocks[block_charge][alpha_slice, beta_slice] += c * tree_block
            else:
                beta_slice = new_domain.tree_block_slice(beta_tree)
                alpha_unc, alpha_in, alpha_mul = (alpha_tree.uncoupled, alpha_tree.inner_sectors,
                                                  alpha_tree.multiplicities)

                left_charge = alpha_unc[0] if index == 1 else alpha_in[index-2]
                right_charge = alpha_tree.coupled if index == ten.num_codomain_legs - 2 else alpha_in[index]

                for f in symmetry.fusion_outcomes(left_charge, alpha_unc[index+1]):
                    if not symmetry.can_fuse_to(f, alpha_unc[index], right_charge):
                        continue

                    if symmetry.braiding_style.value >= 20 and levels[index] > levels[index + 1]:
                        cs = symmetry.c_symbol(left_charge, alpha_unc[index+1], alpha_unc[index],
                                               right_charge, f, alpha_in[index-1])[:, :,
                                               alpha_mul[index-1], alpha_mul[index]]
                    else:
                        cs = symmetry.c_symbol(left_charge, alpha_unc[index], alpha_unc[index+1],
                                               right_charge, alpha_in[index-1], f)[
                                               alpha_mul[index-1], alpha_mul[index], :, :].conj()

                    a = alpha_tree.copy(True)
                    a_unc, a_in, a_mul = a.uncoupled, a.inner_sectors, a.multiplicities

                    a_unc[index:index+2] = a_unc[index:index+2][::-1]
                    a_in[index-1] = f
                    for (kap, lam), c in np.ndenumerate(cs):
                        if abs(c) < backend.eps:
                            continue
                        a_mul[index-1] = kap
                        a_mul[index] = lam

                        alpha_slice = new_codomain.tree_block_slice(a)
                        new_data.blocks[block_charge][alpha_slice, beta_slice] += c * tree_block
    new_data.discard_zero_blocks(block_backend, backend.eps)
    return new_data, new_codomain, new_domain


def cross_check_single_c_symbol_tree_cols(ten: SymmetricTensor, leg: int | str, levels: list[int]
                                          ) -> tuple[fusion_tree_backend.FusionTreeData,
                                                     TensorProduct, TensorProduct]:
    """Naive implementation of a single C symbol for test purposes on the level
    of the tree columns (= tree slices of codomain xor domain). `ten.legs[leg]`
    is exchanged with `ten.legs[leg + 1]`.

    NOTE this function may be deleted at a later stage
    """
    ftb = fusion_tree_backend
    index = ten.get_leg_idcs(leg)[0]
    backend = ten.backend
    block_backend = ten.backend.block_backend
    symmetry = ten.symmetry

    # NOTE do these checks in permute_legs for the actual (efficient) function
    assert index != ten.num_codomain_legs - 1, 'Cannot apply C symbol without applying B symbol first.'
    assert index < ten.num_legs - 1

    in_domain = index > ten.num_codomain_legs - 1

    if in_domain:
        index = ten.num_legs - 1 - (index + 1)  # + 1 because it braids with the leg left of it
        levels = levels[::-1]
        spaces = ten.domain[:index] + ten.domain[index:index+2][::-1] + ten.domain[index+2:]
        new_domain = TensorProduct(spaces, symmetry=symmetry)
        # TODO can re-use:_sectors=ten.domain.sector_decomposition, _multiplicities=ten.domain.multiplicities)
        new_codomain = ten.codomain
        # for permuting the shape of the tree blocks
        shape_perm = np.append([0], np.arange(1, ten.num_domain_legs+1))
        shape_perm[index+1:index+3] = shape_perm[index+1:index+3][::-1]
    else:
        spaces = ten.codomain[:index] + ten.codomain[index:index+2][::-1] + ten.codomain[index+2:]
        new_codomain = TensorProduct(spaces, symmetry=symmetry)
        # TODO can re-use:
        # _sectors=ten.codomain.sector_decomposition,
        # _multiplicities=ten.codomain.multiplicities)
        new_domain = ten.domain
        shape_perm = np.append(np.arange(ten.num_codomain_legs), [ten.num_codomain_legs])
        shape_perm[index:index+2] = shape_perm[index:index+2][::-1]

    shape_perm = list(shape_perm)  # torch does not like np.arrays
    zero_blocks = [block_backend.zeros(block_backend.get_shape(block), dtype=Dtype.complex128)
                   for block in ten.data.blocks]
    new_data = ftb.FusionTreeData(ten.data.block_inds, zero_blocks, ten.data.dtype,
                                  device=block_backend.default_device)
    iter_space = [ten.codomain, ten.domain][in_domain]
    iter_coupled = [ten.codomain.sector_decomposition[ind[0]] for ind in ten.data.block_inds]

    for tree, slc, _ in iter_space.iter_tree_blocks(iter_coupled):
        block_charge = ten.domain.sector_decomposition_where(tree.coupled)
        block_charge = ten.data.block_ind_from_domain_sector_ind(block_charge)

        tree_block = (ten.data.blocks[block_charge][:,slc] if in_domain
                      else ten.data.blocks[block_charge][slc,:])

        initial_shape = block_backend.get_shape(tree_block)

        modified_shape = [iter_space[i].sector_multiplicity(sec) for i, sec in enumerate(tree.uncoupled)]
        modified_shape.insert((not in_domain) * len(modified_shape), initial_shape[not in_domain])

        tree_block = block_backend.reshape(tree_block, tuple(modified_shape))
        tree_block = block_backend.permute_axes(tree_block, shape_perm)
        tree_block = block_backend.reshape(tree_block, initial_shape)

        if index == 0 or index == ten.num_legs - 2:
            new_tree = tree.copy(deep=True)
            _unc, _in, _mul = new_tree.uncoupled, new_tree.inner_sectors, new_tree.multiplicities
            f = new_tree.coupled if len(_in) == 0 else _in[0]

            if symmetry.braiding_style.value >= 20 and levels[index] > levels[index + 1]:
                r = symmetry._r_symbol(_unc[1], _unc[0], f)[_mul[0]]
            else:
                r = symmetry._r_symbol(_unc[0], _unc[1], f)[_mul[0]].conj()

            if in_domain:
                r = r.conj()

            _unc[index:index+2] = _unc[index:index+2][::-1]
            if in_domain:
                new_slc = new_domain.tree_block_slice(new_tree)
            else:
                new_slc = new_codomain.tree_block_slice(new_tree)

            if in_domain:
                new_data.blocks[block_charge][:, new_slc] += r * tree_block
            else:
                new_data.blocks[block_charge][new_slc, :] += r * tree_block
        else:
            _unc, _in, _mul = tree.uncoupled, tree.inner_sectors, tree.multiplicities

            left_charge = _unc[0] if index == 1 else _in[index-2]
            right_charge = tree.coupled if index == _in.shape[0] else _in[index]

            for f in symmetry.fusion_outcomes(left_charge, _unc[index+1]):
                if not symmetry.can_fuse_to(f, _unc[index], right_charge):
                    continue

                if symmetry.braiding_style.value >= 20 and levels[index] > levels[index+1]:
                    cs = symmetry._c_symbol(left_charge, _unc[index+1], _unc[index], right_charge,
                                            f, _in[index-1])[:, :, _mul[index-1], _mul[index]]
                else:
                    cs = symmetry._c_symbol(left_charge, _unc[index], _unc[index+1], right_charge,
                                            _in[index-1], f)[_mul[index-1], _mul[index], :, :].conj()

                if in_domain:
                    cs = cs.conj()

                new_tree = tree.copy(deep=True)
                new_tree.uncoupled[index:index+2] = new_tree.uncoupled[index:index+2][::-1]
                new_tree.inner_sectors[index-1] = f
                for (kap, lam), c in np.ndenumerate(cs):
                    if abs(c) < backend.eps:
                        continue
                    new_tree.multiplicities[index-1] = kap
                    new_tree.multiplicities[index] = lam

                    if in_domain:
                        new_slc = new_domain.tree_block_slice(new_tree)
                    else:
                        new_slc = new_codomain.tree_block_slice(new_tree)

                    if in_domain:
                        new_data.blocks[block_charge][:, new_slc] += c * tree_block
                    else:
                        new_data.blocks[block_charge][new_slc, :] += c * tree_block
    new_data.discard_zero_blocks(block_backend, backend.eps)
    return new_data, new_codomain, new_domain


def cross_check_single_b_symbol(ten: SymmetricTensor, bend_up: bool
                                ) -> tuple[fusion_tree_backend.FusionTreeData,
                                           TensorProduct, TensorProduct]:
    """Naive implementation of a single B symbol for test purposes.
    If `bend_up == True`, the right-most leg in the domain is bent up,
    otherwise the right-most leg in the codomain is bent down.

    NOTE this function may be deleted at a later stage
    """
    ftb = fusion_tree_backend
    backend = ten.backend
    block_backend = ten.backend.block_backend
    symmetry = ten.symmetry
    device = backend.block_backend.default_device

    # NOTE do these checks in permute_legs for the actual (efficient) function
    if bend_up:
        assert ten.num_domain_legs > 0, 'There is no leg to bend in the domain!'
    else:
        assert ten.num_codomain_legs > 0, 'There is no leg to bend in the codomain!'
    assert len(ten.data.blocks) > 0, 'The given tensor has no blocks to act on!'

    spaces = [ten.codomain, ten.domain]
    space1, space2 = spaces[bend_up], spaces[not bend_up]
    new_space1 = TensorProduct(space1.factors[:-1], symmetry=symmetry)
    new_space2 = TensorProduct(space2.factors + [space1.factors[-1].dual], symmetry=symmetry)

    new_codomain = [new_space1, new_space2][bend_up]
    new_domain = [new_space1, new_space2][not bend_up]

    new_data = ten.backend.zero_data(new_codomain, new_domain, dtype=Dtype.complex128,
                                     device=device, all_blocks=True)

    for alpha_tree, beta_tree, tree_block in ftb._tree_block_iter(ten):
        modified_shape = [ten.codomain[i].sector_multiplicity(sec)
                          for i, sec in enumerate(alpha_tree.uncoupled)]
        modified_shape += [ten.domain[i].sector_multiplicity(sec)
                           for i, sec in enumerate(beta_tree.uncoupled)]

        if bend_up:
            if beta_tree.uncoupled.shape[0] == 1:
                coupled = symmetry.trivial_sector
            else:
                coupled = (beta_tree.inner_sectors[-1] if beta_tree.inner_sectors.shape[0] > 0
                           else beta_tree.uncoupled[0])
            modified_shape = (prod(modified_shape[:ten.num_codomain_legs]),
                              prod(modified_shape[ten.num_codomain_legs:ten.num_legs-1]),
                              modified_shape[ten.num_legs-1])
            sec_mul = ten.domain[-1].sector_multiplicity(beta_tree.uncoupled[-1])
            final_shape = (block_backend.get_shape(tree_block)[0] * sec_mul,
                           block_backend.get_shape(tree_block)[1] // sec_mul)
        else:
            if alpha_tree.uncoupled.shape[0] == 1:
                coupled = symmetry.trivial_sector
            else:
                coupled = (alpha_tree.inner_sectors[-1] if alpha_tree.inner_sectors.shape[0] > 0
                           else alpha_tree.uncoupled[0])
            modified_shape = (prod(modified_shape[:ten.num_codomain_legs-1]),
                              modified_shape[ten.num_codomain_legs-1],
                              prod(modified_shape[ten.num_codomain_legs:ten.num_legs]))
            sec_mul = ten.codomain[-1].sector_multiplicity(alpha_tree.uncoupled[-1])
            final_shape = (block_backend.get_shape(tree_block)[0] // sec_mul,
                           block_backend.get_shape(tree_block)[1] * sec_mul)
        block_ind = new_domain.sector_decomposition_where(coupled)
        block_ind = new_data.block_ind_from_domain_sector_ind(block_ind)

        tree_block = block_backend.reshape(tree_block, modified_shape)
        tree_block = block_backend.permute_axes(tree_block, [0, 2, 1])
        tree_block = block_backend.reshape(tree_block, final_shape)

        trees = [alpha_tree, beta_tree]
        tree1, tree2 = trees[bend_up], trees[not bend_up]

        if tree1.uncoupled.shape[0] == 1:
            tree1_in_1 = symmetry.trivial_sector
        else:
            tree1_in_1 = (tree1.inner_sectors[-1] if tree1.inner_sectors.shape[0] > 0
                          else tree1.uncoupled[0])

        new_tree1 = FusionTree(symmetry, tree1.uncoupled[:-1], tree1_in_1, tree1.are_dual[:-1],
                               tree1.inner_sectors[:-1], tree1.multiplicities[:-1])
        new_tree2 = FusionTree(symmetry, np.append(tree2.uncoupled,
                               [symmetry.dual_sector(tree1.uncoupled[-1])], axis=0),
                               tree1_in_1, np.append(tree2.are_dual, [not tree1.are_dual[-1]]),
                               np.append(tree2.inner_sectors, [tree2.coupled], axis=0),
                               np.append(tree2.multiplicities, [0]))

        b_sym = symmetry._b_symbol(tree1_in_1, tree1.uncoupled[-1], tree1.coupled)
        if not bend_up:
            b_sym = b_sym.conj()
        if tree1.are_dual[-1]:
            b_sym *= symmetry.frobenius_schur(tree1.uncoupled[-1])
        mu = tree1.multiplicities[-1] if tree1.multiplicities.shape[0] > 0 else 0
        for nu in range(b_sym.shape[1]):
            if abs(b_sym[mu, nu]) < backend.eps:
                continue
            new_tree2.multiplicities[-1] = nu

            if bend_up:
                alpha_slice = new_codomain.tree_block_slice(new_tree2)
                if new_tree1.uncoupled.shape[0] == 0:
                    beta_slice = slice(0, 1)
                else:
                    beta_slice = new_domain.tree_block_slice(new_tree1)
            else:
                if new_tree1.uncoupled.shape[0] == 0:
                    alpha_slice = slice(0, 1)
                else:
                    alpha_slice = new_codomain.tree_block_slice(new_tree1)
                beta_slice = new_domain.tree_block_slice(new_tree2)

            new_data.blocks[block_ind][alpha_slice, beta_slice] += b_sym[mu, nu] * tree_block
    new_data.discard_zero_blocks(block_backend, backend.eps)
    return new_data, new_codomain, new_domain


def cross_check_transpose(ten: SymmetricTensor
                          ) -> tuple[fusion_tree_backend.FusionTreeData, TensorProduct, TensorProduct]:
    """There are two different ways to compute the transpose when using `permute_legs`.
    The one used in `FusionTreeBackend` corresponds to twisting the legs in the codomain
    and braiding them *over* the legs in the domain. This should be equivalent to twisting
    the codomain legs in the opposite way and and braiding them *under* the legs in the domain.

    This function implements the latter approach.
    """
    ftb_TMD = fusion_tree_backend.TreeMappingDict
    codomain_idcs = list(range(ten.num_codomain_legs, ten.num_legs))
    domain_idcs = list(reversed(range(ten.num_codomain_legs)))
    levels = list(range(ten.num_legs))
    coupled = np.array([ten.domain.sector_decomposition[i[1]] for i in ten.data.block_inds])

    mapping_twists = ftb_TMD.from_topological_twists(ten.codomain, coupled, inverse=True)
    mapping_twists = mapping_twists.add_tensorproduct(ten.domain, coupled, index=1)

    mapping_permute, codomain, domain = ftb_TMD.from_permute_legs(
        a=ten, codomain_idcs=codomain_idcs, domain_idcs=domain_idcs, levels=levels
    )
    full_mapping = mapping_twists.compose(mapping_permute)

    axes_perm = codomain_idcs + domain_idcs
    axes_perm = [i if i < ten.num_codomain_legs else ten.num_legs - 1 - i + ten.num_codomain_legs
                 for i in axes_perm]
    assert axes_perm == list(reversed(range(ten.num_legs)))
    data = full_mapping.apply_to_tensor(ten, codomain, domain, axes_perm, None)
    return data, codomain, domain
