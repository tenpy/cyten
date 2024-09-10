"""A collection of tests for teh tools submodules."""
# Copyright (C) TeNPy Developers, GNU GPLv3

import numpy as np
import numpy.testing as npt
import itertools as it
from cytnx import tools
import warnings
import pytest
import os.path


# TODO use fixtures, e.g. np_random


def test_inverse_permutation(N=10):
    x = np.random.random(N)
    p = np.arange(N)
    np.random.shuffle(p)
    xnew = x[p]
    pinv = tools.misc.inverse_permutation(p)
    npt.assert_equal(x, xnew[pinv])
    npt.assert_equal(pinv[p], np.arange(N))
    npt.assert_equal(p[pinv], np.arange(N))
    pinv2 = tools.misc.inverse_permutation(tuple(p))
    npt.assert_equal(pinv, pinv2)


@pytest.mark.parametrize('stable', [True, False])
def test_rank_data(stable, N=10):
    float_data = np.random.random(N)
    int_data = np.random.randint(2 * N, size=N)
    int_data[-2] = int_data[2]  # make sure there is a duplicate to check stability
    for data in [int_data, float_data]:
        print(f'data={data}')
        ranks = tools.misc.rank_data(data, stable=stable)
        print(f'ranks={ranks}')
        if stable:
            # check vs known implementation
            ranks2 = np.argsort(np.argsort(data, kind='stable'), kind='stable')
            npt.assert_array_equal(ranks, ranks2)
        # check defining property
        for i, ai in enumerate(data):
            for j, aj in enumerate(data):
                if ai < aj:
                    assert ranks[i] < ranks[j]
                elif ai > aj:
                    assert ranks[i] > ranks[j]
                elif stable:
                    assert (i < j) == (ranks[i] < ranks[j])
                    assert (i > j) == (ranks[i] > ranks[j])


def test_argsort():
    x = [1., -1., 1.5, -1.5, 2.j, -2.j]
    npt.assert_equal(tools.misc.argsort(x, 'LM', kind='stable'), [4, 5, 2, 3, 0, 1])
    npt.assert_equal(tools.misc.argsort(x, 'SM', kind='stable'), [0, 1, 2, 3, 4, 5])
    npt.assert_equal(tools.misc.argsort(x, 'LR', kind='stable'), [2, 0, 4, 5, 1, 3])


def test_speigs():
    x = np.array([1., -1.2, 1.5, -1.8, 2.j, -2.2j])
    tol_NULP = len(x)**3
    x_LM = x[tools.misc.argsort(x, 'm>')]
    x_SM = x[tools.misc.argsort(x, 'SM')]
    A = np.diag(x)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')  # disable warngings temporarily
        for k in range(4, 9):
            print(k)
            W, V = tools.math.speigs(A, k, which='LM')
            W = W[tools.misc.argsort(W, 'LM')]
            print(W, x_LM[:k])
            npt.assert_array_almost_equal_nulp(W, x_LM[:k], tol_NULP)
            W, V = tools.math.speigs(A, k, which='SM')
            W = W[tools.misc.argsort(W, 'SM')]
            print(W, x_SM[:k])
            npt.assert_array_almost_equal_nulp(W, x_SM[:k], tol_NULP)


def test_find_subclass():
    pytest.skip()
    BaseCls = tenpy.models.lattice.Lattice
    SimpleLattice = tenpy.models.lattice.SimpleLattice  # direct sublcass of Lattice
    Square = tenpy.models.lattice.Square  # sublcass of SimpleLattice -> recursion necessary

    with pytest.raises(ValueError):
        tools.misc.find_subclass(BaseCls, 'UnknownSubclass')
    simple_found = tools.misc.find_subclass(BaseCls, 'SimpleLattice')
    assert simple_found is SimpleLattice
    square_found = tools.misc.find_subclass(BaseCls, 'Square')
    assert square_found is Square