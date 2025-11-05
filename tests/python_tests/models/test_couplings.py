"""A collection of tests for :mod:`cyten.models.couplings`."""
# Copyright (C) TeNPy Developers, Apache license


import numpy as np
from typing import Literal, Sequence
import itertools as it
import pytest

import cyten
from cyten import backends
from cyten import tensors
from cyten.models import degrees_of_freedom, sites, couplings
from cyten.symmetries import SymmetryError


@pytest.mark.parametrize('codom', [1, 2, 3])
def test_coupling(codom, make_compatible_space):
    legs = [make_compatible_space(max_sectors=3, max_mult=3) for _ in range(codom)]
    labels = [f'p{i}' for i in range(codom)]
    labels = [*labels, *[l + '*' for l in labels[::-1]]]
    T = tensors.SymmetricTensor.from_random_normal(codomain=legs, domain=legs, labels=labels)
    site_list = [degrees_of_freedom.Site(leg) for leg in legs]
    coupling = couplings.Coupling.from_tensor(T, site_list, name='name')
    coupling.test_sanity()
    assert coupling.name == 'name'
    assert coupling.num_sites == codom
    assert tensors.almost_equal(coupling.to_tensor(), T)
    if T.symmetry.can_be_dropped:
        coupling_to_numpy = coupling.to_numpy(understood_braiding=True)
        assert np.allclose(coupling_to_numpy, T.to_numpy(understood_braiding=True))
        coupling2 = couplings.Coupling.from_dense_block(coupling_to_numpy, site_list, understood_braiding=True)
        coupling2.test_sanity()
        assert np.all(coupling2.sites == coupling.sites)
        for i in range(codom):
            assert tensors.almost_equal(coupling2.factorization[i], coupling.factorization[i])
