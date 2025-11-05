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


def generate_spin_dofs(backend: backends.TensorBackend) -> list[degrees_of_freedom.SpinDOF]:
    """Return a list of `SpinDOF` sites whose symmetries are consistent with `backend`."""
    site_list = []
    for spin in [.5, 1, 1.5, 2]:
        site_list.append(sites.SpinSite(S=spin, conserve='None', backend=backend))
        if not isinstance(backend, backends.NoSymmetryBackend):
            site_list.append(sites.SpinSite(S=spin, conserve='parity', backend=backend))
            site_list.append(sites.SpinSite(S=spin, conserve='Sz', backend=backend))
        if isinstance(backend, backends.FusionTreeBackend):
            site_list.append(sites.SpinSite(S=spin, conserve='SU(2)', backend=backend))
    if isinstance(backend, backends.FusionTreeBackend):
        all_conserve_N = ['N', 'parity']
        all_conserve_S = ['SU(2)', 'Sz', 'parity', 'None']
        for conserve_N, conserve_S in it.product(all_conserve_N, all_conserve_S):
            site_list.append(sites.SpinHalfFermionSite(conserve_N, conserve_S, backend=backend))
    return site_list


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


# TEST SPIN COUPLINGS


def test_spin_spin_coupling(any_backend, np_random):
    check_dense_blocks = np_random.choice([True, False])
    site_list = generate_spin_dofs(any_backend)
    num_sites = min(5, len(site_list))
    site_list = np_random.choice(site_list, size=num_sites, replace=False)
    for i, site1 in enumerate(site_list):
        Jx, Jy, Jz = np_random.random(3)
        # either SpinSite or SpinHalfFermionSite
        conserve = site1.conserve if isinstance(site1, sites.SpinSite) else site1.conserve_S
        if conserve in ['Sz']:
            Jx = Jy = 0
        elif conserve in ['SU(2)']:
            Jx = Jy = Jz

        # test different site combinations
        for site2 in site_list[:i + 1]:
            # Note: is_same_symmetry does not work here since it does not distinguish
            # between U(1) fermion number symmetry and Sz spin symmetry for fermions
            if not site1.symmetry == site2.symmetry:
                continue
            coupling = couplings.spin_spin_coupling([site1, site2], Jx=Jx, Jy=Jy, Jz=Jz)
            coupling.test_sanity()
            if check_dense_blocks:
                expect = site1.spin_vector.copy()
                expect[:, :, 0] *= Jx
                expect[:, :, 1] *= Jy
                expect[:, :, 2] *= Jz
                expect = np.tensordot(expect, site2.spin_vector, axes=[2, 2])
                expect = np.transpose(expect, [0, 2, 3, 1])
                assert np.allclose(coupling.to_numpy(understood_braiding=True), expect)

    # test correct number of sites
    with pytest.raises(AssertionError):
        _ = couplings.spin_spin_coupling([site_list[0]], Jx=1., Jy=1., Jz=1.)
    with pytest.raises(AssertionError):
        _ = couplings.spin_spin_coupling([site_list[0]] * 3, Jx=1., Jy=1., Jz=1.)


def test_spin_field_coupling(any_backend, np_random):
    check_dense_blocks = np_random.choice([True, False])
    site_list = generate_spin_dofs(any_backend)
    num_sites = min(5, len(site_list))
    site_list = np_random.choice(site_list, size=num_sites, replace=False)
    for site in site_list:
        hx, hy, hz = np_random.random(3)
        # either SpinSite or SpinHalfFermionSite
        conserve = site.conserve if isinstance(site, sites.SpinSite) else site.conserve_S
        if conserve in ['Sz', 'parity']:
            hx = hy = 0
        elif conserve in ['SU(2)']:
            # coupling not allowed
            continue
        coupling = couplings.spin_field_coupling([site], hx=hx, hy=hy, hz=hz)
        coupling.test_sanity()
        if check_dense_blocks:
            expect = site.spin_vector
            expect = hx * expect[:, :, 0] + hy * expect[:, :, 1] + hz * expect[:, :, 2]
            assert np.allclose(coupling.to_numpy(understood_braiding=True), expect)

    # test correct number of sites
    with pytest.raises(AssertionError):
        _ = couplings.spin_field_coupling([site_list[0]] * 2, hx=1.)


def test_aklt_coupling(any_backend, np_random):
    check_dense_blocks = np_random.choice([True, False])
    site_list = generate_spin_dofs(any_backend)
    num_sites = min(5, len(site_list))
    site_list = np_random.choice(site_list, size=num_sites, replace=False)
    for i, site1 in enumerate(site_list):
        J = np_random.random()
        # test different site combinations
        for site2 in site_list[:i + 1]:
            if not site1.symmetry == site2.symmetry:
                continue
            coupling = couplings.aklt_coupling([site1, site2], J=J)
            coupling.test_sanity()
            if check_dense_blocks:
                expect = np.tensordot(site1.spin_vector, site2.spin_vector, axes=[2, 2])
                expect = np.transpose(expect, [0, 2, 3, 1])
                expect += np.tensordot(expect, expect, axes=[[2, 3], [1, 0]]) / 3.
                assert np.allclose(coupling.to_numpy(understood_braiding=True), J * expect)

    # test correct number of sites
    with pytest.raises(AssertionError):
        _ = couplings.spin_spin_coupling([site_list[0]])
    with pytest.raises(AssertionError):
        _ = couplings.spin_spin_coupling([site_list[0]] * 3)


def test_chiral_3spin_coupling(any_backend, np_random):
    check_dense_blocks = np_random.choice([True, False])
    site_list = generate_spin_dofs(any_backend)
    num_sites = min(3, len(site_list))
    site_list = np_random.choice(site_list, size=num_sites, replace=False)
    for i, site1 in enumerate(site_list):
        chi = np_random.random()
        # test different site combinations
        for site2 in site_list[:i + 1]:
            if not site1.symmetry == site2.symmetry:
                continue
            site3 = np_random.choice([site1, site2])
            coupling = couplings.chiral_3spin_coupling([site1, site2, site3], chi=chi)
            coupling.test_sanity()
            if check_dense_blocks:
                s1, s2, s3 = site1.spin_vector, site2.spin_vector, site3.spin_vector
                expect = 0
                for i in range(3):
                    j = (i + 1) % 3
                    k = (i + 2) % 3
                    expect += (s1[:, None, None, None, None, :, i]
                               * s2[None, :, None, None, :, None, j]
                               * s3[None, None, :, :, None, None, k])
                    expect -= (s1[:, None, None, None, None, :, i]
                               * s2[None, :, None, None, :, None, k]
                               * s3[None, None, :, :, None, None, j])
                assert np.allclose(coupling.to_numpy(understood_braiding=True), chi * expect)

    # test correct number of sites
    with pytest.raises(AssertionError):
        _ = couplings.chiral_3spin_coupling([site_list[0]])
    with pytest.raises(AssertionError):
        _ = couplings.chiral_3spin_coupling([site_list[0]] * 2)
