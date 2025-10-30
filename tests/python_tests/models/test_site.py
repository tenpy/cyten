"""A collection of tests for :mod:`cyten.networks.site`."""
# Copyright (C) TeNPy Developers, Apache license


import numpy as np
import numpy.testing as npt
import itertools as it
import pytest

import cyten
from cyten import backends
from cyten.models import degrees_of_freedom, sites
from cyten.testing import random_symmetry_sectors


def check_same_operators(sites: list[degrees_of_freedom.Site]):
    """Check that the given sites have equivalent operators.

    If operators with matching names exist, we check if they have the same dense array representation.
    If an operator is missing on some of the sites, that is ok.
    """
    ops = {}
    for site in sites:
        for name, op in site.onsite_operators.items():
            op = op.to_numpy(understood_braiding=True)
            if name in ops:  # only as far as defined before
                npt.assert_equal(op, ops[name])
            else:
                ops[name] = op


def check_operator_availability(site: degrees_of_freedom.Site,
                                expect_onsite_ops: dict[str, bool]):
    """Check if the operators on a site are as expected.

    We check if the operators exist and whether they are `DiagonalTensor`s.
    """
    assert set(site.onsite_operators.keys()) == set(expect_onsite_ops.keys())
    for name, is_diag in expect_onsite_ops.items():
        assert isinstance(site.onsite_operators[name], cyten.DiagonalTensor) == is_diag


@pytest.mark.parametrize('symmetry_backend, use_sym',
                         [('abelian', True), ('fusion_tree', True), ('abelian', False), ('no_symmetry', False)])
def test_site(np_random, block_backend, symmetry_backend, use_sym):
    backend = cyten.get_backend(block_backend=block_backend, symmetry=symmetry_backend)
    if use_sym:
        sym = cyten.u1_symmetry * cyten.z3_symmetry
    else:
        sym = cyten.no_symmetry
    dim = 8
    some_sectors = random_symmetry_sectors(sym, num=dim, sort=False, np_random=np_random)
    leg = cyten.ElementarySpace.from_basis(sym, np_random.choice(some_sectors, size=dim, replace=True))
    assert leg.dim == dim
    labels = {f'x{j:d}': i for i, j in enumerate(range(10, 10 + dim))}
    site = degrees_of_freedom.Site(leg, backend=backend, state_labels=labels)
    assert site.state_index('x10') == 0
    assert site.state_index('x17') == 7
    assert set(site.onsite_operators.keys()) == set()
    op1 = cyten.SymmetricTensor.from_random_uniform([leg], [leg], backend=backend, labels=['p', 'p*'])
    site.add_onsite_operator('silly_op', op1)
    assert set(site.onsite_operators.keys()) == {'silly_op'}
    assert site.onsite_operators['silly_op'] is op1

    op2 = cyten.SymmetricTensor.from_random_uniform([leg], [leg], backend=backend, labels=['p', 'p*'])
    site.add_onsite_operator('op2', op2)
    assert site.onsite_operators['op2'] is op2

    op3_dense = np.diag(np.arange(10, 10 + dim))
    site.add_onsite_operator('op3', op3_dense, is_diagonal=True)
    assert isinstance(site.onsite_operators['op3'], cyten.DiagonalTensor)
    npt.assert_equal(site.onsite_operators['op3'].to_numpy(), op3_dense)
    
    # if use_sym:
    #     leg2 = leg.drop_symmetry(1)
    #     leg2 = leg2.change_symmetry(symmetry=cyten.z3_symmetry, sector_map=lambda s: s % 3)
    # else:
    #     leg2 = leg
    # leg2.test_sanity()
    # site2 = copy.deepcopy(site)
    # site2.change_leg(leg2)
    # for name in ['silly_op', 'op2', 'op3']:
    #     site_op = site.onsite_operators[name].to_numpy()
    #     site2_op = site2.onsite_operators[name].to_numpy()
    #     npt.assert_equal(site_op, site2_op)


@pytest.mark.parametrize('spin', [0.5, 1, 1.5, 2, 5])
def test_spin_site(any_backend, spin):
    if isinstance(any_backend, backends.NoSymmetryBackend):
        all_conserve = ['None']
    elif isinstance(any_backend, backends.AbelianBackend):
        all_conserve = ['Sz', 'parity', 'None']
    elif isinstance(any_backend, backends.FusionTreeBackend):
        all_conserve = ['SU(2)', 'Sz', 'parity', 'None']
    else:
        raise ValueError
    site_list = []
    for conserve in all_conserve:
        print("conserve = ", conserve)
        site = sites.SpinSite(spin, conserve, backend=any_backend)
        site.test_sanity()

        sz = site.spin_vector[:, :, 2]
        down = site.state_labels['down']
        up = site.state_labels['up']
        assert np.allclose(sz[down, down], -1 * spin)
        assert np.allclose(sz[up, up], spin)

        expect_ops = {}
        if conserve in ['Sz', 'parity', 'None']:
            expect_ops.update(Sz=True)
            if site.double_total_spin == 1:
                expect_ops.update(Sigmaz=True)
        if conserve in ['None']:
            expect_ops.update(Sx=False, Sy=False, Sp=False, Sm=False)
            if site.double_total_spin == 1:
                expect_ops.update(Sigmax=False, Sigmay=False)
        check_operator_availability(site, expect_ops)

        site_list.append(site)
    check_same_operators(site_list)


@pytest.mark.parametrize('Nmax', [1, 2, 5, 8])
def test_spinless_boson_site(any_backend, np_random, Nmax):
    if isinstance(any_backend, backends.NoSymmetryBackend):
        all_conserve = ['None']
    elif isinstance(any_backend, (backends.AbelianBackend, backends.FusionTreeBackend)):
        all_conserve = ['N', 'parity', 'None']
    else:
        raise ValueError
    filling = np_random.choice([None, np_random.random()])
    site_list = []
    for conserve in all_conserve:
        print("conserve = ", conserve)
        site = sites.SpinlessBosonSite(Nmax, conserve=conserve, backend=any_backend, filling=filling)
        site.test_sanity()

        vac = site.state_labels['vac']
        assert np.allclose(site.n_tot[vac, vac], 0)
        for i in range(Nmax):
            state = site.state_labels[str(i)]
            assert np.allclose(site.n_tot[state, state], i)

        expect_ops = dict(N0=True, N=True, Ntot=True)
        expect_ops.update(N0N0=True, NN=True, NtotNtot=True)
        expect_ops.update(P0=True, P=True, Ptot=True)
        if filling is not None:
            expect_ops.update(dN=True, dNdN=True)
        check_operator_availability(site, expect_ops)

        site_list.append(site)
    check_same_operators(site_list)

    # now do the same tests for two species
    Nmax2 = np_random.choice(5)
    if isinstance(any_backend, backends.NoSymmetryBackend):
        all_conserve = ['None']
    else:
        all_conserve = ['N', 'parity', 'None', ['N', 'N'], ['N', 'parity'], ['N', 'None'],
                        ['parity', 'N'], ['parity', 'parity'], ['parity', 'None'],
                        ['None', 'N'], ['None', 'parity'], ['None', 'None']]
    filling = np_random.choice([None, np_random.random()])
    site_list = []
    for conserve in all_conserve:
        site = sites.SpinlessBosonSite([Nmax, Nmax2], conserve=conserve, backend=any_backend, filling=filling)
        site.test_sanity()

        vac = site.state_labels['vac']
        assert np.allclose(site.n_tot[vac, vac], 0)
        for i in range(Nmax):
            for j in range(Nmax2):
                state = site.state_labels[f'({i}, {j})']
                assert np.allclose(site.number_operators[state, state, 0], i)
                assert np.allclose(site.number_operators[state, state, 1], j)

        expect_ops = dict(N0=True, N1=True, Ntot=True)
        expect_ops.update(N0N0=True, N1N1=True, NtotNtot=True)
        expect_ops.update(P0=True, P1=True, Ptot=True)
        if filling is not None:
            expect_ops.update(dN=True, dNdN=True)
        check_operator_availability(site, expect_ops)

        site_list.append(site)
    check_same_operators(site_list)
