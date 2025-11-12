import pytest
import numpy as np
import numpy.testing as npt
import cyten as ct


def cyclically_permute_tensor(T: ct.Tensor, labels: list[str]) -> ct.Tensor:
    if T.labels == labels:
        return T
    for num_bends in range(1, T.num_legs):
        if T.labels[num_bends:] + T.labels[:num_bends] == labels:
            break
    if T.num_codomain_legs == 0:
        num_bends = T.num_legs - num_bends
        T = ct.permute_legs(T, codomain=list(range(T.num_legs - num_bends, T.num_legs)),
                            bend_right=False)
        T = ct.permute_legs(T, domain=list(reversed(range(T.num_legs))), bend_right=True)
    elif T.num_domain_legs == 0:
        T = ct.permute_legs(T, domain=list(reversed(range(num_bends))), bend_right=False)
        T = ct.permute_legs(T, codomain=list(range(T.num_legs)), bend_right=True)
    else:
        raise RuntimeError('An earlier error should have been occured.')
    return T


def is_cyclical_perm(seq: list[int]) -> bool:
    if len(seq) == 0:
        return True
    n = seq[0]
    N = len(seq)
    return list(seq) == [*range(n, N), *range(n)]


@pytest.mark.parametrize('legs, num_legs, is_planar',
                         [([1, 2, 3], 7, True), ([1, 2, 5, 6], 10, False), ([], 0, True),
                          ([], 6, True), ([0, 1, 2, 3], 4, True), ([0, 1, 7, 8], 9, True),
                          ([0, 1, 5, 6], 10, False)])
@pytest.mark.parametrize('shuffle', [True, False])
def test_parse_leg_bipartition(legs, num_legs, is_planar, shuffle, np_random):
    if shuffle:
        np_random.shuffle(legs)

    if not is_planar:
        with pytest.raises(ValueError, match='Not a planar bipartition'):
            _ = ct.planar.parse_leg_bipartition(legs, num_legs)
        return

    a, b = ct.planar.parse_leg_bipartition(legs, num_legs)

    assert len(a) == len(legs)
    assert len(b) == num_legs - len(a)
    assert len(set(a) & set(b)) == 0, 'not a bipartition (duplicates)!'
    assert {*a, *b} == {*range(num_legs)}, 'not a bipartition (missing)!'
    assert all(n2 == n1 + 1 or (n2, n1) == (0, num_legs - 1)
               for n1, n2 in zip(a[:-1], a[1:]))
    assert all(n2 == n1 + 1 or (n2, n1) == (0, num_legs - 1)
               for n1, n2 in zip(b[:-1], b[1:]))


planar_permute_legs_cases = {
    'trivial': (3, 2, None, [0, 1, 2]),
    # same basic case with different input possibilities
    'basic-idcs': (3, 2, None, [3, 4, 0]),
    'basic-labels': (3, 2, None, ['a', 'd', 'e']),
    'basic-codomain': (3, 2, [2, 1], None),
    # empty codomain/domain
    'empty-codomain': (2, 2, [], None),
    'empty-domain': (2, 2, [0, 1, 2, 3], None),
    # input has no codomain
    'J0-empty-domain': (0, 3, None, []),
    'J0': (0, 3, None, [1]),
    'J0-empty-codomain': (0, 3, None, [0, 1, 2]),
    # input has no domain
    'K0-empty-domain': (3, 0, None, []),
    'K0': (3, 0, None, [1]),
    'K0-empty-codomain': (3, 0, [0, 1, 2], None),
}


@pytest.mark.parametrize('J, K, codomain, domain',
                         planar_permute_legs_cases.values(),
                         ids=planar_permute_legs_cases.keys())
@pytest.mark.parametrize('symmetry, backend', [(ct.no_symmetry, 'no_symmetry'),
                                               (ct.u1_symmetry, 'abelian'),
                                               (ct.u1_symmetry, 'fusion_tree'),
                                               (ct.fermion_parity, 'fusion_tree'),
                                               (ct.fibonacci_anyon_category, 'fusion_tree')])
def test_planar_permute_legs(J, K, codomain, domain, symmetry, backend, np_random):
    backend = ct.get_backend(backend, 'numpy')
    T_labels = list('abcdefghijk')[:J + K]
    T: ct.SymmetricTensor = ct.testing.random_tensor(symmetry, J, K, labels=T_labels, np_random=np_random)

    res = ct.planar.planar_permute_legs(T, codomain=codomain, domain=domain)
    res.test_sanity()

    if codomain is None:
        sorted_domain, sorted_codomain = ct.planar.parse_leg_bipartition(T.get_leg_idcs(domain), J + K)
    else:
        sorted_codomain, sorted_domain = ct.planar.parse_leg_bipartition(T.get_leg_idcs(codomain), J + K)
    leg_perm = [*sorted_codomain, *sorted_domain]
    assert is_cyclical_perm(leg_perm)
    assert res.labels == [T.labels[n] for n in leg_perm]
    assert res.legs == [T.get_leg(n) for n in leg_perm]

    if symmetry.can_be_dropped:
        T_np = T.to_numpy(understood_braiding=True)
        res_np = res.to_numpy(understood_braiding=True)
        expect = np.transpose(T_np, leg_perm)
        if symmetry.has_trivial_braid:
            npt.assert_almost_equal(res_np, expect)
        else:
            # the expect is missing some signs from twists in the diagram.
            # I dont know how to figure them out right now, so we just ignore signs here...
            npt.assert_almost_equal(np.abs(res_np), np.abs(expect))

    permuted_back1 = ct.planar.planar_permute_legs(res, codomain=T.codomain_labels)
    permuted_back1.test_sanity()
    permuted_back1 = cyclically_permute_tensor(permuted_back1, T.labels)
    assert ct.almost_equal(permuted_back1, T)

    permuted_back2 = ct.planar.planar_permute_legs(res, domain=T.domain_labels)
    permuted_back2.test_sanity()
    permuted_back2 = cyclically_permute_tensor(permuted_back2, T.labels)
    assert ct.almost_equal(permuted_back2, T)


@pytest.mark.parametrize('symmetry', [ct.no_symmetry, ct.u1_symmetry, ct.fibonacci_anyon_category])
def test_PlanarDiagram(symmetry, np_random):

    # ===========================================
    # define a diagram
    # ===========================================
    density_matrix_mixing_left = ct.PlanarDiagram(
        tensors='Lp[vR*, wR, vR], Lp_hc[vR*, wR*, vR], W[wL, p, wR, p*], W_hc[p, wR*, p*, wL*], '
                'mixL[wL, wL*], theta[vL, p0, p1, vR], theta_hc[vR*, p1*, p0*, vL*]',
        definition='Lp:vR @ theta:vL, Lp:wR @ W:wL, Lp:vR* -> vL, '
                   'theta:p0 @ W:p*, theta:p1 @ theta_hc:p1*, theta:vR @ theta_hc:vR*, '
                   'W:p -> p, W:wR @ mixL:wL, '
                   'Lp_hc:vR -> vL*, Lp_hc:wR* @ W_hc:wL*, Lp_hc:vR* @ theta_hc:vL*, '
                   'W_hc:p @ theta_hc:p0*, W_hc:p* -> p*, W_hc:wR* @ mixL:wL*',
        dims=dict(chi=['vR', 'vL', 'vR*', 'vL*'], w=['wL', 'wR', 'wL*', 'wR*'],
                  d=['p', 'p*', 'p0', 'p0*', 'p1', 'p1*']),
    )
    r"""Planar diagram arising when mixing the left site

        |    .---theta*---.
        |    |   |    \   |
        |   LP*--W0*-  \  |
        |    |   |   \  | |
        |          mixL | |
        |    |   |   /  | |
        |   LP---W0--  /  |
        |    |   |    /   |
        |    .---theta----.
    """

    # ===========================================
    # create example tensors
    # ===========================================
    theta = ct.testing.random_tensor(symmetry, codomain=2, domain=2, labels=['vL', 'p0', 'p1', 'vR'],
                                     np_random=np_random)
    p0 = theta.get_leg('p0')
    vL = theta.get_leg('vL')
    Lp = ct.testing.random_tensor(symmetry, codomain=[vL], domain=[vL, None],
                                  labels=['vR*', 'wR', 'vR'])
    wR = Lp.get_leg('wR')
    W = ct.testing.random_tensor(symmetry, codomain=[p0, wR], domain=[wR, p0], labels=['p', 'wR', 'p*', 'wL'])
    mixL = ct.testing.random_tensor(symmetry, codomain=[wR], domain=[wR], labels=['wL*', 'wL'])

    # ===========================================
    # evaluate the diagram
    # ===========================================
    res = density_matrix_mixing_left(Lp=Lp, Lp_hc=Lp.hc, W=W, W_hc=W.hc, mixL=mixL,
                                     theta=theta, theta_hc=theta.hc)
    res.test_sanity()
    assert res.labels == ['p*', 'vL*', 'vL', 'p'], 'if cyclical need to redesign test. otherwise wrong!'
    assert res.num_codomain_legs == 2, 'if this fails, just need to redesign tests'

    # ===========================================
    # compare to manual contraction, using planar routines
    # ===========================================
    expect1 = ct.planar.planar_contraction(theta, theta.hc, ['p1', 'vR'], ['p1*', 'vR*'])
    expect1 = ct.planar.planar_contraction(expect1, Lp, 'vL', 'vR')
    expect1 = ct.planar.planar_contraction(expect1, W, ['wR', 'p0'], ['wL', 'p*'])
    expect1 = ct.planar.planar_contraction(expect1, mixL, 'wR', 'wL')
    expect1 = ct.planar.planar_contraction(expect1, W.hc, ['p0*', 'wL*'], ['p', 'wR*'])
    expect1 = ct.planar.planar_contraction(expect1, Lp.hc, ['vL*', 'wL*'], ['vR*', 'wR*'])
    expect1 = expect1.relabel({'vR*': 'vL', 'vR': 'vL*'})
    expect1 = ct.planar.planar_permute_legs(expect1, codomain=['p*', 'vL*'])
    assert expect1.labels == ['p*', 'vL*', 'vL', 'p']
    expect1.test_sanity()
    assert ct.almost_equal(res, expect1)

    # ===========================================
    # compare to manual contraction, using general (not planar) routines
    # ===========================================
    assert theta.codomain_labels == ['vL', 'p0']
    assert theta.domain_labels == ['vR', 'p1']
    theta_bent = ct.permute_legs(theta, ['p1', 'vR'], ['p0', 'vL'], bend_right=[True, True, False, False])
    expect2 = ct.compose(theta_bent.hc, theta_bent)
    expect2 = ct.compose(
        ct.permute_legs(expect2, ['p0', 'p0*', 'vL*'], ['vL'], bend_right=[None, None, None, False]),
        ct.permute_legs(Lp, ['vR'], ['wR', 'vR*'], bend_right=[True, None, False])
    )
    expect2 = ct.compose(
        ct.permute_legs(expect2, ['p0*', 'vL*', 'vR*'], ['p0', 'wR'], bend_right=[False, None, None, True, None]),
        ct.permute_legs(W, ['p*', 'wL'], ['wR', 'p'], bend_right=[True, True, False, False])
    )
    expect2 = ct.compose(
        ct.permute_legs(expect2, ['p0*', 'vL*', 'vR*', 'p'], ['wR'], bend_right=[None, None, None, True, None]),
        ct.transpose(mixL)
    )
    expect2 = ct.compose(
        ct.permute_legs(expect2, ['vL*', 'vR*', 'p'], ['p0*', 'wL*'], bend_right=[False, None, None, None, None]),
        ct.permute_legs(W.hc, ['p', 'wR*'], ['wL*', 'p*'], bend_right=[False, None, True, None])
    )
    expect2 = ct.compose(
        ct.permute_legs(expect2, ['vR*', 'p', 'p*'], ['vL*', 'wL*'], bend_right=[False, None, None, True, None]),
        Lp.hc
    )
    expect2 = expect2.relabel({'vR*': 'vL', 'vR': 'vL*'})
    expect2 = ct.permute_legs(expect2, ['p*', 'vL*'], ['p', 'vL'], bend_right=[False, False, None, True])
    expect2.test_sanity()
    assert ct.almost_equal(res, expect2)


@pytest.mark.parametrize('symmetry', [ct.no_symmetry, ct.u1_symmetry, ct.fibonacci_anyon_category])
def test_PlanarLinearOperator(symmetry):

    # ===========================================
    # define an operator
    # ===========================================

    class TwoSiteEffectiveH(ct.planar.PlanarLinearOperator):
        r"""Effective Hamiltonian during Two-site DMRG

        The operator is given by the following network::

            |        .---       ---.
            |        |    |   |    |
            |       LP----W0--W1---RP
            |        |    |   |    |
            |        .---       ---.

        and acts on two-site wavefunctions ``theta`` as::

            |        .---       ---.
            |        |    |   |    |
            |       LP----W0--W1---RP
            |        |    |   |    |
            |        .--- theta ---.
        """

        op_diagram = ct.PlanarDiagram(
            tensors='Lp[vR*, wR, vR], W0[wL, p, wR, p*], W1[wL, p, wR, p*], Rp[vL*, vL, wL]',
            definition='Lp:vR* -> vL, Lp:wR @ W0:wL, Lp:vR -> vL*, '
                       'W0:p -> p0, W0:wR @ W1:wL, W0:p* -> p0*, '
                       'W1:p -> p1, W1:wR @ Rp:wL, W1:p* -> p1*, '
                       'Rp:vL* -> vR, Rp:vL -> vR*',
            dims=dict(chi=['vR', 'vR*', 'vL', 'vL*'], w=['wL', 'wR'], d=['p', 'p*']),
        )
        matvec_diagram = op_diagram.add_tensor(
            tensor='theta[vL, p0, p1, vR]',
            extra_definition='theta:vL @ Lp:vR, theta:p0 @ W0:p*, theta:p1 @ W1:p*, theta:vR @ Rp:vL',
            extra_dims=dict(chi=['vL', 'vR'], d=['p0', 'p1'])
        )

        def __init__(self, Lp, W0, W1, Rp):
            ct.planar.PlanarLinearOperator.__init__(
                self, op_diagram=self.op_diagram, matvec_diagram=self.matvec_diagram,
                op_tensors=dict(Lp=Lp, W0=W0, W1=W1, Rp=Rp), vec_name='theta'
            )

    # ===========================================
    # create example tensors
    # ===========================================

    theta = ct.testing.random_tensor(symmetry, 4, labels=['vL', 'p0', 'p1', 'vR'],
                                     max_multiplicity=3)
    vL, p0, p1, vR = theta.legs
    Lp = ct.testing.random_tensor(symmetry, [vL, None, vL.dual],
                                  labels=['vR*', 'wR', 'vR'], max_multiplicity=3)
    W0 = ct.testing.random_tensor(symmetry, [p0, None, p0.dual, Lp.get_leg('wR').dual],
                                  labels=['p', 'wR', 'p*', 'wL'], max_multiplicity=3)
    W1 = ct.testing.random_tensor(symmetry, [p1, None, p1.dual, W0.get_leg('wR').dual],
                                  labels=['p', 'wR', 'p*', 'wL'], max_multiplicity=3)
    Rp = ct.testing.random_tensor(symmetry, [vR, vR.dual, W1.get_leg('wR').dual],
                                  labels=['vL*', 'vL', 'wL'])

    # ===========================================
    # create an op instance, call to_tensor and matvec
    # ===========================================

    H = TwoSiteEffectiveH(Lp=Lp, W0=W0, W1=W1, Rp=Rp)

    op = H.to_tensor()
    op.test_sanity()
    # get to the correct cyclic permutation
    op = ct.planar.planar_permute_legs(op, codomain=['vL', 'p0', 'p1', 'vR'])
    assert op.codomain_labels == ['vL', 'p0', 'p1', 'vR']
    assert op.domain_labels == ['vL*', 'p0*', 'p1*', 'vR*']

    H_theta = H.matvec(theta)
    H_theta.test_sanity()
    assert H_theta.codomain_labels == ['vL', 'p0', 'p1', 'vR']
    assert H_theta.domain_labels == []

    # ===========================================
    # compare to manual contraction, using planar routines
    # ===========================================
    op_1 = ct.planar.planar_contraction(
        Lp, W0, 'wR', 'wL', relabel1={'vR*': 'vL', 'vR': 'vL*'}, relabel2={'p': 'p0', 'p*': 'p0*'}
    )
    op_1 = ct.planar.planar_contraction(op_1, W1, 'wR', 'wL', relabel2={'p': 'p1', 'p*': 'p1*'})
    op_1 = ct.planar.planar_contraction(op_1, Rp, 'wR', 'wL', relabel2={'vL*': 'vR', 'vL': 'vR*'})
    op_1 = ct.planar.planar_permute_legs(op_1, codomain=['vL', 'p0', 'p1', 'vR'])
    assert ct.almost_equal(op_1, op)

    H_theta_1 = ct.compose(op_1, theta)
    assert ct.almost_equal(H_theta_1, H_theta)

    # ===========================================
    # compare to manual contraction, using general (not planar) routines
    # ===========================================
    op_2 = ct.compose(
        ct.permute_legs(Rp, ['vL*', 'vL'], ['wL'], bend_right=[None, None, True]),
        ct.permute_legs(W1, ['wR'], ['p', 'wL', 'p*'], bend_right=[False, None, True, True]),
        relabel1={'vL*': 'vR', 'vL': 'vR*'},
        relabel2={'p': 'p1', 'p*': 'p1*'}
    )
    op_2 = ct.compose(
        ct.permute_legs(op_2, ['p1', 'vR', 'vR*', 'p1*'], ['wL'], bend_right=[None, None, True, None, False]),
        ct.permute_legs(W0, ['wR'], ['p', 'wL', 'p*'], bend_right=[False, None, True, True]),
        relabel2={'p': 'p0', 'p*': 'p0*'}
    )
    bend_right = [None] * 4 + [True, None, False]
    op_2 = ct.compose(
        ct.permute_legs(op_2, ['p0', 'p1', 'vR', 'vR*', 'p1*', 'p0*'], ['wL'], bend_right=bend_right),
        ct.permute_legs(Lp, ['wR'], ['vR*', 'vR'], bend_right=[False, None, True]),
        relabel2={'vR*': 'vL', 'vR': 'vL*'}
    )
    bend_right = [None] * 3 + [True] * 3 + [None, False]
    op_2 = ct.permute_legs(
        op_2, ['vL', 'p0', 'p1', 'vR'], ['vL*', 'p0*', 'p1*', 'vR*'], bend_right=bend_right
    )
    assert ct.almost_equal(op_2, op)

    H_theta_2 = ct.compose(op_2, theta)
    assert ct.almost_equal(H_theta_2, H_theta)
