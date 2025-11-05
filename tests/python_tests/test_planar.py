import pytest
import cyten as ct


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
        order='definition',
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
    # compare to manual, using planar routines
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
    # compare to manual, using general (not planar) routines
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
