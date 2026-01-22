import numpy as np
import numpy.testing as npt
import pytest

import cyten as ct


def is_cyclical_perm(seq: list[int]) -> bool:
    if len(seq) == 0:
        return True
    n = seq[0]
    N = len(seq)
    return list(seq) == [*range(n, N), *range(n)]


@pytest.mark.parametrize(
    'legs, num_legs, is_planar',
    [
        ([1, 2, 3], 7, True),
        ([1, 2, 5, 6], 10, False),
        ([], 0, True),
        ([], 6, True),
        ([0, 1, 2, 3], 4, True),
        ([0, 1, 7, 8], 9, True),
        ([0, 1, 5, 6], 10, False),
    ],
)
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
    assert all(n2 == n1 + 1 or (n2, n1) == (0, num_legs - 1) for n1, n2 in zip(a[:-1], a[1:]))
    assert all(n2 == n1 + 1 or (n2, n1) == (0, num_legs - 1) for n1, n2 in zip(b[:-1], b[1:]))


planar_combine_cases = {
    'trivial': ([['a', 'b'], ['c', 'd']], [['a'], ['c']], ['a', 'b'], ['c', 'd'], None),
    # without bending
    'codomain-1': ([['a', 'b', 'c'], ['d']], [['a', 'b']], ['a', 'b', 'c'], ['d'], None),
    'codomain-2': ([['a', 'b', 'c', 'd'], ['e']], [['a', 'b'], ['d', 'c']], ['a', 'b', 'c', 'd'], ['e'], None),
    'domain-1': ([['a'], ['b', 'c', 'd']], [['b', 'c']], ['a'], ['b', 'c', 'd'], None),
    'domain-2': ([['a'], ['b', 'c', 'd', 'e']], [['c', 'b'], ['d', 'e']], ['a'], ['b', 'c', 'd', 'e'], None),
    # bending right
    'bend-right-down-1': ([['a', 'b'], ['c']], [['b', 'c']], ['a', 'b', 'c'], [], True),
    'bend-right-down-2': ([['a', 'b'], ['c', 'd']], [['b', 'd', 'c']], ['a', 'b', 'd', 'c'], [], True),
    'bend-right-up-1': ([['a', 'b'], ['c']], [['c', 'b']], ['a'], ['c', 'b'], True),
    'bend-right-up-2': ([['a', 'b', 'c'], ['d', 'e']], [['e', 'd', 'b', 'c']], ['a'], ['d', 'e', 'c', 'b'], True),
    # bending left
    'bend-left-down-1': ([['a', 'b'], ['c']], [['a', 'c']], ['c', 'a', 'b'], [], False),
    'bend-left-down-2': ([['a', 'b'], ['c', 'd']], [['a', 'd', 'c']], ['d', 'c', 'a', 'b'], [], False),
    'bend-left-up-1': ([['a', 'b'], ['c']], [['c', 'a']], ['b'], ['a', 'c'], False),
    'bend-left-up-2': ([['a', 'b', 'c'], ['d', 'e']], [['e', 'd', 'b', 'a']], ['c'], ['b', 'a', 'd', 'e'], False),
    # bending left and right
    'bend-down-down': (
        [['a', 'b'], ['c', 'd', 'e']],
        [['a', 'c'], ['b', 'e']],
        ['c', 'a', 'b', 'e'],
        ['d'],
        [None, None, True, None, False],
    ),
    'bend-down-up': (
        [['a', 'b'], ['c', 'd', 'e']],
        [['a', 'c'], ['e', 'b']],
        ['c', 'a'],
        ['d', 'e', 'b'],
        [None, True, None, None, False],
    ),
    'bend-up-down': (
        [['a', 'b', 'c'], ['d', 'e', 'f']],
        [['d', 'a'], ['c', 'f', 'e'], ['b']],
        ['b', 'c', 'f', 'e'],
        ['a', 'd'],
        [False, None, None, True, True, None],
    ),
    'bend-up-up': (
        [['a', 'b', 'c'], ['d', 'e', 'f']],
        [['d', 'e', 'a'], ['f', 'c'], ['b']],
        ['b'],
        ['a', 'd', 'e', 'f', 'c'],
        [False, None, True, None, None, None],
    ),
}


@pytest.mark.parametrize(
    'labels, which_legs, new_codomain, new_domain, bend_right',
    planar_combine_cases.values(),
    ids=planar_combine_cases.keys(),
)
@pytest.mark.parametrize(
    'symmetry, backend',
    [
        (ct.no_symmetry, 'no_symmetry'),
        (ct.u1_symmetry, 'abelian'),
        (ct.u1_symmetry, 'fusion_tree'),
        (ct.fermion_parity, 'fusion_tree'),
        (ct.fibonacci_anyon_category, 'fusion_tree'),
    ],
)
def test_planar_combine_legs(
    labels,
    which_legs,
    new_codomain,
    new_domain,
    bend_right,
    symmetry,
    backend,
    np_random,
):
    backend = ct.get_backend(backend, 'numpy')
    T: ct.SymmetricTensor = ct.testing.random_tensor(
        symmetry,
        codomain=len(labels[0]),
        domain=len(labels[1]),
        labels=[*labels[0], *reversed(labels[1])],
        backend=backend,
        np_random=np_random,
    )
    pipe_dualities = np_random.choice([True, False], size=len(which_legs))
    # TODO fix
    pipe_dualities = None

    num_codomain_legs = T.num_codomain_legs
    num_codomain_flat_legs = T.num_codomain_flat_legs
    num_domain_legs = T.num_domain_legs
    num_domain_flat_legs = T.num_domain_flat_legs
    for group in which_legs:
        leg0 = T.get_leg_idcs(group[0])[0]
        for i, leg in enumerate(T.get_leg_idcs(group)):
            if i == 0:
                continue
            if leg < T.num_codomain_legs:
                num_codomain_legs -= 1
            else:
                num_domain_legs -= 1
            flat_num = len(T.get_leg(leg)) if isinstance(T.get_leg(leg), ct.LegPipe) else 1
            if leg0 < T.num_codomain_legs and leg >= T.num_codomain_legs:
                num_codomain_flat_legs += flat_num
                num_domain_flat_legs -= flat_num
            elif leg0 >= T.num_codomain_legs and leg < T.num_codomain_legs:
                num_codomain_flat_legs -= flat_num
                num_domain_flat_legs += flat_num

    T_combined = ct.planar.planar_combine_legs(T, *which_legs, pipe_dualities=pipe_dualities)
    T_combined.test_sanity()
    assert T_combined.num_codomain_legs == num_codomain_legs
    assert T_combined.num_codomain_flat_legs == num_codomain_flat_legs
    assert T_combined.num_domain_legs == num_domain_legs
    assert T_combined.num_domain_flat_legs == num_domain_flat_legs
    assert ct.planar.planar_almost_equal(T, ct.split_legs(T_combined))

    # compare to nonplanar function
    T_combined2 = ct.permute_legs(T, codomain=new_codomain, domain=new_domain, bend_right=bend_right)
    # make sure which_legs is sorted to avoid braids
    which_legs = [sorted(T_combined2.get_leg_idcs(group)) for group in which_legs]
    T_combined2 = ct.combine_legs(T_combined2, *which_legs, pipe_dualities=pipe_dualities)
    assert ct.almost_equal(T_combined, T_combined2)


planar_contraction_cases = {
    '2-0-2-right': ([['a'], ['b']], [['c', 'd'], []], [], [], None, None, True),
    '2-1-2-right': ([['a'], ['b']], [['b', 'c'], []], [1], [0], None, None, True),
    '2-1-2-left': ([['a'], ['b']], [['c', 'b'], []], [1], [1], None, None, False),
    '2-2-2-right': ([['a'], ['b']], [[], ['a', 'b']], [1, 0], [0, 1], None, None, True),
    '3-1-3-right': ([['a'], ['c', 'b']], [['c', 'd'], ['e']], [2], [0], None, None, True),
    '3-1-3-left': ([['a'], ['b', 'c']], [['d', 'c'], ['e']], [1], [1], [2, 0], [0, 2], False),
    '3-2-3-left': ([['b', 'a'], ['c']], [['c'], ['b', 'd']], [0, 2], [2, 0], [1], [1], False),
    '3-3-3-right': ([['a', 'b'], ['c']], [['c'], ['a', 'b']], [2, 1, 0], [0, 1, 2], None, None, True),
    '4-2-3-left': ([['a', 'b', 'c'], ['d']], [['d'], ['a', 'e']], [0, 3], [2, 0], None, None, False),
    '4-2-3-right': ([['a', 'b', 'c'], ['d']], [['d'], ['e', 'c']], [3, 2], [0, 1], None, None, True),
    '4-3-3-right': ([['a', 'b', 'c', 'd'], []], [['d'], ['b', 'c']], [3, 2, 1], [0, 1, 2], None, None, True),
    '4-3-4-right1': ([['a', 'b'], ['d', 'c']], [['d', 'c'], ['e', 'b']], [3, 2, 1], [0, 1, 2], None, None, True),
    '4-3-4-right2': (
        [['b'], ['a', 'd', 'c']],
        [['d', 'c'], ['e', 'b']],
        [2, 1, 0],
        [0, 1, 2],
        None,
        None,
        [True, None, True, False],
    ),
    '4-3-4-left1': ([['a', 'b'], ['d', 'c']], [['d', 'c'], ['a', 'e']], [0, 3, 2], [3, 0, 1], None, None, False),
    '4-3-4-left2': (
        [['a'], ['d', 'c', 'b']],
        [['d', 'c'], ['a', 'e']],
        [0, 3, 2],
        [3, 0, 1],
        None,
        None,
        [False, True, None, False],
    ),
}


@pytest.mark.parametrize(
    'labels_A, labels_B, contr_A, contr_B, A_codomain, B_domain, bend_right',
    planar_contraction_cases.values(),
    ids=planar_contraction_cases.keys(),
)
@pytest.mark.parametrize(
    'symmetry, backend',
    [
        (ct.no_symmetry, 'no_symmetry'),
        (ct.u1_symmetry, 'abelian'),
        (ct.u1_symmetry, 'fusion_tree'),
        (ct.fermion_parity, 'fusion_tree'),
        (ct.fibonacci_anyon_category, 'fusion_tree'),
    ],
)
def test_planar_contraction(
    labels_A: list[list[str]],
    labels_B: list[list[str]],
    contr_A: list[int],
    contr_B: list[int],
    A_codomain,
    B_domain,
    bend_right: bool,
    symmetry,
    backend,
    np_random,
):
    backend = ct.get_backend(backend, 'numpy')
    # same construction as in test_tdot in test_tensors.py
    A: ct.SymmetricTensor = ct.testing.random_tensor(
        symmetry,
        codomain=len(labels_A[0]),
        domain=len(labels_A[1]),
        labels=[*labels_A[0], *reversed(labels_A[1])],
        backend=backend,
        np_random=np_random,
    )

    B: ct.SymmetricTensor = ct.testing.random_tensor(
        symmetry,
        codomain=[A._as_domain_leg(l) if A.has_label(l) else None for l in labels_B[0]],
        domain=[A._as_codomain_leg(l) if A.has_label(l) else None for l in labels_B[1]],
        labels=[*labels_B[0], *reversed(labels_B[1])],
        backend=backend,
        np_random=np_random,
    )

    # make sure we defined compatible legs
    for ia, ib in zip(contr_A, contr_B):
        assert A._as_domain_leg(ia) == B._as_codomain_leg(ib)

    res = ct.planar.planar_contraction(A, B, contr_A, contr_B)

    A_permuted = ct.permute_legs(A, codomain=A_codomain, domain=contr_A, bend_right=bend_right)
    B_permuted = ct.permute_legs(B, codomain=contr_B, domain=B_domain, bend_right=bend_right)
    contr_A = list(reversed(range(A_permuted.num_codomain_legs, A_permuted.num_legs)))
    contr_B = list(range(B_permuted.num_codomain_legs))
    expect = ct.tdot(A_permuted, B_permuted, contr_A, contr_B)

    if isinstance(res, ct.Tensor):
        assert ct.planar.planar_almost_equal(res, expect)
    else:
        assert len(contr_A) == A.num_legs and len(contr_B) == B.num_legs
        assert np.allclose(res, expect)


planar_partial_trace_cases = {
    # traces in codomain
    'codomain-aab': (['a', 'a', 'b'], []),
    'codomain-aabbc': (['a', 'a', 'b', 'b', 'c'], []),
    'codomain-abba-c': (['a', 'b', 'b', 'a'], ['c']),
    # traces in domain
    'domain-b-aa': (['b'], ['a', 'a']),
    'domain-c-aabb': (['c'], ['a', 'a', 'b', 'b']),
    'domain-c-abba': (['c'], ['a', 'b', 'b', 'a']),
    # traces in both codomain and domain
    'co_domain-aac-bb': (['a', 'a', 'c'], ['b', 'b']),
    # left and right
    'co_domain-acb-ab': (['a', 'c', 'b'], ['a', 'b']),
    # two left
    'co_domain-abc-ab': (['a', 'b', 'c'], ['a', 'b']),
    # two right
    'co_domain-cab-ab': (['c', 'a', 'b'], ['a', 'b']),
    # winding
    'codomain-aba': (['a', 'b', 'a'], []),
    'codomain-abcba': (['a', 'b', 'c', 'b', 'a'], []),
    'domain--aba': ([], ['a', 'b', 'a']),
    'domain--abcba': ([], ['a', 'b', 'c', 'b', 'a']),
    'co_domain-abcb-a': (['a', 'b', 'c', 'b'], ['a']),
    'co_domain-acab-b': (['a', 'c', 'a', 'b'], ['b']),
}


@pytest.mark.parametrize('codomain, domain', planar_partial_trace_cases.values(), ids=planar_partial_trace_cases.keys())
@pytest.mark.parametrize(
    'symmetry, backend',
    [
        (ct.no_symmetry, 'no_symmetry'),
        (ct.u1_symmetry, 'abelian'),
        (ct.u1_symmetry, 'fusion_tree'),
        (ct.fermion_parity, 'fusion_tree'),
        (ct.fibonacci_anyon_category, 'fusion_tree'),
    ],
)
def test_planar_partial_trace(codomain, domain, symmetry, backend, np_random):
    # same construction as in test_partial_trace in test_tensors.py
    backend = ct.get_backend(backend, 'numpy')
    trace_legs = {
        l: ct.testing.random_leg(symmetry, backend, False, np_random=np_random)
        for l in ct.tools.misc.duplicate_entries([*codomain, *domain])
    }
    seen_labels = []
    codomain_spaces = []
    codomain_labels = []
    for l in codomain:
        if l in seen_labels:
            codomain_spaces.append(trace_legs[l].dual)
            codomain_labels.append(f'{l}*')
        elif l in trace_legs:
            codomain_spaces.append(trace_legs[l])
            seen_labels.append(l)
            codomain_labels.append(l)
        else:
            codomain_spaces.append(ct.testing.random_leg(symmetry, backend, False, np_random=np_random))
            codomain_labels.append(l)
    domain_spaces = []
    domain_labels = []
    for l in domain:
        if l in seen_labels:
            domain_spaces.append(trace_legs[l])
            domain_labels.append(f'{l}*')
        elif l in trace_legs:
            domain_spaces.append(trace_legs[l].dual)
            domain_labels.append(l)
            seen_labels.append(l)
        else:
            domain_spaces.append(ct.testing.random_leg(symmetry, backend, False, np_random=np_random))
            domain_labels.append(l)

    T: ct.SymmetricTensor = ct.testing.random_tensor(
        symmetry,
        codomain_spaces,
        domain_spaces,
        labels=[*codomain_labels, *reversed(domain_labels)],
        backend=backend,
        np_random=np_random,
    )

    pairs = [(T.labels.index(l), T.labels.index(f'{l}*')) for l in trace_legs]
    res = ct.planar.planar_partial_trace(T, *pairs)
    res.test_sanity()
    assert res.labels == [l for l in T.labels if l[0] not in trace_legs]
    assert res.legs == [T.get_leg(l) for l in T.labels if l[0] not in trace_legs]

    if T.symmetry.has_trivial_braid:
        T_np = T.to_numpy()
        idcs1 = [p[0] for p in pairs]
        idcs2 = [p[1] for p in pairs]
        remaining = [n for n in range(T.num_legs) if n not in idcs1 and n not in idcs2]
        expect = T.backend.block_backend.trace_partial(T_np, idcs1, idcs2, remaining)
        expect = T.backend.block_backend.to_numpy(expect)
        res_np = res.to_numpy()
        npt.assert_almost_equal(res_np, expect)

    levels = None
    if not T.symmetry.has_symmetric_braid:
        flat_pairs = [p for pair in pairs for p in pair]
        # may need to braid with open legs since we do not trace over the left side
        levels = [flat_pairs.index(n) if n in flat_pairs else 10 for n in range(T.num_legs)]

    expect = ct.tensors.partial_trace(T, *pairs, levels=levels)
    assert expect.labels == res.labels
    assert expect.legs == res.legs
    assert ct.almost_equal(res, expect)


planar_permute_legs_cases = {
    'trivial': (3, 2, None, [0, 1, 2][::-1]),
    # same basic case with different input possibilities
    'basic-idcs': (3, 2, None, [3, 4, 0][::-1]),
    'basic-labels': (3, 2, None, ['a', 'e', 'd']),
    'basic-codomain': (3, 2, [1, 2], None),
    # empty codomain/domain
    'empty-codomain': (2, 2, [], [1, 0, 3, 2]),
    'empty-domain': (2, 2, [0, 1, 2, 3], None),
    # input has no codomain
    'J0-empty-domain': (0, 3, [2, 0, 1], []),
    'J0': (0, 3, None, [1]),
    'J0-empty-codomain': (0, 3, None, [0, 1, 2][::-1]),
    # input has no domain
    'K0-empty-domain': (3, 0, [1, 2, 0], []),
    'K0': (3, 0, None, [1]),
    'K0-empty-codomain': (3, 0, [0, 1, 2], None),
}


@pytest.mark.parametrize(
    'J, K, codomain, domain', planar_permute_legs_cases.values(), ids=planar_permute_legs_cases.keys()
)
@pytest.mark.parametrize(
    'symmetry, backend',
    [
        (ct.no_symmetry, 'no_symmetry'),
        (ct.u1_symmetry, 'abelian'),
        (ct.u1_symmetry, 'fusion_tree'),
        (ct.fermion_parity, 'fusion_tree'),
        (ct.fibonacci_anyon_category, 'fusion_tree'),
    ],
)
def test_planar_permute_legs(J, K, codomain, domain, symmetry, backend, np_random):
    backend = ct.get_backend(backend, 'numpy')
    T_labels = list('abcdefghijk')[: J + K]
    T: ct.SymmetricTensor = ct.testing.random_tensor(symmetry, J, K, labels=T_labels, np_random=np_random)

    res = ct.planar.planar_permute_legs(T, codomain=codomain, domain=domain)
    res.test_sanity()

    # test planar_almost_equal; test both ways
    assert ct.planar.planar_almost_equal(res, T)
    assert ct.planar.planar_almost_equal(T, res)

    if codomain is None or len(codomain) == 0:
        domain = T.get_leg_idcs(domain)
        num_codom_legs = T.num_legs - len(domain)
        codomain = [i % T.num_legs for i in range(domain[0] + 1, domain[0] + 1 + num_codom_legs)]
        rev_domain = domain[::-1]
    else:
        codomain = T.get_leg_idcs(codomain)
        num_dom_legs = T.num_legs - len(codomain)
        rev_domain = [i % T.num_legs for i in range(codomain[-1] + 1, codomain[-1] + 1 + num_dom_legs)]
    leg_perm = [*codomain, *rev_domain]
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

    if len(T.codomain_labels) > 0:
        permuted_back1 = ct.planar.planar_permute_legs(res, codomain=T.codomain_labels)
        permuted_back1.test_sanity()
        assert ct.almost_equal(permuted_back1, T)

    if len(T.domain_labels) > 0:
        permuted_back2 = ct.planar.planar_permute_legs(res, domain=T.domain_labels)
        permuted_back2.test_sanity()
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
        dims=dict(
            chi=['vR', 'vL', 'vR*', 'vL*'], w=['wL', 'wR', 'wL*', 'wR*'], d=['p', 'p*', 'p0', 'p0*', 'p1', 'p1*']
        ),
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
    theta = ct.testing.random_tensor(
        symmetry, codomain=2, domain=2, labels=['vL', 'p0', 'p1', 'vR'], np_random=np_random
    )
    p0 = theta.get_leg('p0')
    vL = theta.get_leg('vL')
    Lp = ct.testing.random_tensor(symmetry, codomain=[vL], domain=[vL, None], labels=['vR*', 'wR', 'vR'])
    wR = Lp.get_leg('wR')
    W = ct.testing.random_tensor(symmetry, codomain=[p0, wR], domain=[wR, p0], labels=['p', 'wR', 'p*', 'wL'])
    mixL = ct.testing.random_tensor(symmetry, codomain=[wR], domain=[wR], labels=['wL*', 'wL'])

    # ===========================================
    # evaluate the diagram
    # ===========================================
    res = density_matrix_mixing_left(Lp=Lp, Lp_hc=Lp.hc, W=W, W_hc=W.hc, mixL=mixL, theta=theta, theta_hc=theta.hc)
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
        ct.permute_legs(Lp, ['vR'], ['wR', 'vR*'], bend_right=[True, None, False]),
    )
    expect2 = ct.compose(
        ct.permute_legs(expect2, ['p0*', 'vL*', 'vR*'], ['p0', 'wR'], bend_right=[False, None, None, True, None]),
        ct.permute_legs(W, ['p*', 'wL'], ['wR', 'p'], bend_right=[True, True, False, False]),
    )
    expect2 = ct.compose(
        ct.permute_legs(expect2, ['p0*', 'vL*', 'vR*', 'p'], ['wR'], bend_right=[None, None, None, True, None]),
        ct.transpose(mixL),
    )
    expect2 = ct.compose(
        ct.permute_legs(expect2, ['vL*', 'vR*', 'p'], ['p0*', 'wL*'], bend_right=[False, None, None, None, None]),
        ct.permute_legs(W.hc, ['p', 'wR*'], ['wL*', 'p*'], bend_right=[False, None, True, None]),
    )
    expect2 = ct.compose(
        ct.permute_legs(expect2, ['vR*', 'p', 'p*'], ['vL*', 'wL*'], bend_right=[False, None, None, True, None]), Lp.hc
    )
    expect2 = expect2.relabel({'vR*': 'vL', 'vR': 'vL*'})
    expect2 = ct.permute_legs(expect2, ['p*', 'vL*'], ['p', 'vL'], bend_right=[False, False, None, True])
    expect2.test_sanity()
    assert ct.almost_equal(res, expect2)


@pytest.mark.parametrize('symmetry', [ct.no_symmetry, ct.u1_symmetry, ct.fibonacci_anyon_category])
def test_PlanarDiagram_add_remove_tensor(symmetry, np_random):
    # ===========================================
    # define a diagram
    # ===========================================
    partial_diagram = ct.PlanarDiagram(
        tensors='T1[vL, vR], T2[vL, vR], T3[vL, w, vR]',
        definition=[
            ('T1', 'vL', 'T2', 'vR'),
            ('T1', 'vR', None, 'vR'),
            ('T2', 'vL', 'T3', 'vL'),
            ('T3', 'vR', None, 'vL'),
            ('T3', 'w', None, 'w1'),
        ],
        dims=dict(chi=['vR', 'vL'], w=['w']),
    )
    full_diagram = ct.PlanarDiagram(
        tensors='T1[vL, vR], T2[vL, vR], T3[vL, w, vR], T4[vL, w, vR]',
        definition='T1:vL @ T2:vR, T1:vR @ T4:vR, T2:vL @ T3:vL, T3:vR @ T4:vL, T3:w -> w1, T4:w -> w2',
        dims=dict(chi=['vR', 'vL'], w=['w']),
    )
    r"""Random planar diagram; T4 is used to test adding and removal

        |   .--T1--.
        |   |      |
        |  T2      T4-
        |   |      |
        |   .--T3--.
        |      |
    """

    # ===========================================
    # create example tensors
    # ===========================================
    T1 = ct.testing.random_tensor(symmetry, codomain=2, labels=['vL', 'vR'], np_random=np_random)
    T1vL = T1.get_leg('vL')
    T1vR = T1.get_leg('vR')
    T2 = ct.testing.random_tensor(symmetry, codomain=[None, T1vL.dual], labels=['vL', 'vR'], np_random=np_random)
    T2vL = T2.get_leg('vL')
    T3 = ct.testing.random_tensor(
        symmetry, codomain=[T2vL.dual, None, None], labels=['vL', 'w', 'vR'], np_random=np_random
    )
    T3vR = T3.get_leg('vR')
    T4 = ct.testing.random_tensor(
        symmetry, codomain=[T1vR.dual, T3vR.dual, None], labels=['vR', 'vL', 'w'], np_random=np_random
    )

    # ===========================================
    # evaluate the diagrams
    # ===========================================
    partial_res = partial_diagram(T1=T1, T2=T2, T3=T3)
    partial_res.test_sanity()
    assert partial_res.labels == ['w1', 'vL', 'vR']
    assert partial_res.num_codomain_legs == 2

    full_res = full_diagram(T1=T1, T2=T2, T3=T3, T4=T4)
    full_res.test_sanity()
    assert full_res.labels == ['w2', 'w1']
    assert full_res.num_codomain_legs == 1

    # ===========================================
    # transform between the diagrams
    # ===========================================
    full_diagram2 = partial_diagram.add_tensor(
        tensor={'T4': ct.planar.TensorPlaceholder(['vR', 'vL', 'w'], ['chi', 'chi', 'w'])},
        extra_definition=[('T4', 'vL', 'T3', 'vR'), ('T1', 'vR', 'T4', 'vR'), ('T4', 'w', None, 'w2')],
    )
    full_res2 = full_diagram2(T1=T1, T2=T2, T3=T3, T4=T4)
    full_res2.test_sanity()
    full_res2 = ct.planar.planar_permute_legs(full_res2, codomain=['w2'])
    assert full_res2.labels == ['w2', 'w1']
    assert full_res2.num_codomain_legs == 1
    assert ct.almost_equal(full_res, full_res2)

    partial_diagram2 = full_diagram.remove_tensor(
        'T4', extra_definition=[('T3', 'vR', None, 'vL'), ('T1', 'vR', None, 'vR')]
    )
    partial_res2 = partial_diagram2(T1=T1, T2=T2, T3=T3)
    partial_res2.test_sanity()
    assert partial_res2.labels == ['w1', 'vL', 'vR']
    assert partial_res2.num_codomain_legs == 2
    assert ct.almost_equal(partial_res, partial_res2)


@pytest.mark.parametrize('symmetry', [ct.no_symmetry, ct.u1_symmetry, ct.fibonacci_anyon_category])
def test_PlanarDiagram_contraction_orders(symmetry, np_random):
    # ===========================================
    # define a diagram
    # ===========================================
    diagram1 = ct.PlanarDiagram(
        tensors='T1[vL, vR], T2[vL, vR], T3[vL, w, vR], T4[vL, w, vR]',
        definition=[
            ('T1', 'vL', 'T2', 'vR'),
            ('T1', 'vR', 'T4', 'vR'),
            ('T2', 'vL', 'T3', 'vL'),
            ('T3', 'vR', 'T4', 'vL'),
            ('T3', 'w', None, 'w1'),
            ('T4', 'w', None, 'w2'),
        ],
        dims=dict(chi=['vR', 'vL'], w=['w']),
    )
    tensors = {
        'T1': ct.planar.TensorPlaceholder(['vL', 'vR'], ['chi', 'chi']),
        'T2': ct.planar.TensorPlaceholder(['vR', 'vL'], ['chi', 'chi']),
        'T3': ct.planar.TensorPlaceholder(['vL', 'w', 'vR'], ['chi', 'w', 'chi']),
        'T4': ct.planar.TensorPlaceholder(['vR', 'vL', 'w'], ['chi', 'chi', 'w']),
    }
    diagram2 = ct.PlanarDiagram(
        tensors=tensors,
        definition='T1:vL @ T2:vR, T1:vR @ T4:vR, T2:vL @ T3:vL, T3:vR @ T4:vL, T3:w -> w1, T4:w -> w2',
    )
    r"""Same diagram as for test_PlanarDiagram_add_remove_tensor;
    also test adding and removing tensors in the different ways

        |   .--T1--.
        |   |      |
        |  T2      T4-
        |   |      |
        |   .--T3--.
        |      |
    """

    # ===========================================
    # create example tensors
    # ===========================================
    T1 = ct.testing.random_tensor(symmetry, codomain=2, labels=['vL', 'vR'], np_random=np_random)
    T1vL = T1.get_leg('vL')
    T1vR = T1.get_leg('vR')
    T2 = ct.testing.random_tensor(symmetry, codomain=[None, T1vL.dual], labels=['vL', 'vR'], np_random=np_random)
    T2vL = T2.get_leg('vL')
    T3 = ct.testing.random_tensor(
        symmetry, codomain=[T2vL.dual, None, None], labels=['vL', 'w', 'vR'], np_random=np_random
    )
    T3vR = T3.get_leg('vR')
    T4 = ct.testing.random_tensor(
        symmetry, codomain=[T1vR.dual, T3vR.dual, None], labels=['vR', 'vL', 'w'], np_random=np_random
    )

    # ===========================================
    # evaluate the diagrams
    # ===========================================
    res1 = diagram1(T1=T1, T2=T2, T3=T3, T4=T4)
    res1.test_sanity()
    assert res1.labels == ['w2', 'w1']
    assert res1.num_codomain_legs == 1

    res2 = diagram2(T1=T1, T2=T2, T3=T3, T4=T4)
    res2.test_sanity()
    assert res2.labels == ['w2', 'w1']
    assert res2.num_codomain_legs == 1

    assert ct.almost_equal(res1, res2)

    # ===========================================
    # remove T4
    # ===========================================
    partial_diagram1 = diagram1.remove_tensor('T4', extra_definition='T3:vR -> vL, T1:vR -> vR')
    partial_diagram2 = diagram2.remove_tensor(
        'T4', extra_definition=[('T3', 'vR', None, 'vL'), ('T1', 'vR', None, 'vR')]
    )
    partial_res1 = partial_diagram1(T1=T1, T2=T2, T3=T3)
    partial_res1.test_sanity()
    assert partial_res1.labels == ['w1', 'vL', 'vR']
    assert partial_res1.num_codomain_legs == 2

    partial_res2 = partial_diagram2(T1=T1, T2=T2, T3=T3)
    partial_res2.test_sanity()
    assert partial_res2.labels == ['w1', 'vL', 'vR']
    assert partial_res2.num_codomain_legs == 2

    assert ct.almost_equal(partial_res1, partial_res2)

    # ===========================================
    # add T4 again
    # ===========================================
    diagram1_ = partial_diagram1.add_tensor(
        tensor={'T4': tensors['T4']},
        extra_definition='T4:vL @ T3:vR, T1:vR @ T4:vR, T4:w -> w2',
    )
    diagram2_ = partial_diagram2.add_tensor(
        tensor='T4[vL, w, vR]',
        extra_definition=[('T4', 'vL', 'T3', 'vR'), ('T1', 'vR', 'T4', 'vR'), ('T4', 'w', None, 'w2')],
        extra_dims=dict(chi=['vR', 'vL'], w=['w']),
    )
    res1_ = diagram1_(T1=T1, T2=T2, T3=T3, T4=T4)
    res1_.test_sanity()
    assert res1_.labels == ['w1', 'w2']
    assert res1_.num_codomain_legs == 1

    res2_ = diagram2_(T1=T1, T2=T2, T3=T3, T4=T4)
    res2_.test_sanity()
    assert res2_.labels == ['w1', 'w2']
    assert res2_.num_codomain_legs == 1

    assert ct.almost_equal(res1_, res2_)
    assert ct.planar.planar_almost_equal(res1_, res1)


@pytest.mark.parametrize('symmetry', [ct.no_symmetry, ct.u1_symmetry, ct.fibonacci_anyon_category])
def test_PlanarDiagram_with_traces(symmetry, np_random):
    # ===========================================
    # define a diagram
    # ===========================================
    diagram = ct.PlanarDiagram(
        tensors='T1[vL, vR], T2[w, vL, vR, w*], T3[vR, w1, w1*, w2, w2*, vL], T4[vL, w1, w2, w2*, w1*, vR]',
        definition='T1:vL @ T2:vR, T1:vR @ T3:vL, '
        'T2:w @ T2:w*, T2:vL @ T4:vL,'
        'T3:w1 @ T3:w1*, T3:w2 @ T3:w2*, T3:vR @ T4:vR, '
        'T4:w1 @ T4:w1*, T4:w2* @ T4:w2',
        dims=dict(chi=['vR', 'vL'], w=['w', 'w1', 'w2', 'w*', 'w1*', 'w2*']),
    )
    r"""Random planar diagram with multiple partial traces

        |    .---T1---.
        | .  |        |  .
        | |\ |        | /|
        | | T2        T3-.
        | |/ |        ||\
        | .  |        |.-.
        |    .---T4---.
        |       //\\
        |      /.--.\
        |     .------.
    """

    # ===========================================
    # create example tensors
    # ===========================================
    T1 = ct.testing.random_tensor(symmetry, codomain=2, labels=['vL', 'vR'], np_random=np_random)
    T1vL = T1.get_leg('vL')
    T1vR = T1.get_leg('vR')
    traced_legs = [ct.testing.random_ElementarySpace(symmetry, np_random=np_random) for _ in range(5)]
    T2 = ct.testing.random_tensor(
        symmetry,
        codomain=[traced_legs[0], None, T1vL.dual, traced_legs[0].dual],
        labels=['w', 'vL', 'vR', 'w*'],
        np_random=np_random,
    )
    T3 = ct.testing.random_tensor(
        symmetry,
        codomain=[None, traced_legs[1].dual, traced_legs[1], traced_legs[2]],
        domain=[T1vR, traced_legs[2]],
        labels=['vR', 'w1', 'w1*', 'w2', 'w2*', 'vL'],
        np_random=np_random,
    )
    T4vL = T2.get_leg('vL').dual
    T4vR = T3.get_leg('vR')
    T4 = ct.testing.random_tensor(
        symmetry,
        codomain=[T4vL, traced_legs[3], traced_legs[4], traced_legs[4].dual],
        domain=[T4vR, traced_legs[3]],
        labels=['vL', 'w1', 'w2', 'w2*', 'w1*', 'vR'],
        np_random=np_random,
    )

    # ===========================================
    # evaluate the diagram
    # ===========================================
    res = diagram(T1=T1, T2=T2, T3=T3, T4=T4)
    assert isinstance(res, (float, complex))

    # ===========================================
    # compare to manual contraction, using planar routines
    # ===========================================
    T2_traced = ct.planar.planar_partial_trace(T2, ['w', 'w*'])
    T3_traced = ct.planar.planar_partial_trace(T3, ['w1', 'w1*'], ['w2', 'w2*'])
    T4_traced = ct.planar.planar_partial_trace(T4, ['w1', 'w1*'], ['w2', 'w2*'])

    expect1 = ct.planar.planar_contraction(T1, T2_traced, ['vL'], ['vR'])
    expect1 = ct.planar.planar_contraction(expect1, T3_traced, ['vR'], ['vL'])
    expect1 = ct.planar.planar_contraction(expect1, T4_traced, ['vL', 'vR'], ['vL', 'vR'])
    assert isinstance(expect1, (float, complex))
    assert np.allclose(expect1, res)

    # ===========================================
    # compare to manual contraction, using general (not planar) routines
    # ===========================================
    T2_traced_ = ct.planar.partial_trace(T2, ['w', 'w*'], levels=[3, 1, 2, 3])
    T3_traced_ = ct.planar.partial_trace(T3, ['w1', 'w1*'], ['w2', 'w2*'])
    T4_traced_ = ct.planar.partial_trace(T4, ['w1', 'w1*'], ['w2', 'w2*'], levels=[None, 1, 2, 2, 1, None])
    assert ct.almost_equal(T2_traced, T2_traced_)
    assert ct.almost_equal(T3_traced, T3_traced_)
    assert ct.almost_equal(T4_traced, T4_traced_)

    expect2 = ct.permute_legs(T1, codomain=['vL'], domain=['vR'], bend_right=True)
    expect2 = ct.compose(ct.permute_legs(T2_traced_, codomain=['vL'], domain=['vR'], bend_right=True), expect2)
    expect2 = ct.compose(expect2, ct.permute_legs(T3_traced_, codomain=['vL'], domain=['vR'], bend_right=[True, False]))
    expect2 = ct.compose(
        ct.permute_legs(expect2, domain=['vL', 'vR'], bend_right=False),
        ct.permute_legs(T4_traced_, codomain=['vL', 'vR'], bend_right=True),
    )
    assert isinstance(expect2, (float, complex))
    assert np.allclose(expect2, res)


def test_PlanarDiagram_verify_diagram():
    # only a single tensor
    _ = ct.PlanarDiagram(
        tensors='T1[l1, l2, l3]',
        definition='T1:l2 @ T1:l1, T1:l3 -> l3',
        dims=dict(chi=['l1', 'l2', 'l3']),
    )
    with pytest.raises(ValueError):
        _.remove_tensor('T1')

    # disconnected tensors
    with pytest.raises(ValueError, match='The planar diagram is disconnected'):
        _ = ct.PlanarDiagram(
            tensors='T1[l1, l2, l3], T2[l1, l2], T3[l1, l2]',
            definition='T1:l1 @ T2:l1, T1:l2 -> l2, T1:l3 @ T2:l2, T3:l1 @ T3:l2',
            dims=dict(chi=['l1', 'l2', 'l3']),
        )
    with pytest.raises(ValueError, match='The planar diagram is disconnected'):
        _ = ct.PlanarDiagram(
            tensors='T1[l1, l2, l3], T2[l1, l2]',
            definition='T1:l1 @ T1:l3, T1:l2 -> l2, T2:l1 @ T2:l2',
            dims=dict(chi=['l1', 'l2', 'l3']),
        )

    # add a disconnected tensor
    diagram = ct.PlanarDiagram(
        tensors='T1[l1, l2, l3]',
        definition='T1:l2 @ T1:l1, T1:l3 -> l3',
        dims=dict(chi=['l1', 'l2', 'l3']),
    )
    with pytest.raises(ValueError, match='The planar diagram is disconnected'):
        _ = diagram.add_tensor('T2[l1, l2]', 'T2:l1 -> l1, T2:l2 -> l2')

    # remove a tensor to make the diagram disconnected
    diagram = ct.PlanarDiagram(
        tensors='T1[l1, l2], T2[l1, l2, l3], T3[l1, l2]',
        definition='T1:l1 -> l1, T1:l2 @ T2:l1, T2:l2 @ T3:l1, T2:l3 -> l3, T3:l2 -> l2',
        dims=dict(chi=['l1', 'l2', 'l3']),
    )
    with pytest.raises(ValueError):
        _ = diagram.remove_tensor('T2')

    # open leg = contracted leg
    with pytest.raises(ValueError, match='Number of contracted and open legs does not match the total number of legs'):
        _ = ct.PlanarDiagram(
            tensors='T1[l1, l2, l3], T2[l1, l2]',
            definition='T1:l1 @ T2:l1, T1:l2 -> l2, T1:l3 @ T2:l2, T1:l1 -> l1',
            dims=dict(chi=['l1', 'l2', 'l3']),
        )
    with pytest.raises(ValueError, match='Inconsistent open legs'):
        _ = ct.PlanarDiagram(
            tensors='T1[l1, l2, l3], T2[l1, l2]',
            definition='T1:l1 @ T2:l1, T1:l3 @ T2:l2, T1:l1 -> l1',
            dims=dict(chi=['l1', 'l2', 'l3']),
        )

    # contract leg twice
    with pytest.raises(AssertionError, match='duplicate legs'):
        _ = ct.PlanarDiagram(
            tensors='T1[l1, l2, l3], T2[l1, l2]',
            definition='T1:l1 -> l1, T1:l2 @ T2:l2, T1:l3 @ T2:l2',
            dims=dict(chi=['l1', 'l2', 'l3']),
        )

    # unspecified action on leg
    with pytest.raises(ValueError, match='Number of contracted and open legs does not match the total number of legs'):
        _ = ct.PlanarDiagram(
            tensors='T1[l1, l2, l3], T2[l1, l2]',
            definition='T1:l1 @ T2:l1, T1:l3 @ T2:l2',
            dims=dict(chi=['l1', 'l2', 'l3']),
        )

    # unknown tensor
    with pytest.raises(AssertionError, match='No tensor with name T2'):
        _ = ct.PlanarDiagram(
            tensors='T1[l1, l2, l3]',
            definition='T1:l2 @ T2:l1, T1:l1 -> l1, T1:l3 -> l3',
            dims=dict(chi=['l1', 'l2', 'l3']),
        )

    # unknown label
    with pytest.raises(AssertionError, match='Tensor T1 has no leg l4'):
        _ = ct.PlanarDiagram(
            tensors='T1[l1, l2, l3], T2[l1, l2]',
            definition='T1:l1 @ T2:l1, T1:l3 @ T2:l2, T1:l4 -> l4, T1:l3 -> l3',
            dims=dict(chi=['l1', 'l2', 'l3']),
        )

    # non-planar due to partial trace
    with pytest.raises(ValueError, match='Not a planar trace'):
        _ = ct.PlanarDiagram(
            tensors='T1[l1, l2, l3, l4], T2[l1, l2]',
            definition='T1:l2 @ T1:l4, T1:l1 @ T2:l1, T1:l3 -> l3, T2:l2 -> l2',
            dims=dict(chi=['l1', 'l2', 'l3', 'l4']),
        )

    # non-planar due to contractions
    with pytest.raises(ValueError, match='Not a planar bipartition'):
        _ = ct.PlanarDiagram(
            tensors='T1[l1, l2, l3, l4], T2[l1, l2]',
            definition='T1:l1 @ T2:l1, T1:l3 @ T2:l2, T1:l3 -> l3, T1:l4 -> l4',
            dims=dict(chi=['l1', 'l2', 'l3', 'l4']),
        )


@pytest.mark.parametrize('symmetry', [ct.no_symmetry, ct.u1_symmetry, ct.fibonacci_anyon_category])
def test_PlanarLinearOperator(symmetry):
    # ===========================================
    # define an operator
    # ===========================================

    class TwoSiteEffectiveH(ct.PlanarLinearOperator):
        r"""Effective Hamiltonian during Two-site
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
            extra_dims=dict(chi=['vL', 'vR'], d=['p0', 'p1']),
        )

        def __init__(self, Lp, W0, W1, Rp):
            ct.planar.PlanarLinearOperator.__init__(
                self,
                op_diagram=self.op_diagram,
                matvec_diagram=self.matvec_diagram,
                op_tensors=dict(Lp=Lp, W0=W0, W1=W1, Rp=Rp),
                vec_name='theta',
            )

    # ===========================================
    # create example tensors
    # ===========================================

    theta = ct.testing.random_tensor(symmetry, 4, labels=['vL', 'p0', 'p1', 'vR'], max_multiplicity=3)
    vL, p0, p1, vR = theta.legs
    Lp = ct.testing.random_tensor(symmetry, [vL, None, vL.dual], labels=['vR*', 'wR', 'vR'], max_multiplicity=3)
    W0 = ct.testing.random_tensor(
        symmetry, [p0, None, p0.dual, Lp.get_leg('wR').dual], labels=['p', 'wR', 'p*', 'wL'], max_multiplicity=3
    )
    W1 = ct.testing.random_tensor(
        symmetry, [p1, None, p1.dual, W0.get_leg('wR').dual], labels=['p', 'wR', 'p*', 'wL'], max_multiplicity=3
    )
    Rp = ct.testing.random_tensor(symmetry, [vR, vR.dual, W1.get_leg('wR').dual], labels=['vL*', 'vL', 'wL'])

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
        relabel2={'p': 'p1', 'p*': 'p1*'},
    )
    op_2 = ct.compose(
        ct.permute_legs(op_2, ['p1', 'vR', 'vR*', 'p1*'], ['wL'], bend_right=[None, None, True, None, False]),
        ct.permute_legs(W0, ['wR'], ['p', 'wL', 'p*'], bend_right=[False, None, True, True]),
        relabel2={'p': 'p0', 'p*': 'p0*'},
    )
    bend_right = [None] * 4 + [True, None, False]
    op_2 = ct.compose(
        ct.permute_legs(op_2, ['p0', 'p1', 'vR', 'vR*', 'p1*', 'p0*'], ['wL'], bend_right=bend_right),
        ct.permute_legs(Lp, ['wR'], ['vR*', 'vR'], bend_right=[False, None, True]),
        relabel2={'vR*': 'vL', 'vR': 'vL*'},
    )
    bend_right = [None] * 3 + [True] * 3 + [None, False]
    op_2 = ct.permute_legs(op_2, ['vL', 'p0', 'p1', 'vR'], ['vL*', 'p0*', 'p1*', 'vR*'], bend_right=bend_right)
    assert ct.almost_equal(op_2, op)

    H_theta_2 = ct.compose(op_2, theta)
    assert ct.almost_equal(H_theta_2, H_theta)
