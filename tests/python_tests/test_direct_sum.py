"""Tests for VectorLike / DirectSum and their use in sparse Krylov algorithms."""

# Copyright (C) TeNPy Developers, Apache license
import numpy as np
import numpy.testing as npt
import pytest

from cyten import backends, krylov_based, sparse, tensors
from cyten.tensors import DirectSum, VectorLike, almost_equal, inner, linear_combination, norm


def _to_num(value):
    to_numpy = getattr(value, 'to_numpy', None)
    if callable(to_numpy):
        return to_numpy()
    return value


def _abs_norm(vec):
    return abs(_to_num(norm(vec)))


def _two_component_hermitian(make_compatible_tensor):
    """Two-component DirectSum with a block-diagonal Hermitian operator."""
    vec1 = make_compatible_tensor(codomain=1, labels=['v'], use_pipes=False)
    vec2 = make_compatible_tensor(codomain=1, labels=['w'], use_pipes=False)
    H1 = make_compatible_tensor(codomain=[vec1.legs[0]], domain=[vec1.legs[0]], labels=['w1', 'v'], use_pipes=False)
    H2 = make_compatible_tensor(codomain=[vec2.legs[0]], domain=[vec2.legs[0]], labels=['w2', 'w'], use_pipes=False)
    H1 = H1 + H1.dagger
    H2 = H2 + H2.dagger
    psi = DirectSum([vec1, vec2])
    H_op = sparse.DirectSumLinearOperator([sparse.TensorLinearOperator(H1), sparse.TensorLinearOperator(H2)])
    return psi, H_op, H1, H2


def test_vectorlike_isinstance(make_compatible_tensor):
    tens = make_compatible_tensor(codomain=1, use_pipes=False)
    other = make_compatible_tensor(codomain=1, use_pipes=False)
    ds = DirectSum([tens, other])
    assert isinstance(tens, VectorLike)
    assert isinstance(ds, VectorLike)
    assert isinstance(ds, DirectSum)
    assert tens.compatible_with(tens.copy())
    assert ds.compatible_with(ds.copy())
    assert not tens.compatible_with(ds)
    assert not ds.compatible_with(tens)


def test_direct_sum_algebra(make_compatible_tensor, tol=1.0e-12):
    x1 = make_compatible_tensor(codomain=1, labels=['a'], use_pipes=False)
    x2 = make_compatible_tensor(codomain=1, labels=['b'], use_pipes=False)
    y1 = make_compatible_tensor(like=x1, use_pipes=False)
    y2 = make_compatible_tensor(like=x2, use_pipes=False)
    x = DirectSum([x1, x2])
    y = DirectSum([y1, y2])

    with pytest.raises((ValueError, RuntimeError)):
        DirectSum([])

    assert len(x) == 2
    assert almost_equal(x[0], x1)
    assert almost_equal(x.components[1], x2)

    inner_expect = _to_num(inner(x1, y1)) + _to_num(inner(x2, y2))
    assert abs(_to_num(inner(x, y)) - inner_expect) < tol * (abs(inner_expect) + 1.0)

    n2_expect = abs(_to_num(norm(x1))) ** 2 + abs(_to_num(norm(x2))) ** 2
    assert abs(abs(_to_num(norm(x))) ** 2 - n2_expect) < tol * (n2_expect + 1.0)

    a = 2.5 - 0.3j
    scaled = a * x
    assert isinstance(scaled, DirectSum)
    assert almost_equal(scaled[0], a * x1)
    assert almost_equal(scaled[1], a * x2)

    summed = x + y
    assert isinstance(summed, DirectSum)
    assert almost_equal(summed[0], x1 + y1)
    assert almost_equal(summed[1], x2 + y2)

    axpy = x.axpy(a, y)
    assert isinstance(axpy, DirectSum)
    assert almost_equal(axpy[0], a * x1 + y1)
    assert almost_equal(axpy[1], a * x2 + y2)

    lc = linear_combination(1.0, x, a, y)
    assert isinstance(lc, DirectSum)
    assert almost_equal(lc[0], x1 + a * y1)

    tsum = x1 + y1
    assert isinstance(tsum, tensors.Tensor)


def test_direct_sum_wrappers(make_compatible_tensor, tol=1.0e-10):
    psi, H_op, H1, H2 = _two_component_hermitian(make_compatible_tensor)
    got = H_op.matvec(psi)
    assert isinstance(got, DirectSum)
    assert almost_equal(got[0], sparse.TensorLinearOperator(H1).matvec(psi[0]))
    assert almost_equal(got[1], sparse.TensorLinearOperator(H2).matvec(psi[1]))

    shift = 1.5
    shifted = sparse.ShiftedLinearOperator(H_op, shift)
    got_s = shifted.matvec(psi)
    expect0 = sparse.TensorLinearOperator(H1).matvec(psi[0]) + shift * psi[0]
    expect1 = sparse.TensorLinearOperator(H2).matvec(psi[1]) + shift * psi[1]
    assert almost_equal(got_s[0], expect0)
    assert almost_equal(got_s[1], expect1)

    o1 = DirectSum(
        [
            make_compatible_tensor(like=psi[0], use_pipes=False),
            make_compatible_tensor(like=psi[1], use_pipes=False),
        ]
    )
    o2 = DirectSum(
        [
            make_compatible_tensor(like=psi[0], use_pipes=False),
            make_compatible_tensor(like=psi[1], use_pipes=False),
        ]
    )
    vecs_new = sparse.gram_schmidt([psi, o1, o2])
    assert len(vecs_new) >= 1
    ovs = np.zeros((len(vecs_new), len(vecs_new)), dtype=np.complex128)
    for i, v in enumerate(vecs_new):
        for j, w in enumerate(vecs_new):
            ovs[i, j] = _to_num(inner(v, w))
    npt.assert_allclose(ovs, np.eye(len(vecs_new)), atol=1.0e-8)

    proj = sparse.ProjectedLinearOperator(H_op, [vecs_new[0]], project_operator=True, penalty=None)
    res = proj.matvec(vecs_new[0])
    assert _abs_norm(res) < 1.0e-8 * (_abs_norm(vecs_new[0]) + 1.0)

    with pytest.raises(NotImplementedError):
        H_op.to_tensor()


def test_direct_sum_lanczos_gmres(make_compatible_tensor, tol=1.0e-8):
    psi, H_op, _H1, _H2 = _two_component_hermitian(make_compatible_tensor)
    E0, psi0, _N = krylov_based.lanczos(H_op, psi, {'N_max': 20})
    assert isinstance(psi0, DirectSum)
    assert abs(_abs_norm(psi0) - 1.0) < tol
    residual = _abs_norm(H_op.matvec(psi0) - E0 * psi0)
    assert residual < tol * (abs(E0) + 1.0)

    A = sparse.ShiftedLinearOperator(H_op, 1.5)
    b = psi
    x0 = 0.0 * b
    x, rel_err, _errors, _iters = krylov_based.GMRES(
        A, x0, b, {'N_max': 20, 'restart': 10, 'res': 1.0e-10, 'N_min': 0}
    ).run()
    assert isinstance(x, DirectSum)
    b_n = _abs_norm(b)
    residual_g = _abs_norm(A.matvec(x) - b)
    assert residual_g / b_n < 1.0e-6
    assert rel_err < 1.0e-6


def test_direct_sum_arpack_flatten(make_compatible_tensor, tol=1.0e-8):
    psi, H_op, _H1, _H2 = _two_component_hermitian(make_compatible_tensor)
    if isinstance(psi.backend, backends.FusionTreeBackend):
        pytest.xfail('FTBackend does not support dense-block sector conversions yet')

    op, flat = sparse.NumpyArrayLinearOperator.from_matvec_and_vector(H_op.matvec, psi)
    assert op.component_converters is not None
    assert flat.shape == (op.shape[0],)
    psi2 = op.flat_array_to_tensor(flat)
    assert isinstance(psi2, DirectSum)
    assert len(psi2) == len(psi)
    for a, b in zip(psi.components, psi2.components):
        assert almost_equal(a, b)
    npt.assert_allclose(op.matvec(flat), op.tensor_to_flat_array(H_op.matvec(psi)), atol=tol, rtol=tol)

    E0, psi0 = krylov_based.lanczos_arpack(H_op, psi, {})
    assert isinstance(psi0, DirectSum)
    assert abs(_abs_norm(psi0) - 1.0) < tol
    residual = _abs_norm(H_op.matvec(psi0) - E0 * psi0)
    assert residual < tol * (abs(E0) + 1.0)
