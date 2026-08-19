"""A collection of tests for cyten.krylov_based."""

# Copyright (C) TeNPy Developers, Apache license
import numpy as np
import pytest
from scipy.linalg import expm

from cyten import backends, krylov_based, sparse, tensors
from cyten.tensors import almost_equal, inner, norm


def _op_and_vec(make_compatible_tensor, hermitian=True):
    vec = make_compatible_tensor(codomain=1, labels=['v'], use_pipes=False)
    leg = vec.legs[0]
    H = make_compatible_tensor(codomain=[leg], domain=[leg], labels=['w', 'v'], use_pipes=False)
    if hermitian:
        H = H + H.dagger
    H_op = sparse.TensorLinearOperator(H)
    return H, H_op, vec


def _abs_norm(tens):
    return abs(norm(tens).to_numpy())


def _to_num(value):
    to_numpy = getattr(value, 'to_numpy', None)
    if callable(to_numpy):
        return to_numpy()
    return value


def _try_to_numpy(tensor):
    """Return ``tensor.to_numpy()``, or None if the FusionTree backend cannot densify."""
    try:
        return tensor.to_numpy()
    except Exception:
        if isinstance(tensor.backend, backends.FusionTreeBackend):
            return None
        raise


@pytest.mark.parametrize(['N_cache', 'tol'], [(10, 5.0e-12), (20, 5.0e-12)])
def test_lanczos_gs(make_compatible_tensor, N_cache, tol):
    H, H_op, psi_init = _op_and_vec(make_compatible_tensor, hermitian=True)
    opts = {'N_cache': N_cache, 'N_max': 20}
    E0, psi0, _N = krylov_based.lanczos(H_op, psi_init, opts)
    assert abs(_abs_norm(psi0) - 1.0) < tol
    residual = _abs_norm(H_op.matvec(psi0) - E0 * psi0)
    assert residual < tol * (abs(E0) + 1.0)
    rayleigh = _to_num(inner(psi0, H_op.matvec(psi0)))
    assert abs(rayleigh / E0 - 1.0) < 100 * tol or abs(rayleigh - E0) < tol

    if psi0.num_parameters <= 1:
        return

    print('Now look for a second eigenvector in the same sector')
    psi_init2 = psi_init - inner(psi0, psi_init) * psi0
    if _abs_norm(psi_init2) < 1.0e-12:
        psi_init2 = make_compatible_tensor(like=psi_init, use_pipes=False)
        psi_init2 = psi_init2 - inner(psi0, psi_init2) * psi0
        if _abs_norm(psi_init2) < 1.0e-12:
            return
    lanczos_params = {'reortho': True, 'N_max': 20}
    if np.real(E0) > -0.01:
        lanczos_params['E_shift'] = -2.0 * float(np.real(E0)) - 0.2
    H_proj = sparse.ProjectedLinearOperator(H_op, ortho_vecs=[psi0])
    E1, psi1, _N = krylov_based.lanczos(H_proj, psi_init2, lanczos_params)
    assert abs(_abs_norm(psi1) - 1.0) < tol
    residual1 = _abs_norm(H_proj.matvec(psi1) - E1 * psi1)
    assert residual1 < 100 * tol * (abs(E1) + 1.0)
    ov = abs(inner(psi0, psi1).to_numpy())
    assert ov < 100 * tol


def test_lanczos_arpack(make_compatible_tensor, tol=1.0e-8):
    vec = make_compatible_tensor(codomain=1, labels=['v'], use_pipes=False)
    if isinstance(vec.backend, backends.FusionTreeBackend):
        pytest.xfail('FTBackend does not support dense-block sector conversions yet')
    leg = vec.legs[0]
    H = make_compatible_tensor(codomain=[leg], domain=[leg], labels=['w', 'v'], use_pipes=False)
    H_h = H + H.dagger
    H_op = sparse.TensorLinearOperator(H_h)

    E0, psi0 = krylov_based.lanczos_arpack(H_op, vec, {})
    n = abs(tensors.norm(psi0).to_numpy())
    assert abs(n - 1.0) < tol
    residual = tensors.norm(H_op.matvec(psi0) - E0 * psi0)
    assert abs(residual.to_numpy()) < tol * (abs(E0) + 1.0)


@pytest.mark.parametrize(['N_cache', 'tol'], [(10, 5.0e-12), (20, 5.0e-12)])
def test_lanczos_evolve(make_compatible_tensor, N_cache, tol):
    H, H_op, psi_init = _op_and_vec(make_compatible_tensor, hermitian=True)
    lanc = krylov_based.LanczosEvolution(H_op, psi_init, {'N_cache': N_cache, 'N_max': 20})
    H_np = _try_to_numpy(H)
    psi_init_np = None if H_np is None else _try_to_numpy(psi_init)

    for delta in [-0.1j, 0.1j, 1.0j, 0.1, 1.0]:
        psi_final, _N = lanc.run(delta, normalize=False)
        psi_final2, _N = lanc.run(delta, normalize=True)
        n = _abs_norm(psi_final)
        assert n > 0
        assert _abs_norm(psi_final / n - psi_final2) < max(tol, 1.0e-8)

        if H_np is None or psi_init_np is None:
            continue
        psi_final_np = expm(H_np * delta).dot(psi_init_np)
        ref_norm = np.linalg.norm(psi_final_np)
        got = _try_to_numpy(psi_final)
        if got is None:
            continue
        diff = np.linalg.norm(got - psi_final_np)
        print('norm(|psi_final> - |psi_final_flat>)/norm = ', diff / ref_norm)
        assert diff / ref_norm < max(tol, 1.0e-8)


@pytest.mark.parametrize('which', ['LM', 'SR', 'LR'])
def test_arnoldi(make_compatible_tensor, which, N_max=20):
    hermitian = which[-1] == 'R'
    H, H_op, psi_init = _op_and_vec(make_compatible_tensor, hermitian=hermitian)
    tol = 1.0e-8
    engine = krylov_based.Arnoldi(H_op, psi_init, {'which': which, 'num_ev': 1, 'N_max': N_max})
    (E0,), (psi0,), _N = engine.run()
    assert abs(_abs_norm(psi0) - 1.0) < tol
    residual = _abs_norm(H_op.matvec(psi0) - E0 * psi0)
    assert residual < tol * (abs(E0) + 1.0)
    rayleigh = _to_num(inner(psi0, H_op.matvec(psi0)))
    assert abs(rayleigh - E0) < tol * (abs(E0) + 1.0)


def test_arnoldi_evolve(make_compatible_tensor, tol=5.0e-10):
    H, H_op, psi_init = _op_and_vec(make_compatible_tensor, hermitian=False)
    eng = krylov_based.ArnoldiEvolution(H_op, psi_init, {'N_max': 20})
    H_np = _try_to_numpy(H)
    psi_init_np = None if H_np is None else _try_to_numpy(psi_init)

    for delta in [-0.1j, 0.1j, 0.5j, 0.1, -0.05 - 0.1j]:
        psi_final, _N = eng.run(delta, normalize=False)
        psi_final2, _N = eng.run(delta, normalize=True)
        n = _abs_norm(psi_final)
        assert n > 0
        assert _abs_norm(psi_final / n - psi_final2) < max(tol, 1.0e-8)
        psi_final_default, _N = eng.run(delta)
        assert almost_equal(psi_final_default, psi_final)

        if H_np is None or psi_init_np is None:
            continue
        psi_final_np = expm(H_np * delta).dot(psi_init_np)
        ref_norm = np.linalg.norm(psi_final_np)
        got = _try_to_numpy(psi_final)
        if got is None:
            continue
        diff = np.linalg.norm(got - psi_final_np)
        print(f'delta={delta}, norm(diff)/norm = {diff / ref_norm}')
        assert diff / ref_norm < max(tol, 1.0e-8)


def test_arnoldi_vs_lanczos_nonhermitian(make_compatible_tensor, tol_arnoldi=1.0e-8, tol_lanczos_wrong=1.0e-2):
    """ArnoldiEvolution is accurate for non-Hermitian H; LanczosEvolution is not."""
    H, H_op, psi_init = _op_and_vec(make_compatible_tensor, hermitian=False)
    # Anti-Hermitian: H = 1j * G with G hermitian, so Lanczos eigh is the wrong decomposition.
    G = H + H.dagger
    H_ah = 1j * G
    H_op = sparse.TensorLinearOperator(H_ah)
    H_np = _try_to_numpy(H_ah)
    psi_init_np = None if H_np is None else _try_to_numpy(psi_init)
    if H_np is None or psi_init_np is None:
        pytest.skip('dense conversion needed to compare against expm')

    delta = 1.0
    psi_ref_flat = expm(H_np * delta).dot(psi_init_np)
    norm_ref = np.linalg.norm(psi_ref_flat)

    psi_arnoldi, _ = krylov_based.ArnoldiEvolution(H_op, psi_init, {'N_max': 20}).run(delta, normalize=False)
    got_a = _try_to_numpy(psi_arnoldi)
    assert got_a is not None
    diff_arnoldi = np.linalg.norm(got_a - psi_ref_flat)
    print(f'ArnoldiEvolution diff/norm = {diff_arnoldi / norm_ref}')
    assert diff_arnoldi / norm_ref < tol_arnoldi

    psi_lanczos, _ = krylov_based.LanczosEvolution(H_op, psi_init, {}).run(delta, normalize=False)
    got_l = _try_to_numpy(psi_lanczos)
    assert got_l is not None
    diff_lanczos = np.linalg.norm(got_l - psi_ref_flat)
    print(f'LanczosEvolution diff/norm = {diff_lanczos / norm_ref}  (expected to be WRONG)')
    if H_ah.num_parameters > 4:
        assert diff_lanczos / norm_ref > tol_lanczos_wrong


def test_gmres(make_compatible_tensor, tol=1.0e-8):
    H, H_op, b = _op_and_vec(make_compatible_tensor, hermitian=False)
    # Shift away from zero so A is unlikely to be singular in this sector.
    A = sparse.ShiftedLinearOperator(H_op, 1.5)
    x0 = 0.0 * b
    x, rel_err, _errors, _iters = krylov_based.GMRES(
        A, x0, b, {'N_max': 20, 'restart': 10, 'res': 1.0e-10, 'N_min': 0}
    ).run()
    b_n = _abs_norm(b)
    residual = _abs_norm(A.matvec(x) - b)
    print(f'GMRES rel_err={rel_err}, residual/|b|={residual / b_n}')
    assert residual / b_n < 1.0e-6
    assert rel_err < 1.0e-6
