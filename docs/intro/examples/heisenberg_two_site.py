"""Build a two-site Heisenberg Hamiltonian and print its eigenvalues."""

import numpy as np

import cyten as ct

# Local spin-1/2 Hilbert space (no conserved quantum numbers).
site0 = ct.models.SpinSite(S=0.5, conserve=None)
site1 = ct.models.SpinSite(S=0.5, conserve=None)

# Two-site coupling  H = J S_0 · S_1  with antiferromagnetic J = 1.
coupling = ct.models.heisenberg_coupling([site0, site1], J=1)
print(coupling)

# Contract the MPO-like factors to a two-site Hamiltonian tensor.
H = coupling.to_tensor()
print(H)

# Hermitian eigen-decomposition. W is diagonal (the eigenvalues).
W, _ = ct.eigh(H, new_labels=['e'], new_leg_dual=False)
evals = np.sort(np.real(W.diagonal_as_numpy()))
print('eigenvalues:', evals)

# Singlet  E = -3/4  and triplet  E = +1/4.
np.testing.assert_allclose(evals, [-0.75, 0.25, 0.25, 0.25], atol=1e-12)
