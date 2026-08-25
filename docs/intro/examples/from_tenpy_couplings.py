"""Couplings as mini-MPOs: Heisenberg, c^dagger c, and a chiral three-spin term."""

import numpy as np

import cyten as ct

# --- Heisenberg: two-site mini-MPO -----------------------------------------

site0 = ct.models.SpinSite(S=0.5, conserve='Sz')
site1 = ct.models.SpinSite(S=0.5, conserve='Sz')
heisenberg = ct.models.heisenberg_coupling([site0, site1], J=1)
print(heisenberg)
assert heisenberg.num_sites == 2
assert heisenberg.name == 'S.S'
for W in heisenberg.factorization:
    assert W.labels == ['wL', 'p', 'wR', 'p*']

H = heisenberg.to_tensor()
W, _ = ct.eigh(H, new_labels=['e'], new_leg_dual=False)
evals = np.sort(np.real(W.diagonal_as_numpy()))
np.testing.assert_allclose(evals, [-0.75, 0.25, 0.25, 0.25], atol=1e-12)


# --- chiral three-spin: S · (S × S) ----------------------------------------

chiral = ct.models.chiral_3spin_coupling([site0, site0, site0], chi=1)
print(chiral)
assert chiral.num_sites == 3
assert chiral.name == 'S.SxS'
H3 = chiral.to_tensor()
assert ct.almost_equal(H3.hc, H3)
np.testing.assert_allclose(ct.trace(H3).to_numpy(), 0.0, atol=1e-12)


# --- c^dagger_i c_j as a (non-Hermitian) two-site coupling -----------------

ferm = ct.models.SpinlessFermionSite(num_species=1, conserve='N')
Cd = ferm.get_creator_numpy(species=0, include_JW=True)
C = ferm.get_annihilator_numpy(species=0, include_JW=True)
# Dense axes: p0, p1, p1*, p0*. The left factor carries the on-site JW dressing.
h_cdag_c = (Cd @ ferm.JW)[:, None, None, :] * C[None, :, :, None]
cdag_c = ct.Coupling.from_dense_block(h_cdag_c, [ferm, ferm], name='Cd C', understood_braiding=True)
print(cdag_c)
assert cdag_c.num_sites == 2
wR = cdag_c.factorization[0].get_leg_co_domain('wR')
wL = cdag_c.factorization[1].get_leg_co_domain('wL')
assert wR == wL
# One fermion created on the left and annihilated on the right: bond dimension 1.
assert wR.dim == 1

# Hamiltonian hopping is the Hermitian combination  -t (c^dagger c + h.c.).
hop = ct.models.hopping([ferm, ferm], t=1)
assert hop.num_sites == 2
assert ct.almost_equal(hop.to_tensor().hc, hop.to_tensor())


# --- stretch across a site: virtual leg crosses the physical fermions ------

# Sites 0 and 2 hold Cd and C; site 1 gets identity_tensor(w).
stretched = cdag_c.stretch_with_identities([ferm, ferm, ferm], [0, 2])
assert stretched.num_sites == 3
assert ct.almost_equal(stretched.factorization[0], cdag_c.factorization[0])
assert ct.almost_equal(stretched.factorization[2], cdag_c.factorization[1])

Id_middle = ferm.identity_tensor(wR)
assert ct.almost_equal(stretched.factorization[1], Id_middle)

# For a one-dimensional odd virtual bond, that identity is the local JW
# operator: crossing an odd fermion line past the physical space yields (-1)^n.
jw_block = Id_middle.to_numpy(understood_braiding=True)
np.testing.assert_allclose(jw_block[0, :, 0, :], ferm.JW, atol=1e-12)
