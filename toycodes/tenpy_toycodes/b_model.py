"""Toy code implementing the transverse-field ising model."""
# Copyright (C) TeNPy Developers, Apache license
import cyten as ct
import numpy as np


class TFIModel:
    """TFI: -J XX - g Z"""
    def __init__(self, L, J, g, bc='finite'):
        assert bc in ['finite', 'infinite']
        # TODO should we add subclasse for most important case Z2?
        self.symmetry = sym = ct.ZNSymmetry(2, 'Sz_parity')
        self.phys_leg = ct.ElementarySpace(sym, [[0], [1]])  # basis : [down, up]
        self.L = L
        self.bc = bc
        self.J = J
        self.g = g
        self.init_H_bonds()
        self.init_H_mpo()

    def init_H_bonds(self):
        """Initialize `H_bonds` hamiltonian.

        Called by __init__().
        """
        nbonds = self.L - 1 if self.bc == 'finite' else self.L
        p = self.phys_leg
        sx = np.array([[0, 1], [1, 0]], float)
        XX = ct.SymmetricTensor.from_dense_block(sx[:, None, None, :] * sx[None, :, :, None], [p, p], [p, p],
                                                 labels=['p0', 'p1', 'p1*', 'p0*'])
        Z = ct.SymmetricTensor.from_dense_block([[1, 0], [0, -1]], [p], [p], labels=['p', 'p*'])
        I = ct.SymmetricTensor.from_eye([p], labels=['p'])
        IZ = ct.outer(I, Z, {'p': 'p0', 'p*': 'p0*'}, {'p': 'p1', 'p*': 'p1*'})
        ZI = ct.outer(I, Z, {'p': 'p0', 'p*': 'p0*'}, {'p': 'p1', 'p*': 'p1*'})
        H_list = []
        for i in range(nbonds):
            gL = gR = 0.5 * self.g
            if self.bc == 'finite':
                if i == 0:
                    gL = self.g
                if i + 1 == self.L - 1:
                    gR = self.g
            H_list.append(-self.J * XX - gL * ZI - gR * IZ)
        self.H_bonds = H_list

    # (note: not required for TEBD)
    def init_H_mpo(self):
        """Initialize `H_mpo` Hamiltonian.

        Called by __init__().
        """
        p = self.phys_leg
        v = ct.ElementarySpace.from_basis(self.symmetry, [[0], [1], [0]])  # basis [IdL A IdR]
        w_list = []
        sx = np.array([[0, 1], [1, 0]], float)
        sz = np.array([[1, 0], [0, -1]], float)
        for i in range(self.L):
            w = np.zeros((3, 3, self.d, self.d), dtype=float)
            w[0, 0] = w[2, 2] = np.eye(2)
            w[0, 1] = sx
            w[0, 2] = -self.g * sz
            w[1, 2] = -self.J * sx
            w = ct.SymmetricTensor.from_dense_block(w, [v, p], [p, v], labels=['vL', 'p', 'vR', 'p*'])
            w_list.append(w)
        self.H_mpo = w_list


def main():
    print('Running this example module just initilizes the TFI model.')
    _ = TFIModel(L=10, J=1, g=0.8, bc='finite')
    _ = TFIModel(L=10, J=1, g=0.8, bc='infinite')


if __name__ == '__main__':
    main()
