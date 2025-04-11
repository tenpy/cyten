from cyten import symmetries, spaces, backends, tensors
import numpy as np
import h5py

def create_test_random_symmetric_tensor():
    sym = symmetries.SU2Symmetry()
    sec = np.random.choice(int(1.3 * 3), replace=False, size=(3, 1))
    x1 = spaces.ElementarySpace.from_defining_sectors(sym, sec)
    xp1 = spaces.TensorProduct([x1]*2)
    xp2 = spaces.TensorProduct([x1]*4)
    dat = np.random.normal(size=(11,11,11,11,11,11))
    tens = tensors.SymmetricTensor.from_dense_block(dat, xp1, xp2, tol=10**20, labels=[['p1', 'p2'],
                                                                                       ['p1*', 'p2*', 'p3*', 'p4*']])
    return tens

def create_test_random_SU3_symmetric_tensor():
    cgf = h5py.File('/home/t30/all/ge36xeh/Desktop/Link to ge36xeh/PycharmProjects/pythonProject/cytens/cyten/Test_N_3_HWeight_8.hdf5','r')
    ff = h5py.File('/home/t30/all/ge36xeh/Desktop/Link to ge36xeh/PycharmProjects/pythonProject/cytens/cyten/Test_Fsymb_3_HWeight_3.hdf5','r')
    rf = h5py.File('/home/t30/all/ge36xeh/Desktop/Link to ge36xeh/PycharmProjects/pythonProject/cytens/cyten/Test_Rsymb_3_HWeight_5.hdf5','r')

    sym = symmetries.SUNSymmetry(N=3, CGfile=cgf, Ffile=ff, Rfile=rf)


    sec = np.array([[1, 1, 0]])
    #sec = np.array([[1, 0, 0]])
    print(symmetries.SUNSymmetry.batch_qdim(sym,sec))
    print(symmetries.SUNSymmetry.are_valid_sectors(sym,sec))
    #x1 = spaces.ElementarySpace.from_defining_sectors(sym, sec)
    x1 = spaces.ElementarySpace.from_defining_sectors(sym, sec)
    #x2 = spaces.ElementarySpace.from_defining_sectors(sym, sec2)

    xp1 = spaces.TensorProduct([x1] * 2)
    xp2 = spaces.TensorProduct([x1] * 2)

    dat = np.random.normal(size=(6, 6, 6, 6))
    tens = tensors.SymmetricTensor.from_dense_block(dat, xp1, xp2, tol=10**20, labels=[['p1', 'p2'],
                                                                                       ['p1*', 'p2*']])
    return tens

def SU2_sym_test_tensor():
    sym = symmetries.SU2Symmetry()
    spin_half = spaces.ElementarySpace(sym, np.array([[1]]))
    backend=backends.backend_factory.get_backend(sym, 'numpy')

    sx = .5 * np.array([[0., 1.], [1., 0.]], dtype=complex)
    sy = .5 * np.array([[0., -1.j], [1.j, 0]], dtype=complex)
    sz = .5 * np.array([[1., 0.], [0., -1.]], dtype=complex)

    heisenberg_4 = sum(si[:, :, None, None] * si[None, None, :, :] for si in [sx, sy, sz])  # [p1, p1*, p2, p2*]
    heisenberg_4 = np.transpose(heisenberg_4, [0, 2, 3, 1])  # [p1, p2, p2*, p1*]

    tens = tensors.SymmetricTensor.from_dense_block(
        heisenberg_4, codomain=[spin_half, spin_half], domain=[spin_half, spin_half],
        backend=backend, labels=[['p1', 'p2'], ['p1*', 'p2*']],tol=10**-8
    )

    return tens


def U1_sym_test_tensor():
    sym = symmetries.U1Symmetry()
    spin_half = spaces.ElementarySpace(sym, np.array([[1]]))
    backend = backends.backend_factory.get_backend(sym, 'numpy')

    sx = .5 * np.array([[0., 1.], [1., 0.]], dtype=complex)
    sy = .5 * np.array([[0., -1.j], [1.j, 0]], dtype=complex)
    sz = .5 * np.array([[1., 0.], [0., -1.]], dtype=complex)

    heisenberg_4 = sum(si[:, :, None, None] * si[None, None, :, :] for si in [sx, sy, sz])  # [p1, p1*, p2, p2*]
    heisenberg_4 = np.transpose(heisenberg_4, [0, 2, 3, 1])  # [p1, p2, p2*, p1*]

    tens = tensors.SymmetricTensor.from_dense_block(
        heisenberg_4, codomain=[spin_half, spin_half], domain=[spin_half, spin_half],
        backend=backend, labels=[['p1', 'p2'], ['p1*', 'p2*']], tol=10**-8
    )

    return tens

def create_test_random_symmetric_tensor():
    sym = symmetries.SU2Symmetry()
    sec = np.random.choice(int(1.3 * 3), replace=False, size=(3, 1))

    x1 = spaces.ElementarySpace.from_defining_sectors(sym,sec)
    xp1 = spaces.TensorProduct([x1]*4)
    xp2 = spaces.TensorProduct([x1]*2)

    dat = np.random.normal(size=(11,11,11,11,11,11))

    tens = tensors.SymmetricTensor.from_dense_block(dat, xp1, xp2, tol=10**20, labels=[['p1', 'p2', 'p3', 'p4'],
                                                                                       ['p1*', 'p2*']])

    return tens

def create_test_random_diagonal_tensor():
    sym = symmetries.SU2Symmetry()
    sec = np.random.choice(int(1.3 * 3), replace=False, size=(3, 1))

    x1 = spaces.ElementarySpace.from_defining_sectors(sym, sec)
    dat = np.diag(np.random.normal(size=2000))

    tens = tensors.DiagonalTensor.from_dense_block(dat, x1, tol=10**20)

    return tens

def create_test_Mask():
    sym = symmetries.SU2Symmetry()
    sec = np.random.choice(int(1.3 * 3), replace=False, size=(3, 1))

    x1 = spaces.ElementarySpace.from_defining_sectors(sym, sec)
    dat = np.diag(np.random.choice([True, False], size=2000))

    tens = tensors.DiagonalTensor.from_dense_block(dat, x1, tol=10**20)
    tens = tensors.Mask.from_DiagonalTensor(tens)

    return tens

def flip_leg_duality(tens: tensors.Tensor, leg_indices):
    """
    Flips the duality of a given Tensor leg. The leg can also be a leg pipe in which case
    also the combinestyle is flipped
    """

    for i in leg_indices:
        if isinstance(tens.legs[i], spaces.AbelianLegPipe):
            tens.legs[i] = spaces.AbelianLegPipe.with_opposite_duality_and_combinestyle(tens.legs[i])

        # elif isinstance(tens.legs[i], spaces.LegPipe) and not isinstance(tens.legs[i], spaces.AbelianLegPipe):
        #     raise NotImplementedError('Not yet implemented')
        else:
            tens.legs[i] = tens.legs[i].dual


def te_pipe_dualities(tens: tensors.Tensor):
    """
    Tests the equivalence of tensor leg transformations involving pipe dualities.

    Checks consistency of duality transformations and bendings applied to the tensors legs
    such that the leg duality labels for the following diagram commute:

            ┏─────────────────────────────────────────┓
            ▼                                         ▼
          A    B               A    B               A    B
          ^    ^               ^    ^               ^    ^
        ┏━┷━━━━┷━┓           ┏━┷━━━━┷━┓           ┏━┷━━━━┷━┓
        ┃ tens_lu┃   ◀────▶  ┃  tens  ┃   ◀────▶  ┃ tens_ru┃
        ┗━━━━┯━━━┛           ┗━┯━━━━┯━┛           ┗━━━━┯━━━┛
             ▼                 ^    ^                  ▲
        (V* ⊗ W*)              W    V               (W ⊗ V)
             ▲                   ▲                     ▲
             │                   │                     │
             ▼                   ▼                     ▼
          A B (V* ⊗ W*)        A B V W              A B (W ⊗ V)
          ^ ^  ▲               ^ ^ ^ ^              ^ ^  ▼
        ┏━┷━┷━━┷━┓           ┏━┷━┷━┷━┷┓           ┏━┷━┷━━┷━┓
        ┃ tens_ld┃   ◀────▶  ┃ tens_md┃   ◀────▶  ┃ tens_rd┃
        ┗━━━━━━━━┛           ┗━━━━━━━━┛           ┗━━━━━━━━┛
            ▲                                         ▲
            ┗─────────────────────────────────────────┛
    Parameters:
    ----------
    tens : tensors.Tensor
        The input tensor whose leg dualities are being tested.
    """

    lcd = len(tens.codomain)
    k = lcd
    kn = k+1

    tens_lu = tensors.combine_legs(tens, [k, kn], pipe_dualities=[False])
    tens_ld = tensors.bend_legs(tens_lu, lcd+1)

    tens_md = tensors.bend_legs(tens, lcd+2)
    tens_ld_p = tensors.combine_legs(tens_md, [k, kn], pipe_dualities=[False])

    assert tensors.almost_equal(tens_ld, tens_ld_p)
    assert tens_ld.legs.__eq__(tens_ld_p.legs)

    tens_ru = tensors.combine_legs(tens, [k, kn], pipe_dualities=[True])
    tens_rd = tensors.bend_legs(tens_ru, lcd+1)

    tens_md = tensors.bend_legs(tens, lcd+2)
    tens_rd_p = tensors.combine_legs(tens_md, [k, kn], pipe_dualities=[True])

    assert tensors.almost_equal(tens_rd, tens_rd_p)
    assert tens_rd.legs.__eq__(tens_rd_p.legs)

    flip_leg_duality(tens_ld, [k])
    assert tens_rd.legs.__eq__(tens_ld.legs)

    flip_leg_duality(tens_lu, [k])
    assert tens_ru.legs.__eq__(tens_lu.legs)


    def iter_uncoupled(self) -> Iterator[SectorArray]:
        """Iterate over all combinations of sectors

        For a TensorProduct of zero spaces, i.e. with ``num_space == 0``, we yield an empty
        array once.
        """

        if self.num_factors == 0:
            yield self.symmetry.empty_sector_array
            return

        if any(not isinstance(sp, ElementarySpace) for sp in self.factors):
            tup = []
            for i in self.factors:
                if not isinstance(i, LegPipe):
                    tup.append(i.sector_decomposition)
                else:
                    for leg in i.legs:
                        tup.append(leg.sector_decomposition)
            for unc in it.product(*tup):
                yield np.array(unc, int)

        for unc in it.product(*(s.sector_decomposition for s in self.factors)):
            yield np.array(unc, int)


if __name__ == "__main__":

    # tens = U1_sym_test_tensor()
    # sym = tens.symmetry
    #
    # te_pipe_dualities(tens)

    tens2=SU2_sym_test_tensor()
    #te_pipe_dualities(tens2)

    tens2=tensors.combine_legs(tens2,[0,1])
    tens2.test_sanity()
    print(tens2.ascii_diagram)

    print(tens2.legs[0])

    flip_leg_duality(tens2, [0])

    print(tens2.legs[0])


    #print(tens2.legs[0].legs)

    # tup = []
    # for i in tens2.legs:
    #     if not isinstance(i, spaces.LegPipe):
    #         tup.append(i.sector_decomposition)
    #     else:
    #         for l in i.legs:
    #             tup.append(l.sector_decomposition)
    #
    # print(tup)




    #flip_leg_duality(tens, [2])
    #te_pipe_dualities(tens)











