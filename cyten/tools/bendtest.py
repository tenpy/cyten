from cyten import symmetries, spaces, backends, tensors
import numpy as np
import h5py
#from tests.python_tests.tools import io_test


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

def backbend(tens: tensors.Tensor) -> tensors.Tensor:
    leglist = []
    #print(tens.domain_labels, [i.is_dual for i in tens.domain])
    #print(tens.codomain_labels, [i.is_dual for i in tens.codomain])

    for c, l in enumerate(tens.domain):

        if l.is_dual:
            #print(c, tens.domain_labels[c], tens.get_leg_idcs(tens.domain_labels[c]))
            leglist.append(tens.get_leg_idcs(tens.domain_labels[c])[0])

    for i in reversed(leglist):
        tens = tensors.move_leg(tens, i, codomain_pos=-1)
        #print(tens.ascii_diagram)

    leglist = []
    for c, l in enumerate(tens.codomain):

        if l.is_dual:
            #print(c, tens.codomain_labels[c], tens.get_leg_idcs(tens.codomain_labels[c]))
            leglist.append(tens.get_leg_idcs(tens.codomain_labels[c])[0])

    for i in reversed(leglist):
        tens = tensors.move_leg(tens, i, domain_pos=-1)
        #print(tens.ascii_diagram)

    return tens


def backbend2(tens: tensors.Tensor) -> tensors.Tensor:
    leglist = []
    coleglist=[]
    domainlist= []
    codomainlist=[]

    for c, l in enumerate(tens.domain):

        if l.is_dual:
            #print(c, tens.domain_labels[c], tens.get_leg_idcs(tens.domain_labels[c]))
            leglist.append(tens.get_leg_idcs(tens.domain_labels[c])[0])
        else:
            domainlist.append(tens.get_leg_idcs(tens.domain_labels[c])[0])


    for c, l in enumerate(tens.codomain):

        if l.is_dual:
            #print(c, tens.codomain_labels[c], tens.get_leg_idcs(tens.codomain_labels[c]))
            coleglist.append(tens.get_leg_idcs(tens.codomain_labels[c])[0])
        else:
            codomainlist.append(tens.get_leg_idcs(tens.codomain_labels[c])[0])


    for i in leglist:
        codomainlist.append(i)

    for i in coleglist:
        domainlist.append(i)

    tens = tensors.permute_legs(tens, domain=domainlist, codomain=codomainlist)
    return tens


def backbend3(tens: tensors.Tensor) -> tensors.Tensor:
    domainlist = [tens.get_leg_idcs(label)[0] for label, leg in zip(tens.domain_labels, tens.domain) if not leg.is_dual]
    codomainlist = [tens.get_leg_idcs(label)[0] for label, leg in zip(tens.codomain_labels, tens.codomain) if
                    not leg.is_dual]

    leglist = [tens.get_leg_idcs(label)[0] for label, leg in zip(tens.domain_labels, tens.domain) if leg.is_dual]
    coleglist = [tens.get_leg_idcs(label)[0] for label, leg in zip(tens.codomain_labels, tens.codomain) if leg.is_dual]

    codomainlist.extend(leglist)
    domainlist.extend(coleglist)

    return tensors.permute_legs(tens, domain=sorted(domainlist)[::-1], codomain=sorted(codomainlist))

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
    backend=backends.backend_factory.get_backend(sym, 'numpy')

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



def sanity_check_hdf5(f):

    def has_data_in_group(group):
        # If it's a dataset, check if it has data
        if isinstance(group, h5py.Dataset):
            return group.size > 0  # Dataset is not empty

        # If it's a group, recursively check its contents
        elif isinstance(group, h5py.Group):
            # Iterate through all items in the group and check if any of them has data
            for key in group.keys():
                if has_data_in_group(group[key]):
                    return True
        return False

    H = f.attrs['Highest_Weight']
    N = f.attrs['N']
    type = str(list(f.keys())[0])[0]

    if type == 'F':
        # Check if /F_sym/ group exists
        if '/F_sym/' not in f:
            raise ValueError("HDF5 file does not contain '/F_sym/' group.")

        keys = list(f['/F_sym/'].keys())
        print(f"Found {len(keys)} keys in /F_sym/:", keys[:10])  # Preview first 10 keys

        # Ensure all keys start with 'F['
        valid_keys = [key for key in keys if key.startswith('F[')]
        if not valid_keys:
            raise ValueError("No valid F-symbol keys found in '/F_sym/'.")

        # Determine list length by checking how many '[...]' sections are in the first key
        first_key = valid_keys[0]
        num_lists = first_key.count('[')
        print(f"Detected {num_lists} lists per key (from first key: {first_key})")

        # Check for all-zero key
        zero_key = 'F' + ''.join('[0' + ', 0' * (first_key.count(',') // num_lists) + ']' for _ in range(num_lists))
        if zero_key not in keys:
            raise ValueError(f"Missing key for all-trivial-sector F-symbol: {zero_key}")

        # Check for at least one entry containing [H, H, 0]
        h_key = f"[{H}, {H}, 0]"
        found_h_key = any(h_key in key for key in keys)
        if not found_h_key:
            raise ValueError(f"No key found containing {h_key}.")

    elif type == 'R':
        # Check if /R_sym/ group exists
        if '/R_sym/' not in f:
            raise ValueError("HDF5 file does not contain '/R_sym/' group.")

        keys = list(f['/R_sym/'].keys())
        print(f"Found {len(keys)} keys in /R_sym/:", keys[:10])  # Preview first 10 keys

        # Ensure all keys start with 'R['
        valid_keys = [key for key in keys if key.startswith('R[')]
        if not valid_keys:
            raise ValueError("No valid R-symbol keys found in '/R_sym/'.")

        # Determine list length by checking how many '[...]' sections are in the first key
        first_key = valid_keys[0]
        num_lists = first_key.count('[')
        print(f"Detected {num_lists} lists per key (from first key: {first_key})")

        # Check for all-zero key
        zero_key = 'R' + ''.join('[0' + ', 0' * (first_key.count(',') // num_lists) + ']' for _ in range(num_lists))
        if zero_key not in keys:
            raise ValueError(f"Missing key for all-trivial-sector R-symbol: {zero_key}")

        # Check for at least one entry containing [H, H, 0]
        h_key = f"[{H}, {H}, 0]"
        found_h_key = any(h_key in key for key in keys)
        if not found_h_key:
            raise ValueError(f"No key found containing {h_key}.")


    elif type == 'N':

        if f'/N_{N}/' not in f:
            raise ValueError(f'HDF5 file does not contain /N_{N}/ group.')

        keys = list(f[f'/N_{N}/'].keys())
        print(f"Found {len(keys)} keys in /N_{N}/.")
        assert len(keys) == H+1 # Contains all the keys up to the highest weight

        high=f[f'/N_{N}/'+str(keys[-1])]
        low=f[f'/N_{N}/' + str(keys[0])]

        for group in [high,low]:

            assert len(group.keys()) != 0  # Assert key for loop weight is non-empty

            if not has_data_in_group(group):
                raise ValueError(f"Key exists but contains no data.")

        print(f[f'/N_{N}/'+str(keys[0])])

    print("Sanity checks passed")


def zisotest(a: symmetries.Sector):
    nfile=h5py.File('/home/t30/all/ge36xeh/Desktop/Link to ge36xeh/PycharmProjects/pythonProject/cytens/cyten/Test_N_3_HWeight_8.hdf5','r')
    ffile=h5py.File('/home/t30/all/ge36xeh/Desktop/Link to ge36xeh/PycharmProjects/pythonProject/cytens/cyten/Test_Fsymb_3_HWeight_3.hdf5','r')
    rfile=h5py.File('/home/t30/all/ge36xeh/Desktop/Link to ge36xeh/PycharmProjects/pythonProject/cytens/cyten/Test_Rsymb_3_HWeight_5.hdf5','r')
    sym = symmetries.SUNSymmetry(3,nfile,ffile,rfile)

    print('Ziso:', sym.Z_iso(a))
    return np.sqrt(sym.sector_dim(a))*sym.fusion_tensor( a, sym.dual_sector(a),sym.trivial_sector)[0, :, :, 0]

def checkZiso(a: symmetries.Sector):
    nfile = h5py.File(
        '/home/t30/all/ge36xeh/Desktop/Link to ge36xeh/PycharmProjects/pythonProject/cytens/cyten/Test_N_3_HWeight_8.hdf5',
        'r')
    ffile = h5py.File(
        '/home/t30/all/ge36xeh/Desktop/Link to ge36xeh/PycharmProjects/pythonProject/cytens/cyten/Test_Fsymb_3_HWeight_3.hdf5',
        'r')
    rfile = h5py.File(
        '/home/t30/all/ge36xeh/Desktop/Link to ge36xeh/PycharmProjects/pythonProject/cytens/cyten/Test_Rsymb_3_HWeight_5.hdf5',
        'r')
    sym = symmetries.SUNSymmetry(3, nfile, ffile, rfile)

    # d_a = sym.sector_dim(a)
    # a_bar = sym.dual_sector(a)
    # Z_a = sym.Z_iso(a)
    # Z_a_hc = Z_a.conj().T

    d_a = sym.sector_dim(a)
    a_bar = sym.dual_sector(a)
    Z_a = sym.Z_iso(a)
    Z_a_bar = sym.Z_iso(a_bar)
    Z_a_hc = Z_a.conj().T



    # # relationship to cap
    # Y_a_abar_u = sym.fusion_tensor(a, a_bar, sym.trivial_sector).conj()[0, :, :, 0]  # set mu=0, m_c=0
    # cup = np.eye(d_a)
    # expect_1 = np.tensordot(cup, Z_a, (1, 0)) / np.sqrt(d_a)
    # expect_2 = sym.frobenius_schur(a) * np.tensordot(Z_a_bar, cup, (1, 0)) / np.sqrt(d_a)
    # #assert_array_almost_equal(Y_a_abar_u, expect_1)
    # #assert_array_almost_equal(Y_a_abar_u, expect_2)

    Y_a_abar_u = sym.fusion_tensor(a, a_bar, sym.trivial_sector).conj()[0, :, :, 0]  # [m_a, m_abar]
    cup = np.eye(d_a)
    expect_1 = np.tensordot(cup, Z_a_bar, (1, 1)) / np.sqrt(d_a)  # [m_a, m_a] @ [m_abar, m_a] -> [m_a, m_abar]
    expect_2 = sym.frobenius_schur(a) * np.tensordot(Z_a, cup, (1, 0)) / np.sqrt(
        d_a)  # [m_a, m_abar] @ [m_abar, m_abar] -> [m_a, m_abar]


    print(Z_a)
    print(sym.Z_iso(a_bar))
    #
    # X_a_abar_u = sym.fusion_tensor(a, a_bar, sym.trivial_sector)[0, :, :, 0]  # set mu=0, m_c=0
    # cup = np.eye(d_a)
    # expect_1 = np.tensordot(cup, Z_a_hc, (1, 1)) / np.sqrt(d_a)
    # expect_2 = sym.frobenius_schur(a) * np.tensordot(Z_a_hc, cup, (1, 0)) / np.sqrt(d_a)

    print(sym.frobenius_schur(a))
    #print(np.trace(Z_a_hc@Z_a), d_a)
    print(Y_a_abar_u)
    print(':::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::')
    print(expect_2)
    print('Unitary:', np.all(np.isclose(Z_a @ Z_a_hc, np.eye(d_a))), np.all(np.isclose(Z_a_hc @ Z_a, np.eye(d_a))))
    print('R1:', np.all(np.isclose(Y_a_abar_u, expect_1, atol=10**-5)))
    print('R2:', np.all(np.isclose(Y_a_abar_u, expect_2, atol=10**-5)))

    return


if __name__ == "__main__":
    ###print(sanity_check_hdf5(h5py.File('/home/t30/all/ge36xeh/Desktop/Link to ge36xeh/PycharmProjects/pythonProject/cytens/cyten/Test_N_3_HWeight_8.hdf5','r')))
    #Z=zisotest(np.array([3,1,0]))

    np.set_printoptions(linewidth=300)
    print(checkZiso(np.array([2,1,0])))

    #print(Z@Z.conj().T)
    #print(np.all(np.isclose(Z@Z.conj().T, np.eye(Z.shape[0]))))

    # #print(sanity_check_hdf5(h5py.File('/home/t30/all/ge36xeh/Desktop/Link to ge36xeh/PycharmProjects/pythonProject/cytens/cyten/Test_Rsymb_3_HWeight_5.hdf5','r')))
    # #print(sanity_check_hdf5(h5py.File('/home/t30/all/ge36xeh/Desktop/Link to ge36xeh/PycharmProjects/pythonProject/cytens/cyten/Test_N_3_HWeight_8.hdf5','r')))
    # testtens = create_test_random_SU3_symmetric_tensor()
    # #testdiag = create_test_Mask()
    #
    # print(testtens.ascii_diagram)


    #tensors.combine_legs(testtens,[0,1])
    #print(testtens.ascii_diagram)


    # print(testtens.domain_labels, [i.is_dual for i in testtens.domain])
    # print(testtens.codomain_labels, [i.is_dual for i in testtens.codomain])
    #
    # #for i in testtens.legs:
    #     #print(i)
    #
    #
    #testbend = tensors.bend_legs(testtens, 1)
    # testbend = tensors.permute_legs(testtens, [0, 1, 2, 3])
    # print(testbend.codomain)
    #
    #
    # print(testbend.ascii_diagram)
    #testbend.to_numpy()
    # bendcod = testbend.codomain
    # benddom = testbend.domain
    # bendback = testbend.backend
    # bendlabels = testbend.labels
    #print(testbend.domain)
    #print(testbend.codomain)
    #
    # print(testbend.ascii_diagram)
    # #print(len(testbend.domain),len(testbend.codomain))
    #
    #


    # ret=backbend3(testbend)
    # nparr = ret.to_numpy()
    #
    # print(nparr.shape)
    #
    #
    # cgf = h5py.File('/home/t30/all/ge36xeh/Desktop/Link to ge36xeh/PycharmProjects/pythonProject/cytens/cyten/Test_N_3_HWeight_8.hdf5','r')
    # ff = h5py.File('/home/t30/all/ge36xeh/Desktop/Link to ge36xeh/PycharmProjects/pythonProject/cytens/cyten/Test_Fsymb_3_HWeight_3.hdf5','r')
    # rf = h5py.File('/home/t30/all/ge36xeh/Desktop/Link to ge36xeh/PycharmProjects/pythonProject/cytens/cyten/Test_Rsymb_3_HWeight_5.hdf5','r')
    #
    # sym = symmetries.SUNSymmetry(N=3, CGfile=cgf, Ffile=ff, Rfile=rf)
    #
    # sec = np.array([[1, 1, 0]])
    # print(sym.batch_sector_dim(sec))
    #
    # x1 = spaces.ElementarySpace.from_defining_sectors(sym, sec)
    # xp1 = spaces.TensorProduct([x1] * 2)
    # xp2 = spaces.TensorProduct([x1] * 2)
    # tensfromnp = tensors.SymmetricTensor.from_dense_block(nparr, xp1, xp2, tol=10**20)
    # #
    # print(ret.ascii_diagram)
    # #
    # print(tensors.almost_equal(ret, testtens))
    # print(tensors.almost_equal(tensfromnp, ret))
    #
    #
    #
    #
    #
    #
    # #
    # #
    # # nptens = testbend.to_numpy()
    # #
    # # npfrom = tensors.SymmetricTensor.from_dense_block(nptens, bendcod, benddom, bendback,bendlabels )
    # #
    # #
    # # print(npfrom.ascii_diagram)
    # #
    # # print(tensors.almost_equal(npfrom, testbend))
    # # #print(nptens)
    # # #print(testtens.to_numpy())




