from cyten import spaces, symmetries, tensors, backends, dtypes
import numpy as np
from tests.pytest.tools.hdf5_io.src.python3 import hdf5_io
import h5py
from cyten.tensors import SymmetricTensor, Mask, ChargedTensor
from cyten.backends import NumpyBlockBackend
from tests.pytest.tools import io_test
import pathlib
import pytest

# h5s=hdf5_io.Hdf5Saver
h5l=hdf5_io.Hdf5Loader

def SU2_sym_test_tensor():
    sym = symmetries.SU2Symmetry()
    spin_half = spaces.ElementarySpace(sym, np.array([[1]]))
    backend=backends.backend_factory.get_backend(sym,'numpy')

    sx = .5 * np.array([[0., 1.], [1., 0.]], dtype=complex)
    sy = .5 * np.array([[0., -1.j], [1.j, 0]], dtype=complex)
    sz = .5 * np.array([[1., 0.], [0., -1.]], dtype=complex)
    idm = np.eye(2, dtype=complex)

    heisenberg_4 = sum(si[:, :, None, None] * si[None, None, :, :] for si in [sx, sy, sz])  # [p1, p1*, p2, p2*]

    #print(heisenberg_4.transpose([0, 2, 1, 3]).reshape((4, 4)))
    heisenberg_4 = np.transpose(heisenberg_4, [0, 2, 3, 1])  # [p1, p2, p2*, p1*]
    #
    #
    # heisenberg_6 = sum(idm[:,:,None,None,None,None]*si[None,None,:,:,None,None]*si[None,None,None,None,:,:] for si in [sx,sy,sz])
    # heisenberg_6 += sum(si[:,:,None,None,None,None]*si[None,None,:,:,None,None]*idm[None,None,None,None,:,:] for si in [sx,sy,sz]) # p1, p1*, p2, p2*, p3, p3*
    # heisenberg_6 = np.transpose(heisenberg_6, [1, 3, 5, 4, 2, 0])  # [p1, p2, p3, p1*, p2*, p3*]


    #heisenberg_6=np.zeros((2,2,2,2,2,2))




    tens = SymmetricTensor.from_dense_block(
        heisenberg_4, codomain=[spin_half, spin_half], domain=[spin_half, spin_half],
        backend=backend, labels=[['p1', 'p2'], ['p1*', 'p2*']],tol=10**-8
    )
    #
    #
    # h2=sum(np.kron(si,si) for si in [sx,sy,sz])
    # heisenberg_6 = np.array(np.kron(h2,idm)+np.kron(idm,h2)) #[p1, p1*, p2, p2*, p3, p3*]
    # print(heisenberg_6)
    #
    # heisenberg_6=heisenberg_6.reshape([2,2,2,2,2,2])
    #
    # tens = SymmetricTensor.from_dense_block(
    #     heisenberg_6,
    #     codomain=spaces.ProductSpace([spin_half, spin_half, spin_half]),
    #     domain=spaces.ProductSpace([spin_half, spin_half, spin_half]),
    #     backend=backend, labels=[['p1', 'p2', 'p3'], ['p1*', 'p2*', 'p3*']],
    #     tol=10**20
    # )

    return tens


def U1_sym_test_tensor():
    sym = symmetries.U1Symmetry()
    spin_half = spaces.ElementarySpace(sym, np.array([[1]]))
    backend=backends.backend_factory.get_backend(sym,'numpy')

    sx = .5 * np.array([[0., 1.], [1., 0.]], dtype=complex)
    sy = .5 * np.array([[0., -1.j], [1.j, 0]], dtype=complex)
    sz = .5 * np.array([[1., 0.], [0., -1.]], dtype=complex)
    idm = np.eye(2, dtype=complex)

    heisenberg_4 = sum(si[:, :, None, None] * si[None, None, :, :] for si in [sx, sy, sz])  # [p1, p1*, p2, p2*]

    heisenberg_4 = np.transpose(heisenberg_4, [0, 2, 3, 1])  # [p1, p2, p2*, p1*]



    tens = SymmetricTensor.from_dense_block(
        heisenberg_4, codomain=[spin_half, spin_half], domain=[spin_half, spin_half],
        backend=backend, labels=[['p1', 'p2'], ['p1*', 'p2*']],tol=10**-8
    )

    print(tens.labels)


    return tens



def create_test_random_symmetric_tensor():
    sym=symmetries.SU2Symmetry()
    sec = np.random.choice(int(1.3 * 3), replace=False, size=(3, 1))
    #print(sec)

    x1 = spaces.ElementarySpace.from_sectors(sym, sec)
    xp1= spaces.ProductSpace([x1]*4)
    xp2=spaces.ProductSpace([x1]*2)

    dat = np.random.normal(size=(11,11,11,11,11,11))

    tens = tensors.SymmetricTensor.from_dense_block(dat,xp1,xp2,tol=10**20,labels = [['p1', 'p2', 'p3', 'p4'], ['p1*', 'p2*']])
    print(tens.labels)

    return tens


def create_test_random_diagonal_tensor():
    sym=symmetries.SU2Symmetry()
    sec = np.random.choice(int(1.3 * 3), replace=False, size=(3, 1))

    x1 = spaces.ElementarySpace.from_sectors(sym, sec)
    xp1= spaces.ProductSpace([x1]*4)
    dat = np.diag(np.random.normal(size=2000))

    tens=tensors.DiagonalTensor.from_dense_block(dat,xp1,tol=10**20)
    return tens

def create_test_zero_mask():
    sym = symmetries.SU2Symmetry()
    sec = np.random.choice(int(1.3 * 3), replace=False, size=(3, 1))

    x1 = spaces.ElementarySpace.from_sectors(sym, sec)
    xp1= spaces.ProductSpace([x1]*4)

    return Mask.from_zero(xp1.as_ElementarySpace())

def create_test_mask():
    sym = symmetries.SU2Symmetry()
    sec = np.random.choice(int(1.3 * 3), replace=False, size=(3, 1))
    backend = backends.backend_factory.get_backend(sym, 'numpy')

    x1 = spaces.ElementarySpace.from_sectors(sym, sec)
    xp1= spaces.ProductSpace([x1]*4)

    return Mask.from_eye(x1,True,backend)


def create_test_charged_tensor():
    sym = symmetries.SU2Symmetry()

    spin_half = spaces.ElementarySpace(sym, np.array([[1]]))
    backend = backends.backend_factory.get_backend(sym, 'numpy')

    sx = .5 * np.array([[0., 1.], [1., 0.]], dtype=complex)
    sy = .5 * np.array([[0., -1.j], [1.j, 0]], dtype=complex)
    sz = .5 * np.array([[1., 0.], [0., -1.]], dtype=complex)
    idm = np.eye(2, dtype=complex)

    sp=0.5*(sx+sy)

    sp=np.zeros((2,2,2,2,2,2))

    tens = ChargedTensor.from_dense_block(sp,
        codomain = spaces.ProductSpace([spin_half,spin_half,spin_half]),
        domain = spaces.ProductSpace([spin_half,spin_half,spin_half]),
        charge= spaces.ElementarySpace(sym, np.array([[0]])),
        backend=backend,
        labels = [['p1', 'p2', 'p3'], ['p1*', 'p2*', 'p3*']]
    )

    return tens

# def create_test_charged_tensor():
#     sym=symmetries.SU2Symmetry()
#     sec = np.random.choice(int(1.3 * 3), replace=False, size=(3, 1))
#
#     x1 = spaces.ElementarySpace.from_sectors(sym, sec)
#     xp1= spaces.ProductSpace([x1]*4)
#     xp2=spaces.ProductSpace([x1]*2)
#
#     dat = np.random.normal(size=(11,11,11,11,11,11))
#
#     tens = tensors.ChargedTensor.from_dense_block(dat,xp1,xp2,tol=10**20)
#     print(tens.labels)
#
#     return tens


def zero_su2_tensor():
    sym = symmetries.SU2Symmetry()
    spin_half = spaces.ElementarySpace(sym, np.array([[1]]))
    backend = backends.backend_factory.get_backend(sym, 'numpy')

    tens = SymmetricTensor.from_dense_block(
        np.zeros((2,2,2,2)), codomain=[spin_half, spin_half], domain=[spin_half, spin_half],
        backend=backend, labels=[['p1', 'p2'], ['p1*', 'p2*']], tol=10**-8)

    return tens


def export_tensor(tens: tensors.SymmetricTensor, filename: str):
    #filename='testfile_io.hdf5'
    data=tens

    with h5py.File(str(filename), 'w') as f:
        hdf5_io.save_to_hdf5(f, data)
    return

# def import_own_diagonal(path:str):
#     with h5py.File(path, 'r') as f:
#         data = h5py.File

def import_tensor(path:str):

    with h5py.File(path, 'r') as f:
        data = hdf5_io.load_from_hdf5(f)

    return data


class MyComplicatedClass:
    def __init__(self, x, y, abc):
        self.x = float(x)
        self.y = np.array(y)
        self.abc = abc

    def save_hdf5(self, hdf5_saver, h5gr, subpath):
        hdf5_saver.save(self.y, subpath + "y")  # for big content/data
        hdf5_saver.save(self.abc.abc, subpath + "abc.abc")  # for big content/data

        h5gr.attrs["x"] = x               # for small metadata

    @classmethod
    def from_hdf5(cls, hdf5_loader, h5gr, subpath):
        obj = cls.__new__(cls)                     # create class instance, no __init__() call
        hdf5_loader.memorize_load(h5gr, obj)       # call preferably before loading other data
        x = hdf5_loader.get_attr(h5gr, "x")  # for metadata
        y = hdf5_loader.load(subpath + "y")   # for big content/data
        abc = hdf5_loader.load(subpath + "abc")   # for big content/data
        obj.x = x
        obj.y = y
        obj.abc = abc
        return obj

# class MyComplicatedABC:
#     def __init__(self, a, b, c):
#         self.abc = (a,b,c)
#         self.L =
#
#
#     def save_hdf5(self, hdf5_saver, h5gr, subpath):
#         hdf5_saver.save(self.abc, subpath + "abc")  # for big content/data
#         h5gr.attrs["name"] = info               # for metadata
#
#     @classmethod
#     def from_hdf5(cls, hdf5_loader, h5gr, subpath):

def import_export_tests(tmp_path):
    testU1 = U1_sym_test_tensor()
    testSU2 = SU2_sym_test_tensor()
    testrand = create_test_random_symmetric_tensor()
    testdiag= create_test_random_diagonal_tensor()

    export_tensor(testU1, tmp_path + 'testU1.hdf5')
    export_tensor(testSU2, tmp_path + 'testSU2.hdf5')
    export_tensor(testrand, tmp_path + 'testrand.hdf5')
    export_tensor(testdiag, tmp_path + 'testdiag.hdf5')

    importU1= import_tensor(tmp_path + 'testU1.hdf5')
    importSU2 = import_tensor(tmp_path + 'testSU2.hdf5')
    importrand = import_tensor(tmp_path + 'testrand.hdf5')
    importdiag = import_tensor(tmp_path + 'testdiag.hdf5')


    io_test.assert_equal_data(importU1, testU1)
    io_test.assert_equal_data(importSU2, testSU2)
    io_test.assert_equal_data(importrand, testrand)
    io_test.assert_equal_data(importdiag, testdiag)

    return

@pytest.mark.filterwarnings(r'ignore:Hdf5Saver.* object of type.*:UserWarning')
def test_hdf5_export_import(make_compatible_space, compatible_backend, tmp_path):
    """Try subsequent export and import to pickle."""

    testU1 = U1_sym_test_tensor()
    testSU2 = SU2_sym_test_tensor()
    testrand = create_test_random_symmetric_tensor()
    testdiag= create_test_random_diagonal_tensor()

    for data in [testU1, testSU2, testrand,testdiag]:

        io_test.assert_event_handler_example_works(data)  #if this fails, it's not import/export
        filename = tmp_path / 'test.hdf5'
        with h5py.File(str(filename), 'w') as f:
            hdf5_io.save_to_hdf5(f, data)
        with h5py.File(str(filename), 'r') as f:
            data_imported = hdf5_io.load_from_hdf5(f)
        io_test.assert_equal_data(data_imported, data)
        io_test.assert_event_handler_example_works(data_imported)




# if __name__ == "__main__":
#
#     #testtens=create_test_symmetric_tensor()
#     #testtens=create_test_random_diagonal_tensor()
#     #testtens=SU2_sym_test_tensor()
#     #testtens = U1_sym_test_tensor()
#
#     #testtens=create_test_zero_mask()
#     #testtens=create_test_mask()
#     #testtens=create_test_charged_tensor()
#     #testtens=zero_su2_tensor()
#
#
#     #print(testtens)
#
#     #print(testtens.ascii_diagram)
#     #print(testtens.to_numpy())
#     #print(testtens.shape)
#     #print(testtens.data.blocks)
#     #
#     #export_tensor(testtens, 'testfile_io.hdf5')
#     #testimport = import_tensor('/space/ge36xeh/PycharmProjects/pythonProject/cytens/tests/pytest/tools/hdf5_io/testfile_io.hdf5')
#     #
#     #print(testimport)
#     #tmp = pathlib.Path('./tmp')
#     import_export_tests('/space/ge36xeh/PycharmProjects/pythonProject/cytens/tests/pytest/tools/hdf5_io/')
#
