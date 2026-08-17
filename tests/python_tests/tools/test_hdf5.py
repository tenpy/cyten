"""Test output to and import from hdf5."""

import os
import warnings

import io_test
import numpy as np
import pytest

import cyten
from cyten.tools import hdf5_io

h5py = pytest.importorskip('h5py')

datadir_hdf5 = [f for f in io_test.datadir_files if f.endswith('.hdf5')]


def export_to_datadir():
    filename = io_test.get_datadir_filename('exported_from_tenpy_{0}.hdf5')
    data = io_test.gen_example_data()
    with warnings.catch_warnings(record=True) as caught:
        warnings.filterwarnings('ignore', category=UserWarning)
        with h5py.File(filename, 'w') as f:
            hdf5_io.save_to_hdf5(f, data)
    for w in caught:
        msg = str(w.message)
        expected = 'without explicit HDF5 format' in msg
        if expected:
            expected = any(
                t in msg
                for t in [
                    'io_test.DummyClass',
                    'tenpy.tools.events.EventHandler',
                    'tenpy.tools.events.Listener',
                    'method',
                ]
            )
        if not expected:
            warnings.showwarning(w.message, w.category, w.filename, w.lineno, w.file, w.line)


@pytest.mark.filterwarnings(r'ignore:Hdf5Saver.* object of type.*:UserWarning')
def test_hdf5_block_and_scalar_io(block_backend, tmp_path):
    """Roundtrip C++ NumpyBlockBackend Block and Scalar via HDF5."""

    be = cyten.get_backend('no_symmetry', block_backend).block_backend
    block = be.as_block(np.arange(6, dtype=np.float64).reshape(2, 3))
    scalar = be.as_scalar(3.5)

    filename = tmp_path / 'block_scalar.hdf5'
    with h5py.File(str(filename), 'w') as f:
        hdf5_io.save_to_hdf5(f, {'block': block, 'scalar': scalar})
    with h5py.File(str(filename), 'r') as f:
        loaded = hdf5_io.load_from_hdf5(f)

    block_loaded = loaded['block']
    scalar_loaded = loaded['scalar']
    np.testing.assert_array_equal(block_loaded.to_numpy(), block.to_numpy())
    assert block_loaded.shape == block.shape
    assert scalar_loaded.as_float64() == pytest.approx(3.5)
    assert scalar_loaded.dtype == scalar.dtype


@pytest.mark.filterwarnings(r'ignore:Hdf5Saver.* object of type.*:UserWarning')
def test_hdf5_export_import(tmp_path):
    """Try subsequent export and import to pickle."""
    data = io_test.gen_example_data()
    filename = tmp_path / 'test.hdf5'
    with h5py.File(str(filename), 'w') as f:
        hdf5_io.save_to_hdf5(f, data)
    with h5py.File(str(filename), 'r') as f:
        data_imported = hdf5_io.load_from_hdf5(f)
    io_test.assert_equal_data(data_imported, data)


@pytest.mark.parametrize('fn', datadir_hdf5)
@pytest.mark.filterwarnings('ignore::FutureWarning')
def test_import_from_datadir(fn):
    print('import ', fn)
    filename = os.path.join(io_test.datadir, fn)
    with h5py.File(filename, 'r') as f:
        data = hdf5_io.load_from_hdf5(f)
    if 'version' not in data:
        raise ValueError(f'Version not found in data: {data.keys()}')
    data_expected = io_test.gen_example_data(data['version'])
    io_test.assert_equal_data(data, data_expected)
