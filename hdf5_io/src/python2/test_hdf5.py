"""Test output to and import from hdf5."""

import os
import pytest
import warnings
import tempfile
import hdf5_io  # the local version
import numpy as np
import h5py

datadir = os.path.join(os.path.dirname(__file__), 'data')
datadir_files = []
if os.path.isdir(datadir):
    datadir_files = os.listdir(datadir)
datadir_hdf5 = [f for f in datadir_files if f.endswith('.hdf5')]


def dummy_function():
    pass


class DummyClass(hdf5_io.Hdf5Exportable):
    def __init__(self):
        self.data = []

    def dummy_method(self, obj):
        self.data.append(obj)


def gen_example_data():
    data = {
        'None': None,
        'scalars': [0, np.int64(1), 2., np.float64(3.), 4.j, 'five', True, 2**70,
                    b'a byte string'],
        'arrays': [np.array([6, 66]), np.array([]), np.zeros([]), np.array([True, False])],
        'masked_arrays': [
            np.ma.masked_equal([[1, -1, 3], [-1, 5, 6]], -1),
            np.ma.masked_array([[1, -1, 3], [-1, 5, 6]], # hard: fill_value in data
                               mask=[[True, True, False], [False]*3],
                               fill_value=-1)

        ],
        'iterables': [[], [11, 12],
                      tuple([]),
                      tuple([1, 2, 3]),
                      set([]), set([1, 2, 3])],
        'recursive': [0, None, 2, [3, None, 5]],
        'dict_complicated': {
            0: 1,
            'asdf': 2,
            (1, 2): '3'
        },
        'exportable': hdf5_io.Hdf5Exportable(),
        'range': xrange(2, 8, 3),
        'dtypes': [np.dtype("int64"),
                   np.dtype([('a', np.int32, 8), ('b', np.float64, 5)])],
    }
    data['recursive'][3][1] = data['recursive'][1] = data['recursive']
    data['exportable'].some_attr = "something"
    return data


def assert_equal_data(data_imported, data_expected, max_recursion_depth=10):
    """Check that the imported data is as expected."""
    assert (isinstance(data_imported, type(data_expected))
            or (type(data_expected) is bytes and isinstance(data_imported, str))
            # special case because str is bytes in python2, so no way to distinguish
            )
    if hasattr(data_expected, 'test_sanity'):
        data_imported.test_sanity()
    if isinstance(data_expected, dict):
        assert set(data_imported.keys()) == set(data_expected.keys())
        if max_recursion_depth > 0:
            for ki in data_imported.keys():
                assert_equal_data(data_imported[ki], data_expected[ki], max_recursion_depth - 1)
    elif isinstance(data_expected, (list, tuple)):
        assert len(data_imported) == len(data_expected)
        if max_recursion_depth > 0:
            for vi, ve in zip(data_imported, data_expected):
                assert_equal_data(vi, ve, max_recursion_depth - 1)
    elif isinstance(data_expected, np.ndarray):
        np.testing.assert_array_equal(data_imported, data_expected)
    elif isinstance(data_expected, (int, float, np.int64, np.float64, bool)):
        assert data_imported == data_expected
    elif isinstance(data_expected, xrange):
        assert tuple(data_imported) == tuple(data_expected)


def export_to_datadir():
    filename = "data/exported_python2.hdf5"
    data = gen_example_data()
    with h5py.File(filename, 'w') as f:
        hdf5_io.save_to_hdf5(f, data)


def test_hdf5_export_import():
    """Try subsequent export and import to pickle."""
    data = gen_example_data()
    dc = DummyClass()
    data.update({
        'global_function': dummy_function,
        'global_class': DummyClass,
        'instance': dc,
        # 'method': dc.dummy_method,  # python 2.7 doesn't support pickling methods
        'excluded_from_load': np.arange(3.),
    })
    data_with_ignore = data.copy()
    data_with_ignore['ignore_save'] = hdf5_io.Hdf5Ignored()
    with tempfile.TemporaryFile() as tf:
        with h5py.File(tf, 'w') as f:
            hdf5_io.save_to_hdf5(f, data_with_ignore)
            f['ignore_load'] = "ignore_during_load"
            f['ignore_load'].attrs[hdf5_io.ATTR_TYPE] = hdf5_io.REPR_IGNORED
        tf.seek(0)  # reset pointer to beginning of file for reading
        with h5py.File(tf, 'r') as f:
            data_imported = hdf5_io.load_from_hdf5(f,
                                                   ignore_unknown=False,
                                                   exclude=['/excluded_from_load'])
    # data is a dict with simple keys
    # so 'ignore_load' should be loaded as Hdf5Ignored instance
    assert isinstance(data_imported['ignore_load'], hdf5_io.Hdf5Ignored)
    del data_imported['ignore_load']

    assert isinstance(data_imported['excluded_from_load'], hdf5_io.Hdf5Ignored)
    data['excluded_from_load'] = hdf5_io.Hdf5Ignored('/excluded_from_load')

    assert_equal_data(data_imported, data)


@pytest.mark.parametrize('fn', datadir_hdf5)
def test_import_from_datadir(fn):
    print("import ", fn)
    filename = os.path.join(datadir, fn)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        with h5py.File(filename, 'r') as f:
            data = hdf5_io.load_from_hdf5(f)
    data_expected = gen_example_data()
    assert_equal_data(data, data_expected)


if __name__ == "__main__":
    export_to_datadir()
