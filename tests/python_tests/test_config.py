import getpass

import numpy as np
import pytest

import cyten as ct
from cyten.config import CytenConfig, get_config, get_option, restore_defaults, set_options, temporary_options

FAKE_USER_CONFIG = """
print_linewidth: 93
maxlines_spaces: 11
maxlines_tensors: 22
default_block_backend: gpu
"""

FAKE_LOCAL_CONFIG = """
print_linewidth: 92
print_indent: 3
maxlines_spaces: 10
check_fusion: False
"""


@pytest.fixture
def setup_fake_config(tmp_path_factory, monkeypatch):
    with monkeypatch.context() as m:
        user_config_path = tmp_path_factory.mktemp('home').joinpath('.cytenconfig.yaml')
        user_config_path.write_text(FAKE_USER_CONFIG)
        m.setenv('CYTEN_CONFIG_FILE', str(user_config_path))

        cwd = tmp_path_factory.mktemp('wd')
        cwd.joinpath('.cytenconfig.yaml').write_text(FAKE_LOCAL_CONFIG)
        m.chdir(cwd)

        m.setenv('CYTEN_PRINT_LINEWIDTH', '91')
        m.setenv('CYTEN_PRINT_INDENT', '2')
        m.setenv('CYTEN_MAXLINES_TENSORS', '21')

        restore_defaults()

        yield  # <- this runs the actual test
    # exiting the context restores the old cwd and env

    # cleanup
    restore_defaults()


def test_config_precedence(setup_fake_config):
    """

    ========================  ======  ======  ======  ======  ======  ======
    option                    temp    rntm    env     local   user    deflt
    ========================  ======  ======  ======  ======  ======  ======
    print_linewidth           80      90      91      92      93      100
    ------------------------  ------  ------  ------  ------  ------  ------
    print_indent                      1       2       3               4
    ------------------------  ------  ------  ------  ------  ------  ------
    maxlines_spaces                                   10      11      15
    ------------------------  ------  ------  ------  ------  ------  ------
    maxlines_tensors                          21              22      30
    ------------------------  ------  ------  ------  ------  ------  ------
    check_fusion              True                    False           True
    ------------------------  ------  ------  ------  ------  ------  ------
    default_tensor_backend                                            abel
    ------------------------  ------  ------  ------  ------  ------  ------
    default_block_backend                                     gpu     numpy
    ------------------------  ------  ------  ------  ------  ------  ------
    coupling_cutoff                                                   1e-13
    ========================  ======  ======  ======  ======  ======  ======

    """

    assert get_option('print_linewidth') == 91
    assert get_option('print_indent') == 2
    assert get_option('maxlines_spaces') == 10
    assert get_option('maxlines_tensors') == 21
    assert get_option('check_fusion') is False
    assert get_option('default_tensor_backend') == 'abelian'
    assert get_option('default_block_backend') == 'gpu'
    assert get_option('coupling_cutoff') == 1e-13
    # now, doing an invalid fusion should not raise
    invalid_fusion = ct.u1_symmetry.r_symbol(np.array([1]), np.array([1]), np.array([-1]))

    set_options(print_linewidth=90, print_indent=1)

    assert get_option('print_linewidth') == 90
    assert get_option('print_indent') == 1
    assert get_option('maxlines_spaces') == 10
    assert get_option('maxlines_tensors') == 21
    assert get_option('check_fusion') is False
    assert get_option('default_tensor_backend') == 'abelian'
    assert get_option('default_block_backend') == 'gpu'
    assert get_option('coupling_cutoff') == 1e-13
    # now, doing an invalid fusion should not raise
    invalid_fusion = ct.u1_symmetry.r_symbol(np.array([1]), np.array([1]), np.array([-1]))

    with temporary_options(print_linewidth=80, check_fusion=True):
        assert get_option('print_linewidth') == 80
        assert get_option('print_indent') == 1
        assert get_option('maxlines_spaces') == 10
        assert get_option('maxlines_tensors') == 21
        assert get_option('check_fusion') is True
        assert get_option('default_tensor_backend') == 'abelian'
        assert get_option('default_block_backend') == 'gpu'
        assert get_option('coupling_cutoff') == 1e-13
        # now, doing an invalid fusion should raise
        with pytest.raises(ct.SymmetryError, match='not consistent with fusion rules'):
            invalid_fusion = ct.u1_symmetry.r_symbol(np.array([1]), np.array([1]), np.array([-1]))

    assert get_option('print_linewidth') == 90
    assert get_option('print_indent') == 1
    assert get_option('maxlines_spaces') == 10
    assert get_option('maxlines_tensors') == 21
    assert get_option('check_fusion') is False
    assert get_option('default_tensor_backend') == 'abelian'
    assert get_option('default_block_backend') == 'gpu'
    assert get_option('coupling_cutoff') == 1e-13
    # now, doing an invalid fusion should not raise
    invalid_fusion = ct.u1_symmetry.r_symbol(np.array([1]), np.array([1]), np.array([-1]))


def test_options_consistency():
    config = CytenConfig()  # default config
    for key in get_config().all_option_keys():
        default_val = get_option(key)
        config.set_option(key, default_val)  # make sure that the default is valid


def test_su_n_data_defaults():
    """Default SU(N) data location follows the external generator's convention."""
    assert get_option('su_n_data_path') == f'/home/{getpass.getuser()}/.tenpy/su_n_symmetry_data'
    assert get_option('su_n_data_filename_base') == 'su_n_clebsch_gordan_data'


def test_su_n_data_env_override(monkeypatch):
    with monkeypatch.context() as m:
        m.setenv('CYTEN_SU_N_DATA_PATH', '/some/where/else')
        m.setenv('CYTEN_SU_N_DATA_FILENAME_BASE', 'my_base')
        restore_defaults()
        assert get_option('su_n_data_path') == '/some/where/else'
        assert get_option('su_n_data_filename_base') == 'my_base'
    restore_defaults()


def test_su_n_data_set_option_types():
    config = CytenConfig()
    config.set_option('su_n_data_path', r'C:\Users\x\su_n')  # free-form, no allow-list
    assert config.get_option('su_n_data_path') == r'C:\Users\x\su_n'
    with pytest.raises(TypeError):
        config.set_option('su_n_data_path', 5)
    with pytest.raises(ValueError):
        config.set_option('su_n_data_filename_base', '')


def test_su_n_data_file_path():
    from cyten.symmetries import su_n_data_file_path, su_n_data_filename

    assert su_n_data_filename(3, 'CG', 7) == 'su_n_clebsch_gordan_data_N3_CG_hweight7.hdf5'
    assert su_n_data_filename(3, 'F', 4, filename_base='foo') == 'foo_N3_F_hweight4.hdf5'
    assert su_n_data_filename(3, 'r', 4) == 'su_n_clebsch_gordan_data_N3_R_hweight4.hdf5'
    with pytest.raises(ValueError):
        su_n_data_filename(3, 'X', 4)

    assert su_n_data_file_path(3, 'CG', 7, path='/a/b') == '/a/b/su_n_clebsch_gordan_data_N3_CG_hweight7.hdf5'
    assert su_n_data_file_path(3, 'CG', 7, path='/a/b/') == '/a/b/su_n_clebsch_gordan_data_N3_CG_hweight7.hdf5'
    with temporary_options(su_n_data_path='/x', su_n_data_filename_base='b'):
        assert su_n_data_file_path(3, 'R', 4) == '/x/b_N3_R_hweight4.hdf5'
