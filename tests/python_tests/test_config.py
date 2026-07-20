import numpy as np
import pytest

import cyten as ct
from cyten.config._config import OPTIONS, init_config

FAKE_USER_CONFIG = """
print_linewidth: 93
maxlines_spaces: 11
maxlines_tensors: 21
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

        m.setenv('CYTEN_PRINT_LINEWIDTH', '94')
        m.setenv('CYTEN_PRINT_INDENT', '4')
        m.setenv('CYTEN_MAXLINES_TENSOR', '22')
        m.setenv('CYTEN_CHECK_FUSION', 'False')
        m.setenv('CYTEN_DEFAULT_BLOCK_BACKEND', 'gpu')

        init_config()

        yield  # <- this runs the actual test
    # exiting the context restores the old cwd and env

    # cleanup
    init_config(reinit=True)


def test_config_precedence(setup_fake_config):
    """

    ========================  ======  ======  ======  ======  ======  ======
    option                    temp    rntm    local   user    env     deflt
    ========================  ======  ======  ======  ======  ======  ======
    print_linewidth           80      91      92      93      94      100
    ------------------------  ------  ------  ------  ------  ------  ------
    print_indent                      1       3               4       2
    ------------------------  ------  ------  ------  ------  ------  ------
    maxlines_spaces                           10      11              15
    ------------------------  ------  ------  ------  ------  ------  ------
    maxlines_tensors                                  21      22      30
    ------------------------  ------  ------  ------  ------  ------  ------
    check_fusion              True            False                   True
    ------------------------  ------  ------  ------  ------  ------  ------
    default_tensor_backend                                            abel
    ------------------------  ------  ------  ------  ------  ------  ------
    default_block_backend                                     gpu     numpy
    ========================  ======  ======  ======  ======  ======  ======

    """

    assert ct.get_option('print_linewidth') == 92
    assert ct.get_option('print_indent') == 3
    assert ct.get_option('maxlines_spaces') == 10
    assert ct.get_option('maxlines_tensors') == 21
    assert ct.get_option('check_fusion') is False
    assert ct.get_option('default_tensor_backend') == 'abelian'
    assert ct.get_option('default_block_backend') == 'gpu'
    # now, doing an invalid fusion should not raise
    invalid_fusion = ct.u1_symmetry.r_symbol(np.array([1]), np.array([1]), np.array([-1]))

    ct.set_options(print_linewidth=91, print_indent=1)

    assert ct.get_option('print_linewidth') == 91
    assert ct.get_option('print_indent') == 1
    assert ct.get_option('maxlines_spaces') == 10
    assert ct.get_option('maxlines_tensors') == 21
    assert ct.get_option('check_fusion') is False
    assert ct.get_option('default_tensor_backend') == 'abelian'
    assert ct.get_option('default_block_backend') == 'gpu'
    # now, doing an invalid fusion should not raise
    invalid_fusion = ct.u1_symmetry.r_symbol(np.array([1]), np.array([1]), np.array([-1]))

    with ct.temporary_options(print_linewidth=80, check_fusion=True):
        assert ct.get_option('print_linewidth') == 80
        assert ct.get_option('print_indent') == 1
        assert ct.get_option('maxlines_spaces') == 10
        assert ct.get_option('maxlines_tensors') == 21
        assert ct.get_option('check_fusion') is True
        assert ct.get_option('default_tensor_backend') == 'abelian'
        assert ct.get_option('default_block_backend') == 'gpu'
        # now, doing an invalid fusion should raise
        with pytest.raises(ct.SymmetryError, match='not consistent with fusion rules'):
            invalid_fusion = ct.u1_symmetry.r_symbol(np.array([1]), np.array([1]), np.array([-1]))

    assert ct.get_option('print_linewidth') == 91
    assert ct.get_option('print_indent') == 1
    assert ct.get_option('maxlines_spaces') == 10
    assert ct.get_option('maxlines_tensors') == 21
    assert ct.get_option('check_fusion') is False
    assert ct.get_option('default_tensor_backend') == 'abelian'
    assert ct.get_option('default_block_backend') == 'gpu'
    # now, doing an invalid fusion should not raise
    invalid_fusion = ct.u1_symmetry.r_symbol(np.array([1]), np.array([1]), np.array([-1]))


def test_options_consistency():
    for key, val in OPTIONS.items():
        assert val.name == key
        assert val.env_var == f'CYTEN_{key.upper()}'
        val.coerce(val.default)  # makes sure that the default is valid
