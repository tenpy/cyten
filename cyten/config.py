"""Global config options for cyten.

Implemented in C++ (``include/cyten/config.h``, ``src/config.cpp``).


Config
------

To configure the behavior of cyten at runtime, we offer the following sources, in order of
precedence::

    1. A Python context manager :class:`temporary_options` for scoped overrides

    2. Explicitly setting options at runtime using :func:`set_options`

    3. Read from envvars of the form ``'CYTEN_' + upper(option_name)``, e.g. ``CYTEN_CHECK_FUSION``
       for the ``'check_fusion'`` option.

    4. A YAML file in the working directory with name ``.cytenconfig.yaml``

    5. A YAML file at a fixed location. The location is read from the envvar ``CYTEN_CONFIG_FILE``
       with default ``~/.cytenconfig.yaml``

    6. Hard-coded default values, as returned by :class:`CytenConfig`.

The following options are available::

    ========================  ===========  ======================================================================
    name                      default      meaning
    ========================  ===========  ======================================================================
    print_linewidth           100          Maximum linewidth for printing tensors, spaces, ...
    ------------------------  -----------  ----------------------------------------------------------------------
    print_indent              2            Number of spaces for indenting when printing
    ------------------------  -----------  ----------------------------------------------------------------------
    maxlines_spaces           15           Maximum number of lines for printing spaces
    ------------------------  -----------  ----------------------------------------------------------------------
    maxlines_tensors          30           Maximum number of lines for printing tensors
    ------------------------  -----------  ----------------------------------------------------------------------
    check_fusion              True         If input checks for correct fusion should be enabled.
                                           Disabling can improve performance, but make errors more cryptic.
    ------------------------  -----------  ----------------------------------------------------------------------
    default_tensor_backend    'abelian'    Tensor-backend to be used by default. See :func:`cyten.get_backend`.
    ------------------------  -----------  ----------------------------------------------------------------------
    default_block_backend     'numpy'      Block-backend to be used by default. See :func:`cyten.get_backend`.
    ------------------------  -----------  ----------------------------------------------------------------------
    fusion_tree_eps           5e-14        Threshold for discarding near-zero fusion-tree blocks after
                                           topological moves (braids, bends, twists).
    ------------------------  -----------  ----------------------------------------------------------------------
    su_n_data_path            (see below)  Directory holding the SU(N) Clebsch-Gordan / F-symbol / R-symbol
                                           HDF5 files.
    ------------------------  -----------  ----------------------------------------------------------------------
    su_n_data_filename_base   (see below)  Stem of the SU(N) data file names.
    ------------------------  -----------  ----------------------------------------------------------------------
    coupling_cutoff           1e-13        Default singular-value cutoff when factorizing a Coupling
                                           (``from_dense_block`` / ``from_tensor``).
    ========================  ===========  ======================================================================

The default for ``su_n_data_path`` is the *literal* POSIX path
``/home/<login-name>/.tenpy/su_n_symmetry_data`` on all platforms (login name resolved from the
``LOGNAME``/``USER``/``LNAME``/``USERNAME`` environment variables, in that order), matching the
layout written by the external ``clebsch_gordan_coefficients`` data generator. On Windows this path
will usually not exist by default; override it via the ``su_n_data_path`` config option, the
``CYTEN_SU_N_DATA_PATH`` envvar, or ``.cytenconfig.yaml``.

The default for ``su_n_data_filename_base`` is ``'su_n_clebsch_gordan_data'``. Given ``N``, a data
kind (``'CG'``, ``'F'`` or ``'R'``) and a highest weight ``H``, the expected file name is
``'{su_n_data_filename_base}_N{N}_{kind}_hweight{H}.hdf5'`` inside ``su_n_data_path``. See
:func:`cyten.symmetries.su_n_data_file_path` to resolve this pattern directly.

"""
# Copyright (C) TeNPy Developers, Apache license

from ._core import (
    CytenConfig,
    get_config,
    get_option,
    restore_defaults,
    set_option,
    set_options,
    temporary_options,
)

__all__ = [
    'CytenConfig',
    'get_config',
    'get_option',
    'restore_defaults',
    'set_option',
    'set_options',
    'temporary_options',
]
