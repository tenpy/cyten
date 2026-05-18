"""Configuration for the cyten library.

.. _config:

Config
------

To configure the behavior of cyten at runtime, we offer the following sources, in order of
precedence::

    1. A context manager :class:`temporary_options` for scoped overrides

    2. Explicitly setting options at runtime using :func:`set_options`

    3. A YAML file in the working directory with name either ``.cytenconfig.yaml``

    4. A YAML file at a fixed location. The location is read from the envvar ``CYTEN_CONFIG_FILE``
       with default ``~/.cytenconfig.yaml``

    5. Read from envvars of the form ``'CYTEN_' + upper(option_name)``, e.g. ``CYTEN_CHECK_FUSION``
       for the ``'check_fusion'`` option.

    6. Hard-coded default values. There is a readonly view at :data:`cyten.config.defaults`.

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
    ========================  ===========  ======================================================================

"""

# Copyright (C) TeNPy Developers, Apache license
from ._config import defaults, get_option, restore_defaults, set_options, temporary_options
