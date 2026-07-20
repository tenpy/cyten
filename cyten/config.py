"""Global config options for cyten.

Implemented in C++ (``include/cyten/config.h``, ``src/config.cpp``).
"""
# Copyright (C) TeNPy Developers, Apache license

from ._core import (
    get_config,
    get_option,
    restore_defaults,
    set_option,
    set_options,
    temporary_options,
)

__all__ = [
    'get_config',
    'get_option',
    'restore_defaults',
    'set_option',
    'set_options',
    'temporary_options',
]
