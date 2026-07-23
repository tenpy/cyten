"""Block-backends implement matrix and array algebra on dense blocks, similar to e.g. numpy"""
# Copyright (C) TeNPy Developers, Apache license

# Note: order matters to avoid circular imports!
# pyright: ignore
from .._core import Dtype  # noqa
from .._core import BlockBackend, NumpyBlockBackend
from . import dtypes

Block = BlockBackend.BlockCls
Scalar = BlockBackend.Scalar

# from ._block_backend import BlockBackend
from .array_api import ArrayApiBlockBackend
from .torch import TorchBlockBackend
