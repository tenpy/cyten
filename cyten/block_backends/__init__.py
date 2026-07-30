"""Block-backends implement matrix and array algebra on dense blocks, similar to e.g. numpy"""
# Copyright (C) TeNPy Developers, Apache license

# Note: order matters to avoid circular imports!
# pyright: ignore
from .._core import Dtype  # noqa
from .._core import BlockBackend, NumpyBlockBackend, TorchBlockBackend, ArrayApiBlockBackend
from . import dtypes

Block = BlockBackend.BlockCls
Scalar = BlockBackend.Scalar
