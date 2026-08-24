"""To be used in the `-m` argument of benchmark.py."""
# Copyright (C) TeNPy Developers, Apache license

import numpy as np
import tensordot_cyten
from setup_utils import leg_indices, to_numpy


def setup_benchmark(**kwargs):
    a, b, axes = tensordot_cyten.setup_benchmark(**kwargs)
    axes_a, axes_b = axes
    return to_numpy(a), to_numpy(b), (leg_indices(a, axes_a), leg_indices(b, axes_b))


def benchmark(data):
    a, b, axes = data
    np.tensordot(a, b, axes)
