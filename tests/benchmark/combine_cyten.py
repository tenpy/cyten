"""To be used in the `-m` argument of benchmark.py."""
# Copyright (C) TeNPy Developers, Apache license

import tensordot_cyten
from setup_utils import leg_indices

import cyten as ct


def setup_benchmark(*args, **kwargs):
    a, b, axes = tensordot_cyten.setup_benchmark(*args, **kwargs)
    axes_a, axes_b = axes
    axes_a = leg_indices(a, axes_a)
    axes_b = leg_indices(b, axes_b)
    non_axes_a = [i for i in range(a.num_legs) if i not in axes_a]
    non_axes_b = [i for i in range(b.num_legs) if i not in axes_b]
    return a, b, ((non_axes_a, axes_a), (axes_b, non_axes_b))


def benchmark(data):
    a, b, axes = data
    axes_a, axes_b = axes
    ct.combine_legs(a, axes_a[0], axes_a[1])
    ct.combine_legs(b, axes_b[0], axes_b[1])
