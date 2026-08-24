"""To be used in the `-m` argument of benchmark.py."""
# Copyright (C) TeNPy Developers, Apache license

import numpy as np
import tensordot_cyten
from setup_utils import leg_indices, to_numpy


def setup_benchmark(**kwargs):
    a, b, axes = tensordot_cyten.setup_benchmark(**kwargs)
    axes_a, axes_b = axes
    axes_a = leg_indices(a, axes_a)
    axes_b = leg_indices(b, axes_b)
    non_axes_a = [i for i in range(a.num_legs) if i not in axes_a]
    non_axes_b = [i for i in range(b.num_legs) if i not in axes_b]
    return to_numpy(a), to_numpy(b), ((non_axes_a, axes_a), (axes_b, non_axes_b))


def combine_legs(a, axes):
    axes = list(axes)
    pipe = [[a.shape[i] for i in comb] for comb in axes]
    transp = []
    newshape = []
    for ax in axes:
        transp.extend(ax)
        newshape.append(np.prod([a.shape[i] for i in ax]))
    a = np.transpose(a, transp)
    a = np.reshape(a, newshape)
    return np.ascontiguousarray(a).copy(), pipe


def benchmark(data):
    a, b, axes = data
    axes_a, axes_b = axes
    combine_legs(a, axes_a)
    combine_legs(b, axes_b)
