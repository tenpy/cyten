"""To be used in the `-m` argument of benchmark.py."""
# Copyright (C) TeNPy Developers, Apache license

import numpy as np
import svd_cyten
from setup_utils import dense_as_matrix


def setup_benchmark(**kwargs):
    T = svd_cyten.setup_benchmark(**kwargs)
    return dense_as_matrix(T)


def benchmark(data):
    np.linalg.svd(data, full_matrices=False)
