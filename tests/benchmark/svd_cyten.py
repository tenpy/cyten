"""To be used in the `-m` argument of benchmark.py."""
# Copyright (C) TeNPy Developers, Apache license

import numpy as np
from setup_utils import (
    get_backend_from_kwargs,
    random_elementary_space,
    random_tensor,
    symmetry_from_mod_q,
)

import cyten as ct


def setup_benchmark(mod_q=[1], sectors=3, size=20, legs=2, select_frac=1.0, dtype=np.float64, **kwargs):
    """Random map with ``legs`` legs in (co)domain, for timing ``ct.svd``."""
    symmetry = symmetry_from_mod_q(mod_q)
    backend = get_backend_from_kwargs(symmetry, **kwargs)
    codomain = [random_elementary_space(symmetry, size, sectors) for _ in range(legs)]
    domain = [random_elementary_space(symmetry, size, sectors) for _ in range(legs)]
    labels = [f'c{i}' for i in range(legs)] + [f'd{i}' for i in range(legs - 1, -1, -1)]
    T = random_tensor(
        codomain=codomain,
        domain=domain,
        labels=labels,
        backend=backend,
        dtype=dtype,
        select_frac=select_frac,
    )
    return T


def benchmark(data):
    ct.svd(data)
