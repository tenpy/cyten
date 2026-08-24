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
    """Returns ``a, b, axes`` for timing of ``ct.tdot(a, b, *axes)``.

    Constructed such that ``legs`` legs are contracted, with
        a.num_legs = legs + legs
        b.num_legs = legs + legs
    If `select_frac` < 1, keep only that fraction of stored blocks.
    """
    symmetry = symmetry_from_mod_q(mod_q)
    backend = get_backend_from_kwargs(symmetry, **kwargs)
    legs_contr = [random_elementary_space(symmetry, size, sectors) for _ in range(legs)]
    legs_a_open = [random_elementary_space(symmetry, size, sectors) for _ in range(legs)]
    legs_b_open = [random_elementary_space(symmetry, size, sectors) for _ in range(legs)]

    # labels follow [*codomain, *reversed(domain)]
    labs_a_cod = [f'a{i}' for i in range(legs)]
    labs_contr = [f'c{i}' for i in range(legs)]
    labs_b_dom_flat = [f'b{i}' for i in range(legs - 1, -1, -1)]
    labs_a = labs_a_cod + list(reversed(labs_contr))
    labs_b = labs_contr + labs_b_dom_flat

    a = random_tensor(
        codomain=legs_a_open,
        domain=legs_contr,
        labels=labs_a,
        backend=backend,
        dtype=dtype,
        select_frac=select_frac,
    )
    b = random_tensor(
        codomain=legs_contr,
        domain=legs_b_open,
        labels=labs_b,
        backend=backend,
        dtype=dtype,
        select_frac=select_frac,
    )
    axes = [labs_contr, labs_contr]
    return a, b, axes


def benchmark(data):
    a, b, axes = data
    ct.tdot(a, b, axes[0], axes[1])
