"""TODO write docs"""
# Copyright (C) TeNPy Developers, Apache license

from . import cost_polynomials, math, misc, string
from .cost_polynomials import BigOPolynomial
from .math import speigs, speigsh
from .misc import (
    argsort,
    as_immutable_array,
    combine_constraints,
    combine_permutations,
    duplicate_entries,
    find_row_differences,
    find_subclass,
    inverse_permutation,
    is_iterable,
    is_permutation,
    iter_common_noncommon_sorted,
    iter_common_noncommon_sorted_arrays,
    iter_common_sorted,
    iter_common_sorted_arrays,
    list_to_dict_list,
    make_grid,
    make_stride,
    np_argsort,
    permutation_as_swaps,
    rank_data,
    to_iterable,
    to_valid_idx,
)
from .string import format_like_list


# auto-maintained by scripts/generate_reexport_all.py; do not edit by hand
__all__ = [
    'BigOPolynomial',
    'argsort',
    'as_immutable_array',
    'combine_constraints',
    'combine_permutations',
    'cost_polynomials',
    'duplicate_entries',
    'find_row_differences',
    'find_subclass',
    'format_like_list',
    'inverse_permutation',
    'is_iterable',
    'is_permutation',
    'iter_common_noncommon_sorted',
    'iter_common_noncommon_sorted_arrays',
    'iter_common_sorted',
    'iter_common_sorted_arrays',
    'list_to_dict_list',
    'make_grid',
    'make_stride',
    'math',
    'misc',
    'np_argsort',
    'permutation_as_swaps',
    'rank_data',
    'speigs',
    'speigsh',
    'string',
    'to_iterable',
    'to_valid_idx',
]
