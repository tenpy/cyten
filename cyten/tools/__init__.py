"""TODO write docs"""
# Copyright (C) TeNPy Developers, Apache license

from . import math, misc, string, mappings, cost_polynomials
from .math import speigs, speigsh
from .misc import (
    duplicate_entries, is_iterable, to_iterable, to_valid_idx, as_immutable_array, permutation_as_swaps,
    argsort, combine_constraints, inverse_permutation, rank_data, np_argsort, make_stride,
    make_grid, list_to_dict_list, find_row_differences, iter_common_noncommon_sorted,
    iter_common_sorted, iter_common_sorted_arrays, iter_common_noncommon_sorted_arrays,
    find_subclass
)
from .string import format_like_list
from .mappings import SparseMapping
from .cost_polynomials import BigOPolynomial
