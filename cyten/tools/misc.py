"""Miscellaneous tools, somewhat random mix yet often helpful."""
# Copyright (C) TeNPy Developers, Apache license

import operator
import numpy as np
import os.path
import warnings
from typing import TypeVar, Sequence, Set


__all__ = [
    'UNSPECIFIED', 'duplicate_entries', 'to_iterable', 'as_immutable_array',
    'permutation_as_swaps',
    'argsort', 'combine_constraints', 'inverse_permutation', 'list_to_dict_list',
    'find_subclass',
    'rank_data',
    'np_argsort', 'make_stride', 'find_row_differences', 'unstridify',
    'iter_common_noncommon_sorted', 'iter_common_noncommon_sorted_arrays', 'iter_common_sorted',
    'iter_common_sorted_arrays'
]


UNSPECIFIED = object()  # sentinel, also used elsewhere
_T = TypeVar('_T')  # used in typing some functions
_MAX_INT = np.iinfo(int).max


def duplicate_entries(seq: Sequence[_T], ignore: Sequence[_T] = []) -> Set[_T]:
    return set(ele for idx, ele in enumerate(seq) if ele in seq[idx + 1:] and ele not in ignore)


def to_iterable(a):
    """If `a` is a not iterable or a string, return ``[a]``, else return ``a``."""
    if type(a) == str:
        return [a]
    try:
        iter(a)
    except TypeError:
        return [a]
    else:
        return a


def as_immutable_array(a, dtype=None):
    """Like :func:`numpy.asarray`, but also makes the resulting array immutable."""
    a = np.asarray(a, dtype=dtype)
    a.setflags(write=False)
    return a


def permutation_as_swaps(initial_perm: list, final_perm: list) -> list:
    """Given an initial and final permutation of the same numbers, return a list `swaps`
    of indices such that exchanging the entries of the initial permutation as
    `initial_perm[swaps[i]], initial_perm[swaps[i]+1] = initial_perm[swaps[i]+1], 
    initial_perm[swaps[i]]` leads to the final permutation. The swaps must be applied
    starting from `swaps[0]`.
    
    Consistency of the input is not checked.
    """
    swaps = []
    while final_perm != initial_perm:
        for i in range(len(final_perm)):
            if final_perm[i] != initial_perm[i]:
                ind = initial_perm.index(final_perm[i])
                initial_perm[ind-1:ind+1] = initial_perm[ind-1:ind+1][::-1]
                swaps.append(ind - 1)
                break
    return swaps
    

# TODO remove in favor of backend.block_argsort?
def argsort(a, sort=None, **kwargs):
    """wrapper around np.argsort to allow sorting ascending/descending and by magnitude.

    Parameters
    ----------
    a : array_like
        The array to sort.
    sort : ``'m>', 'm<', '>', '<', None``
        Specify how the arguments should be sorted.

        ==================== =============================
        `sort`               order
        ==================== =============================
        ``'m>', 'LM'``       Largest magnitude first
        -------------------- -----------------------------
        ``'m<', 'SM'``       Smallest magnitude first
        -------------------- -----------------------------
        ``'>', 'LR', 'LA'``  Largest real part first
        -------------------- -----------------------------
        ``'<', 'SR', 'SA'``  Smallest real part first
        -------------------- -----------------------------
        ``'LI'``             Largest imaginary part first
        -------------------- -----------------------------
        ``'SI'``             Smallest imaginary part first
        -------------------- -----------------------------
        ``None``             numpy default: same as '<'
        ==================== =============================

    **kwargs :
        Further keyword arguments given directly to :func:`numpy.argsort`.

    Returns
    -------
    index_array : ndarray, int
        Same shape as `a`, such that ``a[index_array]`` is sorted in the specified way.
    """
    if sort is not None:
        if sort == 'm<' or sort == 'SM':
            a = np.abs(a)
        elif sort == 'm>' or sort == 'LM':
            a = -np.abs(a)
        elif sort == '<' or sort == 'SR' or sort == 'SA':
            a = np.real(a)
        elif sort == '>' or sort == 'LR' or sort == 'LA':
            a = -np.real(a)
        elif sort == 'SI':
            a = np.imag(a)
        elif sort == 'LI':
            a = -np.imag(a)
        else:
            raise ValueError("unknown sort option " + repr(sort))
    return np.argsort(a, **kwargs)


def combine_constraints(good1, good2, warn):
    """Combine constraints, given in the form of 1D numpy bool arrays.

    Return ``logical_and(good1, good2)`` if there remains at least one ``True`` entry.
    Otherwise, emit a warning and return just `good1`.
    """
    assert good1.shape == good2.shape, f'{good1.shape} != {good2.shape}'
    res = np.logical_and(good1, good2)
    if np.any(res):
        return res
    warnings.warn("truncation: can't satisfy constraint for " + warn, stacklevel=3)
    return good1


def inverse_permutation(perm):
    """reverse sorting indices.

    Sort functions (as :meth:`LegCharge.sort`) return a (1D) permutation `perm` array,
    such that ``sorted_array = old_array[perm]``.
    This function inverts the permutation `perm`,
    such that ``old_array = sorted_array[inverse_permutation(perm)]``.

    Parameters
    ----------
    perm : 1D array_like
        The permutation to be reversed. *Assumes* that it is a permutation with unique indices.
        If it is, ``inverse_permutation(inverse_permutation(perm)) == perm``.

    Returns
    -------
    inv_perm : 1D array (int)
        The inverse permutation of `perm` such that ``inv_perm[perm[j]] = j = perm[inv_perm[j]]``.

    Notes
    -----
    This is equivalent to ``numpy.argsort``, but has O(N) complexity instead of O(N log(N))
    """
    perm = np.asarray(perm, dtype=np.intp)
    inv_perm = np.empty_like(perm)
    inv_perm[perm] = np.arange(perm.shape[0], dtype=perm.dtype)
    return inv_perm
     # equivalently: return np.argsort(perm) # would be O(N log(N))


def rank_data(a, stable=True):
    """Assign ranks to data.

    For equal values, the first one has lower rank.
    This is equivalent to ``a.argsort().argsort()`` but should have better scaling.

    Parameters
    ----------
    a : 1D array-like
        The data to rank. TODO support multi-dimensional data?
    stable: bool
        If ``True`` (default), the ranks of equal values are guaranteed increasing by order of
        appearance in `a`. If ``False``, the relative rank of equal elements is arbitrary, which
        may allow faster sorting algorithms.

    Returns
    -------
    ranks : 1D array of int
        The ranks of the data, such that ``a[i] > a[j]`` implies ``ranks[i] > ranks[j]``.
        For equal elements ``a[i] == a[j]``, and only if `stable`, we have ``ranks[i] > ranks[j]``
        iff ``i > j``. Otherwise the relative ranks are arbitrary.
        The result is a permutation of ``range(len(a))``.
    """
    # basically np.argsort(np.argsort(a)),
    # but use same trick as inverse_permutation for the outer argsort call
    order = np_argsort(a, stable=stable)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(len(a))
    return ranks


if int(np.version.version.split('.')[0]) >= 2:
    def np_argsort(a, stable=True):
        """Wrapper around np.argsort, using the ``stable`` kwarg if available"""
        return np.argsort(a, stable=stable)

else:
    def np_argsort(a, stable=True):
        if stable:
            return np.argsort(a, kind='stable')
        return np.argsort(a)


def make_stride(shape, cstyle=True):
    """Create the strides for C- (or F-style) arrays with a given shape.

    Equivalent to ``x = np.zeros(shape); return np.array(x.strides, np.intp) // x.itemsize``.

    Note that ``np.sum(inds * _make_stride(np.max(inds, axis=0), cstyle=False), axis=1)`` is
    sorted for (positive) 2D `inds` if ``np.lexsort(inds.T)`` is sorted.
    """
    L = len(shape)
    stride = 1
    res = np.empty([L], np.intp)
    if cstyle:
        res[L - 1] = 1
        for a in range(L - 1, 0, -1):
            stride *= shape[a]
            res[a - 1] = stride
        assert stride * shape[0] < _MAX_INT
    else:
        res[0] = 1
        for a in range(0, L - 1):
            stride *= shape[a]
            res[a + 1] = stride
        assert stride * shape[0] < _MAX_INT
    return res


def list_to_dict_list(l):
    """Given a list `l` of objects, construct a lookup table.

    This function will handle duplicate entries in `l`.

    Parameters
    ----------
    l: iterable of iterable of immutable
        A list of objects that can be converted to tuples to be used as keys for a dictionary.

    Returns
    -------
    lookup : dict
        A dictionary with (key, value) pairs ``(key):[i1,i2,...]``
        where ``i1, i2, ...`` are the indices where `key` is found in `l`:
        i.e. ``key == tuple(l[i1]) == tuple(l[i2]) == ...``
    """
    d = {}
    for i, r in enumerate(l):
        k = tuple(r)
        try:
            d[k].append(i)
        except KeyError:
            d[k] = [i]
    return d


def find_row_differences(sectors, include_len: bool=False):
    """Return indices where the rows of the 2D array `sectors` change.

    Parameters
    ----------
    sectors : 2D array
        The rows of this array are compared.
    include_len : bool
        If ``len(sectors)`` should be included or not.

    Returns
    -------
    diffs: 1D array
        The indices where rows change, including the first and last. Equivalent to:
        ``[0] + [i for i in range(1, len(sectors)) if np.any(sectors[i-1] != sectors[i])]``
    """
    # note: by default remove last entry [len(sectors)] compared to old.charges
    len_sectors = len(sectors)
    diff = np.ones(len_sectors + int(include_len), dtype=np.bool_)
    diff[1:len_sectors] = np.any(sectors[1:] != sectors[:-1], axis=1)
    return np.nonzero(diff)[0]  # get the indices of True-values


def unstridify(x, strides):
    """Undo applying strides to an index.

    Parameters
    ----------
    x : (..., M) ndarray
        1D array of non-negative integers. Broadcast over leading axis.
    strides : (N,) ndarray
        C-style strides, i.e. positive integers such that ``strides[i]`` is an integer multiple
        of ``strides[i + 1]``.

    Returns
    -------
    (..., M, N) ndarray
        The unique ``ys`` such that ``x == np.sum(strides * ys, axis=-1)``.
    """
    y_list = []
    for s in strides:
        y, x = np.divmod(x, s)
        y_list.append(y)
    return np.stack(y_list, axis=-1)


def iter_common_noncommon_sorted(a, b):
    """Yield the following pairs ``i, j`` of indices:

    - Matching entries, i.e. ``(i, j)`` such that ``a[i] == b[j]``
    - Entries only in `a`, i.e. ``(i, None)`` such that ``a[i]`` is not in `b`
    - Entries only in `b`, i.e. ``(None, j)`` such that ``b[j]`` is not in `a`

    *Assumes* that `a` and `b` are strictly ascending.
    """
    l_a = len(a)
    l_b = len(b)
    i, j = 0, 0
    while i < l_a and j < l_b:
        if a[i] < b[j]:
            yield i, None
            i += 1
        elif a[i] > b[j]:
            yield None, j
            j += 1
        else:
            yield i, j
            i += 1
            j += 1
    # can still have i < l_a or j < l_b but not both
    for i2 in range(i, l_a):
        yield i2, None
    for j2 in range(j, l_b):
        yield None, j2


def iter_common_sorted(a, b):
    """Yield indices ``i, j`` for which ``a[i] == b[j]``.

    *Assumes* that `a` and `b` are strictly ascending 1D arrays.
    Given that, it is equivalent to (but faster than)
    ``[(i, j) for j, i in itertools.product(range(len(b)), range(len(a)) if a[i] == b[j]]``
    """
    # when we call this function, we basically wanted iter_common_sorted_arrays,
    # but used strides to merge multiple columns to avoid too much python loops
    # for C-implementation, this is definitely no longer necessary.
    l_a = len(a)
    l_b = len(b)
    i, j = 0, 0
    while i < l_a and j < l_b:
        if a[i] < b[j]:
            i += 1
        elif b[j] < a[i]:
            j += 1
        else:
            yield i, j
            i += 1
            j += 1


def iter_common_sorted_arrays(a, b, a_strict: bool = True, b_strict: bool = True):
    """Yield indices ``i, j`` for which ``a[i, :] == b[j, :]``.

    *Assumes* that `a` and `b` are lex-sorted (according to ``np.lexsort(a.T)``).
    Given that, it is equivalent to (but faster than)
    ``[(i, j) for j, i in itertools.product(range(len(b)), range(len(a)) if all(a[i,:] == b[j,:])]``

    By default, assume that both are strictly sorted, i.e. contain no duplicate rows.
    Optionally, the strict requirement may be relaxed for *one* of the two arrays.
    """
    if (not a_strict) and (not b_strict):
        raise ValueError('One of the two arrays must be strictly sorted.')

    l_a, d_a = a.shape
    l_b, d_b = b.shape
    assert d_a == d_b
    i, j = 0, 0
    while i < l_a and j < l_b:
        for k in reversed(range(d_a)):
            if a[i, k] < b[j, k]:
                i += 1
                break
            elif b[j, k] < a[i, k]:
                j += 1
                break
        else:
            yield (i, j)
            if b_strict:
                # b is strictly sorted => no further b[j + x, : ] will match the same a[i, :]
                i += 1
            if a_strict:
                # a is strictly sorted => no further a[i + x, : ] will match the same b[j, :]
                j += 1


def iter_common_noncommon_sorted_arrays(a, b):
    """Yield the following pairs ``i, j`` of indices:

    - Matching entries, i.e. ``(i, j)`` such that ``all(a[i, :] == b[j, :])``
    - Entries only in `a`, i.e. ``(i, None)`` such that ``a[i, :]`` is not in `b`
    - Entries only in `b`, i.e. ``(None, j)`` such that ``b[j, :]`` is not in `a`

    *Assumes* that `a` and `b` are strictly lex-sorted (according to ``np.lexsort(a.T)``).
    """
    l_a, d_a = a.shape
    l_b, d_b = b.shape
    assert d_a == d_b
    i, j = 0, 0
    while i < l_a and j < l_b:
        for k in reversed(range(d_a)):
            if a[i, k] < b[j, k]:
                yield i, None
                i += 1
                break
            elif a[i, k] > b[j, k]:
                yield None, j
                j += 1
                break
        else:
            yield i, j
            i += 1
            j += 1
    # can still have i < l_a or j < l_b, but not both
    for i2 in range(i, l_a):
        yield i2, None
    for j2 in range(j, l_b):
        yield None, j2


def find_subclass(base_class, subclass_name):
    """For a given base class, recursively find the subclass with the given name.

    Parameters
    ----------
    base_class : class
        The base class of which `subclass_name` is supposed to be a subclass.
    subclass_name : str | type
        The name (str) of the class to be found.
        Alternatively, if a type is given, it is directly returned. In that case, a warning is
        raised if it is not a subclass of `base_class`.

    Returns
    -------
    subclass : class
        Class with name `subclass_name` which is a subclass of the `base_class`.
        None, if no subclass of the given name is found.

    Raises
    ------
    ValueError: When no or multiple subclasses of `base_class` exists with that `subclass_name`.
    """
    if not isinstance(subclass_name, str):
        subclass = subclass_name
        if not isinstance(subclass, type):
            raise TypeError("expect a str or class for `subclass_name`, got " + repr(subclass))
        if not issubclass(subclass, base_class):
            # still allow it: might intend duck-typing. However, a warning should be raised!
            warnings.warn(f"find_subclass: {subclass!r} is not subclass of {base_class!r}")
        return subclass
    found = set()
    _find_subclass_recursion(base_class, subclass_name, found, set())
    if len(found) == 0:
        raise ValueError(f"No subclass of {base_class.__name__} called {subclass_name!r} defined. "
                         "Maybe missing an import of a file with a custom class definition?")
    elif len(found) == 1:
        return found.pop()
    else:
        found_not_deprecated = [c for c in found if not getattr(c, 'deprecated', False)]
        if len(found_not_deprecated) == 1:
            return found_not_deprecated[0]
        msg = f"There exist multiple subclasses of {base_class!r} with name {subclass_name!r}:"
        raise ValueError('\n'.join([msg] + [repr(c) for c in found]))


def _find_subclass_recursion(base_class, name_to_find, found, checked):
    if base_class.__name__ == name_to_find:
        found.add(base_class)
    for subcls in base_class.__subclasses__():
        if subcls in checked:
            continue
        _find_subclass_recursion(subcls, name_to_find, found, checked)
        checked.add(subcls)
