"""See :mod:`cyten.tensors`."""

# Copyright (C) TeNPy Developers, Apache license
from __future__ import annotations

from .._core import (  # noqa: F401
    CONTRACT_SYMBOL,
    FORBIDDEN_LEG_LABEL_CHARS,
    LEG_SELECT_SYMBOL,
    OPEN_LEG_SYMBOL,
    ChargedTensor,
    DiagonalTensor,
    Identity,
    LabelledLegs,
    Mask,
    SymmetricTensor,
    Tensor,
    _check_compatible_legs,
    _combine_leg_labels,
    _compose_SymmetricTensors,
    _compose_with_Mask,
    _convert_abelian_to_FT,
    _convert_FT_to_abelian,
    _decomposition_labels,
    _decomposition_prepare,
    _dual_label_list,
    _dual_leg_label,
    _get_matching_labels,
    _split_leg_label,
    _svd_new_labels,
    add_trivial_leg,
    almost_equal,
    angle,
    apply_mask,
    apply_mask_DiagonalTensor,
    bend_legs,
    check_same_legs,
    combine_legs,
    combine_to_matrix,
    complex_conj,
    compose,
    conventional_leg_order,
    cutoff_inverse,
    dagger,
    eigh,
    enlarge_leg,
    entropy,
    exp,
    eye,
    get_same_device,
    imag,
    inner,
    is_scalar,
    is_valid_leg_label,
    item,
    linear_combination as _linear_combination,
    lq,
    move_leg,
    norm,
    on_device,
    outer,
    partial_compose,
    partial_trace,
    permute_legs,
    pinv,
    qr,
    real,
    real_if_close,
    scalar_multiply as _scalar_multiply,
    scale_axis,
    split_legs,
    sqrt,
    squeeze_legs,
    stable_log,
    svd,
    svd_apply_mask,
    tdot,
    tensor,
    tensor_from_grid,
    trace,
    transpose,
    truncate_singular_values,
    truncated_svd,
    zero_like,
)


def scalar_multiply(a, v):
    """The scalar multiplication ``a * v``."""
    if a is None:
        raise TypeError("unsupported scalar type: NoneType")
    return _scalar_multiply(a, v)


def linear_combination(a, v, b, w):
    """The linear combination ``a * v + b * w``."""
    if a is None or b is None:
        raise TypeError(
            f"unsupported scalar types: {type(a).__name__}, {type(b).__name__}"
        )
    return _linear_combination(a, v, b, w)

