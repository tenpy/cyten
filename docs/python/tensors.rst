cyten.tensors
=============

Tensor types and operations. Bindings live in ``pybind/tensors/``.

Classes and functions imported from the pybind11 module :mod:`cyten._core`.

cyten.tensors.LabelledLegs
--------------------------

.. autoclass:: cyten.tensors.LabelledLegs
   :members:
   :undoc-members:
   :show-inheritance:

cyten.tensors.Tensor
--------------------

.. autoclass:: cyten.tensors.Tensor
   :members:
   :undoc-members:
   :show-inheritance:

cyten.tensors.SymmetricTensor
-----------------------------

.. autoclass:: cyten.tensors.SymmetricTensor
   :members:
   :undoc-members:
   :show-inheritance:

cyten.tensors.DiagonalTensor
----------------------------

.. autoclass:: cyten.tensors.DiagonalTensor
   :members:
   :undoc-members:
   :show-inheritance:

cyten.tensors.Identity
----------------------

.. autoclass:: cyten.tensors.Identity
   :members:
   :undoc-members:
   :show-inheritance:

cyten.tensors.Mask
------------------

.. autoclass:: cyten.tensors.Mask
   :members:
   :undoc-members:
   :show-inheritance:

cyten.tensors.ChargedTensor
---------------------------

.. autoclass:: cyten.tensors.ChargedTensor
   :members:
   :undoc-members:
   :show-inheritance:

Functions
---------

.. autofunction:: cyten.tensors.add_trivial_leg

.. autofunction:: cyten.tensors.almost_equal

.. autofunction:: cyten.tensors.angle

.. autofunction:: cyten.tensors.apply_mask

.. autofunction:: cyten.tensors.apply_mask_DiagonalTensor

.. autofunction:: cyten.tensors.bend_legs

.. autofunction:: cyten.tensors.check_same_legs

.. autofunction:: cyten.tensors.combine_legs

.. autofunction:: cyten.tensors.combine_to_matrix

.. autofunction:: cyten.tensors.complex_conj

.. autofunction:: cyten.tensors.compose

.. autofunction:: cyten.tensors.cutoff_inverse

.. autofunction:: cyten.tensors.dagger

.. autofunction:: cyten.tensors.eigh

.. autofunction:: cyten.tensors.enlarge_leg

.. autofunction:: cyten.tensors.entropy

.. autofunction:: cyten.tensors.exp

.. autofunction:: cyten.tensors.eye

.. autofunction:: cyten.tensors.get_same_device

.. autofunction:: cyten.tensors.imag

.. autofunction:: cyten.tensors.inner

.. autofunction:: cyten.tensors.is_scalar

.. autofunction:: cyten.tensors.is_valid_leg_label

.. autofunction:: cyten.tensors.item

.. autofunction:: cyten.tensors.linear_combination

.. autofunction:: cyten.tensors.lq

.. autofunction:: cyten.tensors.move_leg

.. autofunction:: cyten.tensors.norm

.. autofunction:: cyten.tensors.on_device

.. autofunction:: cyten.tensors.outer

.. autofunction:: cyten.tensors.partial_compose

.. autofunction:: cyten.tensors.partial_trace

.. autofunction:: cyten.tensors.permute_legs

.. autofunction:: cyten.tensors.pinv

.. autofunction:: cyten.tensors.qr

.. autofunction:: cyten.tensors.real

.. autofunction:: cyten.tensors.real_if_close

.. autofunction:: cyten.tensors.scalar_multiply

.. autofunction:: cyten.tensors.scale_axis

.. autofunction:: cyten.tensors.slice_leg

.. autofunction:: cyten.tensors.split_legs

.. autofunction:: cyten.tensors.sqrt

.. autofunction:: cyten.tensors.squeeze_legs

.. autofunction:: cyten.tensors.stable_log

.. autofunction:: cyten.tensors.svd

.. autofunction:: cyten.tensors.svd_apply_mask

.. autofunction:: cyten.tensors.tdot

.. autofunction:: cyten.tensors.tensor

.. autofunction:: cyten.tensors.tensor_from_grid

.. autofunction:: cyten.tensors.trace

.. autofunction:: cyten.tensors.transpose

.. autofunction:: cyten.tensors.truncate_singular_values

.. autofunction:: cyten.tensors.truncated_svd

.. autofunction:: cyten.tensors.zero_like
