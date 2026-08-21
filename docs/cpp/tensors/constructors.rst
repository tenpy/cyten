constructors.h
==============

Declared in ``include/cyten/tensors/constructors.h``.

Free functions
--------------

.. doxygenfunction:: cyten::eye
   :project: cyten

.. doxygenfunction:: cyten::tensor(TensorCPtr, TensorProduct::Ptr, TensorProduct::Ptr, TensorBackend::Ptr, std::optional<LegLabels>, std::optional<Dtype>, std::optional<std::string>)
   :project: cyten

.. doxygenfunction:: cyten::tensor(BlockBackend::BlockPtr, TensorProduct::Ptr, TensorProduct::Ptr, TensorBackend::Ptr, std::optional<LegLabels>, std::optional<Dtype>, std::optional<std::string>, bool)
   :project: cyten

.. doxygenfunction:: cyten::add_trivial_leg
   :project: cyten

.. doxygenfunction:: cyten::zero_like
   :project: cyten

.. doxygenfunction:: cyten::tensor_from_grid
   :project: cyten
