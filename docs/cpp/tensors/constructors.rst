constructors.h
==============

Declared in ``include/lasyten/tensors/constructors.h``.

Free functions
--------------

.. doxygenfunction:: lasyten::eye
   :project: lasyten

.. doxygenfunction:: lasyten::tensor(TensorCPtr, TensorProduct::Ptr, TensorProduct::Ptr, TensorBackend::Ptr, std::optional<OptionalLabels>, std::optional<Dtype>, std::optional<std::string>)
   :project: lasyten

.. doxygenfunction:: lasyten::tensor(BlockBackend::BlockPtr, TensorProduct::Ptr, TensorProduct::Ptr, TensorBackend::Ptr, std::optional<OptionalLabels>, std::optional<Dtype>, std::optional<std::string>, bool)
   :project: lasyten

.. doxygenfunction:: lasyten::add_trivial_leg
   :project: lasyten

.. doxygenfunction:: lasyten::zero_like
   :project: lasyten

.. doxygenfunction:: lasyten::tensor_from_grid
   :project: lasyten
