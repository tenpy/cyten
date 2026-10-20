tensor_backend.h
================

Declared in ``include/lasyten/backends/tensor_backend.h``.

lasyten::TensorBackend
--------------------

.. doxygenclass:: lasyten::TensorBackend
   :project: lasyten
   :members:
   :undoc-members:

Free functions
--------------

.. doxygenfunction:: lasyten::conventional_leg_order(TensorProduct::Ptr, TensorProduct::Ptr)
   :project: lasyten

.. doxygenfunction:: lasyten::conventional_leg_order(py::object, py::object)
   :project: lasyten

.. doxygenfunction:: lasyten::conventional_leg_order(TensorCPtr)
   :project: lasyten

.. doxygenfunction:: lasyten::get_same_backend(const std::vector< py::object > &, std::string)
   :project: lasyten

.. doxygenfunction:: lasyten::get_same_backend(const std::vector< TensorCPtr > &, std::string)
   :project: lasyten
