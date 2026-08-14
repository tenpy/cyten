tensor_backend.h
================

Declared in ``include/cyten/backends/tensor_backend.h``.

cyten::TensorBackend
--------------------

.. doxygenclass:: cyten::TensorBackend
   :project: cyten
   :members:
   :undoc-members:

Free functions
--------------

.. doxygenfunction:: cyten::conventional_leg_order(TensorProduct::Ptr, TensorProduct::Ptr)
   :project: cyten

.. doxygenfunction:: cyten::conventional_leg_order(py::object, py::object)
   :project: cyten

.. doxygenfunction:: cyten::conventional_leg_order(TensorCPtr)
   :project: cyten

.. doxygenfunction:: cyten::get_same_backend(const std::vector< py::object > &, std::string)
   :project: cyten

.. doxygenfunction:: cyten::get_same_backend(const std::vector< TensorCPtr > &, std::string)
   :project: cyten
