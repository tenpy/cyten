planar.h
========

Declared in ``include/cyten/tensors/planar.h``.

.. doxygenclass:: cyten::TensorPlaceholder
   :project: cyten
   :members:
   :undoc-members:

.. doxygenclass:: cyten::ContractionTreeNode
   :project: cyten
   :members:
   :undoc-members:

.. doxygenclass:: cyten::ContractionTree
   :project: cyten
   :members:
   :undoc-members:

.. doxygenclass:: cyten::PlanarDiagram
   :project: cyten
   :members:
   :undoc-members:

.. doxygenclass:: cyten::PlanarLinearOperator
   :project: cyten
   :members:
   :undoc-members:

Free functions
--------------

.. doxygenfunction:: cyten::parse_leg_bipartition
   :project: cyten

.. doxygenfunction:: cyten::horizontal_factorization
   :project: cyten

.. doxygenfunction:: cyten::planar_almost_equal
   :project: cyten

.. doxygenfunction:: cyten::planar_combine_legs
   :project: cyten

.. doxygenfunction:: cyten::planar_contraction(TensorCPtr, TensorCPtr, std::vector<LegRef>, std::vector<LegRef>, std::map<std::string, std::string>, std::map<std::string, std::string>)
   :project: cyten

.. doxygenfunction:: cyten::planar_contraction(TensorPlaceholder const &, TensorPlaceholder const &, std::vector<LegRef>, std::vector<LegRef>)
   :project: cyten

.. doxygenfunction:: cyten::planar_eigh
   :project: cyten

.. doxygenfunction:: cyten::planar_lq
   :project: cyten

.. doxygenfunction:: cyten::planar_partial_trace(TensorCPtr, std::vector<std::vector<LegRef>>)
   :project: cyten

.. doxygenfunction:: cyten::planar_partial_trace(TensorPlaceholder const &, std::vector<std::vector<LegRef>>)
   :project: cyten

.. doxygenfunction:: cyten::planar_permute_legs
   :project: cyten

.. doxygenfunction:: cyten::planar_qr
   :project: cyten

.. doxygenfunction:: cyten::planar_svd
   :project: cyten

.. doxygenfunction:: cyten::planar_truncated_svd
   :project: cyten
