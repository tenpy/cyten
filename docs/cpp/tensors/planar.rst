planar.h
========

Declared in ``include/lasyten/tensors/planar.h``.

.. doxygenclass:: lasyten::TensorPlaceholder
   :project: lasyten
   :members:
   :undoc-members:

.. doxygenclass:: lasyten::ContractionTreeNode
   :project: lasyten
   :members:
   :undoc-members:

.. doxygenclass:: lasyten::ContractionTree
   :project: lasyten
   :members:
   :undoc-members:

.. doxygenclass:: lasyten::PlanarDiagram
   :project: lasyten
   :members:
   :undoc-members:

.. doxygenclass:: lasyten::PlanarLinearOperator
   :project: lasyten
   :members:
   :undoc-members:

Free functions
--------------

.. doxygenfunction:: lasyten::parse_leg_bipartition
   :project: lasyten

.. doxygenfunction:: lasyten::horizontal_factorization
   :project: lasyten

.. doxygenfunction:: lasyten::planar_almost_equal
   :project: lasyten

.. doxygenfunction:: lasyten::planar_combine_legs
   :project: lasyten

.. doxygenfunction:: lasyten::planar_contraction(TensorCPtr, TensorCPtr, std::vector<LegRef>, std::vector<LegRef>, std::map<std::string, std::string>, std::map<std::string, std::string>)
   :project: lasyten

.. doxygenfunction:: lasyten::planar_contraction(TensorPlaceholder const &, TensorPlaceholder const &, std::vector<LegRef>, std::vector<LegRef>)
   :project: lasyten

.. doxygenfunction:: lasyten::planar_eigh
   :project: lasyten

.. doxygenfunction:: lasyten::planar_eig
   :project: lasyten

.. doxygenfunction:: lasyten::planar_eigvals
   :project: lasyten

.. doxygenfunction:: lasyten::planar_lq
   :project: lasyten

.. doxygenfunction:: lasyten::planar_partial_trace(TensorCPtr, std::vector<std::vector<LegRef>>)
   :project: lasyten

.. doxygenfunction:: lasyten::planar_partial_trace(TensorPlaceholder const &, std::vector<std::vector<LegRef>>)
   :project: lasyten

.. doxygenfunction:: lasyten::planar_permute_legs
   :project: lasyten

.. doxygenfunction:: lasyten::planar_qr
   :project: lasyten

.. doxygenfunction:: lasyten::planar_svd
   :project: lasyten

.. doxygenfunction:: lasyten::planar_truncated_svd
   :project: lasyten
