cyten.backends
==============

Tensor backends. Bindings live in ``pybind/backends/``.

Classes and functions imported from the pybind11 module :mod:`cyten._core`.

.. toctree::
   :maxdepth: 1

   backends/abelian
   backends/no_symmetry
   backends/fusion_tree_backend

cyten.backends.TensorBackend
----------------------------

.. autoclass:: cyten.backends.TensorBackend
   :members:
   :undoc-members:
   :show-inheritance:

Functions
---------

.. autofunction:: cyten.backends.get_backend

.. autofunction:: cyten.backends.conventional_leg_order

.. autofunction:: cyten.backends.get_same_backend
