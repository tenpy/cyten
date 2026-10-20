backends
========

- full name: lasyten.backends
- parent module: :mod:`lasyten`
- type: module

.. py:module:: lasyten.backends

Tensor backends. Bindings live in ``pybind/backends/``.

Classes and functions imported from the pybind11 module :mod:`lasyten._core`.

.. toctree::
   :maxdepth: 1

   backends/abelian
   backends/no_symmetry
   backends/fusion_tree_backend

.. rubric:: Inheritance Diagram

.. lasyten-inheritance-diagram::

lasyten.backends.TensorBackend
----------------------------

.. autoclass:: lasyten.backends.TensorBackend
   :members:
   :undoc-members:
   :show-inheritance:

Functions
---------

.. autofunction:: lasyten.backends.get_backend

.. autofunction:: lasyten.backends.conventional_leg_order

.. autofunction:: lasyten.backends.get_same_backend
