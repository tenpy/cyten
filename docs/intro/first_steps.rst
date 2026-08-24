First steps
===========

This page walks through a first calculation with Cyten: define a local spin,
build a two-site Heisenberg coupling, contract it to a Hamiltonian, and
read off the eigenvalues.

The same steps are shown in Python and in C++. Both listings live in
``docs/intro/examples/`` and are run as part of the test suite.

If Cyten is not installed yet, see :doc:`/INSTALL`.

Sites and couplings
-------------------

A lattice model is built from two ingredients:

- A :class:`~cyten.models.SpinSite` (more generally a :class:`~cyten.models.Site`)
  holds the local Hilbert space and the on-site operators that are compatible
  with the chosen symmetry.
- A :class:`~cyten.models.Coupling` is a few-site operator, stored as one tensor
  per site in an MPO-like factorization. Contracting those tensors recovers
  the dense few-site Hamiltonian.

The Heisenberg interaction on two spins is

.. math::

    h_{ij} = J\, \vec{S}_i \cdot \vec{S}_j.

On two spin-:math:`1/2` sites the spectrum is a singlet at :math:`E = -3/4`
and a triplet at :math:`E = +1/4` (for :math:`J = 1`).

Python
------

.. literalinclude:: examples/heisenberg_two_site.py
   :language: python

Run it with::

    python docs/intro/examples/heisenberg_two_site.py

:func:`~cyten.models.heisenberg_coupling` builds the factorized coupling.
:meth:`~cyten.models.Coupling.to_tensor` contracts the virtual legs and
returns a four-leg tensor with labels ``p0, p1`` (outgoing) and
``p0*, p1*`` (incoming). :func:`~cyten.eigh` then diagonalizes that
hermitian map; :meth:`~cyten.tensors.DiagonalTensor.diagonal_as_numpy`
extracts the eigenvalues as a NumPy array.

C++
---

The C++ API mirrors the Python one. A standalone program must start an
embedded Python interpreter and ``import cyten`` so that backends and
pybind11 type casters are registered. Compile with ``-fvisibility=hidden``
(the CMake ``cyten`` target passes this on); otherwise GCC/Clang warn that
classes such as :cpp:class:`cyten::SpinSite` have greater visibility than
their pybind11 members.

.. literalinclude:: examples/heisenberg_two_site.cpp
   :language: c++

The corresponding types and functions are
:cpp:class:`cyten::SpinSite`,
:cpp:func:`cyten::heisenberg_coupling`,
:cpp:func:`cyten::Coupling::to_tensor()`,
and :cpp:func:`cyten::eigh`.

From the CMake build tree the example is built and tested as
``example_heisenberg_two_site``::

    cmake --build build --target example_heisenberg_two_site
    ./build/tests/ctest/example_heisenberg_two_site

What to try next
----------------

- Pass ``conserve='Sz'`` or ``conserve='SU(2)'`` to :class:`~cyten.models.SpinSite`
  to keep the corresponding spin symmetry. :func:`~cyten.eigh` still applies;
  eigenvalues are then sorted *within* each charge block, so sort them globally
  if you want to compare against the four numbers above.
- Browse the on-site operators on ``site0.onsite_operators`` (for example
  ``Sz`` when :math:`S^z` is conserved).
- The :mod:`cyten.models` reference lists further couplings (fields, hopping,
  density-density, …). Tensor operations live in :mod:`cyten.tensors`.
- Coming from TeNPy's :mod:`tenpy.linalg.np_conserved`? See
  :doc:`from_np_conserved` for what changed and how to update existing code.
