From TeNPy ``np_conserved`` to Cyten
====================================

Cyten replaces :external+tenpy_v1:py:mod:`tenpy.linalg.np_conserved` as the tensor library underneath
TeNPy. The Python interface is deliberately similar — labelled legs, block-sparse
linear algebra, ``tensordot``-style contractions — but it is **not** a drop-in
import rename. This page lists what changed and how to update existing code.

If you are new to Cyten, start with :doc:`first_steps`. This page assumes you
already know TeNPy's :external+tenpy_v1:py:class:`~tenpy.linalg.np_conserved.Array`.

The two scripts under ``docs/intro/examples/`` are Cyten ports of TeNPy's
``a_npc_arrays_triv.py`` and ``b_npc_arrays.py`` userguide examples.


Why the interface changed
-------------------------

TeNPy v1 implements **abelian** charge conservation only. An
:external+tenpy_v1:py:class:`~tenpy.linalg.np_conserved.Array` is a numpy-like tensor: a list of legs,
each with a :external+tenpy_v1:py:class:`~tenpy.linalg.charges.LegCharge` and a ``qconj`` arrow, plus a
total charge :external+tenpy_v1:py:attr:`~tenpy.linalg.np_conserved.Array.qtotal`.

Cyten is written for a broader class of symmetries (abelian groups, non-abelian
groups such as :math:`\mathrm{SU}(2)`, fermions, and anyon categories). That
forces a few design changes:

- The symmetry is a first-class object (:class:`~cyten.symmetries.Symmetry`),
  not a list of ``qmod`` integers.
- A tensor is a **linear map** from a :attr:`~cyten.tensors.Tensor.domain` to a
  :attr:`~cyten.tensors.Tensor.codomain`, not a structureless multi-index array.
- There are several tensor types (symmetric, diagonal, mask, identity, charged)
  sharing the abstract :class:`~cyten.tensors.Tensor` base, instead of a single
  :external+tenpy_v1:py:class:`~tenpy.linalg.np_conserved.Array`.
- Charge-non-conserving operators are a separate type
  (:class:`~cyten.tensors.ChargedTensor`) instead of a non-zero ``qtotal``.
- Operations that were :external+tenpy_v1:py:class:`~tenpy.linalg.np_conserved.Array` methods
  (``combine_legs``, ``split_legs``, ``conj``, ``transpose``, …) are **functions**
  (``ct.combine_legs(A, ...)``). In-place ``i*`` methods are gone.

.. warning ::

    Symmetries can have a non-trivial braiding: legs that cross in a tensor-network
    diagram then have non-trivial consequences. The simplest example is fermions,
    where a crossing implies a sign change on the odd-parity sector.
    Unlike anyon categories, this braiding does not distinguish over- and
    under-braids (*symmetric* braiding); all group symmetries have that property
    as well.

The C++ API mirrors the Python one. The notes below are written for Python;
the corresponding C++ names live in ``cyten::``.


Imports and types
-----------------

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - TeNPy
     - Cyten
   * - ``import tenpy.linalg.np_conserved as npc``
     - ``import cyten as ct``
   * - :external+tenpy_v1:py:class:`~tenpy.linalg.np_conserved.Array`
     - :class:`~cyten.tensors.Tensor` (abstract) /
       :class:`~cyten.tensors.SymmetricTensor` (the usual case)
   * - :external+tenpy_v1:py:class:`~tenpy.linalg.charges.ChargeInfo`
     - :class:`~cyten.symmetries.Symmetry` (product of
       :class:`~cyten.symmetries.SymmetryFactor`\s)
   * - :external+tenpy_v1:py:class:`~tenpy.linalg.charges.LegCharge`
     - :class:`~cyten.symmetries.spaces.ElementarySpace`
   * - :external+tenpy_v1:py:class:`~tenpy.linalg.charges.LegPipe`
     - :class:`~cyten.symmetries.spaces.LegPipe` /
       :class:`~cyten.symmetries.spaces.AbelianLegPipe`
   * - ``Array.rank`` / ``ndim``
     - ``tensor.num_legs``
   * - ``Array.chinfo``
     - ``tensor.symmetry``
   * - ``Array.qtotal``
     - no analogue on :class:`~cyten.tensors.SymmetricTensor`;
       use :class:`~cyten.tensors.ChargedTensor`
   * - ``Array.dtype`` (numpy dtype)
     - :class:`~cyten.block_backends.Dtype` (``ct.float64``, ``ct.complex128``, …)

Additional Cyten types you will meet:

:class:`~cyten.tensors.DiagonalTensor`
    SVD singular values, eigenvalues, scaling vectors. The replacement for a
    1D numpy array of singular values plus a :external+tenpy_v1:py:func:`~tenpy.linalg.np_conserved.diag`
    call.
:class:`~cyten.tensors.Identity`
    The identity map on a space.
:class:`~cyten.tensors.Mask`
    A projection / inclusion between a space and a subspace. Replaces
    :external+tenpy_v1:py:meth:`~tenpy.linalg.np_conserved.Array.iproject` boolean masks.
:class:`~cyten.tensors.ChargedTensor`
    A tensor that transforms in a definite non-trivial sector (e.g. :math:`S^+`
    when :math:`S^z` is conserved).


Tensors as maps
---------------

This is the change that affects the most call sites.

A Cyten tensor is a map :math:`\mathrm{domain} \to \mathrm{codomain}`. The
flat list :attr:`~cyten.tensors.Tensor.legs` is **derived** from that
partition::

    legs == [*codomain, *reversed(leg.dual for leg in domain)]

Graphically, with ``codomain == [V, W, Z]`` and ``domain == [X, Y]``::

    |            X   Y
    |         ┏━━┷━━━┷━━━┓
    |         ┃    T     ┃
    |         ┗┯━━━┯━━━┯━┛
    |          V   W   Z
    |
    |     legs[0], legs[1], legs[2]  ==  V, W, Z          (codomain)
    |     legs[3], legs[4]           ==  Y.dual, X.dual   (domain, reversed)

.. note ::

    The tensor-diagram notation here is stricter than for generic tensor networks:
    the top legs are always the domain and the bottom legs always the codomain.
    Compositions such as :func:`~cyten.tensors.tdot` therefore stack tensors
    vertically.

Integer indices and string labels still refer to positions in ``legs``, so you
can often ignore the (co)domain split. It becomes visible when you:

- **create** a tensor: you pass ``codomain=`` and ``domain=`` instead of a list
  of :external+tenpy_v1:py:class:`~tenpy.linalg.charges.LegCharge`;
- **compose** maps with ``A @ B`` / :func:`~cyten.tensors.compose`, which
  requires ``A.domain == B.codomain``;
- run **decompositions** (:func:`~cyten.tensors.svd`, :func:`~cyten.tensors.eigh`,
  :func:`~cyten.tensors.qr`): they factor the map as it is, without an extra
  reshape to a matrix.

TeNPy's ``qconj = +1`` (ket / inward) corresponds to an
:class:`~cyten.symmetries.spaces.ElementarySpace` with ``is_dual=False``.
``qconj = -1`` (bra / dual) corresponds to ``space.dual`` (``is_dual=True``).
You rarely set ``is_dual`` by hand: put ket spaces in the (co)domain and let
``legs`` insert the duals on the domain side.

To inspect the layout, print the tensor (it draws an ascii diagram) or call
:meth:`~cyten.tensors.Tensor.dbg`.


Symmetries instead of ``ChargeInfo``
------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - TeNPy
     - Cyten
   * - ``npc.ChargeInfo()`` (trivial)
     - ``ct.NoSymmetry()``
   * - ``npc.ChargeInfo([1])`` (:math:`U(1)`)
     - ``ct.U1()`` / :class:`~cyten.symmetries.U1`
   * - ``npc.ChargeInfo([2])`` (:math:`\mathbb{Z}_2`)
     - ``ct.ZN(2)`` / :class:`~cyten.symmetries.ZN` ``(2)``
   * - ``npc.ChargeInfo([1, 2], names=['N', 'P'])``
     - ``ct.U1("N") * ct.ZN(2, "P")``
   * - (not available)
     - ``ct.SU2()``, ``ct.FermionParity()``, anyon categories, …

Sectors are 1D integer arrays. For a single :math:`U(1)` they look like
``[n]``, same as TeNPy's 1-column charge arrays. For a product symmetry they
are concatenations of the factor sectors. Use
:func:`~cyten.symmetries.as_sector` when you need a ``Sector`` rather than a
Python list (the C++ bindings do not always convert ``[2]``).

A local space is built from the sector of every basis state — the analogue of
:external+tenpy_v1:py:meth:`~tenpy.linalg.charges.LegCharge.from_qflat`::

    # TeNPy
    chinfo = npc.ChargeInfo([1])
    p_leg = npc.LegCharge.from_qflat(chinfo, [[1], [-1]])  # |↑⟩, |↓⟩

    # Cyten
    p = ct.ElementarySpace.from_basis(ct.U1(), [[1], [-1]])

:meth:`~cyten.symmetries.spaces.ElementarySpace.from_defining_sectors` is
closer to :external+tenpy_v1:py:meth:`~tenpy.linalg.charges.LegCharge.from_qind`: list each sector
once, with a multiplicity. For non-abelian symmetries that is the natural
constructor (an :math:`\mathrm{SU}(2)` spin-:math:`1/2` is one copy of the
2-dimensional sector, not two basis entries).

Cyten always stores sectors in a canonical order internally and records the
public basis order as :attr:`~cyten.symmetries.spaces.Leg.basis_perm`. You do
not call :external+tenpy_v1:py:meth:`~tenpy.linalg.np_conserved.Array.sort_legcharge`.

:external+tenpy_v1:py:meth:`~tenpy.linalg.charges.LegCharge.conj` becomes the ``dual`` of a space
(a property, not a method)::

    p.dual          # ElementarySpace with is_dual=True
    p_leg.conj()    # TeNPy equivalent


Creating tensors
----------------

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - TeNPy
     - Cyten
   * - ``npc.Array.from_ndarray_trivial(x)``
     - ``ct.tensor(x, codomain=[leg], domain=[...])``
       with :meth:`~cyten.symmetries.spaces.ElementarySpace.from_trivial_sector`
   * - ``npc.Array.from_ndarray(x, legs, qtotal=...)``
     - ``ct.tensor(x, codomain=..., domain=...)``
       (symmetric, ``qtotal`` trivial) or
       :meth:`~cyten.tensors.ChargedTensor.from_dense_block`
   * - ``npc.Array.from_func(np.ones, legs)``
     - :meth:`~cyten.tensors.SymmetricTensor.from_block_func`
   * - ``npc.zeros(legs)``
     - :meth:`~cyten.tensors.SymmetricTensor.from_zero`
   * - ``npc.eye_like(a)`` / ``npc.diag(s, leg)``
     - :func:`~cyten.tensors.eye` /
       :meth:`~cyten.tensors.DiagonalTensor.from_diag_block`
   * - ``npc.ones(...)``
     - ``SymmetricTensor.from_block_func(np.ones, ...)``
   * - ``a.to_ndarray()``
     - :meth:`~cyten.tensors.Tensor.to_numpy` /
       :meth:`~cyten.tensors.Tensor.to_dense_block`
   * - ``a.astype(dtype)``
     - :meth:`~cyten.tensors.Tensor.as_dtype`
   * - ``a.zeros_like()``
     - :func:`~cyten.tensors.zero_like`

:func:`~cyten.tensors.tensor` converts a numpy (or backend) dense block to a
:class:`~cyten.tensors.SymmetricTensor`. Axis order must match
:attr:`~cyten.tensors.Tensor.legs`: **codomain first, then domain reversed**.
Entries that violate the symmetry raise; Cyten does not silently drop them
the way a ``cutoff`` in :external+tenpy_v1:py:meth:`~tenpy.linalg.np_conserved.Array.from_ndarray`
does.

A vector is a map :math:`\mathbb{C} \to V``, i.e. one leg in the codomain and
an empty domain::

    v = ct.tensor([3.0, 4.0 + 1.0j], codomain=[leg], labels=['i'])

A square operator on :math:`V` (an endomorphism :math:`V \to V`) has
``codomain == domain == [V]``, so :attr:`~cyten.tensors.Tensor.legs` is
``[V, V.dual]``::

    simgaX = ct.tensor([[0.0, 1.0], [1.0, 0.0]],
                       codomain=[leg], domain=[leg], labels=['i', 'j'])

Labels may be a flat list in ``legs`` order (``['i', 'j']``) or a pair
``[codomain_labels, domain_labels]``. For an endomorphism the domain labels
are often the duals of the codomain labels (``'p'`` and ``'p*'``).


Labels
------

String labels still name legs, and almost every operation accepts a label
wherever it accepts an integer index.

Differences:

- Dual labels use a trailing ``*``, same as TeNPy ``Array.conj()``. Combined
  pipes still use ``(a.b)``. Avoid ``( ) . ? ! *`` inside a *single* (not
  combined/dual) label; see :func:`~cyten.tensors.is_valid_leg_label`.
- Set labels with :meth:`~cyten.tensors.LabelledLegs.set_labels` /
  :meth:`~cyten.tensors.LabelledLegs.set_label` /
  :meth:`~cyten.tensors.LabelledLegs.relabel` (in-place). There is no
  ``iset_leg_labels``.
- Read them from :attr:`~cyten.tensors.LabelledLegs.labels`,
  :attr:`~cyten.tensors.Tensor.codomain_labels`,
  :attr:`~cyten.tensors.Tensor.domain_labels`. There is no
  ``get_leg_labels()``.
- Duplicate labels after a contraction are dropped (as in TeNPy). Optional
  ``relabel1`` / ``relabel2`` dicts on :func:`~cyten.tensors.tdot` and
  :func:`~cyten.tensors.outer` rename *before* the operation, which is the
  usual way to avoid collisions.

.. code-block:: python

    A.set_labels(['p', 'vL', 'vR'])           # TeNPy: A.iset_leg_labels(...)
    A.relabel({'vL': 'vR', 'vR': 'vL'})
    A.set_label(0, 'p0')


Contractions
------------

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - TeNPy
     - Cyten
   * - ``npc.tensordot(A, B, axes=[['a'], ['b']])``
     - ``ct.tdot(A, B, 'a', 'b')``
   * - ``npc.tensordot(A, B, axes=2)``
     - no integer default; pass the legs explicitly
   * - ``npc.inner(A, B, axes='labels', do_conj=True)``
     - ``ct.inner(A, B)`` (same (co)domains; daggers ``A``)
   * - ``npc.inner(..., do_conj=False)``
     - ``ct.inner(A, B, do_dagger=False)``
   * - ``npc.outer(A, B)``
     - ``ct.outer(A, B, relabel1=..., relabel2=...)``
   * - ``npc.trace(A, 'a', 'a*')``
     - :func:`~cyten.tensors.trace` (full) or
       :func:`~cyten.tensors.partial_trace` (pairs of legs)
   * - (no map composition)
     - ``A @ B`` / :func:`~cyten.tensors.compose` if
       ``A.domain == B.codomain``

:func:`~cyten.tensors.tdot` **always** takes the contracted legs of each
tensor. It does not accept ``axes=2``. After the contraction, the remaining
legs of the first tensor form the codomain and those of the second form the
domain (reversed, as usual).

:func:`~cyten.tensors.inner` no longer has ``axes='range'`` / ``axes='labels'``.
The two tensors must have matching (co)domains; the product is
:math:`\mathrm{Tr}[A^\dagger \circ B]`. The result is a
:class:`~cyten.block_backends.BlockBackend.Scalar` (see
:ref:`scalars_from_norm_inner_trace`).

``A @ B`` is :func:`~cyten.tensors.compose`, **not** a numpy-style
``tensordot`` over the last / first axis. It is the right tool for applying
an operator to a state::

    M_v = M @ v          # requires M.domain == v.codomain
    M_v = ct.tdot(M, v, 'j', 'i')   # same, by labels


``conj``, ``dagger``, and ``transpose``
---------------------------------------

TeNPy's :external+tenpy_v1:py:meth:`~tenpy.linalg.np_conserved.Array.conj` complex-conjugates the
data, flips every ``qconj``, negates ``qtotal``, and toggles ``*`` on labels
*without* reversing the legs. That is a per-leg dual, not the hermitian
conjugate of a linear map.

Cyten splits those ideas:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Call
     - Meaning
   * - :func:`~cyten.tensors.dagger` / ``A.hc``
     - Hermitian conjugate of the map: swap domain and codomain, reverse and
       dualize ``legs``, toggle ``*`` on labels, complex-conjugate data.
   * - :func:`~cyten.tensors.transpose`
     - Same leg permutation as the dagger, without complex conjugation.
   * - :func:`~cyten.tensors.complex_conj`
     - Element-wise complex conjugation on a
       :class:`~cyten.tensors.DiagonalTensor` (or a scalar). Not the TeNPy
       ``Array.complex_conj``.

For inner products, SVDs, and "apply :math:`H^\dagger`", use ``dagger`` /
``.hc``. Do not write ``A.conj()``; that method does not exist.


Charged operators (``qtotal``)
------------------------------

A TeNPy array with non-trivial ``qtotal`` (for example :math:`S^+` at
conserved :math:`S^z`) cannot be a
:class:`~cyten.tensors.SymmetricTensor`. Create a
:class:`~cyten.tensors.ChargedTensor` instead: a symmetric *invariant part*
plus a hidden charge leg.

.. code-block:: python

    # TeNPy: Sp has qtotal = [2]
    Sp = npc.Array.from_ndarray([[0, 1], [0, 0]], [p_leg, p_leg.conj()])

    # Cyten
    charge_plus = ct.ElementarySpace.from_defining_sectors(sym, [[2]])
    Sp = ct.ChargedTensor.from_dense_block(
        [[0.0, 1.0], [0.0, 0.0]],
        codomain=[p], domain=[p],
        charge=charge_plus,
        labels=['p', 'p*'],
    )

Passing the same dense block to :func:`~cyten.tensors.tensor` raises, because
it is not invariant.

A product of charged operators that *is* invariant (e.g. the two-site
Heisenberg coupling) should be built as a :class:`~cyten.tensors.SymmetricTensor`
directly. :func:`~cyten.tensors.outer` of two :class:`~cyten.tensors.ChargedTensor`\s
is not the usual replacement for ``npc.outer(Sp, Sm)``, because it is not
clear how the two charged legs would have to be connected.

.. note ::

    Cyten introduces :class:`~cyten.models.Coupling` for this: a few-site operator
    stored as one tensor per site, i.e. essentially a mini-MPO. Each factor has
    physical legs ``p, p*`` and virtual legs ``wL, wR``; contracting the virtual
    bonds (which carry the charge that used to live in ``qtotal``) recovers a
    symmetric multi-site map. Factories such as
    :func:`~cyten.models.heisenberg_coupling` build that factorization for you.
    :meth:`~cyten.models.Coupling.to_tensor` contracts it to a
    :class:`~cyten.tensors.SymmetricTensor` if you need the dense few-site
    operator. See :doc:`from_tenpy_models` for the high-level picture
    (Heisenberg, :math:`c^\dagger c`, chiral three-spin, and why MPS no
    longer carry Jordan-Wigner strings).


``combine_legs`` / ``split_legs``
---------------------------------

The operations still exist, as functions::

    # TeNPy
    B = A.combine_legs([['s1', 's2'], ['t1', 't2']], qconj=[+1, -1])
    A2 = B.split_legs()

    # Cyten
    B = ct.combine_legs(A, ['s1', 's2'], ['t1', 't2'])
    A2 = ct.split_legs(B)

Differences:

- Groups are *separate arguments* (``*which_legs``), not a nested list.
- ``qconj`` is :attr:`~cyten.symmetries.spaces.LegPipe.is_dual` via
  ``pipe_dualities``. The default is all ``False``.
- There is no ``new_axes``. Each pipe replaces the first leg of its group;
  other legs may be permuted to make the group contiguous.
- For non-symmetric braiding (anyons), pass ``levels`` to fix braid
  chirality, same as :func:`~cyten.tensors.permute_legs`.
- :func:`~cyten.tensors.combine_to_matrix` combines *all* legs into one
  codomain pipe and one domain pipe. Prefer calling :func:`~cyten.tensors.svd`
  / :func:`~cyten.tensors.eigh` on the uncombined map when you only need a
  decomposition: they treat ``domain → codomain`` as the matrix bipartition.

:func:`~cyten.tensors.squeeze_legs` replaces
:external+tenpy_v1:py:meth:`~tenpy.linalg.np_conserved.Array.squeeze`.
:func:`~cyten.tensors.add_trivial_leg` replaces
:external+tenpy_v1:py:meth:`~tenpy.linalg.np_conserved.Array.add_trivial_leg` (``qconj`` →
``is_dual``, and you choose ``legs_pos`` / ``codomain_pos`` / ``domain_pos``).


Decompositions
--------------

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - TeNPy
     - Cyten
   * - ``U, S, VH = npc.svd(A, inner_labels=['a', 'b'], inner_qconj=+1)``
     - ``U, S, VH = ct.svd(A, new_labels=['a', 'b'], new_leg_dual=False)``
   * - ``E, V = npc.eigh(A)``  (``E`` a 1D numpy array)
     - ``W, V = ct.eigh(A, new_labels=['e'], new_leg_dual=False)``
       (``W`` a :class:`~cyten.tensors.DiagonalTensor`)
   * - ``npc.eig`` / ``eigvals`` / ``eigvalsh`` / ``speigs``
     - no dense non-hermitian eigensolver; use
       :mod:`~cyten.tensors.krylov_based` for iterative problems
   * - ``Q, R = npc.qr(A, inner_labels=..., inner_qconj=...)``
     - ``Q, R = ct.qr(A, new_labels=..., new_leg_dual=...)``
   * - ``npc.expm(A)``
     - :func:`~cyten.tensors.exp`
   * - ``npc.pinv(A)``
     - :func:`~cyten.tensors.pinv`
   * - (truncation in ``tenpy.linalg.truncation``)
     - :func:`~cyten.tensors.truncated_svd` /
       :func:`~cyten.tensors.truncate_singular_values`

``new_leg_dual`` replaces ``inner_qconj``: ``False`` is a ket space
(``qconj=+1``), ``True`` a bra space.
:func:`~cyten.tensors.eigh` requires ``tensor.domain == tensor.codomain``
(the map is an endomorphism). :func:`~cyten.tensors.svd` always returns the
reduced ("economic") factorization.

.. note ::

    Eigenvalues ``W`` from :func:`~cyten.tensors.eigh` and singular values
    ``S`` from :func:`~cyten.tensors.svd` are :class:`~cyten.tensors.DiagonalTensor`\s,
    not 1D numpy arrays. They are sorted only *within each charge block*
    (same ``sort=`` codes as TeNPy for ``eigh``), not globally.
    :meth:`~cyten.tensors.DiagonalTensor.diagonal_as_numpy` recovers a flat
    numpy array if needed, but :class:`~cyten.tensors.DiagonalTensor` also
    supports many element-wise operations natively — for example a comparison
    yields a boolean :class:`~cyten.tensors.DiagonalTensor`, which
    :meth:`~cyten.tensors.Mask.from_DiagonalTensor` turns into a projection
    :class:`~cyten.tensors.Mask` for truncating a leg.


.. _scalars_from_norm_inner_trace:

Scalars from ``norm``, ``inner``, and ``trace``
-----------------------------------------------

:func:`~cyten.tensors.norm`, :func:`~cyten.tensors.inner`, and
:func:`~cyten.tensors.trace` (and a full :func:`~cyten.tensors.partial_trace`)
do not return a Python ``float`` / ``complex``. They return a
:class:`~cyten.block_backends.BlockBackend.Scalar`, which can keep the value
on the same device as the tensors (e.g. a GPU). That design will also be
needed for automatic differentiation in a future version.
Convert to a host number with ``.to_numpy()`` when you need a plain Python or
numpy scalar.

TeNPy's ``npc.norm(A)`` maps to :func:`~cyten.tensors.norm`.


Indexing, scaling, and slicing
------------------------------

Cyten tensors are not numpy arrays.

- ``T[i, j, k]`` returns a single entry, and only if the symmetry
  :attr:`~cyten.symmetries.Symmetry.can_be_dropped`. Slices, boolean masks,
  and ``T[i, j] = x`` are **not** supported.
- Project a leg with a :class:`~cyten.tensors.Mask` and
  :func:`~cyten.tensors.apply_mask` (TeNPy ``iproject``).
- Take a sector / multiplicity slice with :func:`~cyten.tensors.slice_leg`.
- Scale one leg with a :class:`~cyten.tensors.DiagonalTensor` via
  :func:`~cyten.tensors.scale_axis` (TeNPy ``scale_axis`` / ``iscale_axis``).
- Permute legs with :func:`~cyten.tensors.permute_legs` or
  :func:`~cyten.tensors.move_leg`. There is no ``itranspose``; prefer
  addressing legs by label so a transpose is unnecessary.
- Arithmetic ``+``, ``-``, ``*`` scalar, ``/`` scalar still work and return
  new tensors. There are no in-place ``iadd`` / ``iscale_prefactor`` methods;
  :func:`~cyten.tensors.linear_combination` and
  :func:`~cyten.tensors.scalar_multiply` are the explicit forms.

:func:`~cyten.tensors.almost_equal` replaces comparing arrays with
``(A - B).norm() < eps``.


Grid concatenation
------------------

``npc.concatenate`` / ``npc.grid_concat`` / ``npc.grid_outer`` become
:func:`~cyten.tensors.tensor_from_grid`: stack a 2D grid of tensors (``None``
= zero) along the first codomain leg and the last domain leg. Direct sums of
spaces are also available as :class:`~cyten.tensors.DirectSum`.


Backends
--------

TeNPy has a compiled Cython helper for the abelian block sparse format.
Cyten chooses a **tensor backend**
(:class:`~cyten.backends.TensorBackend`; no-symmetry, abelian, or fusion-tree)
from the symmetry and a **block backend**
(:class:`~cyten.block_backends.BlockBackend`; e.g.
:class:`~cyten.block_backends.NumpyBlockBackend` or
:class:`~cyten.block_backends.TorchBlockBackend`) for the dense blocks.
You usually do not pass a backend; :func:`~cyten.backends.get_backend` picks
one. To run on a GPU, use a torch block backend and
:func:`~cyten.tensors.on_device`.


What has no replacement
-----------------------

These TeNPy APIs have no direct Cyten counterpart. In most cases the
surrounding algorithm should be rewritten against the map interface rather
than emulated.

- In-place ``i*`` methods (``itranspose``, ``iscale_axis``, ``iconj``, …).
- ``Array.__setitem__`` and numpy-style slicing / advanced indexing.
- ``qtotal``, ``gauge_total_charge``, ``detect_qtotal``, ``detect_legcharge``.
- ``sort_legcharge``, ``is_completely_blocked``, ``get_block``, ``_data`` /
  ``_qdata`` (backend-private).
- ``eig`` / ``eigvals`` / ``eigvalsh`` / ``polar`` / ``orthogonal_columns``.
- ``from_ndarray(..., cutoff=..., warn_wrong_sector=False)`` (illegal blocks
  are an error).
- Sharing a mutable :external+tenpy_v1:py:class:`~tenpy.linalg.charges.LegCharge` across arrays and
  mutating it. Spaces are immutable enough that you build a new one instead.


Worked example: trivial tensors
-------------------------------

TeNPy (``examples/userguide/a_npc_arrays_triv.py``)::

    import tenpy.linalg.np_conserved as npc

    M = npc.Array.from_ndarray_trivial([[0.0, 1.0], [1.0, 0.0]])
    v = npc.Array.from_ndarray_trivial([2.0, 4.0 + 1.0j])
    v[0] = 3.0
    M_v = npc.tensordot(M, v, axes=[1, 0])
    npc.inner(v.conj(), M_v, axes='range')

Cyten:

.. literalinclude:: examples/from_npc_trivial.py
   :language: python

Item assignment (``v[0] = 3.0``) is gone; put the entries in the constructor.
``v.conj()`` is replaced by the default ``do_dagger=True`` of
:func:`~cyten.tensors.inner`.


Worked example: :math:`U(1)` spin-1/2
-------------------------------------

TeNPy (``examples/userguide/b_npc_arrays.py``) builds :math:`S^z`, :math:`S^\pm`,
a four-leg Hamiltonian by ``outer``, ``combine_legs``, and ``eigh``.

Cyten:

.. literalinclude:: examples/from_npc_u1.py
   :language: python

:math:`S^+` is a :class:`~cyten.tensors.ChargedTensor`. The two-site
Hamiltonian conserves total :math:`S^z`, so it is a
:class:`~cyten.tensors.SymmetricTensor` and can be diagonalized as a map
:math:`p\otimes p \to p\otimes p` without ``combine_legs``.


Porting checklist
-----------------

1. Replace ``import tenpy.linalg.np_conserved as npc`` with ``import cyten as ct``.
2. Replace ``ChargeInfo([1, 2, ...])`` with ``ct.U1() * ct.ZN(2) * ...``.
3. Replace ``LegCharge.from_qflat(...)`` with
   :meth:`~cyten.symmetries.spaces.ElementarySpace.from_basis` (or
   :meth:`~cyten.symmetries.spaces.ElementarySpace.from_defining_sectors`).
4. Decide domain / codomain for every tensor. Operators are maps
   ``codomain=[p], domain=[p]``; vectors are ``codomain=[p]`` with empty domain.
5. Create data with :func:`~cyten.tensors.tensor` /
   :meth:`~cyten.tensors.SymmetricTensor.from_zero` /
   :meth:`~cyten.tensors.ChargedTensor.from_dense_block`. Dense axis order is
   ``legs``.
6. Change ``npc.tensordot(A, B, axes=[...])`` to :func:`~cyten.tensors.tdot`
   with explicit legs, or to ``A @ B`` when composing maps.
7. Change ``A.combine_legs(...)``, ``A.split_legs()``, ``A.conj()``,
   ``A.transpose()`` to :func:`~cyten.tensors.combine_legs`,
   :func:`~cyten.tensors.split_legs`, :func:`~cyten.tensors.dagger`,
   :func:`~cyten.tensors.transpose`.
8. Change ``npc.svd`` / ``eigh`` / ``qr`` inner-leg arguments to
   ``new_labels`` and ``new_leg_dual``. Treat singular values / eigenvalues as
   :class:`~cyten.tensors.DiagonalTensor`.
9. Drop in-place methods, ``qtotal``, and numpy-style item assignment.
10. Reactivate tests for what has been ported.
