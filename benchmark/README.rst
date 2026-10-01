Cyten backend benchmarks
========================

Standalone scripts that assume ``cyten`` is installed and ``import cyten`` works.
They sweep tensor size across selectable **symmetry backends**, **block backends**,
and **devices**, timing contraction, decompositions, and leg combine/split.

Results are written as JSON for analysis in ``plot_results.ipynb``
(requires Jupyter / matplotlib; not part of cyten install requirements).


Quick start
-----------

From the repository root::

    python benchmark/run_benchmarks.py \
        --ops tdot,svd,eigh,combine_legs,split_legs \
        --cases nosym,u1,su2 \
        --symmetry-backends no_symmetry,abelian,fusion_tree \
        --block-backends numpy,torch \
        --devices cpu \
        --dims 16,32,64,128,256 \
        --warmup 2 --repeats 5 \
        --numpy-ref \
        --output benchmark/results/run.json

Or as a module::

    python -m benchmark.run_benchmarks --devices cpu --dims 32,64 --numpy-ref

Then open ``benchmark/plot_results.ipynb`` and point ``RESULT_PATHS`` at the JSON file.


What is compared
----------------

Cases (deterministic legs, total dimension near ``--dims``):

======= ============ ================================ ========================================
Case    Symmetry     Allowed symmetry backends        Leg construction
======= ============ ================================ ========================================
nosym   NoSymmetry   no_symmetry                      ``from_trivial_sector(dim)``
u1      U1           abelian, fusion_tree             charges ``-2..2``, equal multiplicities
su2     _SU2          fusion_tree                      sectors ``0,1,2`` (j=0,½,1), equal mults
======= ============ ================================ ========================================

Comparing ``u1`` + ``abelian`` vs ``u1`` + ``fusion_tree`` isolates symmetry-backend
overhead on the same block structure. ``nosym`` is the dense cyten baseline.

Operations:

* ``tdot`` — square-map contraction
* ``svd`` / ``eigh`` — decompositions on square maps (``eigh`` uses ``A + A.hc``)
* ``combine_legs`` / ``split_legs`` — rank-4 tensors (two legs per (co)domain)

Dense NumPy reference (``--numpy-ref``):

* Emits extra JSON rows with ``impl="numpy"``, ``symmetry_backend="dense"``,
  ``block_backend="numpy"``, ``device="cpu"``
* Timed once per ``(op, case, dim, dtype)`` (deduped across cyten backend/device combos)
* Uses ``to_numpy(understood_braiding=True)`` then ``numpy.tensordot`` /
  ``numpy.linalg.svd`` / ``eigh`` / transpose+reshape combine/split

Block backends / aliases (via ``cyten.get_backend``):

* ``numpy`` / ``cpu`` — NumPy on CPU
* ``torch`` — PyTorch (pass ``--devices cpu``, ``cuda:0``, or ``mps:0``)
* ``gpu`` — Torch with default CUDA device (pair with ``--devices cuda:0``)
* ``apple_silicon`` — Torch MPS (pair with ``--devices mps:0``)

Incompatible combinations (e.g. ``numpy`` + ``cuda``, ``su2`` + ``abelian``) are skipped.


CLI flags
---------

==================== ===========================================================
Flag                 Meaning
==================== ===========================================================
``--ops``            ``tdot``, ``svd``, ``eigh``, ``combine_legs``, ``split_legs``
``--cases``          ``nosym``, ``u1``, ``su2``
``--symmetry-backends``  ``no_symmetry``, ``abelian``, ``fusion_tree``
``--block-backends`` ``numpy``, ``torch``, ``cpu``, ``gpu``, ``apple_silicon``
``--devices``        e.g. ``cpu``, ``cuda:0``, ``mps:0``
``--dims``           Target leg dimensions (actual dim may differ for charged cases)
``--dtype``          ``float64`` (default), ``float32``, ``complex64``, ``complex128``
``--warmup`` / ``--repeats``  Timing iterations
``--numpy-ref``      Also time dense NumPy analogues (deduped)
``--output``         JSON path (default ``benchmark/results/run.json``)
``--list-devices``   Print availability of requested devices and exit
==================== ===========================================================


Interpreting results
--------------------

JSON layout::

    {
      "metadata": {
        "cyten_version": "...",
        "cyten_commit_id": "...",
        "hostname": "...",
        "started_at": "YYYY-MM-DDTHH:MM",
        "compile": { "cmake_build_type": "Debug", "...": "..." },
        "..."
      },
      "results": [ { "op": "tdot", "impl": "cyten", "...": "..." }, ... ]
    }

Each result row includes ``impl`` (``cyten`` or ``numpy``), ``mean`` / ``median`` /
``min`` / ``stdev`` wall times, plus ``actual_dim``, ``num_blocks``, and backend metadata.

* At **small** ``actual_dim``, symmetry-backend bookkeeping often dominates: expect
  ``fusion_tree`` slower than ``abelian``, and both slower than ``nosym``.
* At **large** ``actual_dim``, block GEMM/SVD cost dominates: Torch CUDA/MPS can
  pull ahead of NumPy/CPU when the dense blocks are large enough.
* With ``--numpy-ref``, compare cyten rows to the matching ``impl=numpy`` baseline
  for the same ``op`` / ``case`` / ``actual_dim``.


Layout
------

* ``common.py`` — timing, device checks, JSON I/O
* ``cases.py`` — deterministic spaces / tensor builders
* ``dense_numpy.py`` — dense export and NumPy combine/split helpers
* ``bench_*.py`` — per-op runners
* ``run_benchmarks.py`` — CLI
* ``plot_results.ipynb`` — plots
* ``results/`` — generated JSON (gitignored)
