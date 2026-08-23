Cyten
-----

.. image:: https://img.shields.io/github/last-commit/tenpy/cyten
   :alt: GitHub last commit
   :target: https://github.com/tenpy/cyten
.. image:: https://github.com/tenpy/cyten/actions/workflows/pytest_numpy.yml/badge.svg?branch=main
   :alt: Pytest
   :target: https://github.com/tenpy/cyten/actions/
.. image:: https://img.shields.io/github/issues/tenpy/cyten
   :alt: GitHub issues
   :target: https://github.com/tenpy/cyten/issues

.. warning::

   Cyten is still in **alpha** and under heavy development.
   The API is not stable and may change without notice.

Cyten (pronounced like "sci-ten") is a C++ library with Python bindings for tensors whose
block-sparse structure is imposed by symmetries.
It serves as the linear algebra backend for `TeNPy <https://github.com/tenpy/tenpy>`_,
a Python library for the simulation of strongly correlated quantum systems with tensor networks.

The library supports both abelian and non-abelian symmetries (via a fusion-tree backend).
The dense blocks can be handled by different backends, for example NumPy or PyTorch
(including GPUs).

How do I get set up?
--------------------
Create a conda environment from the provided ``environment.yml`` and install Cyten in
editable mode::

    conda env create -f environment.yml
    conda activate cyten
    pip install -e .

Further details and alternative methods can be found in the file `docs/INSTALL.rst`.
The latest version of the source code can be obtained from https://github.com/tenpy/cyten.

How to read the documentation
-----------------------------
The **documentation is hosted** at https://cyten.readthedocs.io/.
It is roughly split into a "user guide" with additional explanations, and a full
"reference" of the Python interface and the C++ API.

The documentation is based on Python's docstrings, C++ comments, and some additional
``*.rst`` files located in the folder `docs/` of the repository.
All documentation is formatted as `reStructuredText <http://www.sphinx-doc.org/en/stable/rest.html>`_,
which means it is quite readable in the source plain text, but can also be converted to other formats.
If you like it simple, you can just use interactive python `help()`, Python IDEs of your choice
or jupyter notebooks, or just read the source.
Moreover, the documentation gets converted into HTML using `Sphinx <http://www.sphinx-doc.org>`_,
and is made available online at https://cyten.readthedocs.io/.
The big advantages of the (online) HTML documentation are a lot of cross-links between different
functions, and even a search function.
If you prefer yet another format, you can try to build the documentation yourself, as described
in ``docs/README.md``.

I found a bug
-------------
You might want to check the `github issues <https://github.com/tenpy/cyten/issues>`_,
if someone else already reported the same problem.
To report a new bug, just `open a new issue <https://github.com/tenpy/cyten/issues/new>`_ on github.
If you already know how to fix it, you can just create a pull request :)

License
-------
The code is licensed under Apache v2 given in the file ``LICENSE`` of the repository.
