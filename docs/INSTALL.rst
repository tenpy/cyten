Installation instructions
=========================

.. todo ::
   This doesn't work yet, we still need to set this up, once we have the first (beta) release.
   Meanwhile, install the build requirements (including PyTorch; see below) and run
   ``pip install --no-build-isolation .`` from the top folder of the repo.

With the `conda package manager <https://docs.conda.io>`_ you can install python with::

    conda install --channel=conda-forge cyten  # TODO doesn't work yet

If you don't have conda, but you have `pip <https://pip.pypa.io>`_, you can::

    pip install cyten   # TODO doesn't work yet

Building from source
++++++++++++++++++++

To build cyten locally on your machine, install the following requirements
(currently only tested on standard linux distros like ubuntu - no Windows support yet, use WSL):

- C++ compiler with at least C++17 standard. In a conda env this **must** be
  conda-forge ``cxx-compiler`` (included in ``environment.yml``), not a newer
  system ``g++``. Python already loads conda's ``libstdc++``; objects compiled
  with a newer GCC fail at ``import cyten`` with missing ``GLIBCXX_*`` symbols.
- CMake, make
- Python >= 3.12, with numpy>=2.0, scipy and a few other python packages as listed in `environment.yml`
- **PyTorch** (``torch`` / conda-forge ``pytorch``): required at build and runtime.
  CMake finds libtorch via ``torch.utils.cmake_prefix_path`` and links ``cyten._core`` against
  the same shared libraries as ``import torch``, so PyTorch must be installed in the build
  environment before compiling.
  Avoid a plain isolated ``pip install pytorch`` against PyPI ``torch`` on machines without the
  CUDA toolkit: the default Linux wheels are CUDA-enabled and CMake's ``find_package(Torch)``
  then fails looking for CUDA.
  Prefer conda-forge ``pytorch`` (CPU) or install a CPU wheel, e.g.
  ``pip install torch --index-url https://download.pytorch.org/whl/cpu``.
- scikit-build

The easiest way to install all of those is to create a conda environment from the `environment.yml`
(which includes PyTorch) and then pip-install the package
(use `docs/environment.yml` if you plan to build the documentation as well)::

    conda env create -f environment.yml -n cyten
    conda activate cyten
    conda install -c conda-forge _openmp_mutex=*=*_llvm # on Linux/WSL only
    conda install -c conda-forge llvm-openmp # on MacOS only
    pip install -v --no-build-isolation .

Use ``--no-build-isolation`` so the build sees the conda-installed PyTorch (and other build
deps) instead of resolving them in an isolated pip environment.

If needed, you can add defines for the CMake build as options to pip, e.g. `pip install -v -C cmake.define.=ON .`.


For a debug build, you can even enable automatic rebuild upon python import::

    pip install -v --no-build-isolation -C editable.rebuild=true -e .
