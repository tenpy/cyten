# Cyten - a combination of Cytnx and TeNPy

The name Cyten is pronounced like sci-ten, and refers to scientific tensor networks implmented in C++ with a focus on the provided Python bindings, and the python-style library in modern C++.

## About: combining forces of the TeNPy and Cytnx team
This repo originates from a collaboration between the [Cytnx](https://github.com/cytnx-dev/cytnx) and [TeNPy](https://github.com/tenpy/tenpy) developers.
Cytnx is a C++ library for Tensors with Abelian symmetries with pybind11 bindings to Python.
TeNPy is a python library for Tensor networks, in the current version 1.0 also for abelian symmetries only.

## current status
The goal of this repository is to refactor and replace the linear algebra part of TeNPy, combined with the knowledge of Cytnx, to provide a high-level interface for Tensors with block-diagonal structue imposed by symmetries. At the same time, we heavly generalize to also support non-abelian symmetries with a fusion-tree backend. Moreover, we allow to switch the backend for the underlying blocks e.g. to torch to allow an efficient implementation on GPUs.

By now, we have completed a draft of the new Tensor library/interface, including working backends in Python, and are (on this branch) converting this python library code to C++ to avoid unnecessary Python overhead.
The goal is to keep the code structure and python interface mostly intact, and in particular keep the existing and extensive python test suite working while step by step conver individual modules/classes.

## Repo overview
The overview of the repository is given in the file `docs/intro/overview.rst`.

## Setup
Once released, we will provide pre-compiled packages on conda/pip.
Until then, you need to build the package yourself on your local machine, as detailed in `docs/INSTALL.rst`.

## Testing
The python interface is tested with `pytest` run from the `tests/` folder in the repo.
C++ tests can be run via CMake CTest by invoking `ctest` in the `build/` folder, provided that the project has been build with `-DBUILD_TESTING=ON` (the default).

## Documentation
Will eventually be online, but so far you also need to build it locally.
See `README.md` in the `docs/` folder on how - essentially just install `cyten`, run `doxygen` and then `make html` in the `docs/` folder.

## Pre-commit
Formatting and linting can be enabled with pre-commit using git hooks, so you don't need to worry about formatting.

## Code style
Details are given in `docs/guidelines/code_style.rst`.
