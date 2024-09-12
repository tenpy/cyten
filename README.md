# Cyten - a combination of Cytnx and TeNPy

The name Cyten is pronounced like sci-ten, and refers to scientific tensor networks implmented in C++ with a focus on the provided Python bindings, and the python-style library in modern C++.

## About: combining forces of the TeNPy and Cytnx team
This repo originates from a collaboration between the [Cytnx](https://github.com/cytnx-dev/cytnx) and [TeNPy](https://github.com/tenpy/tenpy) developers.
Cytnx is a C++ library for Tensors with Abelian symmetries with pybind11 bindings to Python. 
The goal is to use that as a basis to translate the refactoring and implementation of non-Abelian symmetries from the `v2_alpha` branch in the tenpy repository (currenlty in pure Python) into C++, thereby extending the capabilities of the cytnx library, and providing a backend for TeNpy which will then focus on the higher-levels (defining MPS and algorithms like DMRG  etc).
At the same time, the code from cytnx to be included will be refactored and cleaned a bit.

## Setup
Once released, we will provide pre-compiled packages on conda/pip.
Until then, you need to build the package yourself on your local machine, as detailed in `docs/INSTALL.rst`.

## Testing
The python interface is tested with `pytest` run from the `tests/` folder in the repo.
To test the C++ code, run `gtest` (from GoogleTest).

## Documentation
Will eventually be online, but so far you also need to build it locally.
See `README.md` in the `docs/` folder on how - essentially just install `cyten`, run `doxygen` and then `make html` in the `docs/` folder.
