# Cytnx version 2.0

The name cytnx is pronounced like sci-tens, and refers to scientific tensor networks implmented in C++ with a focus on the provided Python bindings, and the python-style library in modern C++.

## About: combining forces of the TeNPy and Cytnx team
This repo originates from a collaboration between the [Cytnx](https://github.com/cytnx-dev/cytnx) and [TeNPy](https://github.com/tenpy/tenpy) developers.
Cytnx is a C++ library for Tensors with Abelian symmetries with pybind11 bindings to Python. 
The goal is to use that as a basis to translate the refactoring and implementation of non-Abelian symmetries from the `v2_alpha` branch in the tenpy repository (currenlty in pure Python) into C++, thereby extending the capabilities of the cytnx library, and providing a backend for TeNpy which will then focus on the higher-levels (defining MPS and algorithms like DMRG  etc).
At the same time, the code from cytnx to be included will be refactored and cleaned a bit.

## Setup
Once released, we will provide pre-compiled packages on conda/pip. Until then, you need to build the package yourself on your local machine.
For that, install the following requirements (currently only tested on standard linux distros like ubuntu - no Windows support yet, use WSL):

- C++ compiler with at least C++17 standard  (can be installed manually with `conda install -c conda-forge compilers` if needed)
- CMake, make
- boost library (only required for the intrusive pointer - can we get rid of that requirement?)
- Python >= 3.9, with numpy>=2.0, scipy and a few other python packages as listed `environment.yml`
- scikit-build

The easiest way to install is to create a conda envrironment from the `environment.yml` to install all requirements
and then pip install the package (from the base folder of the repository):

```
conda env create -f environment.yml -n cytnx_v2
conda activate cytnx_v2
conda install -c conda-forge _openmp_mutex=*=*_llvm # on Linux/WSL only
conda install -c conda-forge llvm-openmp # on MacOS only
pip install -v .
```

If needed, you can add defines for the CMake build as options to pip, e.g. `pip install -v -C cmake.define.=ON .`.


For a debug build, you can even enable automatic rebuild upon python import:
```
pip install -v --no-build-isolation -C editable.rebuild=true -e .
```

## Testing
The python interface is tested with `pytest` run from the base folder in the repo.


## Documentation
C++ functions are documented with doxygen in `docs/doxygen`.
Python parts are documented with `sphinx` from the base `docs/` folder.


