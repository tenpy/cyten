# Cyten documentation

C++ functions are documented with [doxygen](https://doxygen.nl),
python parts with [sphinx](https://www.sphinx-doc.org).

## Building the docs

1. Install Sphinx/Breathe/doxygen and the packages in [`docs/environment.yml`](environment.yml)
   (a full `pip install .` / C++ build is **not** required).
2. From the repo root, generate version metadata and the autodoc stub if needed:
   ```bash
   python -c "from setuptools_scm import get_version; get_version(write_to='cyten/_version.py')"
   python scripts/generate_core_stubs.py   # or --from-doxygen if pybind/docstrings/ is empty
   ```
   [`docs/conf.py`](conf.py) will also generate the stub automatically when
   `cyten._core` is not a compiled extension.
3. Run doxygen from the `docs/` folder (Breathe C++ API pages).
4. Run sphinx: `make html` or `sphinx-build -M html . build_docs` from `docs/`.

HTML output (including the Doxygen C++ API reference) is under `docs/build_docs/html/`.

If you already have a compiled `cyten._core` extension installed, Sphinx uses that
instead of the stub.
