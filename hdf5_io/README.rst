Input and output to and from hdf5
=================================

This repository is about the import and export features to HDF5 files included in `TeNPy <https://github.com/tenpy/tenpy>`_,
in particular the module ``tenpy.tools.hdf5_io``.
The introduction/specification of the format is at https://tenpy.readthedocs.io/en/latest/intro/input_output.html.

Since the general technique is not bound to the particular classes of TeNPy, I've separated the code into this repository.

The file ``src/python3/hdf5_io.py`` should just be the corresponding file of the ``tenpy.tools.hdf5_io`` module in `TeNPy`_.

To make the export and import feature really usefull, we need scripts to convert the data such that it can be
imported/exported with other libraries except TeNPy.
We keep scripts for such a conversion in this repository, see the files ``src/python3/conversion*.py``.
