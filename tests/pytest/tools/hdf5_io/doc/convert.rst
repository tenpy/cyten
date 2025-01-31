Converting files: from the old TeNPy to the new TeNPy (and vice versa)
======================================================================

Here, "new TeNPy" referst to https://github.com/tenpy/tenpy, and
"old TeNPy" refers to the precurser by Mike Zaletel, Frank Pollmann and Roger Mong, originally written in Python 2.

You need to be able to export/import to HDF5 from both the old and the new TeNPy.
Hence you need to have `h5py` installed for both Python 2 and Python 3 (the ones where you can import TeNPy, respectively).
Moreover, you need to have up-to-date versions of *both* the new and old TeNPy, which support the HDF5 export.

Given that, you need to do the following steps.

1. Gather and export the data you want to convert to HDF5::

    # in Python2 with old TeNPy
    import h5py

    from tools import hdf5_io
    from models.spin_chain import spin_chain_model

    M = spin_chain_model({'conserve_Sz': True})
    initial_state = np.array([M.dn, M.dn])
    psi = iMPS.product_imps(M.d, initial_state, dtype=float, conserve=M, bc='periodic')
    data = {'model': M, 'psi': psi}
    
    with h5py.File('exported.h5', 'w') as f:
        hdf5_io.save_to_hdf5(f, data)

2. Call the appropriate conversion script with Python 3::

    # in terminal
    python3 convert_from_old_tenpy_to_tenpy.py exported.h5

3. Import the data in the new TeNPy::

    # in Python3 with new TeNPy

    import h5py

    with h5py.File('exported.h5', 'r') as f:
        data = hdf5_io.load_from_hdf5(f)
    psi = data['psi']  # do whatever you need to do with the data...
    print(psi.entanglement_entropy())
