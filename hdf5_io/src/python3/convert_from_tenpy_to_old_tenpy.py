#!/usr/bin/env python3
# Try command line argument ``--help`` for options.

import convert
from convert import Hdf5Converter
import hdf5_io

import tenpy  # requirement: can import the new tenpy
import tenpy.linalg.np_conserved as npc
import numpy as np


class Converter(Hdf5Converter):
    """Convert from the "new" TeNPy into the "old" TeNpy (in python 2)."""

    mappings = {}
    mappings[('tenpy.tools.hdf5_io', 'Hdf5Exportable')] = \
        (('tools.hdf5_io', 'Hdf5Exportable'), Hdf5Converter.convert_Hdf5Exportable)

    def convert_array(self, h5gr_orig, subpath_orig, h5gr_new, subpath_new):
        # copy (hardlinks avoid copies)
        h5gr_new["dtype"] = h5gr_orig["dtype"]
        h5gr_new["blocks"] = h5gr_orig["blocks"]
        total_charge = self.load(subpath_orig + "total_charge")
        h5gr_new.attrs["total_charge"] = total_charge

        block_inds = np.asarray(self.load(subpath_orig + "block_inds"), np.uint)
        self.save(block_inds, subpath_new + "block_inds")
        h5gr_new.attrs["block_inds_sorted"] = h5gr_orig.attrs["block_inds_sorted"]
        h5gr_new.attrs["rank"] = rank = self.get_attr(h5gr_orig, "rank")
        h5gr_new.attrs["shape"] = self.get_attr(h5gr_orig, "shape")

        # convert charges
        chinfo = self.load(subpath_orig + "chinfo")
        legs = self.load(subpath_orig + "legs")
        h5gr_new.attrs["num_charges"] = qnumber = int(chinfo.qnumber)
        h5gr_new.attrs["U1_ZN"] = np.asarray(chinfo.mod, int)
        qconj = np.array([leg.qconj for leg in legs], int)
        h5gr_new.attrs["qconj"] = qconj
        leg_charges = []
        for leg in legs:
            q_ind = np.zeros((leg.block_number, 2 + qnumber), int)
            q_ind[:, 0] = leg.slices[:-1]
            q_ind[:, 1] = leg.slices[1:]
            q_ind[:, 2:] = leg.charges
            leg_charges.append(q_ind)
        self.save(leg_charges, subpath_new + "leg_charges")

        labels = self.load(subpath_orig + "labels")
        # convert [str] => {str: int}
        labels_new = {}
        for k, v in enumerate(labels):
            if v is not None:
                labels_new[v] = k
        self.save(labels_new, subpath_new + "labels")

    mappings[('tenpy.linalg.np_conserved', 'Array')] = \
        (('algorithms.linalg.np_conserved', 'array'), convert_array)

    def convert_bc(self, bc):
        """MPS/MPO boundary conditions."""
        # new tenpy:      'finite', 'segment', 'infinite'
        # old tenpy: bc = 'finite', 'segment', 'periodic'
        if bc not in ['finite', 'segment', 'infinite']:
            raise ValueError("unexpected boundary conditions bc={0!r}".format(bc))
        if bc == 'infinite':
            bc = 'periodic'
        return bc

    def convert_MPS(self, h5gr_orig, subpath_orig, h5gr_new, subpath_new):
        L = self.get_attr(h5gr_orig, "L")

        # tensors
        tensors = self.load(subpath_orig + "tensors")
        # convert leg labels and order
        tensors = [
            B.transpose(['p', 'vL', 'vR']).iset_leg_labels(['p', 'b', 'b*']) for B in tensors
        ]
        self.save(tensors, subpath_new + "tensors")
        self.convert_group(h5gr_new["tensors"])  # implicitly calls self.convert_array

        # boundary conditions
        bc = self.convert_bc(self.load(subpath_orig + "boundary_condition"))
        self.save(bc, subpath_new + "boundary_condition")

        # singular values
        SVs = self.load(subpath_orig + "singular_values")
        # new tenpy: bond b left  of site b, always L+1 entries
        # old tenpy: bond b right of site b, length L+1 for segment bc, others L
        SV0, SVs = SVs[0], SVs[1:]  # now entry b is right of site b
        if bc == "segment":
            SVs.append(SV0)
        # else: already correct L entries
        self.save(SVs, subpath_new + "singular_values")

        # canonical form
        form = self.load(subpath_orig + "canonical_form")
        form = [f.copy() for f in form]  # old tenpy expects list of shape (2,) arrays
        self.save(form, subpath_new + "canonical_form")

        # copy metadata
        h5gr_new.attrs["L"] = self.get_attr(h5gr_orig, "L")
        h5gr_new.attrs["max_bond_dimension"] = self.get_attr(h5gr_orig, "max_bond_dimension")

        h5gr_new.attrs["grouped"] = grouped = self.get_attr(h5gr_orig, "grouped")
        if grouped > 1:
            warnings.warn("MPS with grouped sites: " "splitting afterwards not supported")
        if "site_pipes" in h5gr_orig:  # (only if it was originially converted from old tenpy)
            h5gr_new["site_pipes"] = h5gr_orig["site_pipes"]
        else:
            self.save([], subpath_new + "site_pipes")
        h5gr_new.attrs["transfermatrix_keep"] = self.get_attr(h5gr_orig, "transfermatrix_keep")
        self.save(None, subpath_new + "translate_Q1_data")

    mappings[('tenpy.networks.mps', 'MPS')] = \
        (('mps.mps', 'iMPS'), convert_MPS)

    def convert_index_identity(self, Id_LR, bc):
        """Convert  `IdL`->`vL`and `IdR`->`vR` for an MPO."""
        # new tenpy: bond b left  of site b, always L+1 entries
        # old tenpy: bond b right of site b, length L+1 for segment bc, others L
        Id_LR0, Id_LR = Id_LR[0], Id_LR[1:]  # now entry b is left of site b
        if bc == 'infinite' or bc == 'periodic':
            pass  # fine
        elif bc == 'finite':
            pass  # fine
        elif bc == 'segment':
            Id_LR.append(Id_LR0)
        else:
            raise ValueError("don't recognise bc_MPS=" + repr(bc))
        return Id_LR

    def convert_MPO(self, h5gr_orig, subpath_orig, h5gr_new, subpath_new):
        L = self.get_attr(h5gr_orig, "L")
        # tensors
        tensors = self.load(subpath_orig + "tensors")
        # convert leg labels and order
        tensors = [
            B.transpose(['wL', 'wR', 'p', 'p*']).iset_leg_labels(['w', 'w*', 'p', 'p*'])
            for B in tensors
        ]
        self.save(tensors, subpath_new + "tensors")
        self.convert_group(h5gr_new["tensors"])  # implicitly calls self.convert_array

        # boundary conditions
        bc = self.convert_bc(self.load(subpath_orig + "boundary_condition"))
        self.save(bc, subpath_new + "boundary_condition")

        # IdL and IdR
        IdL = self.load(subpath_orig + "index_identity_left")
        IdR = self.load(subpath_orig + "index_identity_right")
        # new tenpy: bond b left  of site b, always L+1 entries
        # old tenpy: bond b right of site b, length L+1 for segment bc, others L
        IdL = self.convert_index_identity(IdL, bc)
        IdR = self.convert_index_identity(IdR, bc)
        self.save(IdL, subpath_new + "index_identity_left")
        self.save(IdR, subpath_new + "index_identity_right")

        # copy metadata
        h5gr_new.attrs["L"] = self.get_attr(h5gr_orig, "L")
        h5gr_new.attrs["max_bond_dimension"] = self.get_attr(h5gr_orig, "max_bond_dimension")
        h5gr_new.attrs["grouped"] = self.get_attr(h5gr_orig, "grouped")
        self.save(None, subpath_new + "translate_Q1_data")

    mappings[('tenpy.networks.mpo', 'MPO')] = \
        (('mps.mpo', 'MPO'), convert_MPO)

    def convert_model(self, h5gr_orig, subpath_orig, h5gr_new, subpath_new):
        assert h5gr_orig.attrs[hdf5_io.ATTR_FORMAT] == hdf5_io.REPR_DICT_SIMPLE
        lat = self.load(subpath_orig + "lat")
        sites = lat.mps_sites()
        L = len(sites)
        bc = lat.bc_MPS
        self.save(self.convert_bc(bc), subpath_new + "boundary_condition")

        if "H_bond" in h5gr_orig:
            # have nearest-neighbor Hamiltonian
            H_bond = list(self.load(subpath_orig + "H_bond"))
            # new tenpy: always L entries, entry b is sites (b-1, b), None if not to be used.
            # old tenpy: always L entries, entry b is sites (b, b+1), for finite: last entry 0.
            H_bond = H_bond[1:] + [H_bond[0]]  # now entry b is sites (b, b+1)
            if bc == 'finite':
                p0 = H_bond[-2].get_leg('p1')
                p1 = H_bond[0].get_leg('p0')
                H_bond[-1] = npc.zeros([p0, p1, p0.conj(), p1.conj()],
                                       labels=['p0', 'p1', 'p0*', 'p1*'])
            H_bond = [H.transpose(['p0', 'p1', 'p0*', 'p1*']) for H in H_bond]
            self.save(H_bond, subpath_new + "H")
            self.convert_group(h5gr_new["H"])  # convert npc Arrays
        else:
            self.save(None, subpath_new + "H")

        # convert MPO directly to extract the converted tensors
        # (conversion might also have happend before starting the conversion of the model!)
        H_mpo_gr = self.convert_group(h5gr_orig['H_MPO'])  # convert MPO: does most of the job
        h5gr_new["H_mpo"] = H_mpo_gr["tensors"]
        h5gr_new["index_identity_left"] = H_mpo_gr["index_identity_left"]
        h5gr_new["index_identity_right"] = H_mpo_gr["index_identity_right"]
        h5gr_new["translate_Q1_data"] = H_mpo_gr["translate_Q1_data"]
        h5gr_new.attrs['grouped'] = self.get_attr(H_mpo_gr, "grouped")

        states = [s.state_labels for s in sites]
        self.save(states, subpath_new + "states")
        onsite_ops = {}
        for i, s in enumerate(sites):
            for opname in s.opnames:
                op_list = onsite_ops.setdefault(opname, [None] * L)
                op_list[i] = s.get_op(opname)
        self.save(onsite_ops, subpath_new + "onsite_operators")
        self.save({}, subpath_new + "bond_operators")  # doesn't exist (yet?) in new TeNPy
        self.save(False, subpath_new + "add_conj")  # doesn't exist (yet?) in new TeNPy
        h5gr_new.attrs['L'] = L
        dims = np.array([s.dim for s in sites])
        self.save(dims, subpath_new + "dimensions")

    for _model in [
        ('tenpy.models.model', 'Model'),  # base class
        ('tenpy.models.model', 'NearestNeighborModel'),  # base class
        ('tenpy.models.model', 'MPOModel'),  # base class
        ('tenpy.models.model', 'CouplingMPOModel'),  # base class
            # and derived classes defined in TeNPy
            # TODO: hard-coded list of models
            # needs to be updated if new models are implemented in TeNPy
        ('tenpy.models.fermions_spinless', 'FermionModel'),
        ('tenpy.models.fermions_spinless', 'FermionChain'),
        ('tenpy.models.haldane', 'BosonicHaldaneModel'),
        ('tenpy.models.haldane', 'FermionicHaldaneModel'),
        ('tenpy.models.hofstadter', 'HofstadterFermions'),
        ('tenpy.models.hofstadter', 'HofstadterBosons'),
        ('tenpy.models.hubbard', 'BoseHubbardModel'),
        ('tenpy.models.hubbard', 'BoseHubbardChain'),
        ('tenpy.models.hubbard', 'FermiHubbardModel'),
        ('tenpy.models.hubbard', 'FermiHubbardChain'),
        ('tenpy.models.spins_nnn', 'SpinChainNNN'),
        ('tenpy.models.spins_nnn', 'SpinChainNNN2'),
        ('tenpy.models.spins', 'SpinModel'),
        ('tenpy.models.spins', 'SpinChain'),
        ('tenpy.models.tf_ising', 'TFIModel'),
        ('tenpy.models.tf_ising', 'TFIChain'),
        ('tenpy.models.toric_code', 'ToricCode'),
        ('tenpy.models.xxz_chain', 'XXZChain'),
        ('tenpy.models.xxz_chain', 'XXZChain2'),
    ]:
        mappings[_model] = (('models.model', 'model'), convert_model)
    del _model


if __name__ == "__main__":
    args = convert.parse_args(converter_cls=Converter)
    convert.main(args)
