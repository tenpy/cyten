#!/usr/bin/env python3
# Try command line argument ``--help`` for options.

import convert
from convert import Hdf5Converter
import hdf5_io

import tenpy  # requirement: can import the new tenpy
import tenpy.linalg.np_conserved as npc
import numpy as np


class Converter(Hdf5Converter):
    """Convert from the "old" TeNpy (in python 2) to the "new" TeNPy."""

    mappings = {}
    mappings[('tools.hdf5_io', 'Hdf5Exportable')] = \
        (('tenpy.tools.hdf5_io', 'Hdf5Exportable'), Hdf5Converter.convert_Hdf5Exportable)

    def convert_array(self, h5gr_orig, subpath_orig, h5gr_new, subpath_new):
        # copy (hardlinks avoid copies)
        h5gr_new["dtype"] = h5gr_orig["dtype"]
        h5gr_new["blocks"] = h5gr_orig["blocks"]
        total_charge = self.get_attr(h5gr_orig, "total_charge")
        self.save(total_charge, subpath_new + "total_charge")
        block_inds = np.asarray(self.load(subpath_orig + "block_inds"), np.intp)
        self.save(block_inds, subpath_new + "block_inds")
        h5gr_new.attrs["block_inds_sorted"] = h5gr_orig.attrs["block_inds_sorted"]
        h5gr_new.attrs["rank"] = rank = self.get_attr(h5gr_orig, "rank")
        h5gr_new.attrs["shape"] = self.get_attr(h5gr_orig, "shape")

        # convert charges
        num_q = self.get_attr(h5gr_orig, "num_charges")
        mod_q = self.get_attr(h5gr_orig, "U1_ZN")
        leg_charges = self.load(subpath_orig + "leg_charges")
        qconj = self.get_attr(h5gr_orig, "qconj")

        # ChargeInfo
        chinfo = npc.ChargeInfo(mod_q)  # no names available
        assert chinfo.qnumber == num_q
        # legs: convert leg_charges, qconj into list of LegCharge instances
        legs_new = []
        for q_ind, q_conj in zip(leg_charges, qconj):
            slices = np.concatenate((q_ind[:, 0], [q_ind[-1, 1]]))
            charges = q_ind[:, 2:]
            leg = npc.LegCharge(chinfo, slices, charges, q_conj)
            legs_new.append(leg)
        self.save(chinfo, subpath_new + "chinfo")
        self.save(legs_new, subpath_new + "legs")

        labels = self.load(subpath_orig + "labels")
        # convert {str: int} -> [str]
        labels_new = [None] * rank
        if labels is not None:
            for k, v in labels.items():
                labels_new[v] = k
        self.save(labels_new, subpath_new + "labels")

    mappings[('algorithms.linalg.np_conserved', 'array')] = \
        (('tenpy.linalg.np_conserved', 'Array'), convert_array)

    def convert_bc(self, bc):
        """MPS/MPO boundary conditions."""
        # old tenpy: bc = 'finite', 'segment', 'periodic'
        # new tenpy:      'finite', 'segment', 'infinite'
        if bc == 'periodic':
            bc = 'infinite'
        if bc not in ['finite', 'segment', 'infinite']:
            raise ValueError("unexpected boundary conditions bc={0!r}".format(bc))
        return bc

    def convert_MPS(self, h5gr_orig, subpath_orig, h5gr_new, subpath_new):
        L = self.get_attr(h5gr_orig, "L")
        if self.get_attr(h5gr_orig["translate_Q1_data"], hdf5_io.ATTR_TYPE) != hdf5_io.REPR_NONE:
            raise ValueError("TODO: non-trivial 'translate_Q1_data' not supported in new TeNPy")
        # tensors
        tensors = self.convert_group(h5gr_orig["tensors"])  # implicitly calls self.convert_array
        tensors = self.load(tensors.name)
        # convert leg labels and order
        tensors = [
            B.transpose(['b', 'p', 'b*']).iset_leg_labels(['vL', 'p', 'vR']) for B in tensors
        ]
        chinfo = tensors[0].legs[0].chinfo
        for B in tensors:
            for leg in B.legs:
                leg.chinfo = chinfo  # share chinfo
        self.save(tensors, subpath_new + "tensors")
        self.save(chinfo, subpath_new + "chinfo")
        # make trivial sites with just the physical leg
        sites = [tenpy.networks.site.Site(B.legs[1]) for B in tensors]
        self.save(sites, subpath_new + "sites")

        # boundary conditions
        bc = self.convert_bc(self.load(subpath_orig + "boundary_condition"))
        self.save(bc, subpath_new + "boundary_condition")

        # singular values
        SVs = self.load(subpath_orig + "singular_values")
        # new tenpy: bond b left  of site b, always L+1 entries
        # old tenpy: bond b right of site b, length L+1 for segment bc, others L
        SVs = [None] + SVs  # now entry b is left of site b
        # fine for infinite bc with L+1 entries
        if bc == 'infinite':
            SVs[0] = SVs[-1]
        elif bc == 'finite':
            SVs[0] = SVs[-1] = np.ones([1])
        else:  # segment
            SVs[0] = SVs.pop()
        self.save(SVs, subpath_new + "singular_values")

        # canonical form
        form = self.load(subpath_orig + "canonical_form")
        if type(form) == np.ndarray:
            assert form.shape == (2, )
            form = np.array([form] * L)  # old tenpy with single 2-tuple, valid for all sites
        else:  # old tenpy: list of array
            form = np.array(form)
            assert form.shape == (L, 2)
        self.save(form, subpath_new + "canonical_form")

        # copy metadata
        h5gr_new.attrs["L"] = self.get_attr(h5gr_orig, "L")
        h5gr_new.attrs["max_bond_dimension"] = self.get_attr(h5gr_orig, "max_bond_dimension")

        h5gr_new.attrs["norm"] = 1  # required by new TeNPy
        h5gr_new.attrs["grouped"] = grouped = self.get_attr(h5gr_orig, "grouped")
        if grouped > 1:
            warnings.warn("MPS with grouped sites: " "splitting after conversion not supported")
        # still copy site_pipes: allow converting back to old tenpy for splitting
        h5gr_new["site_pipes"] = h5gr_orig["site_pipes"]
        h5gr_new.attrs["transfermatrix_keep"] = self.get_attr(h5gr_orig, "transfermatrix_keep")

    mappings[('mps.mps', 'iMPS')] = (('tenpy.networks.mps', 'MPS'), convert_MPS)

    def convert_index_identity(self, Id_LR, bc):
        """Convert `vL`-> `IdL` and `vR`->`IdR` for an MPO."""
        # new tenpy: bond b left  of site b, always L+1 entries
        # old tenpy: bond b right of site b, length L+1 for segment bc, others L
        Id_LR = [None] + list(Id_LR) # now entry b is left of site b
        if bc == 'infinite' or bc == 'periodic':
            Id_LR[0] = Id_LR[-1]
        elif bc == 'finite':
            Id_LR[0] = Id_LR[-1] = 0
        elif bc == 'segment':
            Id_LR[0] = Id_LR.pop()
        else:
            raise ValueError("don't recognise bc_MPS=" + repr(bc))
        return Id_LR

    def convert_MPO(self, h5gr_orig, subpath_orig, h5gr_new, subpath_new):
        L = self.get_attr(h5gr_orig, "L")
        if self.get_attr(h5gr_orig["translate_Q1_data"], hdf5_io.ATTR_TYPE) != hdf5_io.REPR_NONE:
            warnings.warn("non-trivial 'translate_Q1_data' not supported in new TeNPy")
        # tensors: (implicitly) call above conversion function
        tensors = self.convert_group(h5gr_orig["tensors"])  # convert the arrays
        tensors = self.load(tensors.name)
        # convert leg labels.
        tensors = [
            W.transpose(['w', 'w*', 'p', 'p*']).iset_leg_labels(['wL', 'wR', 'p', 'p*'])
            for W in tensors
        ]
        chinfo = tensors[0].legs[0].chinfo
        for W in tensors:
            for leg in W.legs:
                leg.chinfo = chinfo  # share chinfo
        p_legs = [W.legs[2] for W in tensors]
        self.save(tensors, subpath_new + "tensors")
        self.save(chinfo, subpath_new + "chinfo")
        # make trivial sites with just the physical leg
        sites = [tenpy.networks.site.Site(W.legs[2]) for W in tensors]
        self.save(sites, subpath_new + "sites")

        # boundary conditions
        bc = self.convert_bc(self.load(subpath_orig + "boundary_condition"))
        self.save(bc, subpath_new + "boundary_condition")

        # IdL and IdR
        IdL = list(self.load(subpath_orig + "index_identity_left"))
        IdR = list(self.load(subpath_orig + "index_identity_right"))
        IdL = self.convert_index_identity(IdL, bc)
        IdR = self.convert_index_identity(IdR, bc)
        self.save(IdL, subpath_new + "index_identity_left")
        self.save(IdR, subpath_new + "index_identity_right")

        # copy metadata
        h5gr_new.attrs["L"] = self.get_attr(h5gr_orig, "L")
        h5gr_new.attrs["max_bond_dimension"] = self.get_attr(h5gr_orig, "max_bond_dimension")
        h5gr_new.attrs["grouped"] = self.get_attr(h5gr_orig, "grouped")
        self.save(None, subpath_new + "max_range")  # unknown

    mappings[('mps.mpo', 'MPO')] = (('tenpy.networks.mpo', 'MPO'), convert_MPO)

    def convert_model(self, h5gr_orig, subpath_orig, h5gr_new, subpath_new):
        L = self.get_attr(h5gr_orig, "L")
        if self.load(subpath_orig + "add_conj"):
            raise NotImplementedError("add_conj=True corresponds to `explicit_plus_hc` flag")
        if self.get_attr(h5gr_orig["translate_Q1_data"], hdf5_io.ATTR_TYPE) != hdf5_io.REPR_NONE:
            warnings.warn("non-trivial 'translate_Q1_data' not supported in new TeNPy")
        # convert the npc arrays
        H_mpo_tensors = self.convert_group(h5gr_orig["H_mpo"])
        H_mpo_tensors = self.load(H_mpo_tensors.name)
        assert H_mpo_tensors is not None
        # construct sites
        chinfo = H_mpo_tensors[0].chinfo
        states = self.load(subpath_orig + "states")
        onsite_ops = self.convert_group(h5gr_orig["onsite_operators"])  # converts npc Arrays
        onsite_ops = self.load(onsite_ops.name)
        bond_ops = self.convert_group(h5gr_orig["bond_operators"])  # converts npc Arrays
        h5gr_new["bond_ops"] = bond_ops  # just hardcopy the hdf5 group
        # TODO: so far, there is no well-defined format for bond operators in the new TeNPy.

        # generate sites
        sites = []
        for i in range(L):
            H_mpo_tensors[i] = Hi = H_mpo_tensors[i].transpose(['w', 'w*', 'p', 'p*'])  # copy
            Hi.iset_leg_labels(['wL', 'wR', 'p', 'p*'])
            p_leg = Hi.legs[2]
            p_leg.chinfo = chinfo
            Hi.legs[3].test_contractible(p_leg)
            Hi.legs[3] = p_leg.conj()
            d = p_leg.ind_len
            state_labels = [None] * d
            for name, j in states[i].items():
                state_labels[j] = name
            local_site_ops = {}
            for name, op_list in onsite_ops.items():
                op = op_list[i].transpose(['p', 'p*'])
                op.legs[0].test_equal(p_leg)
                op.chinfo = chinfo
                op.legs = [p_leg, p_leg.conj()]
                local_site_ops[name] = op
            if 'Id' in local_site_ops:
                del local_site_ops['Id']
            site = tenpy.networks.site.Site(p_leg, state_labels, **local_site_ops)
            sites.append(site)

        # get `data` dictionary with content to be saved as model.__dict__
        data = {}
        data['grouped'] = grouped = self.get_attr(h5gr_orig, "grouped")
        # lat
        bc_MPS = self.convert_bc(self.load(subpath_orig + "boundary_condition"))
        bc = [("periodic" if bc_MPS == 'infinite' else "open")]
        data['lat'] = tenpy.models.lattice.TrivialLattice(sites, bc_MPS=bc_MPS, bc=bc)
        # H_MPO
        IdL = self.load(subpath_orig + "index_identity_left")
        IdR = self.load(subpath_orig + "index_identity_right")
        IdL = self.convert_index_identity(IdL, bc_MPS)
        IdR = self.convert_index_identity(IdR, bc_MPS)
        data["H_MPO"] = tenpy.networks.mpo.MPO(sites, H_mpo_tensors, bc_MPS, IdL, IdR)

        # H_bond
        if "H" in h5gr_orig:
            H_bond = self.convert_group(h5gr_orig["H"])  # convert npc Arrays
            H_bond = self.load(H_bond.name)
            if H_bond is not None:
                H_bond = list(H_bond)
                # new tenpy: always L entries, entry b is sites (b-1, b), None if not to be used.
                # old tenpy: always L entries, entry b is sites (b, b+1), for finite: last entry 0.
                H_bond = [H_bond[-1]] + H_bond[:-1]  # now entry b is sites (b-1, b)
                for i, H in enumerate(H_bond):
                    if H is not None:
                        H_bond[i] = H.transpose(['p0', 'p1', 'p0*', 'p1*'])  # copy, same labels
                if bc_MPS == 'finite':
                    H_bond[0] = None
                data["H_bond"] = H_bond
        # save the contents of the model
        type_repr = self.save_dict_content(data, h5gr_new, subpath_new)
        h5gr_new.attrs[hdf5_io.ATTR_FORMAT] = type_repr

    for _model in [
        ('models.model', 'model'),  # base class
            # and derived classes defined in tenpy
        ('models.bhf', 'bhf_model'),
        ('models.boson', 'boson_model'),
        ('models.boson2d', 'boson2d_model'),
        ('models.double_model', 'double_model'),
        ('models.dual_ising', 'dual_ising_model'),
        ('models.fermions_chain', 'sf_model'),
        ('models.fermions_hubbard', 'fh_model'),
        ('models.fermions_ladder', 'fermionic_model'),
        ('models.height_models', 'height_model'),
        ('models.levingu', 'levingu_model'),
        ('models.long_range_spin_chain', 'spin_chain_model'),
        ('models.majorana_island', 'majorana_island_model'),
        ('models.multilayer_qh', 'QH_model'),
        ('models.potts', 'potts_model'),
        ('models.quantum_hall', 'QH_model'),
        ('models.spin_chain', 'spin_chain_model'),
        ('models.xxz_tfi', 'xxz_tfi_model'),
    ]:
        mappings[_model] = (('tenpy.models.model', 'MPOModel'), convert_model)
    del _model


if __name__ == "__main__":
    args = convert.parse_args(converter_cls=Converter)
    convert.main(args)
