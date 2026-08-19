#include <cyten/models/couplings.h>

#include "../py_cyten_pybind11.h"

#include <optional>
#include <string>
#include <vector>

namespace cyten {

namespace {

py::object
not_implemented()
{
    return py::module_::import("builtins").attr("NotImplemented");
}

} // namespace

void
bind_models_couplings(py::module_& m)
{
    m.def(
      "freeze",
      &freeze,
      py::arg("obj"),
      R"pydoc(Recursively turn the output of the ``_*_to_dict`` helpers into hashable nested tuples.)pydoc");

    m.def("_adjacent_transpositions",
          &adjacent_transpositions,
          py::arg("permutation"),
          R"pydoc(
Decompose a permutation into a sequence of adjacent position swaps.

Parameters
----------
permutation : list of int
    A permutation of ``range(len(permutation))``; ``permutation[k]`` is the value that ends
    up at position `k`.

Returns
-------
swap_positions : list of int
    Positions `pos` such that applying the swaps ``(pos, pos + 1)`` in order to
    ``list(range(len(permutation)))`` produces `permutation`. Realizes `permutation` with the
    minimal number of adjacent transpositions, i.e. its number of inversions.

)pydoc");

    m.def("space_to_dict", &space_to_dict, py::arg("space"));

    py::class_<Coupling, py::smart_holder> coupling(m, "Coupling");
    coupling.doc() =
      R"pydoc(A coupling is an operator on a few :class:`Site` s, factorized as one tensor per site.

A coupling represents an operator of the following form::

    |        p0   p1   ..   pN
    |        │    │    │    │
    |       ┏┷━━━━┷━━━━┷━━━━┷┓
    |       ┃       h        ┃
    |       ┗┯━━━━┯━━━━┯━━━━┯┛
    |        │    │    │    │
    |        p0*  p1*  ..  pN*

The intended use case is to build tensor network representations (e.g. MPOs) of Hamiltonians.

Attributes
----------
sites : list of :class:`Site`
    The sites that the operators act on.
factorization : list of :class:`SymmetricTensor`
    A list of tensors that, if contracted, give the operator that is represented.
    Each tensor ``factorization[i]`` has legs ``[wL, p, wR, p*]``, where ``p`` and ``p*``
    are the physical :attr:`Site.leg` of the corresponding ``sites[i]``, and where contracting
    the ``wL`` and ``wR`` legs in an MPO-like geometry gives the multi-site operator.
name : str, optional
    A descriptive name that can be used when pretty-printing, to identify the coupling.
    For example, a Heisenberg coupling is usually initialized with name ``'S.S'``.

)pydoc";

    coupling
      .def(py::init<std::vector<Site::Ptr>,
                    std::vector<SymmetricTensorPtr>,
                    std::optional<std::string>>(),
           py::arg("sites"),
           py::arg("factorization"),
           py::arg("name") = py::none())
      .def_readwrite("sites", &Coupling::sites)
      .def_readwrite("factorization", &Coupling::factorization)
      .def_readwrite("name", &Coupling::name)
      .def_static("from_dense_block",
                  &Coupling::from_dense_block,
                  py::arg("operator"),
                  py::arg("sites"),
                  py::arg("name") = py::none(),
                  py::arg("dtype") = py::none(),
                  py::arg("understood_braiding") = false,
                  py::arg("cutoff_singular_values") = py::none(),
                  R"pydoc(
Convert a dense block to a :class:`Coupling`.

Parameters
----------
operator : Block
    The data to be converted to a Coupling as a backend-specific block or some data that
    can be converted using :meth:`BlockBackend.as_block`. The order of axes must match the
    `sites`, that is, the axes correspond to ``[p0, p1, ..., p1*, p0*]`` (codomain legs
    ascending, domain legs descending), where ``pi`` corresponds to site ``sites[i]``.
    The block should be given in the "public" basis order of the sites, i.e.,
    according to ``sites[i].sectors_of_basis``.
sites : list of :class:`Site`
    The sites that the operators act on.
name : str, optional
    A descriptive name that can be used when pretty-printing, to identify the coupling.
dtype : :class:`Dtype`, optional
    If given, the block is converted to that dtype and the resulting tensors in the
    factorization will have that dtype. By default, we detect the dtype from the block.
cutoff_singular_values : float, optional
    If given, truncate singular values (see :func:`cyten.horizontal_factorization`)
    below this threshold.

)pydoc")
      .def_static("from_tensor",
                  &Coupling::from_tensor,
                  py::arg("operator"),
                  py::arg("sites"),
                  py::arg("name") = py::none(),
                  py::arg("cutoff") = py::none(),
                  R"pydoc(
Convert an operator / tensor to a :class:Coupling.

Decomposes an operator into factors using :func:`cyten.horizontal_factorization` to
obtain the :attr:`factorization` of the coupling.

Parameters
----------
operator : :class:`SymmetricTensor`
    Operator to be converted to a coupling. The legs should be ordered as
    ``[p0, p1, ..., p1*, p0*]``, where ``pi`` and ``pi*`` correspond to the legs associated
    with site ``sites[i]``.
sites : list of :class:`Site`
    The sites that the operator acts on.
name : str, optional
    A descriptive name that can be used when pretty-printing, to identify the coupling.
    For example, a Heisenberg coupling is usually initialized with name ``'S.S'``.
cutoff_singular_values : float, optional
    If given, truncate singular values (see :func:`cyten.horizontal_factorization`)
    below this threshold.

)pydoc")
      .def("to_tensor", &Coupling::to_tensor, R"pydoc(Convert to a tensor.)pydoc")
      .def("to_numpy",
           &Coupling::to_numpy,
           py::arg("leg_order") = py::none(),
           py::arg("dtype") = py::none(),
           py::arg("understood_braiding") = false,
           R"pydoc(Convert to a numpy array.)pydoc")
      .def("stretch_with_identities",
           &Coupling::stretch_with_identities,
           py::arg("all_sites"),
           py::arg("coupling_positions"),
           R"pydoc(
Place this coupling's tensors among `all_sites`, filling the gaps with identities.

Returns a new coupling spanning `all_sites` from the first to the last of
`coupling_positions` (inclusive). `self`'s tensors sit at those positions and every site in
between gets an identity tensor independent of what site it is.

Parameters
----------
all_sites : list of Site
    The sites the returned coupling should be based on.
    Only the positions defined by `coupling_positions` are used.
coupling_positions : list of int
    Strictly ascending, one entry per :attr:`factorization` tensor: the index into
    `all_sites` where that tensor should sit.

Returns
-------
Coupling
    Spans all_sites[coupling_positions[0] to coupling_positions[-1] + 1].

)pydoc")
      .def("permute",
           &Coupling::permute,
           py::arg("permutation"),
           py::arg("levels") = py::none(),
           py::arg("over_braid") = py::none(),
           R"pydoc(
Permute the sites of this coupling, braiding through the (possibly anyonic) legs.

Contracts `self` to a single tensor (:meth:`to_tensor`), realizes `permutation` as a
sequence of elementary adjacent-site transpositions (each one braiding the full ``(p, p*)``
leg pair of one site past that of its neighbour, as a single unit -- the two legs of one
site never cross each other), and re-factorizes the result (:meth:`from_tensor`) with the
sites reordered accordingly. This is analogous to how
:class:`~cyten.backends.fusion_tree_backend.PermuteLegsInstructionEngine` realizes a leg
permutation as a sequence of elementary swaps, tracking a `levels` list that is itself
reordered as legs move.

Results are cached on `self` (not shared with `self`'s other permutations, or with the
result of this call): calling ``self.permute(permutation, ...)`` twice with the same
`permutation` returns the same (cached) result, using `levels`/`over_braid` only the first
time; a different `permutation` triggers a new computation.

Parameters
----------
permutation : list of int
    A permutation of ``range(len(self.sites))``. ``permutation[k]`` is the index (in
    `self`'s current order) of the site that ends up at new position `k`.
levels : list of int | None
    One entry per site of `self` (in `self`'s current order): its "height", used to
    derive the braid chirality (over/under) for any elementary transposition whose
    `over_braid` entry is ``None``, the same way :attr:`~cyten.symmetries.Symmetry` legs
    with a higher level braid over those with a lower one. Only needed for symmetries
    without a symmetric braid (see :attr:`~cyten.symmetries.Symmetry.braiding_style`);
    ignored otherwise.
over_braid : list of bool | None
    One entry per elementary adjacent-site transposition needed to realize `permutation`
    (i.e. NOT one entry per site -- the number of transpositions depends on
    `permutation`, e.g. via the number of its inversions). Explicitly fixes the braid
    chirality for that transposition (``True`` = the site moving from the lower position
    over the one moving from the higher position); ``None`` derives it from `levels`.

Returns
-------
Coupling
    A new coupling with :attr:`sites` (and the represented operator) reordered according
    to `permutation`.

)pydoc")
      .def(
        "_key",
        [](Coupling const& self) { return std::get<0>(self.key()); },
        R"pydoc(Structural identity used by :meth:`__hash__`/ :meth:`__eq__`.)pydoc")
      .def(
        "__eq__",
        [](Coupling const& self, py::handle other) -> py::object {
            if (!py::isinstance<Coupling>(other)) {
                return not_implemented();
            }
            return py::cast(self == other.cast<Coupling>());
        },
        py::arg("other"))
      .def_property_readonly("num_sites", &Coupling::num_sites)
      .def("__hash__", &Coupling::hash)
      .def("__repr__", &Coupling::repr)
      .def("test_sanity", &Coupling::test_sanity, R"pydoc(Perform sanity checks.)pydoc");

    m.def("spin_spin_coupling",
          &spin_spin_coupling,
          py::arg("sites"),
          py::arg("Jx") = 0,
          py::arg("Jy") = 0,
          py::arg("Jz") = 0,
          py::arg("backend") = py::none(),
          py::arg("device") = py::none(),
          py::arg("name") = py::none(),
          R"pydoc(
Two-site coupling between spins.

.. math ::
    h_{ij} = \mathtt{Jx} S_i^x S_j^x + \mathtt{Jy} S_i^y S_j^y + \mathtt{Jz} S_i^z S_j^z

Parameters
----------
sites: list of Site
    The sites that the coupling acts on. Note that the order matters for the final leg order.
Jx, Jy, Jz: float
    Prefactor, as given above. By default, all prefactors vanish.

)pydoc");

    m.def("spin_field_coupling",
          &spin_field_coupling,
          py::arg("sites"),
          py::arg("hx") = 0,
          py::arg("hy") = 0,
          py::arg("hz") = 0,
          py::arg("backend") = py::none(),
          py::arg("device") = py::none(),
          py::arg("name") = py::none(),
          R"pydoc(
Single-site coupling of a spin to an external field.

.. math ::
    h_i = \mathtt{hx} S_i^x + \mathtt{hy} S_i^y + \mathtt{hz} S_i^z

Parameters
----------
sites: list of Site
    The sites that the coupling acts on. Note that the order matters for the final leg order.
hx, hy, hz: float
    Prefactor, as given above. By default, all prefactors vanish.

)pydoc");

    m.def("aklt_coupling",
          &aklt_coupling,
          py::arg("sites"),
          py::arg("J") = 1,
          py::arg("backend") = py::none(),
          py::arg("device") = py::none(),
          py::arg("name") = py::none(),
          R"pydoc(
Two-site AKLT coupling between spins.

.. math ::
    h_{ij} = \mathtt{J} [\vec{S}_i \cdot \vec{S}_j + \frac{1}{3} (\vec{S}_i \cdot \vec{S}_j)^2]

This is the coupling originally defined by Affleck, Kennedy, Lieb, Tasaki
in :cite:`affleck1987`, except we drop the constant part of 1/3 per bond and rescale with a
factor of 2, i.e. :math:`h_{ij} = 2 P^{S=2}_{i, j} + const.`.

It was defined for spin-1 degrees of freedom in the original work, but we allow any site
with a spin DOF. Note that the coupling simplifies to a Heisenberg coupling for spin-1/2.

Parameters
----------
sites: list of Site
    The sites that the coupling acts on. Note that the order matters for the final leg order.
J: float
    Prefactor, as given above. By default use ``1``.

)pydoc");

    m.def("heisenberg_coupling",
          &heisenberg_coupling,
          py::arg("sites"),
          py::arg("J") = 1,
          py::arg("backend") = py::none(),
          py::arg("device") = py::none(),
          py::arg("name") = py::none(),
          R"pydoc(
Two-site Heisenberg coupling between spins.

.. math ::
    h_{ij} = \mathtt{J} \vec{S}_i \cdot \vec{S}_j

Parameters
----------
sites: list of Site
    The sites that the coupling acts on. Note that the order matters for the final leg order.
J: float
    Prefactor, as given above. By default use ``1``, i.e. an anti-ferromagnetic coupling.

)pydoc");

    m.def("chiral_3spin_coupling",
          &chiral_3spin_coupling,
          py::arg("sites"),
          py::arg("chi") = 1,
          py::arg("backend") = py::none(),
          py::arg("device") = py::none(),
          py::arg("name") = py::none(),
          R"pydoc(
Chiral coupling of three spins.

.. math ::
    h_{ijk} = \mathtt{chi} \vec{S}_i \cdot ( \vec{S}_j \times \vec{S}_k )

Parameters
----------
sites: list of Site
    The sites that the coupling acts on. Note that the order matters for the final leg order.
chi: float
    Prefactor, as given above. By default use ``1``.

)pydoc");

    m.def("chemical_potential",
          &chemical_potential,
          py::arg("sites"),
          py::arg("mu"),
          py::arg("species") = all_species_sentinel(),
          py::arg("backend") = py::none(),
          py::arg("device") = py::none(),
          py::arg("name") = py::none(),
          R"pydoc(
Chemical potential for bosons or fermions. Single-site coupling.

.. math ::
    h_i = -\mathtt{mu} \sum_{k \in \mathtt{species} n_{i, k}

where :math:`n_{i, k}` is the occupation number of species :math:`k` on site :math:`i`.

Parameters
----------
sites: list of Site
    The sites that the coupling acts on. Note that the order matters for the final leg order.
mu: float
    Chemical potential, as defined above.
species: (list of) int | str, optional
    If given, the chemical potential only couples to the occupation of this species.
    By default, it couples to the total occupation of all species.

)pydoc");

    m.def("onsite_interaction",
          &onsite_interaction,
          py::arg("sites"),
          py::arg("U") = 1,
          py::arg("species") = all_species_sentinel(),
          py::arg("backend") = py::none(),
          py::arg("device") = py::none(),
          py::arg("name") = py::none(),
          R"pydoc(
Onsite interaction for bosons or fermions. Single-site coupling.

.. math ::
    h_i = \frac{U}{2} n_i^2

where :math:`n_i` is the total occupation number, or the occupation of a single `species`.

Parameters
----------
sites: list of Site
    The sites that the coupling acts on. Note that the order matters for the final leg order.
U: float
    Prefactor, as defined above. By default, use ``1``, i.e. a repulsive interaction.
species: int | str, optional
    If given, we use only the occupation of this one species as the density :math:`n_i`.
    By default, we use the total occupation of all species.

)pydoc");

    m.def("density_density_interaction",
          &density_density_interaction,
          py::arg("sites"),
          py::arg("V") = 1,
          py::arg("species_i") = all_species_sentinel(),
          py::arg("species_j") = all_species_sentinel(),
          py::arg("backend") = py::none(),
          py::arg("device") = py::none(),
          py::arg("name") = py::none(),
          R"pydoc(
Density-density interaction. Two-site coupling.

.. math ::
    h_{ij} = \mathtt{V} n_i n_j

where :math:`n_i` is the total occupation number.

Parameters
----------
sites: list of Site
    The sites that the coupling acts on. Note that the order matters for the final leg order.
V: float
    Prefactor, as defined above. By default, use ``1``, i.e. a repulsive interaction.
species_i, species_j: int | str, optional
    If given, we use only the occupation of this one species as the density :math:`n_{i/j}`.
    By default, we use the total occupation of all species.
    Note that if the two species are different, this coupling alone is not hermitian!

)pydoc");

    m.def("hopping",
          &hopping,
          py::arg("sites"),
          py::arg("t") = 1,
          py::arg("species") = py::none(),
          py::arg("backend") = py::none(),
          py::arg("device") = py::none(),
          py::arg("name") = py::none(),
          R"pydoc(
Hopping of fermions or bosons. Two-site coupling.

.. math ::
    h_{ij} = -\mathtt{t} \sum_{k \in \mathtt{species}} a_{i, k_i}^\dagger a_{j, k_j} + h.c.

Parameters
----------
sites: list of Site
    The sites that the coupling acts on. Note that the order matters for the final leg order.
t : float
    Prefactor, as given above. By default ``1``.
species : tuple of list of (int | str), optional
    Which species should participate (the sum above goes over ``k_i, k_j in zip(*species)``).
    By default, we let :math:`k_i = k_j` go over all species, i.e. include all
    "species preserving" hoppings.

)pydoc");

    m.def("pairing",
          &pairing,
          py::arg("sites"),
          py::arg("Delta") = 1,
          py::arg("species") = py::none(),
          py::arg("backend") = py::none(),
          py::arg("device") = py::none(),
          py::arg("name") = py::none(),
          R"pydoc(
Superconducting pairing of fermions or bosons. Two-site coupling.

.. math ::
    h_{ij} = \mathtt{Delta} \sum_{k\in\mathtt{species}} a_{i, k_i}^\dagger a_{j, k_j}^\dagger + h.c.

.. note ::
    This coupling assumes distinct sites :math:`i \neq j`.
    Use :func:`onsite_pairing` for :math:`i = j`.

Parameters
----------
sites: list of Site
    The sites that the coupling acts on. Note that the order matters for the final leg order.
Delta : float
    Prefactor, as given above. By default ``1``.
species : tuple of list of (int | str), optional
    Which species should participate (the sum above goes over ``k_i, k_j in zip(*species)``).
    By default, we let :math:`k_i = k_j` go over all species, i.e. include all "same-species"
    pairings.

See Also
--------
onsite_pairing

)pydoc");

    m.def("onsite_pairing",
          &onsite_pairing,
          py::arg("sites"),
          py::arg("Delta") = 1,
          py::arg("species") = py::none(),
          py::arg("backend") = py::none(),
          py::arg("device") = py::none(),
          py::arg("name") = py::none(),
          R"pydoc(
Superconducting pairing of fermions or bosons. Single-site coupling.

.. math ::
    h_i = \mathtt{Delta} \sum_{k\in\mathtt{species}} a_{i, k_1}^\dagger a_{i, k_2}^\dagger + h.c.

Parameters
----------
sites: list of Site
    The sites that the coupling acts on. Note that the order matters for the final leg order.
Delta : float
    Prefactor, as given above. By default ``1``.
species : tuple of list of (int | str), optional
    Which species should participate (the sum above goes over ``k_1, k_2 in zip(*species)``).
    By default, we let :math:`k_1 = k_2` go over all species, i.e. include all "same-species"
    pairings.

See Also
--------
pairing

)pydoc");

    m.def("clock_clock_coupling",
          &clock_clock_coupling,
          py::arg("sites"),
          py::arg("Jx") = 0,
          py::arg("Jz") = 0,
          py::arg("backend") = py::none(),
          py::arg("device") = py::none(),
          py::arg("name") = py::none(),
          R"pydoc(
Two-site coupling between quantum clocks.

.. math ::
    h_{ij} = \mathtt{Jx} X_i X_j^\dagger + \mathtt{Jz} Z_i Z_j^\dagger + h.c.

Parameters
----------
sites: list of Site
    The sites that the coupling acts on. Note that the order matters for the final leg order.
Jx, Jz: float
    Prefactor, as given above. By default, all prefactors vanish.

)pydoc");

    m.def("clock_field_coupling",
          &clock_field_coupling,
          py::arg("sites"),
          py::arg("hx") = py::none(),
          py::arg("hz") = py::none(),
          py::arg("backend") = py::none(),
          py::arg("device") = py::none(),
          py::arg("name") = py::none(),
          R"pydoc(
Single-site coupling of a quantum clock to an external field.

.. math ::
    h_i = \mathtt{hx} X_i + \mathtt{hz} Z_i + h.c.

Parameters
----------
sites: list of Site
    The sites that the coupling acts on. Note that the order matters for the final leg order.
hx, hz: float
    Prefactor, as given above. By default, all prefactors vanish.

)pydoc");

    m.def("sector_projection_coupling",
          &sector_projection_coupling,
          py::arg("sites"),
          py::arg("J"),
          py::arg("sector"),
          py::arg("name"),
          py::arg("backend") = py::none(),
          py::arg("device") = py::none(),
          R"pydoc(
Coupling that is given by the projector onto a single sector

The number of sites is arbitrary and the operator :math:`h_{ij...}` is given
by :meth:`cyten.SymmetricTensor.from_sector_projection`, with prefactor `J`.
Note that positive `J` mean that states that fuse to the given `sector` are energetically
*disfavored*.
)pydoc");

    m.def("gold_coupling",
          &gold_coupling,
          py::arg("sites"),
          py::arg("J") = 1,
          py::arg("backend") = py::none(),
          py::arg("device") = py::none(),
          py::arg("name") = py::none(),
          R"pydoc(
Two-site coupling of Fibonacci anyons that energy splits fusion to vacuum or tau.

.. math ::
    h_{ij} = -J P^\text{vac}_{i, j}

Parameters
----------
sites: list of Site
    The sites that the coupling acts on. Note that the order matters for the final leg order.
J: float
    Prefactor, as given above. By default ``1``. Positive `J` energetically favor the
    trivial fusion channel, i.e. they are the "antiferromagnetic" analog.

)pydoc");
}

} // namespace cyten
