#include <cyten/models/degrees_of_freedom.h>

#include "../py_cyten_pybind11.h"

#include <cmath>
#include <optional>
#include <string>
#include <vector>

namespace cyten {

namespace {

class PySite
  : public Site
  , public py::trampoline_self_life_support
{
  public:
    using Site::Site;

    void test_sanity() override { PYBIND11_OVERRIDE(void, Site, test_sanity); }
};

class PyOccupationDOF
  : public OccupationDOF
  , public py::trampoline_self_life_support
{
  public:
    using OccupationDOF::OccupationDOF;

    py::array get_annihilator_numpy(py::object species, bool include_JW = false) override
    {
        PYBIND11_OVERRIDE_PURE(
          py::array, OccupationDOF, get_annihilator_numpy, species, include_JW);
    }

    py::array get_creator_numpy(py::object species, bool include_JW = false) override
    {
        PYBIND11_OVERRIDE_PURE(py::array, OccupationDOF, get_creator_numpy, species, include_JW);
    }
};

py::object
dim_to_python(float64 dim)
{
    if (std::isfinite(dim) && std::floor(dim) == dim) {
        return py::int_(static_cast<long long>(dim));
    }
    return py::float_(dim);
}

} // namespace

void
bind_models_degrees_of_freedom(py::module_& m)
{
    m.attr("ALL_SPECIES") = all_species_sentinel();

    py::class_<Site, PySite, py::smart_holder> site(m, "Site");
    site.doc() = R"pydoc(Collects necessary information about a local site of a lattice model.

A site defines the local Hilbert space in terms of its :attr:`leg`.
This involves a choice for the local basis.
Moreover, it exposes the symmetric single-site operators.
Multi-site operators, on the other hand, are represented by :class:`Coupling` s.

Attributes
----------
leg : ElementarySpace
    The local physical Hilbert space.
state_labels : {str: int}
    Optional labels for the local basis states. Any state may have multiple labels, or none.
onsite_operators : {str: SymmetricTensor}
    The available on-site operators. Note: which operators are available typically depends
    on what symmetry is enforced. Operators that are symmetric under a small symmetry may
    not be symmetric under a larger symmetry, and are thus not available as `onsite_operators`.
    Each must have the :attr:`leg` of the site as the only factor in its domain and codomain.

Examples
--------
TODO put some

)pydoc";

    site
      .def(py::init<ElementarySpace::Ptr,
                    std::map<std::string, int64>,
                    std::map<std::string, SymmetricTensorPtr>,
                    TensorBackend::Ptr,
                    std::optional<std::string>>(),
           py::arg("leg"),
           py::arg("state_labels") = std::map<std::string, int64>{},
           py::arg("onsite_operators") = std::map<std::string, SymmetricTensorPtr>{},
           py::arg("backend") = nullptr,
           py::arg("default_device") = py::none())
      .def_readwrite("leg", &Site::leg)
      .def_readwrite("state_labels", &Site::state_labels)
      .def_readwrite("backend", &Site::backend)
      .def_readwrite("default_device", &Site::default_device)
      .def_readwrite("onsite_operators", &Site::onsite_operators)
      .def("test_sanity", &Site::test_sanity, R"pydoc(Perform sanity checks.)pydoc")
      .def_property_readonly("symmetry", &Site::symmetry)
      .def_property_readonly("dim", [](Site const& self) { return dim_to_python(self.dim()); })
      .def("add_onsite_operator",
           &Site::add_onsite_operator,
           py::arg("name"),
           py::arg("op"),
           py::arg("is_diagonal") = py::none(),
           py::arg("understood_braiding") = false,
           R"pydoc(Add an operator to the :attr:`onsite_operators`.)pydoc")
      .def("valid_opname", &Site::valid_opname, py::arg("name"), R"pydoc(
Whether `name` labels a valid onsite operator of this site.)pydoc")
      .def("get_op",
           &Site::get_op,
           py::arg("name"),
           R"pydoc(
Return operator of given name.

Parameters
----------
name : str
    The name of the operator to be returned.
    In case of multiple operator names separated by whitespace,
    we multiply them together to a single on-site operator
    (with the one on the right acting first).

Returns
-------
op : :class:`~cyten.SymmetricTensor`
    The operator given by `name`, with labels ``'p', 'p*'``.
    If name already was an onsite operator, it's directly returned.

)pydoc")
      .def("multiply_op_names",
           &Site::multiply_op_names,
           py::arg("names"),
           R"pydoc(
Multiply operator names together.

Join the operator names in `names` such that `get_op` returns the product of the
corresponding operators.

Parameters
----------
names : list of str
    List of valid operator labels.

Returns
-------
combined_opname : str
    A valid operator name
    Operator name representing the product of operators in `names`.

)pydoc")
      .def("multiply_operators",
           &Site::multiply_operators,
           py::arg("operators"),
           R"pydoc(
Multiply local operators (possibly given by their names) together.

Parameters
----------
operators : list of {str | :class:`~cyten.SymmetricTensor`}
    List of valid operator names (to be translated with :meth:`get_op`) or
    directly on-site operators in the form of tensors with ``'p', 'p*'`` labels.
    The operators are multiplied left-to-right.

Returns
-------
combined_operator : :class:`~cyten.SymmetricTensor`
    The product of the given `operators` in a left-to-right multiplication following the
    usual mathematical convention. For example, if ``operators=['Sz', 'Sp', 'Sx']``,
    the final operator is equivalent to ``site.get_op('Sz Sp Sx')``, with the ``'Sx'``
    operator acting first on any physical state.

)pydoc")
      .def("identity_tensor",
           &Site::identity_tensor,
           py::arg("w"),
           py::arg("overbraid") = true,
           R"pydoc(
Build an identity MPO tensor for this site with the given virtual legs.

Returns a 4-leg tensor with legs ``[wL, p, wR, p*]``; the physical legs carry :attr:`leg`.
This tensor acts as identity map between wL and wR, and is symmetric under the symmetry of this site.

Parameters
----------
w : ElementarySpace
    Virtual leg for the `wL` and `wR` legs (they are the same) of the returned tensor.
overbraid : bool
    Braiding direction when the virtual leg is permuted past the physical
    leg. ``True`` (default) uses an over-braid (``bend_right=False`` in
    :func:`~cyten.tensors.permute_legs`); ``False`` uses an under-braid (``bend_right=True``).

Returns
-------
SymmetricTensor
    Identity tensor with legs ``[wL, p, wR, p*]``.

)pydoc")
      .def("state_index",
           &Site::state_index,
           py::arg("label"),
           R"pydoc(The index of a basis state.)pydoc")
      .def("state_indices",
           &Site::state_indices,
           py::arg("labels"),
           R"pydoc(The indices of multiple basis states)pydoc")
      .def("__repr__", &Site::repr);

    py::class_<SpinDOF, Site, py::smart_holder> spin_dof(m, "SpinDOF");
    spin_dof.doc() = R"pydoc(Common base class for sites that have a spin degree of freedom.

Attributes
----------
spin_vector : 3D array
    The vector of spin operators as a numpy array with axes ``[p, p*, i]`` and shape
    ``(dim, dim, 3)``. These operators include the factor of the total spin,
    e.g. for spin-1/2, these are ``.5`` times the pauli matrices.

)pydoc";

    spin_dof
      .def(py::init<ElementarySpace::Ptr,
                    py::array,
                    std::map<std::string, int64>,
                    std::map<std::string, SymmetricTensorPtr>,
                    TensorBackend::Ptr,
                    std::optional<std::string>>(),
           py::arg("leg"),
           py::arg("spin_vector"),
           py::arg("state_labels") = std::map<std::string, int64>{},
           py::arg("onsite_operators") = std::map<std::string, SymmetricTensorPtr>{},
           py::arg("backend") = nullptr,
           py::arg("default_device") = py::none())
      .def_readwrite("spin_vector", &SpinDOF::spin_vector)
      .def("test_sanity", &SpinDOF::test_sanity, R"pydoc(Perform sanity checks.)pydoc")
      .def_static("spin_vector_from_Sp",
                  &SpinDOF::spin_vector_from_Sp,
                  py::arg("Sz"),
                  py::arg("Sp"),
                  R"pydoc(Build the spin_vector from ``Sz`` and ``Sp = Sx + i Sy``)pydoc")
      .def_static("conservation_law_to_symmetry",
                  &SpinDOF::conservation_law_to_symmetry,
                  py::arg("conserve"),
                  R"pydoc(Translate conservation law for a spin to a symmetry.)pydoc");

    py::class_<ClockDOF, Site, py::smart_holder> clock_dof(m, "ClockDOF");
    clock_dof.doc() =
      R"pydoc(Common base class for sites that have a quantum clock degree of freedom.)pydoc";

    clock_dof
      .def(py::init<ElementarySpace::Ptr,
                    py::array,
                    std::map<std::string, int64>,
                    std::map<std::string, SymmetricTensorPtr>,
                    TensorBackend::Ptr,
                    std::optional<std::string>>(),
           py::arg("leg"),
           py::arg("clock_operators"),
           py::arg("state_labels") = std::map<std::string, int64>{},
           py::arg("onsite_operators") = std::map<std::string, SymmetricTensorPtr>{},
           py::arg("backend") = nullptr,
           py::arg("default_device") = py::none())
      .def_readwrite("clock_operators", &ClockDOF::clock_operators)
      .def("test_sanity", &ClockDOF::test_sanity, R"pydoc(Perform sanity checks.)pydoc")
      .def_static("conservation_law_to_symmetry",
                  &ClockDOF::conservation_law_to_symmetry,
                  py::arg("conserve"),
                  R"pydoc(Translate conservation law for a clock to a symmetry.)pydoc");

    py::class_<AnyonDOF, Site, py::smart_holder> anyon_dof(m, "AnyonDOF");
    anyon_dof.doc() =
      R"pydoc(Common base class for sites that have an anyonic degree of freedom.)pydoc";

    anyon_dof
      .def(py::init<ElementarySpace::Ptr,
                    std::vector<std::string>,
                    std::map<std::string, int64>,
                    std::map<std::string, SymmetricTensorPtr>,
                    TensorBackend::Ptr,
                    std::optional<std::string>>(),
           py::arg("leg"),
           py::arg("sector_names") = std::vector<std::string>{},
           py::arg("state_labels") = std::map<std::string, int64>{},
           py::arg("onsite_operators") = std::map<std::string, SymmetricTensorPtr>{},
           py::arg("backend") = nullptr,
           py::arg("default_device") = py::none())
      .def_readwrite("sector_names", &AnyonDOF::sector_names)
      .def("test_sanity", &AnyonDOF::test_sanity, R"pydoc(Perform sanity checks.)pydoc");

    py::class_<OccupationDOF, Site, PyOccupationDOF, py::smart_holder> occupation_dof(
      m, "OccupationDOF");
    occupation_dof.doc() =
      R"pydoc(Common base class for sites that have a bosonic or fermionic degree of freedom.

Requires that the local basis is such that the :attr:`number_operators` of all species
are diagonal.

Attributes
----------
num_species : int
    Number of boson species.
creators : 3D array
    The vector of creation operators as a numpy array with shape ``(dim, dim, num_species)``
    and axes ``[p, p*, i]``, where `i` corresponds to the different species of bosons (i.e.,
    ``[Bd0, Bd1`, ...]`` stacked along axis 2).
annihilators : 3D array
    The vector of annihilation operators as a numpy array with shape ``(dim, dim, num_species)``
    and axes ``[p, p*, i]``, where `i` corresponds to the different species of bosons (i.e.,
    ``[B0, B1`, ...]`` stacked along axis 2).
anti_commute_sign : float
    ``+1`` for bosons, ``-1`` for fermions.
species_names : list of (str | None)
    Names for each of the species.
number_operators : 3D array
    The vector of occupation number operators with shape ``(dim, dim, num_species)``.
n_tot : 2D array
    The total occupation number operator with shape ``(dim, dim)``.

)pydoc";

    occupation_dof.def_readwrite("num_species", &OccupationDOF::num_species)
      .def_readwrite("creators", &OccupationDOF::creators)
      .def_readwrite("annihilators", &OccupationDOF::annihilators)
      .def_readwrite("anti_commute_sign", &OccupationDOF::anti_commute_sign)
      .def_readwrite("species_names", &OccupationDOF::species_names)
      .def_readwrite("number_operators", &OccupationDOF::number_operators)
      .def_readwrite("n_tot", &OccupationDOF::n_tot)
      .def("test_sanity", &OccupationDOF::test_sanity, R"pydoc(Perform sanity checks.)pydoc")
      .def("add_individual_occupation_ops",
           &OccupationDOF::add_individual_occupation_ops,
           R"pydoc(
Add occupation and parity operators for each species as symmetric onsite operators.

The added operators include::
    - occupation operators ``Ni`` for each species ``i``
    - parity operators ``Pi`` for each species ``i``

If there is only a single species, also the aliases ``N`` for ``N0`` and ``P`` for ``P0``.
)pydoc")
      .def("add_total_occupation_ops",
           &OccupationDOF::add_total_occupation_ops,
           R"pydoc(
Add total occupation and parity operators as symmetric onsite operators.

The added operators include:
- total occupation operator `Ntot`
- total parity operator `Ptot`
- squared total occupation operator `NtotNtot`
)pydoc")
      .def("get_annihilator_numpy",
           &OccupationDOF::get_annihilator_numpy,
           py::arg("species"),
           py::arg("include_JW") = false,
           R"pydoc(
Wrapper around ``annihilators[:, :, species]``, optionally including JW strings.

If `include_JW`, we include the ``(-1) ** n_k`` from all ``k < species``.
)pydoc")
      .def("get_creator_numpy",
           &OccupationDOF::get_creator_numpy,
           py::arg("species"),
           py::arg("include_JW") = false,
           R"pydoc(
Wrapper around ``creators[:, :, species]``, optionally including JW strings.

If `include_JW`, we include the ``(-1) ** n_k`` from all ``k < species``.
)pydoc")
      .def(
        "get_occupation_numpy",
        &OccupationDOF::get_occupation_numpy,
        py::arg("species") = all_species_sentinel(),
        R"pydoc(Get the occupation number operator for some or multiple species as a numpy array.)pydoc")
      .def("get_species_idx", &OccupationDOF::get_species_idx, py::arg("species"));

    py::class_<BosonicDOF, OccupationDOF, py::smart_holder> bosonic_dof(m, "BosonicDOF");
    bosonic_dof.doc() = R"pydoc(Common base class for sites that have a bosonic degree of freedom.

Requires that the local basis is such that the :attr:`number_operators` of all species
are diagonal.

Mutually exclusive with :class:`FermionicDOF`. Sites containing both bosonic and fermionic
degrees of freedom can be realized by grouping of a bosonic site with a fermionic one.

Attributes
----------
Nmax : 1D array of int
    Cutoff defining the maximum number of bosons per species and site. ``Nmax[i]`` corresponds
    to the cutoff for the `i`th species; a value of ``Nmax[i] = 1`` describes hard-core bosons.

)pydoc";

    bosonic_dof
      .def(py::init<ElementarySpace::Ptr,
                    py::array,
                    py::array,
                    py::array,
                    std::vector<std::optional<std::string>>,
                    std::map<std::string, int64>,
                    std::map<std::string, SymmetricTensorPtr>,
                    TensorBackend::Ptr,
                    std::optional<std::string>>(),
           py::arg("leg"),
           py::arg("Nmax"),
           py::arg("creators"),
           py::arg("annihilators"),
           py::arg("species_names") = std::vector<std::optional<std::string>>{},
           py::arg("state_labels") = std::map<std::string, int64>{},
           py::arg("onsite_operators") = std::map<std::string, SymmetricTensorPtr>{},
           py::arg("backend") = nullptr,
           py::arg("default_device") = py::none())
      .def_readwrite("Nmax", &BosonicDOF::Nmax)
      .def_readwrite("JW", &BosonicDOF::JW)
      .def_readwrite("_JW", &BosonicDOF::JW)
      .def("test_sanity", &BosonicDOF::test_sanity, R"pydoc(Perform sanity checks.)pydoc")
      .def("add_individual_occupation_ops",
           &BosonicDOF::add_individual_occupation_ops,
           R"pydoc(Add occupation and parity operators for each species.)pydoc")
      .def("get_annihilator_numpy",
           &BosonicDOF::get_annihilator_numpy,
           py::arg("species"),
           py::arg("include_JW") = false)
      .def("get_creator_numpy",
           &BosonicDOF::get_creator_numpy,
           py::arg("species"),
           py::arg("include_JW") = false)
      .def_static(
        "conservation_law_to_symmetry",
        &BosonicDOF::conservation_law_to_symmetry,
        py::arg("conserve"),
        R"pydoc(Translate conservation law for individual / all bosons to a symmetry.)pydoc")
      .def_static("creation_annihilation_op_from_single_Nmax",
                  &BosonicDOF::creation_annihilation_op_from_single_Nmax,
                  py::arg("Nmax"),
                  py::arg("dim"))
      .def_static("creation_annihilation_ops_from_Nmax",
                  &BosonicDOF::creation_annihilation_ops_from_Nmax,
                  py::arg("Nmax"),
                  py::arg("dim"))
      .def_static("creation_annihilation_ops",
                  &BosonicDOF::creation_annihilation_ops,
                  py::arg("num_species"),
                  py::arg("Nmax"),
                  py::arg("dim"));

    py::class_<FermionicDOF, OccupationDOF, py::smart_holder> fermionic_dof(m, "FermionicDOF");
    fermionic_dof.doc() =
      R"pydoc(Common base class for sites that have a fermionic degree of freedom.

Requires that the local basis is such that the :attr:`number_operators` of all species
are diagonal.

Mutually exclusive with :class:`BosonicDOF`. Sites containing both bosonic and fermionic
degrees of freedom can be realized by grouping of a bosonic site with a fermionic one.
)pydoc";

    fermionic_dof
      .def(py::init<ElementarySpace::Ptr,
                    py::array,
                    py::array,
                    std::vector<std::optional<std::string>>,
                    std::map<std::string, int64>,
                    std::map<std::string, SymmetricTensorPtr>,
                    TensorBackend::Ptr,
                    std::optional<std::string>>(),
           py::arg("leg"),
           py::arg("creators"),
           py::arg("annihilators"),
           py::arg("species_names") = std::vector<std::optional<std::string>>{},
           py::arg("state_labels") = std::map<std::string, int64>{},
           py::arg("onsite_operators") = std::map<std::string, SymmetricTensorPtr>{},
           py::arg("backend") = nullptr,
           py::arg("default_device") = py::none())
      .def_readwrite("partial_JWs", &FermionicDOF::partial_JWs)
      .def_readwrite("JW", &FermionicDOF::JW)
      .def_readwrite("_JW", &FermionicDOF::JW)
      .def("test_sanity", &FermionicDOF::test_sanity, R"pydoc(Perform sanity checks.)pydoc")
      .def("get_annihilator_numpy",
           &FermionicDOF::get_annihilator_numpy,
           py::arg("species"),
           py::arg("include_JW") = false)
      .def("get_creator_numpy",
           &FermionicDOF::get_creator_numpy,
           py::arg("species"),
           py::arg("include_JW") = false)
      .def_static(
        "conservation_law_to_symmetry",
        &FermionicDOF::conservation_law_to_symmetry,
        py::arg("conserve"),
        R"pydoc(Translate conservation law for individual / all fermions to a symmetry.)pydoc")
      .def_static("creation_annihilation_ops",
                  &FermionicDOF::creation_annihilation_ops,
                  py::arg("num_species"));
}

} // namespace cyten
