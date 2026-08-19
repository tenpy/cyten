#include <cyten/models/sites.h>

#include "../py_cyten_pybind11.h"

#include <optional>
#include <string>

namespace cyten {

void
bind_models_sites(py::module_& m)
{
    py::class_<SpinSite, SpinDOF, py::smart_holder> spin_site(m, "SpinSite");
    spin_site.doc() = R"pydoc(Class for sites that have a single spin degree of freedom.

TODO find a good format to doc the onsite operators that exist in a site

Attributes
----------
S : float
    The total spin.
double_total_spin : int
    Twice the :attr:`S`. We store this in addition because it is an integer.
conserve : Literal['SU(2)', 'Sz', 'parity', 'None']
    The symmetry to be conserved. We can conserve::

        - SU(2), the full spin rotation symmetry.
        - Sz (= U(1) symmetry), with sector labels corresponding to ``2 * Sz``.
        - Sz parity (= Z_2 symmetry), with sector labels corresponding to ``(Sz + S_tot) % 2``.
        - nothing.

    Conserves nothing by default.

)pydoc";

    spin_site
      .def(py::init<float64,
                    std::optional<std::string>,
                    TensorBackend::Ptr,
                    std::optional<std::string>>(),
           py::arg("S") = 0.5,
           py::arg("conserve") = py::none(),
           py::arg("backend") = nullptr,
           py::arg("default_device") = py::none())
      .def_readwrite("S", &SpinSite::S)
      .def_readwrite("double_total_spin", &SpinSite::double_total_spin)
      .def_readwrite("conserve", &SpinSite::conserve)
      .def("test_sanity", &SpinSite::test_sanity, R"pydoc(Perform sanity checks.)pydoc")
      .def("__repr__", &SpinSite::repr);

    py::class_<SpinlessBosonSite, BosonicDOF, py::smart_holder> spinless_boson_site(
      m, "SpinlessBosonSite");
    spinless_boson_site.doc() = R"pydoc(Site for (possibly multiple) spinless bosons.

TODO describe onsite operators

Parameters
----------
Nmax : int | Sequence[int]
    The maximum occupation of each of the boson species. An `int` corresponds to a single boson
    species. Otherwise, the number of boson species corresponds to `len(Nmax)`.
conserve : Literal['N', 'parity', 'None'] | Sequence[Literal['N', 'parity', 'None']]
    The symmetry to be conserved. We can conserve::

        - total particle number sum_k N_k (``conserve == 'N'``).
        - individual particle numbers N_k (``conserve[i] == 'N'``).
        - total parity (sum_i N_k) % 2 (``conserve == 'parity'``).
        - individual parities N_k % 2 (``conserve[i] == 'parity'``).
        - nothing (``conserve == 'None'`` or ``conserve[i] == 'None'``).

    A `Literal` corresponds to symmetries involving all boson species, such as the total
    particle number (``conserve == 'N'``) or the total parity (``conserve == 'parity'``).
    For a sequence, the entry ``conserve[i]`` corresponds to the symmetry of boson species `k`,
    such that, e.g., ``conserve[k] == 'N'`` signifies that its particle number is conserved.

    Conserves nothing by default.
filling : float | None
    Average total filling (that is, filling of all species together). Used to define the
    on-site operators ``dN`` and ``dNdN`` if ``filling is not None``.

Attributes
----------
conserve : Literal['N', 'parity', 'None'] | list[Literal['N', 'parity', 'None']]
    The conserved symmetry, see above.
filling : float | None
    Average total filling.
num_species, Nmax, creators, annihilators
    see :class:`BosonicDOF`

)pydoc";

    spinless_boson_site
      .def(py::init<py::object,
                    py::object,
                    std::optional<float64>,
                    TensorBackend::Ptr,
                    std::optional<std::string>>(),
           py::arg("Nmax") = 1,
           py::arg("conserve") = py::none(),
           py::arg("filling") = py::none(),
           py::arg("backend") = nullptr,
           py::arg("default_device") = py::none())
      .def_readwrite("conserve", &SpinlessBosonSite::conserve)
      .def_readwrite("filling", &SpinlessBosonSite::filling)
      .def("__repr__", &SpinlessBosonSite::repr);

    py::class_<SpinlessFermionSite, FermionicDOF, py::smart_holder> spinless_fermion_site(
      m, "SpinlessFermionSite");
    spinless_fermion_site.doc() = R"pydoc(Site for (possibly multiple) spinless fermions.

TODO describe onsite operators

.. todo ::
    For now, assume that the symmetry needs to capture the fermionic statistics.
    Do not think about JW strings yet...
    That is also the reason why NoSymmetry is not an option here

Parameters
----------
num_species : int
    Number of fermion species.
conserve : Literal['N', 'parity'] | Sequence[Literal['N', 'parity', 'None']]
    The symmetry to be conserved. We can conserve::

        - total fermion number sum_i N_k (``conserve == 'N'``).
        - individual fermion numbers N_k (``conserve[i] == 'N'``).
        - total fermion parity (sum_i N_k) % 2 (``conserve == 'parity'``).
        - individual fermion parities N_k % 2 (``conserve[i] == 'parity'``).
        - nothing for an individual fermion (``conserve[i] == 'None'``); .

    A `Literal` corresponds to symmetries involving all fermion species, such as the total
    fermion number (``conserve == 'N'``) or the total fermion parity
    (``conserve == 'parity'``). For a sequence, the entry ``conserve[k]`` corresponds to the
    symmetry of fermion species `k`, such that, e.g., ``conserve[k] == 'N'`` signifies that
    its fermion number is conserved.

    Note that the total fermion parity is always conserved. It is thus always part of the
    symmetry. Hence, ``conserve == 'None'`` is not a valid value. On the other hand,
    ``conserve = ['None']`` is interpreted as valid and the resulting symmetry conserves the
    fermionic parity.

    Conserves total fermion parity by default.
filling : float | None
    Average total filling (that is, filling of all species together). Used to define the
    on-site operators ``dN`` and ``dNdN`` if ``filling is not None``.

Attributes
----------
num_species : int
    Number of fermion species.
conserve : Literal['N', 'parity'] | list[Literal['N', 'parity', 'None']]
    The conserved symmetry, see above.
filling : float, optional
    Average total filling.
creators, annihilators
    see :class:`FermionicDOF`

)pydoc";

    spinless_fermion_site
      .def(py::init<int64,
                    py::object,
                    std::optional<float64>,
                    TensorBackend::Ptr,
                    std::optional<std::string>>(),
           py::arg("num_species") = 1,
           py::arg("conserve") = "parity",
           py::arg("filling") = py::none(),
           py::arg("backend") = nullptr,
           py::arg("default_device") = py::none())
      .def_readwrite("num_species", &SpinlessFermionSite::num_species)
      .def_readwrite("conserve", &SpinlessFermionSite::conserve)
      .def_readwrite("filling", &SpinlessFermionSite::filling)
      .def("__repr__", &SpinlessFermionSite::repr);

    py::class_<SpinHalfFermionSite, SpinDOF, FermionicDOF, py::smart_holder>
      spin_half_fermion_site(m, "SpinHalfFermionSite");
    spin_half_fermion_site.doc() = R"pydoc(Site for spin-1/2 fermions.

TODO describe onsite operators

Parameters
----------
conserve_N : Literal['N', 'parity']
    The fermion symmetry to be conserved. We can conserve::

        - total fermion number N_up + N_down (``conserve == 'N'``).
        - total fermion parity (N_up + N_down) % 2 (``conserve == 'parity'``).

    Note that the total fermion parity is always conserved and is thus always part of the
    total symmetry. Hence, ``conserve == 'None'`` is not a valid choice.
    Conserves total fermion parity by default.
conserve_S : Literal['SU(2)', 'Sz', 'parity', 'None']
    The spin symmetry to be conserved. We can conserve::

        - SU(2), the full spin rotation symmetry.
        - Sz (= U(1) symmetry), with sector labels corresponding to ``2 * Sz``.
        - Sz parity (= Z_2 symmetry), with sector labels corresponding to ``(Sz + S_tot) % 2``.
        - nothing.

    Conserves nothing by default.
filling : float | None
    Average total filling (that is, filling of spin up and spin down fermions together). Used
    to define the on-site operators ``dN`` and ``dNdN`` if ``filling is not None``.

Attributes
----------
conserve_N : Literal['N', 'parity']
    The conserved symmetry, see above.
conserve_S : Literal['SU(2)', 'Sz', 'parity', 'None']
    The conserved spin symmetry, see above.
filling : float, optional
    Average total filling.
creators, annihilators
    see :class:`FermionicDOF`
spin_vector
    see :class:`SpinDOF`

)pydoc";

    spin_half_fermion_site
      .def(py::init<std::string,
                    std::optional<std::string>,
                    std::optional<float64>,
                    TensorBackend::Ptr,
                    std::optional<std::string>>(),
           py::arg("conserve_N") = "parity",
           py::arg("conserve_S") = py::none(),
           py::arg("filling") = py::none(),
           py::arg("backend") = nullptr,
           py::arg("default_device") = py::none())
      .def_readwrite("conserve_N", &SpinHalfFermionSite::conserve_N)
      .def_readwrite("conserve_S", &SpinHalfFermionSite::conserve_S)
      .def_readwrite("filling", &SpinHalfFermionSite::filling)
      .def("test_sanity", &SpinHalfFermionSite::test_sanity, R"pydoc(Perform sanity checks.)pydoc")
      .def("__repr__", &SpinHalfFermionSite::repr);

    py::class_<ClockSite, ClockDOF, py::smart_holder> clock_site(m, "ClockSite");
    clock_site.doc() = R"pydoc(Class for sites that have a single quantum clock degree of freedom.

TODO describe onsite operators

Parameters
----------
q : int
    Number of states per site.
conserve : Literal['Z_N', 'None']
    The symmetry to be conserved. We can conserve::

        - Z_N symmetry.
        - nothing.

Attributes
----------
conserve : Literal['Z_N', 'None']
    The conserved symmetry, see above.
q, clock_operators
    see :class:`ClockDOF`

)pydoc";

    clock_site
      .def(py::init<int64,
                    std::optional<std::string>,
                    TensorBackend::Ptr,
                    std::optional<std::string>>(),
           py::arg("q"),
           py::arg("conserve") = py::none(),
           py::arg("backend") = nullptr,
           py::arg("default_device") = py::none())
      .def_readwrite("q", &ClockSite::q)
      .def_readwrite("conserve", &ClockSite::conserve)
      .def("__repr__", &ClockSite::repr);

    py::class_<AnyonSite, AnyonDOF, py::smart_holder> anyon_site(m, "AnyonSite");
    anyon_site.doc() =
      R"pydoc(Class for anyon models where the local Hilbert space contains all sectors once.

Parameters
----------
symmetry : Symmetry
    The symmetry describing the anyons.
sector_names : sequence of str or None
    The sector names that appear in the onsite projection operators. The `i`th operator is
    called `f'P_{sector_names[i]}'` and projects onto the `i`th sector in
    `leg.sector_decomposition`. For `None` entries (default), no projection operators are
    constructed.

)pydoc";

    anyon_site
      .def(py::init<Symmetry::Ptr, TensorBackend::Ptr, std::optional<std::string>>(),
           py::arg("symmetry"),
           py::arg("backend") = nullptr,
           py::arg("default_device") = py::none())
      .def("__repr__", &AnyonSite::repr);

    py::class_<FibonacciAnyonSite, AnyonSite, py::smart_holder> fibonacci_anyon_site(
      m, "FibonacciAnyonSite");
    fibonacci_anyon_site.doc() =
      R"pydoc(Class for sites containing the trivial and the Fibonacci / tau sectors.

Projectors onto the onsite vacuum and tau sectors are automatically constructed
and are named `'P_vac'` and `'P_tau'`, respectively.

Parameters
----------
handedness: Literal['left', 'right']
    The handedness of the anyons.

)pydoc";

    fibonacci_anyon_site
      .def(py::init<TensorBackend::Ptr, std::optional<std::string>>(),
           py::arg("backend") = nullptr,
           py::arg("default_device") = py::none())
      .def("__repr__", &FibonacciAnyonSite::repr);

    py::class_<IsingAnyonSite, AnyonSite, py::smart_holder> ising_anyon_site(m, "IsingAnyonSite");
    ising_anyon_site.doc() =
      R"pydoc(Class for sites containing the trivial, the Ising / sigma, and the fermion / psi sectors.

Projectors onto the onsite vacuum, sigma and psi sectors are automatically constructed and are
named `'P_vac'`, `'P_sigma'`, and `'P_psi'`, respectively.

Parameters
----------
`nu`: odd int
    Specifies the Ising anyons as different `nu` correspond to different topological twists.

)pydoc";

    ising_anyon_site
      .def(py::init<int, TensorBackend::Ptr, std::optional<std::string>>(),
           py::arg("nu") = 1,
           py::arg("backend") = nullptr,
           py::arg("default_device") = py::none())
      .def("__repr__", &IsingAnyonSite::repr);

    py::class_<GoldenSite, AnyonDOF, py::smart_holder> golden_site(m, "GoldenSite");
    golden_site.doc() =
      R"pydoc(Class for Fibonacci anyon models where the local Hilbert space only contains the tau sector.

Parameters
----------
handedness: Literal['left', 'right']
    The handedness of the anyons.

)pydoc";

    golden_site
      .def(py::init<std::string, TensorBackend::Ptr, std::optional<std::string>>(),
           py::arg("handedness") = "left",
           py::arg("backend") = nullptr,
           py::arg("default_device") = py::none())
      .def("__repr__", &GoldenSite::repr);

    py::class_<SU2kSpin1Site, AnyonDOF, py::smart_holder> su2k_spin1_site(m, "SU2kSpin1Site");
    su2k_spin1_site.doc() =
      R"pydoc(Class for SU(2)_k anyon models where the local Hilbert space only contains the spin-1 sector.

Parameters
----------
k : int
    Level of the SU(2)_k anyon model / symmetry.
handedness: Literal['left', 'right']
    The handedness of the anyons.

)pydoc";

    su2k_spin1_site
      .def(py::init<int64, TensorBackend::Ptr, std::optional<std::string>>(),
           py::arg("k"),
           py::arg("backend") = nullptr,
           py::arg("default_device") = py::none())
      .def_readwrite("k", &SU2kSpin1Site::k)
      .def("__repr__", &SU2kSpin1Site::repr);
}

} // namespace cyten
