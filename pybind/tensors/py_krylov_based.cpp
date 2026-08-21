#include <cyten/tensors/krylov_based.h>

#include "../doc_plus.h"
#include "../py_cyten_pybind11.h"

#include "docstrings/tensors/krylov_based.h"

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <optional>
#include <utility>
#include <vector>

namespace cyten {

namespace {

class PyKrylovBased
  : public KrylovBased
  , public py::trampoline_self_life_support
{
  public:
    using KrylovBased::KrylovBased;

    int64 _build_krylov() override { PYBIND11_OVERRIDE_PURE(int64, KrylovBased, _build_krylov); }

    void _calc_result_krylov(int64 k) override
    {
        PYBIND11_OVERRIDE_PURE(void, KrylovBased, _calc_result_krylov, k);
    }

    bool _converged(int64 k) override { PYBIND11_OVERRIDE_PURE(bool, KrylovBased, _converged, k); }

    void _to_cache(VectorLike::Ptr psi) override
    {
        PYBIND11_OVERRIDE(void, KrylovBased, _to_cache, psi);
    }

    VectorLike::Ptr _rebuild_krylov_for_result_full(VectorLike::Ptr psif, int64 N_max) override
    {
        PYBIND11_OVERRIDE(
          VectorLike::Ptr, KrylovBased, _rebuild_krylov_for_result_full, psif, N_max);
    }
};

py::object
optional_float_to_py(std::optional<float64> const& v)
{
    if (!v.has_value()) {
        return py::none();
    }
    return py::float_(*v);
}

py::object
optional_complex_to_py(std::optional<complex128> const& v)
{
    if (!v.has_value()) {
        return py::none();
    }
    return py::cast(*v);
}

py::array
eigenvalues_to_numpy(std::vector<complex128> const& E0s)
{
    py::array_t<complex128> arr(static_cast<py::ssize_t>(E0s.size()));
    auto r = arr.mutable_unchecked<1>();
    for (py::ssize_t i = 0; i < arr.shape(0); ++i) {
        r(i) = E0s[static_cast<std::size_t>(i)];
    }
    return arr;
}

} // namespace

void
bind_tensors_krylov_based(py::module_& m)
{
    py::class_<KrylovBased, PyKrylovBased, py::smart_holder> krylov_based(m, "KrylovBased");
    krylov_based.doc() = DOC(cyten, KrylovBased);

    krylov_based
      .def(py::init<LinearOperator::Ptr, VectorLike::Ptr, py::object>(),
           py::arg("H"),
           py::arg("psi0"),
           py::arg("options") = py::none())
      .def_readwrite("H", &KrylovBased::H)
      .def_readwrite("psi0", &KrylovBased::psi0)
      .def_readwrite("options", &KrylovBased::options)
      .def_readwrite("N_min", &KrylovBased::N_min)
      .def_readwrite("N_max", &KrylovBased::N_max)
      .def_readwrite("N_cache", &KrylovBased::N_cache)
      .def_readwrite("P_tol", &KrylovBased::P_tol)
      .def_readwrite("min_gap", &KrylovBased::min_gap)
      .def_readwrite("reortho", &KrylovBased::reortho)
      .def_property(
        "E_shift",
        [](KrylovBased const& self) { return optional_float_to_py(self.E_shift); },
        [](KrylovBased& self, py::object v) {
            if (v.is_none()) {
                self.E_shift = std::nullopt;
            } else {
                self.E_shift = v.cast<float64>();
            }
        })
      .def_readwrite("_cutoff", &KrylovBased::_cutoff)
      .def_property_readonly("Es", &KrylovBased::Es_numpy)
      .def_property_readonly("_h_krylov", &KrylovBased::h_krylov_numpy)
      .def_property_readonly("_result_krylov", &KrylovBased::result_krylov_numpy)
      .def("_reset_krylov_state",
           &KrylovBased::_reset_krylov_state,
           "Clear cached Krylov vectors and the projected Hessenberg matrix.");

    py::class_<GMRES, py::smart_holder> gmres(m, "GMRES");
    gmres.doc() = DOC(cyten, GMRES);

    gmres
      .def(py::init<LinearOperator::Ptr, VectorLike::Ptr, VectorLike::Ptr, py::object>(),
           py::arg("A"),
           py::arg("x"),
           py::arg("b"),
           py::arg("options") = py::none())
      .def_readwrite("A", &GMRES::A)
      .def_readwrite("x", &GMRES::x)
      .def_readwrite("b", &GMRES::b)
      .def_readwrite("options", &GMRES::options)
      .def_readwrite("N_min", &GMRES::N_min)
      .def_readwrite("N_max", &GMRES::N_max)
      .def_readwrite("restart", &GMRES::restart)
      .def_readwrite("res", &GMRES::res)
      .def("run", &GMRES::run)
      .def("arnoldi", &GMRES::arnoldi, py::arg("k"))
      .def("apply_givens_rotation", &GMRES::apply_givens_rotation, py::arg("k"))
      .def("givens_rotation", &GMRES::givens_rotation, py::arg("k"))
      .def("backsolve", &GMRES::backsolve, py::arg("k"))
      .def("reset", &GMRES::reset);

    py::class_<Arnoldi, KrylovBased, py::smart_holder> arnoldi(m, "Arnoldi");
    arnoldi.doc() = DOC(cyten, Arnoldi);

    arnoldi
      .def(py::init<LinearOperator::Ptr, VectorLike::Ptr, py::object>(),
           py::arg("H"),
           py::arg("psi0"),
           py::arg("options") = py::none())
      .def_readwrite("E_tol", &Arnoldi::E_tol)
      .def_readwrite("which", &Arnoldi::which)
      .def_readwrite("num_ev", &Arnoldi::num_ev)
      .def(
        "run",
        [](Arnoldi& self) {
            auto [E0s, psis, N] = self.run();
            return py::make_tuple(eigenvalues_to_numpy(E0s), std::move(psis), N);
        },
        DOC(cyten, Arnoldi, run));

    py::class_<ArnoldiEvolution, Arnoldi, py::smart_holder> arnoldi_evolution(m,
                                                                              "ArnoldiEvolution");
    arnoldi_evolution.doc() = DOC(cyten, ArnoldiEvolution);

    arnoldi_evolution
      .def(py::init<LinearOperator::Ptr, VectorLike::Ptr, py::object>(),
           py::arg("H"),
           py::arg("psi0"),
           py::arg("options") = py::none())
      .def_readwrite("_result_norm", &ArnoldiEvolution::_result_norm)
      .def_property(
        "delta",
        [](ArnoldiEvolution const& self) { return optional_complex_to_py(self.delta); },
        [](ArnoldiEvolution& self, py::object v) {
            if (v.is_none()) {
                self.delta = std::nullopt;
            } else {
                self.delta = v.cast<complex128>();
            }
        })
      .def("run",
           &ArnoldiEvolution::run,
           py::arg("delta"),
           py::arg("normalize") = py::none(),
           DOC(cyten, ArnoldiEvolution, run));

    py::class_<LanczosGroundState, KrylovBased, py::smart_holder> lanczos_ground_state(
      m, "LanczosGroundState");
    lanczos_ground_state.doc() = DOC(cyten, LanczosGroundState);

    lanczos_ground_state
      .def(py::init<LinearOperator::Ptr, VectorLike::Ptr, py::object>(),
           py::arg("H"),
           py::arg("psi0"),
           py::arg("options") = py::none())
      .def_readwrite("E_tol", &LanczosGroundState::E_tol)
      .def("run",
           &LanczosGroundState::run,
           DOC(cyten, LanczosGroundState, run));

    py::class_<LanczosEvolution, LanczosGroundState, py::smart_holder> lanczos_evolution(
      m, "LanczosEvolution");
    lanczos_evolution.doc() = DOC(cyten, LanczosEvolution);

    lanczos_evolution
      .def(py::init<LinearOperator::Ptr, VectorLike::Ptr, py::object>(),
           py::arg("H"),
           py::arg("psi0"),
           py::arg("options") = py::none())
      .def_readwrite("_result_norm", &LanczosEvolution::_result_norm)
      .def_property(
        "delta",
        [](LanczosEvolution const& self) { return optional_complex_to_py(self.delta); },
        [](LanczosEvolution& self, py::object v) {
            if (v.is_none()) {
                self.delta = std::nullopt;
            } else {
                self.delta = v.cast<complex128>();
            }
        })
      .def("run",
           &LanczosEvolution::run,
           py::arg("delta"),
           py::arg("normalize") = py::none(),
           DOC(cyten, LanczosEvolution, run));

    m.def("lanczos",
          &lanczos,
          py::arg("H"),
          py::arg("psi"),
          py::arg("options") = py::none(),
          DOC(cyten, lanczos));
}

} // namespace cyten
