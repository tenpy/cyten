#include <cyten/tensors/decompositions.h>

#include "../py_cyten_pybind11.h"

#include <optional>
#include <string>
#include <tuple>

namespace cyten {

void
bind_tensors_decompositions(py::module_& m)
{
    m.def("apply_mask_DiagonalTensor",
          &apply_mask_DiagonalTensor,
          py::arg("tensor"),
          py::arg("mask"),
          R"pydoc(Apply a mask to *both* legs of a diagonal tensor.)pydoc");

    m.def("eigh",
          &eigh,
          py::arg("tensor"),
          py::arg("new_labels"),
          py::arg("new_leg_dual"),
          py::arg("sort") = py::none(),
          R"pydoc(The eigen-decomposition of a hermitian tensor.)pydoc");

    m.def("entropy",
          &entropy,
          py::arg("p"),
          py::arg("n") = 1,
          R"pydoc(The entropy of a probability distribution.)pydoc");

    m.def("lq",
          &lq,
          py::arg("tensor"),
          py::arg("new_labels") = py::none(),
          py::arg("new_leg_dual") = false,
          py::arg("charge_leg_top") = true,
          R"pydoc(The LQ decomposition of a tensor.)pydoc");

    m.def("qr",
          &qr,
          py::arg("tensor"),
          py::arg("new_labels") = py::none(),
          py::arg("new_leg_dual") = false,
          py::arg("charge_leg_top") = true,
          R"pydoc(The QR decomposition of a tensor.)pydoc");

    m.def("svd",
          &svd,
          py::arg("tensor"),
          py::arg("new_labels") = py::none(),
          py::arg("new_leg_dual") = false,
          py::arg("charge_leg_top") = true,
          py::arg("algorithm") = py::none(),
          R"pydoc(The singular value decomposition (SVD) of a tensor.)pydoc");

    m.def("svd_apply_mask",
          &svd_apply_mask,
          py::arg("U"),
          py::arg("S"),
          py::arg("Vh"),
          py::arg("mask"),
          R"pydoc(Truncate an existing SVD)pydoc");

    m.def(
      "truncate_singular_values",
      [](py::object S,
         py::object chi_max,
         py::object chi_min,
         py::object degeneracy_tol,
         py::object trunc_cut,
         py::object svd_min,
         bool minimize_error,
         py::object mask_labels) {
          std::optional<int64> chi_max_opt;
          if (!chi_max.is_none()) {
              chi_max_opt = chi_max.cast<int64>();
          }
          int64 chi_min_v = chi_min.is_none() ? 1 : chi_min.cast<int64>();
          float64 degeneracy_tol_v =
            degeneracy_tol.is_none() ? 0. : degeneracy_tol.cast<float64>();
          float64 trunc_cut_v = trunc_cut.is_none() ? 0. : trunc_cut.cast<float64>();
          float64 svd_min_v = svd_min.is_none() ? 0. : svd_min.cast<float64>();
          return truncate_singular_values(std::move(S),
                                          chi_max_opt,
                                          chi_min_v,
                                          degeneracy_tol_v,
                                          trunc_cut_v,
                                          svd_min_v,
                                          minimize_error,
                                          std::move(mask_labels));
      },
      py::arg("S"),
      py::arg("chi_max") = py::none(),
      py::arg("chi_min") = 1,
      py::arg("degeneracy_tol") = 0,
      py::arg("trunc_cut") = 0,
      py::arg("svd_min") = 0,
      py::arg("minimize_error") = true,
      py::arg("mask_labels") = py::none(),
      R"pydoc(Given *normalized* singular values, determine which to keep.)pydoc");

    m.def(
      "truncated_svd",
      [](py::object tensor,
         py::object new_labels,
         bool new_leg_dual,
         bool charge_leg_top,
         py::object algorithm,
         py::object normalize_to,
         py::object chi_max,
         py::object chi_min,
         py::object degeneracy_tol,
         py::object trunc_cut,
         py::object svd_min) {
          std::optional<float64> normalize_opt;
          if (!normalize_to.is_none()) {
              normalize_opt = normalize_to.cast<float64>();
          }
          std::optional<int64> chi_max_opt;
          if (!chi_max.is_none()) {
              chi_max_opt = chi_max.cast<int64>();
          }
          int64 chi_min_v = chi_min.is_none() ? 1 : chi_min.cast<int64>();
          float64 degeneracy_tol_v =
            degeneracy_tol.is_none() ? 0. : degeneracy_tol.cast<float64>();
          float64 trunc_cut_v = trunc_cut.is_none() ? 0. : trunc_cut.cast<float64>();
          float64 svd_min_v = svd_min.is_none() ? 0. : svd_min.cast<float64>();
          return truncated_svd(std::move(tensor),
                               std::move(new_labels),
                               new_leg_dual,
                               charge_leg_top,
                               std::move(algorithm),
                               normalize_opt,
                               chi_max_opt,
                               chi_min_v,
                               degeneracy_tol_v,
                               trunc_cut_v,
                               svd_min_v);
      },
      py::arg("tensor"),
      py::arg("new_labels") = py::none(),
      py::arg("new_leg_dual") = false,
      py::arg("charge_leg_top") = true,
      py::arg("algorithm") = py::none(),
      py::arg("normalize_to") = py::none(),
      py::arg("chi_max") = py::none(),
      py::arg("chi_min") = 1,
      py::arg("degeneracy_tol") = 0,
      py::arg("trunc_cut") = 0,
      py::arg("svd_min") = 0,
      R"pydoc(Truncated version of :func:`svd`.)pydoc");
}

} // namespace cyten
