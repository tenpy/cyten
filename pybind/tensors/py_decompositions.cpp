#include <cyten/tensors/decompositions.h>
#include <cyten/tensors/diagonal_tensor.h>
#include <cyten/tensors/mask.h>
#include <cyten/tensors/tensor.h>
#include <cyten/tools.h>

#include "../py_cyten_pybind11.h"

#include <cmath>
#include <limits>
#include <optional>
#include <string>
#include <tuple>

namespace cyten {

namespace {

LegLabels
py_leg_labels(py::object seq)
{
    LegLabels out;
    for (auto item : py::reinterpret_borrow<py::iterable>(seq)) {
        if (item.is_none()) {
            out.push_back(std::nullopt);
        } else {
            out.push_back(item.cast<std::string>());
        }
    }
    return out;
}

std::optional<LegLabels>
py_opt_labels(py::object obj)
{
    if (obj.is_none()) {
        return std::nullopt;
    }
    return py_leg_labels(to_iterable(obj));
}

std::optional<std::string>
py_opt_string(py::object obj)
{
    if (obj.is_none()) {
        return std::nullopt;
    }
    return obj.cast<std::string>();
}

py::object
entropy_numpy(py::object p, py::object n)
{
    auto np = py::module_::import("numpy");
    p = np.attr("asarray")(p);
    p = np.attr("real_if_close")(p);
    p = p.attr("__getitem__")(p.attr("__gt__")(1e-30));
    auto is_inf = py::module_::import("math").attr("isinf");
    if (n.equal(py::int_(1))) {
        return -np.attr("inner")(np.attr("log")(p), p);
    }
    if (is_inf(n).cast<bool>()) {
        return -np.attr("log")(np.attr("max")(p));
    }
    float64 n_f = n.cast<float64>();
    return np.attr("log")(np.attr("sum")(p.attr("__pow__")(n))).attr("__truediv__")(1.0 - n_f);
}

} // namespace

void
bind_tensors_decompositions(py::module_& m)
{
    m.def("apply_mask_DiagonalTensor",
          &apply_mask_DiagonalTensor,
          py::arg("tensor"),
          py::arg("mask"),
          R"pydoc(Apply a mask to *both* legs of a diagonal tensor.)pydoc");

    m.def(
      "eigh",
      [](TensorCPtr tensor, py::object new_labels, bool new_leg_dual, py::object sort) {
          return eigh(std::move(tensor),
                      py_leg_labels(to_iterable(new_labels)),
                      new_leg_dual,
                      py_opt_string(sort));
      },
      py::arg("tensor"),
      py::arg("new_labels"),
      py::arg("new_leg_dual"),
      py::arg("sort") = py::none(),
      R"pydoc(The eigen-decomposition of a hermitian tensor.)pydoc");

    m.def(
      "entropy",
      [](py::object p, py::object n) {
          if (py::isinstance<DiagonalTensor>(p)) {
              float64 n_f = 1.;
              if (py::module_::import("math").attr("isinf")(n).cast<bool>()) {
                  n_f = std::numeric_limits<float64>::infinity();
              } else {
                  n_f = n.cast<float64>();
              }
              return py::cast(entropy(p.cast<DiagonalTensorCPtr>(), n_f)).attr("to_numpy")();
          }
          return entropy_numpy(std::move(p), std::move(n));
      },
      py::arg("p"),
      py::arg("n") = 1,
      R"pydoc(The entropy of a probability distribution.)pydoc");

    m.def(
      "lq",
      [](TensorCPtr tensor, py::object new_labels, bool new_leg_dual, bool charge_leg_top) {
          return lq(std::move(tensor), py_opt_labels(new_labels), new_leg_dual, charge_leg_top);
      },
      py::arg("tensor"),
      py::arg("new_labels") = py::none(),
      py::arg("new_leg_dual") = false,
      py::arg("charge_leg_top") = true,
      R"pydoc(The LQ decomposition of a tensor.)pydoc");

    m.def(
      "qr",
      [](TensorCPtr tensor, py::object new_labels, bool new_leg_dual, bool charge_leg_top) {
          return qr(std::move(tensor), py_opt_labels(new_labels), new_leg_dual, charge_leg_top);
      },
      py::arg("tensor"),
      py::arg("new_labels") = py::none(),
      py::arg("new_leg_dual") = false,
      py::arg("charge_leg_top") = true,
      R"pydoc(The QR decomposition of a tensor.)pydoc");

    m.def(
      "svd",
      [](TensorCPtr tensor,
         py::object new_labels,
         bool new_leg_dual,
         bool charge_leg_top,
         py::object algorithm) {
          return svd(std::move(tensor),
                     py_opt_labels(new_labels),
                     new_leg_dual,
                     charge_leg_top,
                     py_opt_string(algorithm));
      },
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
      [](DiagonalTensorCPtr S,
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
                                          py_opt_labels(mask_labels));
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
      [](TensorCPtr tensor,
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
                               py_opt_labels(new_labels),
                               new_leg_dual,
                               charge_leg_top,
                               py_opt_string(algorithm),
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
