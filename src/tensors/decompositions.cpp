#include <cyten/tensors/decompositions.h>

#include <cyten/backends/no_symmetry.h>
#include <cyten/backends/tensor_backend.h>
#include <cyten/block_backend/dtypes.h>
#include <cyten/symmetries/spaces.h>
#include <cyten/tensors/charged_tensor.h>
#include <cyten/tensors/diagonal_tensor.h>
#include <cyten/tensors/helpers.h>
#include <cyten/tensors/labels.h>
#include <cyten/tensors/mask.h>
#include <cyten/tensors/ops_algebra.h>
#include <cyten/tensors/ops_elementwise.h>
#include <cyten/tensors/ops_legs.h>
#include <cyten/tensors/symmetric_tensor.h>
#include <cyten/tensors/tensor.h>
#include <cyten/tools.h>

#include <cassert>
#include <cmath>
#include <format>
#include <stdexcept>
#include <utility>

namespace cyten {

namespace {

py::object
tensors_mod()
{
    return py::module_::import("cyten.tensors._tensors");
}

bool
is_python_instance(py::object obj, char const* class_name)
{
    return py::isinstance(obj, tensors_mod().attr(class_name));
}

bool
is_Mask(py::object obj)
{
    return is_python_instance(obj, "Mask") || py::isinstance<Mask>(obj);
}

bool
is_DiagonalTensor(py::object obj)
{
    return is_python_instance(obj, "DiagonalTensor") || py::isinstance<DiagonalTensor>(obj);
}

bool
is_Identity(py::object obj)
{
    return is_python_instance(obj, "Identity") || py::isinstance<Identity>(obj);
}

bool
is_SymmetricTensor(py::object obj)
{
    return is_python_instance(obj, "SymmetricTensor") || py::isinstance<SymmetricTensor>(obj);
}

bool
is_ChargedTensor(py::object obj)
{
    return is_python_instance(obj, "ChargedTensor") || py::isinstance<ChargedTensor>(obj);
}

bool
py_eq(py::object a, py::object b)
{
    py::object eq = a.attr("__eq__")(b);
    if (eq.is(py::reinterpret_borrow<py::object>(Py_NotImplemented))) {
        return false;
    }
    return eq.cast<bool>();
}

py::object
data_as_python(TensorBackend::DataPtr data, TensorBackend::Ptr const& /*backend*/)
{
    // C++ SymmetricTensor/Mask/DiagonalTensor ctors take DataPtr (including NoSymmetry BlockData).
    return py::cast(std::move(data));
}

py::object
make_python_symmetric_tensor(TensorBackend::DataPtr data,
                             py::object codomain,
                             py::object domain,
                             TensorBackend::Ptr backend,
                             py::object labels)
{
    return tensors_mod().attr("SymmetricTensor")(data_as_python(std::move(data), backend),
                                                 codomain,
                                                 domain,
                                                 py::arg("backend") = py::cast(backend),
                                                 py::arg("labels") = labels);
}

py::object
make_python_diagonal_tensor(TensorBackend::DataPtr data,
                            py::object leg,
                            TensorBackend::Ptr backend,
                            py::object labels)
{
    return tensors_mod().attr("DiagonalTensor")(data_as_python(std::move(data), backend),
                                                leg,
                                                py::arg("backend") = py::cast(backend),
                                                py::arg("labels") = labels);
}

py::object
make_python_mask(TensorBackend::DataPtr data,
                 py::object space_in,
                 py::object space_out,
                 bool is_projection,
                 TensorBackend::Ptr backend,
                 py::object labels)
{
    return tensors_mod().attr("Mask")(data_as_python(std::move(data), backend),
                                      space_in,
                                      space_out,
                                      py::arg("is_projection") = is_projection,
                                      py::arg("backend") = py::cast(backend),
                                      py::arg("labels") = labels);
}

py::object
make_python_charged_tensor(py::object invariant_part, py::object charged_state)
{
    return tensors_mod().attr("ChargedTensor")(invariant_part, charged_state);
}

py::object
labels_to_py(LegLabels const& labels)
{
    py::list out;
    for (auto const& lab : labels) {
        if (lab.has_value()) {
            out.append(*lab);
        } else {
            out.append(py::none());
        }
    }
    return out;
}

py::object
leg_label_to_py(LegLabel const& lab)
{
    if (lab.has_value()) {
        return py::cast(*lab);
    }
    return py::none();
}

template<typename... Args>
py::list
py_list(Args&&... args)
{
    py::list out;
    (out.append(std::forward<Args>(args)), ...);
    return out;
}

py::object
nested_labels_codomain_domain(py::object codomain_labels, py::object domain_one_label)
{
    py::list out;
    out.append(codomain_labels);
    out.append(py_list(domain_one_label));
    return out;
}

py::object
nested_labels_one_and_domain(py::object one_label, py::object domain_labels)
{
    py::list out;
    out.append(py_list(one_label));
    out.append(domain_labels);
    return out;
}

py::object
charge_leg_label_py(py::object tensor)
{
    return tensor.attr("_CHARGE_LEG_LABEL");
}

bool
is_inf(py::object n)
{
    try {
        return py::module_::import("math").attr("isinf")(n).cast<bool>() ||
               (py::hasattr(n, "__float__") && std::isinf(n.cast<float64>()));
    } catch (...) {
        return false;
    }
}

std::string
same_device2(py::object t1, py::object t2, std::string const& error_msg = "Incompatible devices.")
{
    std::string device = t1.attr("device").cast<std::string>();
    if (t2.attr("device").cast<std::string>() != device) {
        throw std::invalid_argument(error_msg);
    }
    return device;
}

} // namespace

py::object
apply_mask_DiagonalTensor(py::object tensor, py::object mask)
{
    same_device2(tensor, mask);
    assert(mask.attr("is_projection").cast<bool>());
    assert(py_eq(mask.attr("large_leg"), tensor.attr("leg")));
    auto backend = get_same_backend({ tensor, mask });
    if (is_Identity(tensor)) {
        return tensors_mod().attr("Identity")(mask.attr("small_leg"),
                                              py::arg("backend") = py::cast(backend),
                                              py::arg("labels") = tensor.attr("labels"));
    }
    auto data = backend->apply_mask_to_DiagonalTensor(tensor.cast<DiagonalTensorCPtr>(),
                                                      mask.cast<MaskCPtr>());
    return make_python_diagonal_tensor(
      std::move(data), mask.attr("small_leg"), backend, tensor.attr("labels"));
}

std::tuple<py::object, py::object>
eigh(py::object tensor, py::object new_labels, bool new_leg_dual, py::object sort)
{
    // --- hints from Python eigh ---
    // do not define decompositions for ChargedTensors.
    // If the backend requires it, combine legs first
    // first, compute a decomposition where the new leg is a ket space
    // undo the combine
    // if required, flip the leg duality
    // ---
    py::object labels_iter = to_iterable(new_labels);
    py::ssize_t nlab = py::len(labels_iter);
    LegLabel a;
    LegLabel b;
    LegLabel c;
    if (nlab == 1) {
        a = c = labels_iter.attr("__getitem__")(0).is_none()
                  ? std::nullopt
                  : std::optional(labels_iter.attr("__getitem__")(0).cast<std::string>());
        b = _dual_leg_label(a);
    } else if (nlab == 2) {
        a = c = labels_iter.attr("__getitem__")(0).is_none()
                  ? std::nullopt
                  : std::optional(labels_iter.attr("__getitem__")(0).cast<std::string>());
        b = labels_iter.attr("__getitem__")(1).is_none()
              ? std::nullopt
              : std::optional(labels_iter.attr("__getitem__")(1).cast<std::string>());
    } else if (nlab == 3) {
        auto lab0 = labels_iter.attr("__getitem__")(0);
        auto lab1 = labels_iter.attr("__getitem__")(1);
        auto lab2 = labels_iter.attr("__getitem__")(2);
        a = lab0.is_none() ? std::nullopt : std::optional(lab0.cast<std::string>());
        b = lab1.is_none() ? std::nullopt : std::optional(lab1.cast<std::string>());
        c = lab2.is_none() ? std::nullopt : std::optional(lab2.cast<std::string>());
    } else {
        throw std::invalid_argument(
          std::format("Expected 1, 2 or 3 new_labels. Got {}.", static_cast<int>(nlab)));
    }

    assert(py_eq(tensor.attr("domain"), tensor.attr("codomain")));
    if (is_ChargedTensor(tensor)) {
        // do not define decompositions for ChargedTensors.
        throw NotImplemented("eigh for ChargedTensor");
    }
    if (is_DiagonalTensor(tensor)) {
        py::object V =
          tensors_mod()
            .attr("SymmetricTensor")
            .attr("from_eye")(
              py_list(tensor.attr("leg")),
              py::arg("backend") = tensor.attr("backend"),
              py::arg("labels") =
                py_list(tensor.attr("codomain_labels").attr("__getitem__")(0), leg_label_to_py(a)),
              py::arg("dtype") = tensor.attr("dtype"),
              py::arg("device") = tensor.attr("device"));
        py::object W = tensor.attr("as_DiagonalTensor")(py::arg("guarantee_copy") = true)
                         .attr("set_labels")(py_list(leg_label_to_py(b), leg_label_to_py(c)));
        return { W, V };
    }
    tensor = tensor.attr("as_SymmetricTensor")();

    auto backend = tensor.attr("backend").cast<TensorBackend::Ptr>();
    // If the backend requires it, combine legs first
    if (!backend->can_decompose_tensors) {
        int64 n_cod = tensor.attr("num_codomain_legs").cast<int64>();
        int64 n_legs = tensor.attr("num_legs").cast<int64>();
        py::list cod_range;
        for (int64 i = 0; i < n_cod; ++i) {
            cod_range.append(i);
        }
        py::list dom_range;
        for (int64 i = n_cod; i < n_legs; ++i) {
            dom_range.append(i);
        }
        tensor =
          combine_legs(tensor, { cod_range, dom_range }, py_list(new_leg_dual, !new_leg_dual));
        backend = tensor.attr("backend").cast<TensorBackend::Ptr>();
    }

    std::optional<std::string> sort_opt;
    if (!sort.is_none()) {
        sort_opt = sort.cast<std::string>();
    }
    // first, compute a decomposition where the new leg is a ket space
    auto [w_data, v_data, new_leg] =
      backend->eigh(tensor.cast<SymmetricTensorCPtr>(), new_leg_dual, sort_opt);
    py::object W = make_python_diagonal_tensor(std::move(w_data),
                                               py::cast(new_leg),
                                               backend,
                                               py_list(leg_label_to_py(b), leg_label_to_py(c)));
    py::object V = make_python_symmetric_tensor(
      std::move(v_data),
      tensor.attr("codomain"),
      py_list(py::cast(new_leg)),
      backend,
      nested_labels_codomain_domain(tensor.attr("codomain_labels"), leg_label_to_py(a)));

    // undo the combine
    if (!backend->can_decompose_tensors) {
        V = split_legs(V, py::int_(0));
    }

    // if required, flip the leg duality
    if (new_leg_dual != new_leg->is_dual) {
        throw NotImplemented("eigh flip new_leg duality");
    }

    return { W, V };
}

py::object
entropy(py::object p, py::object n)
{
    // --- hints from Python entropy ---
    // for stability of log
    // ---
    if (is_Identity(p)) {
        throw py::type_error(
          "entropy does not support Identity. It is never a normalized distribution.");
    }
    if (is_DiagonalTensor(p)) {
        assert(p.attr("dtype").attr("is_real").cast<bool>());
        if (py_eq(n, py::int_(1))) {
            py::object logged = stable_log(p, 1e-30);
            py::object prod = p.attr("__mul__")(logged);
            return (-trace(prod)).attr("to_numpy")();
        }
        if (is_inf(n)) {
            return (-py::module_::import("numpy").attr("log")(p.attr("max")().attr("to_numpy")()));
        }
        float64 n_f = n.cast<float64>();
        py::object logged =
          py::module_::import("numpy").attr("log")(trace(p.attr("__pow__")(n)).attr("to_numpy")());
        return logged.attr("__truediv__")(1.0 - n_f);
    }
    // else: sequence of floats
    auto np = py::module_::import("numpy");
    p = np.attr("asarray")(p);
    p = np.attr("real_if_close")(p);
    p = p.attr("__getitem__")(p.attr("__gt__")(1e-30)); // for stability of log
    if (py_eq(n, py::int_(1))) {
        return -np.attr("inner")(np.attr("log")(p), p);
    }
    if (is_inf(n)) {
        return -np.attr("log")(np.attr("max")(p));
    }
    float64 n_f = n.cast<float64>();
    return np.attr("log")(np.attr("sum")(p.attr("__pow__")(n))).attr("__truediv__")(1.0 - n_f);
}

std::tuple<py::object, py::object>
qr(py::object tensor, py::object new_labels, bool new_leg_dual, bool charge_leg_top)
{
    if (is_ChargedTensor(tensor)) {
        py::object inv_part = tensor.attr("invariant_part");
        if (!charge_leg_top) {
            inv_part = move_leg(inv_part,
                                charge_leg_label_py(tensor),
                                /*codomain_pos=*/0,
                                /*domain_pos=*/std::nullopt,
                                py::none(),
                                py::bool_(false));
        }
        auto [Q, R] = qr(inv_part, new_labels, new_leg_dual, true);
        if (charge_leg_top) {
            R = make_python_charged_tensor(R, tensor.attr("charged_state"));
        } else {
            Q = move_leg(Q,
                         charge_leg_label_py(tensor),
                         /*codomain_pos=*/std::nullopt,
                         /*domain_pos=*/0,
                         py::none(),
                         py::bool_(false));
            Q = make_python_charged_tensor(Q, tensor.attr("charged_state"));
        }
        return { Q, R };
    }

    auto [a, b] = _decomposition_labels(new_labels);
    auto [tens, new_co_domain, combine_codomain, combine_domain] =
      _decomposition_prepare(tensor, new_leg_dual);
    auto backend = tens.attr("backend").cast<TensorBackend::Ptr>();
    auto [q_data, r_data] = backend->qr(tens.cast<SymmetricTensorCPtr>(), new_co_domain);
    py::object Q = make_python_symmetric_tensor(
      std::move(q_data),
      tens.attr("codomain"),
      py::cast(new_co_domain),
      backend,
      nested_labels_codomain_domain(tens.attr("codomain_labels"), leg_label_to_py(a)));
    py::object R = make_python_symmetric_tensor(
      std::move(r_data),
      py::cast(new_co_domain),
      tens.attr("domain"),
      backend,
      nested_labels_one_and_domain(leg_label_to_py(b), tens.attr("domain_labels")));
    if (combine_codomain) {
        Q = split_legs(Q, py::int_(0));
    }
    if (combine_domain) {
        R = split_legs(R, py::int_(-1));
    }
    return { Q, R };
}

std::tuple<py::object, py::object>
lq(py::object tensor, py::object new_labels, bool new_leg_dual, bool charge_leg_top)
{
    if (is_ChargedTensor(tensor)) {
        py::object inv_part = tensor.attr("invariant_part");
        if (!charge_leg_top) {
            inv_part = move_leg(inv_part,
                                charge_leg_label_py(tensor),
                                /*codomain_pos=*/0,
                                /*domain_pos=*/std::nullopt,
                                py::none(),
                                py::bool_(false));
        }
        auto [L, Q] = lq(inv_part, new_labels, new_leg_dual, true);
        if (charge_leg_top) {
            Q = make_python_charged_tensor(Q, tensor.attr("charged_state"));
        } else {
            L = move_leg(L,
                         charge_leg_label_py(tensor),
                         /*codomain_pos=*/std::nullopt,
                         /*domain_pos=*/0,
                         py::none(),
                         py::bool_(false));
            L = make_python_charged_tensor(L, tensor.attr("charged_state"));
        }
        return { L, Q };
    }

    auto [a, b] = _decomposition_labels(new_labels);
    auto [tens, new_co_domain, combine_codomain, combine_domain] =
      _decomposition_prepare(tensor, new_leg_dual);
    auto backend = tens.attr("backend").cast<TensorBackend::Ptr>();
    auto [l_data, q_data] = backend->lq(tens.cast<SymmetricTensorCPtr>(), new_co_domain);
    py::object L = make_python_symmetric_tensor(
      std::move(l_data),
      tens.attr("codomain"),
      py::cast(new_co_domain),
      backend,
      nested_labels_codomain_domain(tens.attr("codomain_labels"), leg_label_to_py(a)));
    py::object Q = make_python_symmetric_tensor(
      std::move(q_data),
      py::cast(new_co_domain),
      tens.attr("domain"),
      backend,
      nested_labels_one_and_domain(leg_label_to_py(b), tens.attr("domain_labels")));
    if (combine_codomain) {
        L = split_legs(L, py::int_(0));
    }
    if (combine_domain) {
        Q = split_legs(Q, py::int_(-1));
    }
    return { L, Q };
}

std::tuple<py::object, py::object, py::object>
svd(py::object tensor,
    py::object new_labels,
    bool new_leg_dual,
    bool charge_leg_top,
    py::object algorithm)
{
    // --- hints from Python svd ---
    // split legs, if they were previously combined
    // ---
    if (is_ChargedTensor(tensor)) {
        py::object inv_part = tensor.attr("invariant_part");
        if (!charge_leg_top) {
            inv_part = move_leg(inv_part,
                                charge_leg_label_py(tensor),
                                /*codomain_pos=*/0,
                                /*domain_pos=*/std::nullopt,
                                py::none(),
                                py::bool_(false));
        }
        // Intentional: pass algorithm by keyword (Python positional call was ambiguous).
        auto [U, S, Vh] = svd(inv_part, new_labels, new_leg_dual, true, algorithm);
        if (charge_leg_top) {
            Vh = make_python_charged_tensor(Vh, tensor.attr("charged_state"));
        } else {
            U = move_leg(U,
                         charge_leg_label_py(tensor),
                         /*codomain_pos=*/std::nullopt,
                         /*domain_pos=*/0,
                         py::none(),
                         py::bool_(false));
            U = make_python_charged_tensor(U, tensor.attr("charged_state"));
        }
        return { U, S, Vh };
    }

    auto [a, b, c, d] = _svd_new_labels(new_labels);
    auto [tens, new_co_domain, combine_codomain, combine_domain] =
      _decomposition_prepare(tensor, new_leg_dual);
    auto backend = tens.attr("backend").cast<TensorBackend::Ptr>();
    std::optional<std::string> algo;
    if (!algorithm.is_none()) {
        algo = algorithm.cast<std::string>();
    }
    auto [u_data, s_data, vh_data] =
      backend->svd(tens.cast<SymmetricTensorCPtr>(), new_co_domain, algo);
    py::object U = make_python_symmetric_tensor(
      std::move(u_data),
      tens.attr("codomain"),
      py::cast(new_co_domain),
      backend,
      nested_labels_codomain_domain(tens.attr("codomain_labels"), leg_label_to_py(a)));
    py::object S = make_python_diagonal_tensor(std::move(s_data),
                                               py::cast(new_co_domain).attr("__getitem__")(0),
                                               backend,
                                               py_list(leg_label_to_py(b), leg_label_to_py(c)));
    py::object Vh = make_python_symmetric_tensor(
      std::move(vh_data),
      py::cast(new_co_domain),
      tens.attr("domain"),
      backend,
      nested_labels_one_and_domain(leg_label_to_py(d), tens.attr("domain_labels")));
    // split legs, if they were previously combined
    if (combine_codomain) {
        U = split_legs(U, py::int_(0));
    }
    if (combine_domain) {
        Vh = split_legs(Vh, py::int_(-1));
    }
    return { U, S, Vh };
}

std::tuple<py::object, py::object, py::object>
svd_apply_mask(py::object U, py::object S, py::object Vh, py::object mask)
{
    assert(mask.attr("is_projection").cast<bool>());
    assert(
      py_eq(mask.attr("domain").attr("__getitem__")(0), S.attr("domain").attr("__getitem__")(0)));

    U = _compose_with_Mask(U, dagger(mask), -1);
    S = apply_mask_DiagonalTensor(S, mask);
    Vh = _compose_with_Mask(Vh, mask, 0);
    return { U, S, Vh };
}

std::tuple<py::object, float64, float64>
truncate_singular_values(py::object S,
                         std::optional<int64> chi_max,
                         int64 chi_min,
                         float64 degeneracy_tol,
                         float64 trunc_cut,
                         float64 svd_min,
                         bool minimize_error,
                         py::object mask_labels)
{
    assert(S.attr("dtype").attr("is_real").cast<bool>());
    auto backend = S.attr("backend").cast<TensorBackend::Ptr>();
    auto [mask_data, new_leg, err, new_norm] =
      backend->truncate_singular_values(S.cast<DiagonalTensorCPtr>(),
                                        chi_max,
                                        chi_min,
                                        degeneracy_tol,
                                        trunc_cut,
                                        std::optional<float64>{ svd_min },
                                        minimize_error);
    if (mask_labels.is_none()) {
        py::object lab0 = S.attr("labels").attr("__getitem__")(0);
        LegLabel dual;
        if (lab0.is_none()) {
            dual = _dual_leg_label(std::nullopt);
        } else {
            dual = _dual_leg_label(lab0.cast<std::string>());
        }
        mask_labels = py_list(lab0, leg_label_to_py(dual));
    }
    py::object mask = make_python_mask(std::move(mask_data),
                                       S.attr("leg"),
                                       py::cast(new_leg),
                                       /*is_projection=*/true,
                                       backend,
                                       mask_labels);
    return { mask, err, new_norm };
}

std::tuple<py::object, py::object, py::object, float64, float64>
truncated_svd(py::object tensor,
              py::object new_labels,
              bool new_leg_dual,
              bool charge_leg_top,
              py::object algorithm,
              std::optional<float64> normalize_to,
              std::optional<int64> chi_max,
              int64 chi_min,
              float64 degeneracy_tol,
              float64 trunc_cut,
              float64 svd_min)
{
    // --- hints from Python truncated_svd ---
    // norm(S[mask]) == S_norm * new_norm
    // ---
    auto [U, S, Vh] = svd(tensor, new_labels, new_leg_dual, charge_leg_top, algorithm);
    py::object S_norm_obj = norm(S);
    // norm() returns a BlockBackend.Scalar (or number); normalize to float64.
    float64 S_norm;
    if (py::hasattr(S_norm_obj, "to_numpy")) {
        S_norm = S_norm_obj.attr("to_numpy")().cast<float64>();
    } else {
        S_norm = S_norm_obj.cast<float64>();
    }
    // S / S_norm via Python
    py::object S_normed = S.attr("__truediv__")(S_norm_obj);
    auto [mask, err, new_norm] =
      truncate_singular_values(S_normed, chi_max, chi_min, degeneracy_tol, trunc_cut, svd_min);
    std::tie(U, S, Vh) = svd_apply_mask(U, S, Vh, mask);
    float64 renormalize = 1.;
    if (normalize_to.has_value()) {
        // norm(S[mask]) == S_norm * new_norm
        renormalize = *normalize_to / S_norm / new_norm;
        S = S.attr("__mul__")(renormalize);
    }
    return { U, S, Vh, err, renormalize };
}

} // namespace cyten
