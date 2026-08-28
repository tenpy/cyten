#include <cyten/tensors/decompositions.h>
#include <cyten/tensors/ops_algebra.h>

#include <cyten/backends/no_symmetry.h>
#include <cyten/backends/tensor_backend.h>
#include <cyten/block_backend/dtypes.h>
#include <cyten/symmetries/exceptions.h>
#include <cyten/symmetries/spaces.h>
#include <cyten/tensors/charged_tensor.h>
#include <cyten/tensors/diagonal_tensor.h>
#include <cyten/tensors/helpers.h>
#include <cyten/tensors/hidden_leg_tensor.h>
#include <cyten/tensors/labels.h>
#include <cyten/tensors/mask.h>
#include <cyten/tensors/ops_elementwise.h>
#include <cyten/tensors/symmetric_tensor.h>
#include <cyten/tensors/tensor.h>
#include <cyten/tools.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <format>
#include <numeric>
#include <stdexcept>
#include <unordered_set>
#include <utility>
#include <variant>

namespace cyten {

namespace {

char const* _USE_PERMUTE_LEGS_ERR_MSG =
  "Legs can not be permuted automatically. Explicitly use permute_legs()";

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
is_HiddenLegTensor(py::object obj)
{
    return is_python_instance(obj, "HiddenLegTensor") || py::isinstance<HiddenLegTensor>(obj);
}

bool
is_Tensor(py::object obj)
{
    return is_python_instance(obj, "Tensor") || py::isinstance<Tensor>(obj);
}

/// Cast a TensorCPtr to a Python object of the most-derived bound type.
/// Plain `py::cast(TensorCPtr)` can lose HiddenLegTensor / ChargedTensor / … identity.
[[nodiscard]] py::object
tensor_as_py(TensorCPtr const& tensor)
{
    if (auto p = std::dynamic_pointer_cast<HiddenLegTensor const>(tensor)) {
        return py::cast(p);
    }
    if (auto p = std::dynamic_pointer_cast<ChargedTensor const>(tensor)) {
        return py::cast(p);
    }
    if (auto p = std::dynamic_pointer_cast<Mask const>(tensor)) {
        return py::cast(p);
    }
    if (auto p = std::dynamic_pointer_cast<Identity const>(tensor)) {
        return py::cast(p);
    }
    if (auto p = std::dynamic_pointer_cast<DiagonalTensor const>(tensor)) {
        return py::cast(p);
    }
    if (auto p = std::dynamic_pointer_cast<SymmetricTensor const>(tensor)) {
        // Might still be a HiddenLegTensor that failed the first cast if typeinfo differs —
        // check labels as a fallback.
        if (HiddenLegTensor::has_hidden_leg_labels(p->labels())) {
            return py::cast(std::make_shared<HiddenLegTensor>(
              std::const_pointer_cast<SymmetricTensor>(p)));
        }
        return py::cast(p);
    }
    return py::cast(tensor);
}

bool is_Number_or_Scalar(py::object obj); // defined below
LegLabels leg_labels_from_py(py::object seq); // defined below

/// Raise if any of `leg_idcs` refers to a hidden leg on `tensor`.
void
reject_hidden_leg_arguments(py::object tensor, std::vector<int64> const& leg_idcs, char const* op)
{
    if (!is_HiddenLegTensor(tensor)) {
        return;
    }
    auto labs = leg_labels_from_py(tensor.attr("_labels"));
    for (auto idx : leg_idcs) {
        if (idx < 0) {
            idx += static_cast<int64>(labs.size());
        }
        if (idx < 0 || idx >= static_cast<int64>(labs.size())) {
            continue;
        }
        if (HiddenLegTensor::is_hidden_leg_label(labs[static_cast<std::size_t>(idx)])) {
            throw std::invalid_argument(std::format(
              "{}: cannot specify hidden leg '{}' (index {}) in arguments. "
              "Hidden legs are handled implicitly.",
              op,
              labs[static_cast<std::size_t>(idx)].value_or("?"),
              idx));
        }
    }
}

/// Find pairs of hidden-leg indices to contract between two HiddenLegTensors.
/// Raises on equal (non-dual) matching hidden labels.
[[nodiscard]] std::vector<std::pair<int64, int64>>
implicit_hidden_contraction_pairs(py::object tensor1, py::object tensor2)
{
    std::vector<std::pair<int64, int64>> pairs;
    if (!is_HiddenLegTensor(tensor1) || !is_HiddenLegTensor(tensor2)) {
        return pairs;
    }
    auto labs1 = leg_labels_from_py(tensor1.attr("_labels"));
    auto labs2 = leg_labels_from_py(tensor2.attr("_labels"));
    std::vector<std::pair<int64, std::string>> hidden1;
    std::vector<std::pair<int64, std::string>> hidden2;
    for (int64 i = 0; i < static_cast<int64>(labs1.size()); ++i) {
        if (HiddenLegTensor::is_hidden_leg_label(labs1[static_cast<std::size_t>(i)])) {
            hidden1.emplace_back(i, *labs1[static_cast<std::size_t>(i)]);
        }
    }
    for (int64 i = 0; i < static_cast<int64>(labs2.size()); ++i) {
        if (HiddenLegTensor::is_hidden_leg_label(labs2[static_cast<std::size_t>(i)])) {
            hidden2.emplace_back(i, *labs2[static_cast<std::size_t>(i)]);
        }
    }
    std::vector<bool> used2(hidden2.size(), false);
    for (auto const& [i1, lab1] : hidden1) {
        auto dual1 = _dual_leg_label(LegLabel{ lab1 });
        for (std::size_t j = 0; j < hidden2.size(); ++j) {
            if (used2[j]) {
                continue;
            }
            auto const& [i2, lab2] = hidden2[j];
            if (lab1 == lab2) {
                throw std::invalid_argument(std::format(
                  "Cannot contract HiddenLegTensors with equal hidden label '{}' "
                  "(both or neither starred). Dual pairs like '!a' with '!a*' are contracted "
                  "implicitly.",
                  lab1));
            }
            if (dual1 && *dual1 == lab2) {
                pairs.emplace_back(i1, i2);
                used2[j] = true;
                break;
            }
        }
    }
    return pairs;
}

[[nodiscard]] py::object
maybe_wrap_hidden(py::object result, bool wrap_if_hidden_labels)
{
    if (!wrap_if_hidden_labels) {
        return result;
    }
    if (result.is_none() || is_Number_or_Scalar(result)) {
        return result;
    }
    if (!is_Tensor(result) || is_HiddenLegTensor(result) || is_ChargedTensor(result)) {
        return result;
    }
    if (!is_SymmetricTensor(result) && !is_DiagonalTensor(result) && !is_Mask(result) &&
        !is_Identity(result)) {
        return result;
    }
    auto labs = leg_labels_from_py(result.attr("_labels"));
    if (HiddenLegTensor::has_hidden_leg_labels(labs)) {
        if (is_DiagonalTensor(result) || is_Mask(result) || is_Identity(result)) {
            throw std::runtime_error(
              "Internal error: DiagonalTensor/Mask/Identity with hidden labels");
        }
        return py::cast(std::make_shared<HiddenLegTensor>(result.cast<SymmetricTensor::Ptr>()));
    }
    return result;
}

void
require_no_remaining_hidden(py::object tensor, char const* op)
{
    if (!is_HiddenLegTensor(tensor)) {
        return;
    }
    throw std::invalid_argument(std::format(
      "{} requires that no hidden legs remain. Unmatched hidden labels: use partial_trace "
      "or contract them with a dual HiddenLegTensor first.",
      op));
}

bool
is_Number_or_Scalar(py::object obj)
{
    return py::isinstance(obj, py::module_::import("numbers").attr("Number")) ||
           py::isinstance(obj, py::module_::import("cyten.block_backends").attr("Scalar"));
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
py_from_compose_sym(std::variant<SymmetricTensorPtr, BlockBackend::Scalar> const& v)
{
    return std::visit([](auto const& x) -> py::object { return py::cast(x); }, v);
}

py::object
py_compose_with_mask(py::object tensor, py::object mask, int64 leg_idx)
{
    return py::cast(_compose_with_Mask(tensor.cast<TensorCPtr>(), mask.cast<MaskCPtr>(), leg_idx));
}

void
check_spaces(std::initializer_list<py::object> a,
             std::initializer_list<py::object> b,
             bool expect_equal = true)
{
    std::vector<Space::Ptr> va;
    std::vector<Space::Ptr> vb;
    va.reserve(a.size());
    vb.reserve(b.size());
    for (auto const& o : a) {
        va.push_back(o.cast<Space::Ptr>());
    }
    for (auto const& o : b) {
        vb.push_back(o.cast<Space::Ptr>());
    }
    _check_compatible_legs(va, vb, expect_equal);
}

void
check_leg_seq(py::handle seq1, py::handle seq2, bool expect_equal = true)
{
    std::vector<Leg::Ptr> a;
    std::vector<Leg::Ptr> b;
    for (auto item : py::reinterpret_borrow<py::iterable>(seq1)) {
        a.push_back(item.cast<Leg::Ptr>());
    }
    for (auto item : py::reinterpret_borrow<py::iterable>(seq2)) {
        b.push_back(item.cast<Leg::Ptr>());
    }
    _check_compatible_legs(a, b, expect_equal);
}

void
check_legs(std::vector<py::object> const& a,
           std::vector<py::object> const& b,
           bool expect_equal = true)
{
    std::vector<Leg::Ptr> va;
    std::vector<Leg::Ptr> vb;
    va.reserve(a.size());
    vb.reserve(b.size());
    for (auto const& o : a) {
        va.push_back(o.cast<Leg::Ptr>());
    }
    for (auto const& o : b) {
        vb.push_back(o.cast<Leg::Ptr>());
    }
    _check_compatible_legs(va, vb, expect_equal);
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
make_python_charged_tensor(py::object invariant_part, py::object charged_state)
{
    return tensors_mod().attr("ChargedTensor")(invariant_part, charged_state);
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
make_python_identity(py::object leg, TensorBackend::Ptr backend, py::object labels)
{
    return tensors_mod().attr("Identity")(
      leg, py::arg("backend") = py::cast(backend), py::arg("labels") = labels);
}

LegLabels
leg_labels_from_py(py::object seq)
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

LegLabel
relabel_one(LegLabel lab, std::optional<std::map<std::string, std::string>> const& relabel)
{
    if (!lab.has_value() || !relabel.has_value()) {
        return lab;
    }
    auto it = relabel->find(*lab);
    if (it != relabel->end()) {
        return it->second;
    }
    return lab;
}

LegLabels
apply_relabel(LegLabels labels, std::optional<std::map<std::string, std::string>> const& relabel)
{
    if (!relabel.has_value()) {
        return labels;
    }
    for (auto& lab : labels) {
        lab = relabel_one(lab, relabel);
    }
    return labels;
}

py::object
labels_to_py(LegLabels const& labels)
{
    return py::cast(labels);
}

py::object
nested_labels_to_py(LegLabels const& codomain_labels, LegLabels const& domain_labels)
{
    py::list out;
    out.append(labels_to_py(codomain_labels));
    out.append(labels_to_py(domain_labels));
    return out;
}

py::object
duplicate_entries(py::object seq)
{
    return py::module_::import("cyten.tools.misc").attr("duplicate_entries")(seq);
}

void
check_same_legs_py(py::object t1, py::object t2)
{
    tensors_mod().attr("check_same_legs")(t1, t2);
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

py::object
scalar_to_py(BlockBackend::Scalar const& s)
{
    return py::cast(s);
}

std::map<std::string, std::string>
relabel_or_empty(std::optional<std::map<std::string, std::string>> const& relabel)
{
    return relabel.value_or(std::map<std::string, std::string>{});
}

[[noreturn]] void
rethrow_permute_legs_err()
{
    throw SymmetryError(_USE_PERMUTE_LEGS_ERR_MSG);
}

py::object
symmetry_error_type()
{
    return py::module_::import("cyten.symmetries").attr("SymmetryError");
}

/// Catch SymmetryError from C++ or from Python (via ``error_already_set``) and rewrite message.
[[noreturn]] void
handle_permute_legs_symmetry_error()
{
    try {
        throw;
    } catch (SymmetryError const&) {
        rethrow_permute_legs_err();
    } catch (py::error_already_set& e) {
        if (e.matches(symmetry_error_type())) {
            throw SymmetryError(_USE_PERMUTE_LEGS_ERR_MSG);
        }
        throw;
    }
}

/// Call ``tensor.backend.item(tensor)`` via Python (NoSymmetry overrides wrapping Block data).
py::object
backend_item_py(py::object tensor)
{
    return tensor.attr("backend").attr("item")(tensor);
}

char const*
charge_leg_label()
{
    return ChargedTensor::_CHARGE_LEG_LABEL;
}

} // namespace

bool almost_equal_py(py::object tensor_1,
                     py::object tensor_2,
                     float64 rtol,
                     float64 atol,
                     bool allow_different_types = false);
py::object apply_mask_py(py::object tensor, py::object mask, py::object leg);
py::object enlarge_leg_py(py::object tensor, py::object mask, py::object leg);
py::object dagger_py(py::object tensor);
py::object compose_py(py::object tensor1,
                      py::object tensor2,
                      std::optional<std::map<std::string, std::string>> relabel1 = std::nullopt,
                      std::optional<std::map<std::string, std::string>> relabel2 = std::nullopt);
py::object inner_py(py::object A, py::object B, bool do_dagger = true);
bool is_scalar_py(py::object obj);
py::object item_py(py::object tensor);
py::object linear_combination_py(py::object a, py::object v, py::object b, py::object w);
py::object norm_py(py::object tensor);
py::object on_device_py(py::object tensor, std::string device, bool copy);
py::object outer_py(py::object tensor1,
                    py::object tensor2,
                    std::optional<std::map<std::string, std::string>> relabel1 = std::nullopt,
                    std::optional<std::map<std::string, std::string>> relabel2 = std::nullopt);
py::object partial_compose_py(
  py::object tensor1,
  py::object tensor2,
  py::object tensor1_first_leg,
  std::optional<std::map<std::string, std::string>> relabel1 = std::nullopt,
  std::optional<std::map<std::string, std::string>> relabel2 = std::nullopt);
py::object partial_trace_py(py::object tensor,
                            std::vector<py::object> pairs,
                            py::object levels = py::none());
py::object pinv_py(py::object tensor, float64 cutoff);
py::object scalar_multiply_py(py::object a, py::object v);
py::object scale_axis_py(py::object tensor, py::object diag, py::object leg);
py::object tdot_py(py::object tensor1,
                   py::object tensor2,
                   py::object legs1,
                   py::object legs2,
                   std::optional<std::map<std::string, std::string>> relabel1 = std::nullopt,
                   std::optional<std::map<std::string, std::string>> relabel2 = std::nullopt);
py::object trace_py(py::object tensor);
py::object transpose_py(py::object tensor);

bool
almost_equal_py(py::object tensor_1,
                py::object tensor_2,
                float64 rtol,
                float64 atol,
                bool allow_different_types)
{
    // --- hints from Python almost_equal ---
    // TODO this is not strictly correct, since definition is not symmetric...
    // we implement the mixed type comparison SymmetricTensor and ChargedTensor only once.
    // to swap the arguments we need to adjust the definition, to use abs(a2)
    // ---
    check_same_legs_py(tensor_1, tensor_2);
    (void)same_device2(tensor_1, tensor_2);

    if (is_Mask(tensor_1)) {
        if (is_Mask(tensor_2)) {
            // Match Python ``Mask.all(t1 == t2)`` via instance method on the equality Mask.
            return tensor_1.attr("__eq__")(tensor_2).attr("all")().cast<bool>();
        }
        if (is_DiagonalTensor(tensor_2) && allow_different_types) {
            return almost_equal_py(tensor_1.attr("as_DiagonalTensor")(), tensor_2, rtol, atol);
        }
        if ((is_SymmetricTensor(tensor_2) || is_ChargedTensor(tensor_2)) &&
            allow_different_types) {
            return almost_equal_py(tensor_1.attr("as_SymmetricTensor")(), tensor_2, rtol, atol);
        }
    }

    if (is_DiagonalTensor(tensor_1)) {
        if (is_Mask(tensor_2) && allow_different_types) {
            return almost_equal_py(tensor_1, tensor_2.attr("as_DiagonalTensor")(), rtol, atol);
        }
        if (is_DiagonalTensor(tensor_2)) {
            return tensor_1
              .attr("elementwise_almost_equal")(
                tensor_2, py::arg("rtol") = rtol, py::arg("atol") = atol)
              .attr("all")()
              .cast<bool>();
        }
        if ((is_SymmetricTensor(tensor_2) || is_ChargedTensor(tensor_2)) &&
            allow_different_types) {
            return almost_equal_py(tensor_1.attr("as_SymmetricTensor")(), tensor_2, rtol, atol);
        }
    }

    if (is_SymmetricTensor(tensor_1)) {
        if ((is_Mask(tensor_2) || is_DiagonalTensor(tensor_2)) && allow_different_types) {
            return almost_equal_py(tensor_1, tensor_2.attr("as_SymmetricTensor")(), rtol, atol);
        }
        if (is_SymmetricTensor(tensor_2)) {
            auto backend = get_same_backend({ tensor_1, tensor_2 });
            return backend->almost_equal(
              tensor_1.cast<TensorCPtr>(), tensor_2.cast<TensorCPtr>(), rtol, atol);
        }
        if (is_ChargedTensor(tensor_2) && allow_different_types) {
            try {
                py::object t2_symm = tensor_2.attr("as_SymmetricTensor")();
                return almost_equal_py(tensor_1, t2_symm, rtol, atol);
            } catch (SymmetryError const&) {
            } catch (py::error_already_set& e) {
                if (!e.matches(symmetry_error_type())) {
                    throw;
                }
            }
            throw NotImplemented("almost_equal");
        }
    }

    if (is_ChargedTensor(tensor_1)) {
        if ((is_Mask(tensor_2) || is_DiagonalTensor(tensor_2)) && allow_different_types) {
            return almost_equal_py(tensor_1, tensor_2.attr("as_SymmetricTensor")(), rtol, atol);
        }
        if (is_SymmetricTensor(tensor_2)) {
            // TODO this is not strictly correct, since definition is not symmetric...
            // we implement the mixed type comparison SymmetricTensor and ChargedTensor only once.
            // to swap the arguments we need to adjust the definition, to use abs(a2)
            return almost_equal_py(tensor_2, tensor_1, rtol, atol);
        }
        if (is_ChargedTensor(tensor_2)) {
            if (!py_eq(tensor_1.attr("charge_leg"), tensor_2.attr("charge_leg"))) {
                throw std::invalid_argument("Mismatched charge_leg");
            }
            auto backend = get_same_backend({ tensor_1, tensor_2 });
            if (tensor_1.attr("charge_leg").attr("dim").cast<int64>() == 1) {
                auto bb = backend->block_backend;
                auto s2 = bb->item(tensor_2.attr("charged_state").cast<BlockBackend::BlockPtr>());
                auto s1 = bb->item(tensor_1.attr("charged_state").cast<BlockBackend::BlockPtr>());
                return almost_equal_py(
                  scalar_multiply_py(scalar_to_py(s2), tensor_1.attr("invariant_part")),
                  scalar_multiply_py(scalar_to_py(s1), tensor_2.attr("invariant_part")),
                  rtol,
                  atol);
            }
            throw NotImplemented("almost_equal");
        }
    }

    throw py::type_error(
      std::format("Incompatible types: {} and {}",
                  std::string(py::str(tensor_1.attr("__class__").attr("__name__"))),
                  std::string(py::str(tensor_2.attr("__class__").attr("__name__")))));
}

py::object
apply_mask_py(py::object tensor, py::object mask, py::object leg)
{
    (void)same_device2(tensor, mask);
    auto parsed = tensor.attr("_parse_leg_idx")(leg);
    bool in_domain = parsed.attr("__getitem__")(0).cast<bool>();
    int64 leg_idx = parsed.attr("__getitem__")(2).cast<int64>();
    if (!mask.attr("is_projection").cast<bool>()) {
        throw std::invalid_argument("mask must be a projection");
    }
    if (in_domain) {
        mask = transpose_py(mask);
    }
    return py::cast(_compose_with_Mask(tensor.cast<TensorCPtr>(), mask.cast<MaskCPtr>(), leg_idx));
}

py::object
enlarge_leg_py(py::object tensor, py::object mask, py::object leg)
{
    // --- hints from Python enlarge_leg ---
    // parse inputs
    // ---
    (void)same_device2(tensor, mask);
    auto parsed = tensor.attr("_parse_leg_idx")(leg);
    bool in_domain = parsed.attr("__getitem__")(0).cast<bool>();
    int64 leg_idx = parsed.attr("__getitem__")(2).cast<int64>();
    if (mask.attr("is_projection").cast<bool>()) {
        throw std::invalid_argument("enlarge_leg requires a non-projection mask");
    }
    if (in_domain) {
        mask = transpose_py(mask);
    }
    return py::cast(_compose_with_Mask(tensor.cast<TensorCPtr>(), mask.cast<MaskCPtr>(), leg_idx));
}

py::object
dagger_py(py::object tensor)
{
    // --- hints from Python dagger ---
    // charge_leg ends up as codomain[0] and is dual.
    // ---
    if (is_Mask(tensor)) {
        auto backend = tensor.attr("backend").cast<TensorBackend::Ptr>();
        auto data = backend->mask_dagger(tensor.cast<MaskCPtr>());
        LegLabels labs = leg_labels_from_py(tensor.attr("_labels"));
        LegLabels dual_labs;
        for (auto it = labs.rbegin(); it != labs.rend(); ++it) {
            dual_labs.push_back(_dual_leg_label(*it));
        }
        return make_python_mask(std::move(data),
                                tensor.attr("codomain").attr("__getitem__")(0),
                                tensor.attr("domain").attr("__getitem__")(0),
                                !tensor.attr("is_projection").cast<bool>(),
                                backend,
                                labels_to_py(dual_labs));
    }
    if (is_Identity(tensor)) {
        return tensor;
    }
    if (is_DiagonalTensor(tensor)) {
        LegLabels dual_labs;
        LegLabels labs = leg_labels_from_py(tensor.attr("_labels"));
        for (auto it = labs.rbegin(); it != labs.rend(); ++it) {
            dual_labs.push_back(_dual_leg_label(*it));
        }
        if (tensor.attr("dtype").cast<Dtype>() == Dtype::Bool) {
            py::object res = tensor.attr("copy")(py::arg("deep") = false);
            res.attr("set_labels")(labels_to_py(dual_labs));
            return res;
        }
        py::object res = py::cast(complex_conj(tensor.cast<DiagonalTensorCPtr>()));
        res.attr("set_labels")(labels_to_py(dual_labs));
        return res;
    }
    if (is_HiddenLegTensor(tensor)) {
        return py::cast(tensor.cast<HiddenLegTensorCPtr>()->dagger());
    }
    if (is_SymmetricTensor(tensor)) {
        auto backend = tensor.attr("backend").cast<TensorBackend::Ptr>();
        auto data = backend->dagger(tensor.cast<TensorCPtr>());
        LegLabels dual_labs;
        LegLabels labs = leg_labels_from_py(tensor.attr("_labels"));
        for (auto it = labs.rbegin(); it != labs.rend(); ++it) {
            dual_labs.push_back(_dual_leg_label(*it));
        }
        return make_python_symmetric_tensor(std::move(data),
                                            tensor.attr("domain"),
                                            tensor.attr("codomain"),
                                            backend,
                                            labels_to_py(dual_labs));
    }
    if (is_ChargedTensor(tensor)) {
        // charge_leg ends up as codomain[0] and is dual.
        py::object inv_part = dagger_py(tensor.attr("invariant_part"));
        inv_part.attr("set_label")(0, charge_leg_label());
        inv_part = tensors_mod().attr("move_leg")(
          inv_part, 0, py::arg("domain_pos") = 0, py::arg("bend_right") = true);
        auto backend = tensor.attr("backend").cast<TensorBackend::Ptr>();
        py::object charged_state =
          py::cast(backend->block_backend->conj(tensor.attr("charged_state").cast<BlockBackend::BlockPtr>()));
        return make_python_charged_tensor(inv_part, charged_state);
    }
    throw py::type_error("Invalid type for tensor. Expected a Tensor subtype");
}

py::object
compose_py(py::object tensor1,
           py::object tensor2,
           std::optional<std::map<std::string, std::string>> relabel1,
           std::optional<std::map<std::string, std::string>> relabel2)
{
    // --- hints from Python compose ---
    // only tensor2 is ChargedTensor
    // ---
    (void)same_device2(tensor1, tensor2);
    check_spaces({ tensor1.attr("domain") }, { tensor2.attr("codomain") });

    LegLabels codomain_labels =
      apply_relabel(leg_labels_from_py(tensor1.attr("codomain_labels")), relabel1);
    LegLabels domain_labels =
      apply_relabel(leg_labels_from_py(tensor2.attr("domain_labels")), relabel2);
    py::object res_labels = nested_labels_to_py(codomain_labels, domain_labels);

    if (is_Mask(tensor1)) {
        return py::cast(
                 _compose_with_Mask(tensor2.cast<TensorCPtr>(), tensor1.cast<MaskCPtr>(), 0))
          .attr("set_label")(0, tensor1.attr("labels").attr("__getitem__")(0));
    }
    if (is_Mask(tensor2)) {
        return py::cast(
                 _compose_with_Mask(tensor1.cast<TensorCPtr>(), tensor2.cast<MaskCPtr>(), -1))
          .attr("set_label")(-1, tensor2.attr("labels").attr("__getitem__")(1));
    }

    if (is_Identity(tensor1)) {
        return tensor2.attr("copy")(py::arg("deep") = false).attr("set_labels")(res_labels);
    }
    if (is_Identity(tensor2)) {
        return tensor1.attr("copy")(py::arg("deep") = false).attr("set_labels")(res_labels);
    }

    if (is_DiagonalTensor(tensor1)) {
        return scale_axis_py(tensor2, tensor1, py::int_(0)).attr("set_labels")(res_labels);
    }
    if (is_DiagonalTensor(tensor2)) {
        // --- hints from Python scale_axis ---
        // transpose if needed
        // ---
        return scale_axis_py(tensor1, tensor2, py::int_(-1)).attr("set_labels")(res_labels);
    }

    if (is_ChargedTensor(tensor1)) {
        return partial_compose_py(
          tensor1, tensor2, tensor1.attr("num_codomain_legs"), relabel1, relabel2);
    }
    if (is_ChargedTensor(tensor2)) {
        // --- hints from Python partial_compose ---
        // do these cases first since the charged_legs do not count towards the num_domain_legs
        // in ChargedTensors, but we need them for the consistency checks below
        // need to bend down charge leg first
        // domain_pos 1 since domain_pos 0 would mean braiding with c1
        // OPTIMIZE we may add this in the future when we find an actual use case
        // tensor1 cannot be Mask or DiagonalTensor due to num_legs constraint
        // ---
        // only tensor2 is ChargedTensor
        return make_python_charged_tensor(
          compose_py(tensor1, tensor2.attr("invariant_part"), relabel1, relabel2),
          tensor2.attr("charged_state"));
    }

    return py_from_compose_sym(_compose_SymmetricTensors(tensor1.cast<SymmetricTensorCPtr>(),
                                                         tensor2.cast<SymmetricTensorCPtr>(),
                                                         relabel1,
                                                         relabel2));
}

std::string
get_same_device(std::vector<TensorCPtr> const& tensors, std::string const& error_msg)
{
    if (tensors.empty()) {
        throw std::invalid_argument("Need at least one tensor");
    }
    std::string device = tensors[0]->device;
    for (std::size_t i = 1; i < tensors.size(); ++i) {
        if (tensors[i]->device != device) {
            throw std::invalid_argument(error_msg);
        }
    }
    return device;
}

py::object
inner_py(py::object A, py::object B, bool do_dagger)
{
    // --- hints from Python inner ---
    // in this case, there is no benefit to having a dedicated backend function,
    // as the dot is cheap
    // same argument as above.
    // remaining cases: both are either SymmetricTensor or ChargedTensor
    // ['!*'] <- [*a_legs]
    // [*b_legs] <- ['!']
    // ['!*', '!']
    // OPTIMIZE: like GEMM, should we offer an interface where dagger is implicitly done during
    // tdot?
    // [!A, !B] @ [!B*] -> [!A]
    // [!A] @ [!A*] -> []
    // and B is a SymmetricTensor
    // reduce to the case where B is charged and A is not  # OPTIMIZE write it out instead...
    // OPTIMIZE: by charge rule, only components in the trivial sector of the charge_leg contribute
    // could exploit by projecting to those components first.
    // remaining case: both are SymmetricTensor
    // ---
    (void)same_device2(A, B);

    if (do_dagger) {
        check_spaces({ A.attr("codomain"), A.attr("domain") },
                     { B.attr("codomain"), B.attr("domain") });
    } else {
        check_spaces({ A.attr("codomain"), A.attr("domain") },
                     { B.attr("domain"), B.attr("codomain") });
    }

    if (is_Identity(A)) {
        return trace_py(B);
    }

    if (is_Identity(B)) {
        // --- hints from Python trace ---
        // OPTIMIZE can project to trivial sector on charge leg first
        // ---
        if (do_dagger) {
            return py::module_::import("numpy").attr("conj")(trace_py(A));
        }
        return trace_py(A);
    }

    if (is_DiagonalTensor(A) || is_Mask(A)) {
        // in this case, there is no benefit to having a dedicated backend function,
        // as the dot is cheap
        if (do_dagger) {
            return trace_py(compose_py(dagger_py(A), B));
        }
        return trace_py(compose_py(A, B));
    }
    if (is_DiagonalTensor(B) || is_Mask(B)) {
        // same argument as above.
        if (do_dagger) {
            return py::module_::import("numpy").attr("conj")(
              trace_py(compose_py(dagger_py(B), A)));
        }
        return trace_py(compose_py(A, B));
    }

    // remaining cases: both are either SymmetricTensor or ChargedTensor
    auto backend = get_same_backend({ A, B });

    if (is_HiddenLegTensor(A) || is_HiddenLegTensor(B)) {
        py::object left = do_dagger ? dagger_py(A) : A;
        // Raises on equal (non-dual) hidden labels.
        (void)implicit_hidden_contraction_pairs(left, B);
        // Full Frobenius inner on the underlying SymmetricTensors (includes hidden legs).
        return scalar_to_py(backend->inner(left.cast<SymmetricTensorCPtr>(),
                                           B.cast<SymmetricTensorCPtr>(),
                                           /*do_dagger=*/false));
    }

    if (is_ChargedTensor(A) && is_ChargedTensor(B)) {
        auto bb = backend->block_backend;
        if (do_dagger) {
            py::object inv_part = py_from_compose_sym(_compose_SymmetricTensors(
              tensors_mod()
                .attr("bend_legs")(dagger_py(A.attr("invariant_part")),
                                   py::arg("num_codomain_legs") = 1)
                .cast<SymmetricTensorCPtr>(),
              tensors_mod()
                .attr("bend_legs")(B.attr("invariant_part"), py::arg("num_domain_legs") = 1)
                .cast<SymmetricTensorCPtr>())); // ['!*', '!']
            // OPTIMIZE: like GEMM, should we offer an interface where dagger is implicitly done
            // during tdot?
            py::object inv_block =
              inv_part.attr("to_dense_block")(py::arg("understood_braiding") = true);
            auto inv_b = inv_block.cast<BlockBackend::BlockPtr>();
            auto b_state = B.attr("charged_state").cast<BlockBackend::BlockPtr>();
            auto a_state = A.attr("charged_state").cast<BlockBackend::BlockPtr>();
            auto tmp = bb->tdot(inv_b, b_state, { 1 }, { 0 });
            auto res = bb->tdot(bb->conj(a_state), tmp, { 0 }, { 0 });
            return scalar_to_py(bb->item(res));
        }
        {
            int64 n_legs = A.attr("num_legs").cast<int64>();
            std::vector<int64> rev_legs;
            rev_legs.reserve(static_cast<std::size_t>(n_legs));
            for (int64 i = n_legs - 1; i >= 0; --i) {
                rev_legs.push_back(i);
            }
            std::vector<py::object> bend_right;
            bend_right.reserve(static_cast<std::size_t>(n_legs + 1));
            for (int64 i = 0; i < n_legs; ++i) {
                bend_right.push_back(py::bool_(true));
            }
            bend_right.push_back(py::bool_(false));
            py::object A_inv =
              tensors_mod().attr("permute_legs")(A.attr("invariant_part"),
                                                 py::make_tuple(-1),
                                                 py::cast(rev_legs),
                                                 py::arg("bend_right") = py::cast(bend_right));
            std::vector<int64> fwd_legs(static_cast<std::size_t>(n_legs));
            std::iota(fwd_legs.begin(), fwd_legs.end(), 0);
            py::object B_inv = tensors_mod().attr("permute_legs")(B.attr("invariant_part"),
                                                                  py::cast(fwd_legs),
                                                                  py::make_tuple(-1),
                                                                  py::arg("bend_right") = true);
            py::object inv_part = py_from_compose_sym(
              _compose_SymmetricTensors(A_inv.cast<SymmetricTensorCPtr>(),
                                        B_inv.cast<SymmetricTensorCPtr>(),
                                        std::map<std::string, std::string>{ { "!", "!A" } },
                                        std::map<std::string, std::string>{ { "!", "!B" } }));
            assert(
              py_eq(inv_part.attr("labels"), py::cast(std::vector<std::string>{ "!A", "!B" })));
            py::object inv_block =
              inv_part.attr("to_dense_block")(py::arg("understood_braiding") = true);
            auto inv_b = inv_block.cast<BlockBackend::BlockPtr>();
            auto b_state = B.attr("charged_state").cast<BlockBackend::BlockPtr>();
            auto a_state = A.attr("charged_state").cast<BlockBackend::BlockPtr>();
            // [!A, !B] @ [!B*] -> [!A]
            auto res = bb->tdot(inv_b, b_state, { 1 }, { 0 });
            // [!A] @ [!A*] -> []
            res = bb->tdot(a_state, res, { 0 }, { 0 });
            return scalar_to_py(bb->item(res));
        }
    }

    if (is_ChargedTensor(A)) { // and B is a SymmetricTensor
        // reduce to the case where B is charged and A is not  # OPTIMIZE write it out instead...
        if (do_dagger) {
            return py::module_::import("numpy").attr("conj")(inner_py(B, A, true));
        }
        return inner_py(B, A, false);
    }

    if (is_ChargedTensor(B)) {
        auto bb = backend->block_backend;
        if (B.attr("charge_leg")
              .attr("sector_multiplicity")(B.attr("symmetry").attr("trivial_sector"))
              .cast<int64>() == 0) {
            Dtype dt =
              dtype::common({ A.attr("dtype").cast<Dtype>(), B.attr("dtype").cast<Dtype>() });
            return scalar_to_py(bb->as_scalar(dtype::zero_scalar(dt), dt));
        }
        // OPTIMIZE: by charge rule, only components in the trivial sector of the charge_leg
        // contribute
        //           could exploit by projecting to those components first.
        int64 nA = A.attr("num_legs").cast<int64>();
        std::vector<int64> legsA(static_cast<std::size_t>(nA));
        std::iota(legsA.begin(), legsA.end(), 0);
        std::vector<int64> legsB(static_cast<std::size_t>(nA));
        for (int64 i = 0; i < nA; ++i) {
            legsB[static_cast<std::size_t>(i)] = nA - 1 - i;
        }
        if (do_dagger) {
            py::object inv_part =
              tdot_py(dagger_py(A), B.attr("invariant_part"), py::cast(legsA), py::cast(legsB));
            auto B_state = bb->conj(B.attr("charged_state").cast<BlockBackend::BlockPtr>());
            auto res = bb->tdot(inv_part.attr("to_dense_block")().cast<BlockBackend::BlockPtr>(),
                                B_state,
                                { 0 },
                                { 0 });
            return scalar_to_py(bb->item(res));
        }
        py::object inv_part =
          tdot_py(A, B.attr("invariant_part"), py::cast(legsA), py::cast(legsB));
        auto res = bb->tdot(inv_part.attr("to_dense_block")().cast<BlockBackend::BlockPtr>(),
                            B.attr("charged_state").cast<BlockBackend::BlockPtr>(),
                            { 0 },
                            { 0 });
        return scalar_to_py(bb->item(res));
    }

    // remaining case: both are SymmetricTensor
    return scalar_to_py(
      backend->inner(A.cast<SymmetricTensorCPtr>(), B.cast<SymmetricTensorCPtr>(), do_dagger));
}

bool
is_scalar_py(py::object obj)
{
    if (is_Tensor(obj)) {
        if (obj.attr("domain").attr("num_sectors").cast<int64>() != 1) {
            return false;
        }
        if (obj.attr("codomain").attr("num_sectors").cast<int64>() != 1) {
            return false;
        }
        if (!py_eq(obj.attr("domain").attr("sector_decomposition"),
                   obj.attr("codomain").attr("sector_decomposition"))) {
            return false;
        }
        auto np = py::module_::import("numpy");
        if (!np.attr("all")(obj.attr("domain").attr("multiplicities").attr("__eq__")(1))
               .cast<bool>()) {
            return false;
        }
        if (!np.attr("all")(obj.attr("codomain").attr("multiplicities").attr("__eq__")(1))
               .cast<bool>()) {
            return false;
        }
        return true;
    }
    return py::isinstance(obj, py::module_::import("numbers").attr("Number"));
}

py::object
item_py(py::object tensor)
{
    if (!is_scalar_py(tensor)) {
        throw std::invalid_argument("Not a scalar");
    }
    if (is_Mask(tensor)) {
        return tensors_mod().attr("Mask").attr("any")(tensor);
    }
    if (is_Identity(tensor)) {
        return dtype::one_scalar(tensor.attr("dtype").cast<Dtype>());
    }
    if (is_HiddenLegTensor(tensor)) {
        require_no_remaining_hidden(tensor, "item");
    }
    if (is_ChargedTensor(tensor)) {
        auto backend = tensor.attr("backend").cast<TensorBackend::Ptr>();
        auto bb = backend->block_backend;
        py::object inv_block = tensor.attr("invariant_part")
                                 .attr("to_dense_block")(py::arg("understood_braiding") = true);
        auto res = bb->tdot(tensor.attr("charged_state").cast<BlockBackend::BlockPtr>(),
                            inv_block.cast<BlockBackend::BlockPtr>(),
                            { 0 },
                            { -1 });
        return scalar_to_py(bb->item(res));
    }
    if (is_DiagonalTensor(tensor) || is_SymmetricTensor(tensor)) {
        return backend_item_py(tensor);
    }
    throw py::type_error("Invalid type for tensor.");
}

py::object
linear_combination_py(py::object a, py::object v, py::object b, py::object w)
{
    // --- hints from Python linear_combination ---
    // Note: We implement Tensor.__add__ and Tensor.__sub__ in terms of this function, so we cant
    // use them (or the ``+`` and ``-`` operations) here.
    // we treat the following cases independently:
    // DiagonalTensor + DiagonalTensor  ->  DiagonalTensor
    // ChargedTensor + ChargedTensor  ->  ChargedTensor (if compatible)
    // all other cases  ->  SymmetricTensor
    // Remaining case: convert to SymmetricTensor
    // ---
    (void)same_device2(v, w);
    check_spaces({ v.attr("codomain"), v.attr("domain") },
                 { w.attr("codomain"), w.attr("domain") });
    // Note: We implement Tensor.__add__ and Tensor.__sub__ in terms of this function, so we cant
    //       use them (or the ``+`` and ``-`` operations) here.
    if (!is_Number_or_Scalar(a) || !is_Number_or_Scalar(b)) {
        throw py::type_error(std::format("unsupported scalar types: {}, {}",
                                         std::string(py::str(py::type::of(a).attr("__name__"))),
                                         std::string(py::str(py::type::of(b).attr("__name__")))));
    }
    auto backend = get_same_backend({ v, w });
    py::object bb = py::cast(backend).attr("block_backend");
    a = bb.attr("as_scalar")(a);
    b = bb.attr("as_scalar")(b);

    // we treat the following cases independently:
    //  DiagonalTensor + DiagonalTensor  ->  DiagonalTensor
    //  ChargedTensor + ChargedTensor  ->  ChargedTensor (if compatible)
    //  all other cases  ->  SymmetricTensor

    if (is_DiagonalTensor(v) && is_DiagonalTensor(w)) {
        auto a_sc = a.cast<BlockBackend::Scalar>();
        auto b_sc = b.cast<BlockBackend::Scalar>();
        BlockBinaryFn func = [a_sc, b_sc](BlockBackend::BlockPtr const& _v,
                                          BlockBackend::BlockPtr const& _w) {
            auto left = a_sc * (*_v);
            auto right = b_sc * (*_w);
            return (*left) + (*right);
        };
        auto v_d = std::const_pointer_cast<DiagonalTensor>(v.cast<DiagonalTensorCPtr>());
        return py::cast(v_d->_binary_operand(
          w.cast<DiagonalTensorCPtr>(), std::move(func), "linear_combination"));
    }
    if (is_ChargedTensor(v) && is_ChargedTensor(w)) {
        if (!py_eq(v.attr("charge_leg"), w.attr("charge_leg"))) {
            throw std::invalid_argument("Can not add ChargedTensors with different dummy legs");
        }
        if (v.attr("charge_leg").attr("dim").cast<int64>() == 1) {
            auto bb_ptr = backend->block_backend;
            auto factor = bb_ptr->item(w.attr("charged_state").cast<BlockBackend::BlockPtr>()) /
                          bb_ptr->item(v.attr("charged_state").cast<BlockBackend::BlockPtr>());
            py::object inv_part = linear_combination_py(a,
                                                        v.attr("invariant_part"),
                                                        b.attr("__mul__")(scalar_to_py(factor)),
                                                        w.attr("invariant_part"));
            return make_python_charged_tensor(inv_part, v.attr("charged_state"));
        }
        throw NotImplemented("linear_combination");
    }
    if (is_HiddenLegTensor(v) && is_HiddenLegTensor(w)) {
        // Add as SymmetricTensors then re-wrap if labels still hidden.
        py::object res = linear_combination_py(
          a, v.attr("as_SymmetricTensor")(), b, w.attr("as_SymmetricTensor")());
        // Preserve hidden labels from v (must match w).
        res.attr("set_labels")(v.attr("_labels"));
        return maybe_wrap_hidden(res, true);
    }
    if (is_ChargedTensor(v) || is_ChargedTensor(w)) {
        throw py::type_error("Can not add ChargedTensor and non-charged tensor.");
    }

    // Remaining case: convert to SymmetricTensor
    v = v.attr("as_SymmetricTensor")();
    w = w.attr("as_SymmetricTensor")();

    auto a_sc = a.cast<BlockBackend::Scalar>();
    auto b_sc = b.cast<BlockBackend::Scalar>();
    auto data =
      backend->linear_combination(a_sc, v.cast<TensorCPtr>(), b_sc, w.cast<TensorCPtr>());
    LegLabels labels = _get_matching_labels(leg_labels_from_py(v.attr("_labels")),
                                            leg_labels_from_py(w.attr("_labels")));
    return make_python_symmetric_tensor(
      std::move(data), v.attr("codomain"), v.attr("domain"), backend, labels_to_py(labels));
}

py::object
norm_py(py::object tensor)
{
    // --- hints from Python norm ---
    // norm ** 2 = Tr(m^\dagger . m) = Tr(id_{small_leg}) = dim(small_leg)
    // OPTIMIZE
    // ---
    if (is_Mask(tensor)) {
        // norm ** 2 = Tr(m^\dagger . m) = Tr(id_{small_leg}) = dim(small_leg)
        auto backend = tensor.attr("backend").cast<TensorBackend::Ptr>();
        auto np = py::module_::import("numpy");
        return scalar_to_py(backend->block_backend->as_scalar(
          np.attr("sqrt")(tensor.attr("small_leg").attr("dim")).cast<float64>()));
    }
    if (is_Identity(tensor)) {
        auto backend = tensor.attr("backend").cast<TensorBackend::Ptr>();
        auto np = py::module_::import("numpy");
        return scalar_to_py(backend->block_backend->as_scalar(
          np.attr("sqrt")(tensor.attr("leg").attr("dim")).cast<float64>()));
    }
    if (is_DiagonalTensor(tensor) || is_SymmetricTensor(tensor)) {
        auto backend = tensor.attr("backend").cast<TensorBackend::Ptr>();
        return scalar_to_py(backend->norm(tensor.cast<TensorCPtr>()));
    }
    if (is_ChargedTensor(tensor)) {
        auto backend = tensor.attr("backend").cast<TensorBackend::Ptr>();
        bool const one_dim_charge = tensor.attr("charge_leg").attr("dim").cast<int64>() == 1;
        if (one_dim_charge) {
            auto factor = backend->block_backend
                            ->item(tensor.attr("charged_state").cast<BlockBackend::BlockPtr>())
                            .abs();
            return scalar_to_py(factor *
                                backend->norm(tensor.attr("invariant_part").cast<TensorCPtr>()));
        }
        // OPTIMIZE
        py::module_::import("warnings")
          .attr("warn")("Converting ChargedTensor to dense block for `norm`",
                        py::arg("stacklevel") = 2);
        py::object block = tensor.attr("to_dense_block")(py::arg("understood_braiding") = true);
        return scalar_to_py(backend->block_backend->norm(block.cast<BlockBackend::BlockPtr>(), 2));
    }
    throw py::type_error("Invalid type for tensor.");
}

py::object
on_device_py(py::object tensor, std::string device, bool copy)
{
    if (copy) {
        return tensor.attr("copy")(py::arg("device") = device);
    }
    tensor.attr("move_to_device")(device);
    return tensor;
}

py::object
outer_py(py::object tensor1,
         py::object tensor2,
         std::optional<std::map<std::string, std::string>> relabel1,
         std::optional<std::map<std::string, std::string>> relabel2)
{
    // --- hints from Python outer ---
    // construct new labels
    // ---
    (void)same_device2(tensor1, tensor2);
    if (!tensor1.attr("symmetry")
           .attr("is_equivalent_to")(tensor2.attr("symmetry"))
           .cast<bool>()) {
        throw SymmetryError("outer requires equivalent symmetries");
    }

    if (is_Mask(tensor1) || is_DiagonalTensor(tensor1)) {
        char const* msg =
          "Converting to SymmetricTensor for outer. Use as_SymmetricTensor() explicitly to "
          "suppress the warning.";
        tensor1 = tensor1.attr("as_SymmetricTensor")(py::arg("warning") = msg);
    }
    if (is_Mask(tensor2) || is_DiagonalTensor(tensor2)) {
        char const* msg =
          "Converting to SymmetricTensor for outer. Use as_SymmetricTensor() explicitly to "
          "suppress the warning.";
        tensor2 = tensor2.attr("as_SymmetricTensor")(py::arg("warning") = msg);
    }
    if (is_ChargedTensor(tensor1)) {
        if (is_ChargedTensor(tensor2)) {
            std::string bang = charge_leg_label();
            auto r1 = relabel_or_empty(relabel1);
            auto r2 = relabel_or_empty(relabel2);
            r1[bang] = bang + "1";
            r2[bang] = bang + "2";
            py::object inv_part =
              outer_py(tensor1.attr("invariant_part"), tensor2.attr("invariant_part"), r1, r2);
            inv_part =
              tensors_mod().attr("move_leg")(inv_part, bang + "2", py::arg("domain_pos") = 1);
            return tensors_mod()
              .attr("ChargedTensor")
              .attr("from_two_charge_legs")(
                inv_part, tensor1.attr("charged_state"), tensor2.attr("charged_state"));
        }
        py::object inv_part =
          outer_py(tensor1.attr("invariant_part"), tensor2, relabel1, relabel2);
        return make_python_charged_tensor(inv_part, tensor1.attr("charged_state"));
    }
    if (is_ChargedTensor(tensor2)) {
        py::object inv_part =
          outer_py(tensor1, tensor2.attr("invariant_part"), relabel1, relabel2);
        inv_part = tensors_mod().attr("move_leg")(inv_part,
                                                  tensor1.attr("num_codomain_legs").cast<int64>() +
                                                    tensor2.attr("num_legs").cast<int64>(),
                                                  py::arg("domain_pos") = 0);
        return make_python_charged_tensor(inv_part, tensor2.attr("charged_state"));
    }
    if ((is_HiddenLegTensor(tensor1) && is_ChargedTensor(tensor2)) ||
        (is_ChargedTensor(tensor1) && is_HiddenLegTensor(tensor2))) {
        throw std::invalid_argument(
          "Cannot outer ChargedTensor with HiddenLegTensor. Unhide or convert first.");
    }
    if (is_HiddenLegTensor(tensor1) || is_HiddenLegTensor(tensor2)) {
        // Implicitly contract dual hidden labels (same rules as tdot with no public legs).
        auto pairs = implicit_hidden_contraction_pairs(tensor1, tensor2);
        if (!pairs.empty()) {
            std::vector<int64> legs1;
            std::vector<int64> legs2;
            for (auto const& [i1, i2] : pairs) {
                legs1.push_back(i1);
                legs2.push_back(i2);
            }
            return tdot_py(tensor1, tensor2, py::cast(legs1), py::cast(legs2), relabel1, relabel2);
        }
    }
    auto backend = get_same_backend({ tensor1, tensor2 });
    auto data =
      backend->outer(tensor1.cast<SymmetricTensorCPtr>(), tensor2.cast<SymmetricTensorCPtr>());
    auto codomain = TensorProduct::from_partial_products(
      { tensor1.attr("codomain").cast<TensorProduct::Ptr>(),
        tensor2.attr("codomain").cast<TensorProduct::Ptr>() });
    auto domain =
      TensorProduct::from_partial_products({ tensor1.attr("domain").cast<TensorProduct::Ptr>(),
                                             tensor2.attr("domain").cast<TensorProduct::Ptr>() });
    // construct new labels
    LegLabels codomain_labels;
    LegLabels domain_labels;
    {
        auto c1 = apply_relabel(leg_labels_from_py(tensor1.attr("codomain_labels")), relabel1);
        auto d1 = apply_relabel(leg_labels_from_py(tensor1.attr("domain_labels")), relabel1);
        auto c2 = apply_relabel(leg_labels_from_py(tensor2.attr("codomain_labels")), relabel2);
        auto d2 = apply_relabel(leg_labels_from_py(tensor2.attr("domain_labels")), relabel2);
        codomain_labels = std::move(c1);
        codomain_labels.insert(codomain_labels.end(), c2.begin(), c2.end());
        domain_labels = std::move(d1);
        domain_labels.insert(domain_labels.end(), d2.begin(), d2.end());
    }
    return maybe_wrap_hidden(
      make_python_symmetric_tensor(std::move(data),
                                   py::cast(codomain),
                                   py::cast(domain),
                                   backend,
                                   nested_labels_to_py(codomain_labels, domain_labels)),
      is_HiddenLegTensor(tensor1) || is_HiddenLegTensor(tensor2));
}

py::object
partial_compose_py(py::object tensor1,
                   py::object tensor2,
                   py::object tensor1_first_leg,
                   std::optional<std::map<std::string, std::string>> relabel1,
                   std::optional<std::map<std::string, std::string>> relabel2)
{
    // do these cases first since the charged_legs do not count towards the num_domain_legs
    // in ChargedTensors, but we need them for the consistency checks below
    if (is_ChargedTensor(tensor1) && is_ChargedTensor(tensor2)) {
        std::string c = charge_leg_label();
        std::string c1 = c + "1";
        std::string c2 = c + "2";
        auto r1 = relabel_or_empty(relabel1);
        auto r2 = relabel_or_empty(relabel2);
        r1[c] = c1;
        r2[c] = c2;
        py::object inv_part = tensor2.attr("invariant_part");
        if (tensor1_first_leg.cast<int64>() < tensor1.attr("num_codomain_legs").cast<int64>()) {
            // need to bend down charge leg first
            inv_part = tensors_mod().attr("move_leg")(
              inv_part,
              c,
              py::arg("codomain_pos") = tensor2.attr("num_codomain_legs").cast<int64>() - 1,
              py::arg("bend_right") = true);
        }
        inv_part =
          partial_compose_py(tensor1.attr("invariant_part"), inv_part, tensor1_first_leg, r1, r2);
        // domain_pos 1 since domain_pos 0 would mean braiding with c1
        inv_part = tensors_mod().attr("move_leg")(
          inv_part, c2, py::arg("domain_pos") = 1, py::arg("bend_right") = true);
        return tensors_mod()
          .attr("ChargedTensor")
          .attr("from_two_charge_legs")(inv_part,
                                        py::arg("state1") = tensor1.attr("charged_state"),
                                        py::arg("state2") = tensor2.attr("charged_state"));
    }
    if (is_ChargedTensor(tensor1)) {
        py::object inv_part = partial_compose_py(
          tensor1.attr("invariant_part"), tensor2, tensor1_first_leg, relabel1, relabel2);
        return tensors_mod()
          .attr("ChargedTensor")
          .attr("from_invariant_part")(inv_part, tensor1.attr("charged_state"));
    }
    if (is_ChargedTensor(tensor2)) {
        py::object inv_part = tensor2.attr("invariant_part");
        // Note: Python has a leftover debug print here; omit it.
        if (tensor1_first_leg.cast<int64>() < tensor1.attr("num_codomain_legs").cast<int64>()) {
            // need to bend down charge leg first
            inv_part = tensors_mod().attr("move_leg")(
              inv_part,
              charge_leg_label(),
              py::arg("codomain_pos") = tensor2.attr("num_codomain_legs").cast<int64>() - 1,
              py::arg("bend_right") = true);
        }
        inv_part = partial_compose_py(tensor1, inv_part, tensor1_first_leg, relabel1, relabel2);
        inv_part = tensors_mod().attr("move_leg")(
          inv_part, charge_leg_label(), py::arg("domain_pos") = 0, py::arg("bend_right") = true);
        return tensors_mod()
          .attr("ChargedTensor")
          .attr("from_invariant_part")(inv_part, tensor2.attr("charged_state"));
    }

    (void)same_device2(tensor1, tensor2);
    int64 t1_first =
      tensor1.attr("get_leg_idcs")(tensor1_first_leg).attr("__getitem__")(0).cast<int64>();

    LegLabels codomain_labels =
      apply_relabel(leg_labels_from_py(tensor1.attr("codomain_labels")), relabel1);
    LegLabels domain_labels =
      apply_relabel(leg_labels_from_py(tensor1.attr("domain_labels")), relabel1);

    char const* leg_msg = "Not all legs to be contracted are in the (co)domain";
    char const* compose_msg = "Use compose for contracting the full (co)domain";
    // OPTIMIZE we may add this in the future when we find an actual use case
    char const* contract_msg = "Use compose or outer when no legs are to be contracted";

    py::object new_codomain;
    py::object new_domain;
    int64 num_codomain_legs = tensor1.attr("num_codomain_legs").cast<int64>();

    if (t1_first < num_codomain_legs) {
        int64 num_legs = tensor2.attr("num_domain_legs").cast<int64>();
        int64 t1_last = t1_first + num_legs - 1;
        if (!(num_legs > 0)) {
            throw std::runtime_error(contract_msg);
        }
        if (!(t1_last < num_codomain_legs)) {
            throw std::runtime_error(leg_msg);
        }
        if (!(num_legs < num_codomain_legs)) {
            throw std::runtime_error(compose_msg);
        }
        py::object factors1 =
          tensor1.attr("codomain")
            .attr("factors")
            .attr("__getitem__")(py::slice(
              static_cast<py::ssize_t>(t1_first), static_cast<py::ssize_t>(t1_last + 1), 1));
        check_leg_seq(factors1, tensor2.attr("domain").attr("factors"));
        LegLabels tensor2_labels =
          apply_relabel(leg_labels_from_py(tensor2.attr("codomain_labels")), relabel2);
        codomain_labels.erase(codomain_labels.begin() + t1_first,
                              codomain_labels.begin() + t1_last + 1);
        codomain_labels.insert(
          codomain_labels.begin() + t1_first, tensor2_labels.begin(), tensor2_labels.end());

        py::list new_cod_list = py::list(tensor1.attr("codomain").attr("factors"));
        py::list t2_cod = py::list(tensor2.attr("codomain"));
        new_cod_list.attr("__setitem__")(
          py::slice(static_cast<py::ssize_t>(t1_first), static_cast<py::ssize_t>(t1_last + 1), 1),
          t2_cod);
        new_codomain = py::module_::import("cyten.symmetries.spaces")
                         .attr("TensorProduct")(new_cod_list, tensor1.attr("symmetry"));
        new_domain = tensor1.attr("domain");
    } else {
        int64 num_legs = tensor2.attr("num_codomain_legs").cast<int64>();
        int64 t1_last = t1_first + num_legs - 1;
        int64 num_legs_t1 = tensor1.attr("num_legs").cast<int64>();
        int64 num_domain_legs = tensor1.attr("num_domain_legs").cast<int64>();
        if (!(num_legs > 0)) {
            throw std::runtime_error(contract_msg);
        }
        if (!(t1_last < num_legs_t1)) {
            throw std::runtime_error(leg_msg);
        }
        if (!(num_legs < num_domain_legs)) {
            throw std::runtime_error(compose_msg);
        }
        int64 domain_first_leg = num_legs_t1 - 1 - t1_last;
        int64 domain_last_leg = num_legs_t1 - 1 - t1_first;
        py::object factors1 = tensor1.attr("domain").attr("factors").attr("__getitem__")(
          py::slice(static_cast<py::ssize_t>(domain_first_leg),
                    static_cast<py::ssize_t>(domain_last_leg + 1),
                    1));
        check_leg_seq(factors1, tensor2.attr("codomain").attr("factors"));
        LegLabels tensor2_labels =
          apply_relabel(leg_labels_from_py(tensor2.attr("domain_labels")), relabel2);
        domain_labels.erase(domain_labels.begin() + domain_first_leg,
                            domain_labels.begin() + domain_last_leg + 1);
        domain_labels.insert(
          domain_labels.begin() + domain_first_leg, tensor2_labels.begin(), tensor2_labels.end());

        new_codomain = tensor1.attr("codomain");
        py::list new_dom_list = py::list(tensor1.attr("domain"));
        py::list t2_dom = py::list(tensor2.attr("domain"));
        new_dom_list.attr("__setitem__")(py::slice(static_cast<py::ssize_t>(domain_first_leg),
                                                   static_cast<py::ssize_t>(domain_last_leg + 1),
                                                   1),
                                         t2_dom);
        new_domain = py::module_::import("cyten.symmetries.spaces")
                       .attr("TensorProduct")(new_dom_list, tensor1.attr("symmetry"));
    }

    LegLabels res_labels = codomain_labels;
    for (auto it = domain_labels.rbegin(); it != domain_labels.rend(); ++it) {
        res_labels.push_back(*it);
    }
    py::object res_labels_py = labels_to_py(res_labels);
    if (py::len(py::module_::import("cyten.tools.misc")
                  .attr("duplicate_entries")(
                    res_labels_py, py::arg("ignore") = py::make_tuple(py::none()))) > 0) {
        throw std::runtime_error("duplicate labels");
    }

    if (is_Identity(tensor1)) {
        return tensor2.attr("copy")(py::arg("deep") = false).attr("set_labels")(res_labels_py);
    }
    if (is_Identity(tensor2)) {
        return tensor1.attr("copy")(py::arg("deep") = false).attr("set_labels")(res_labels_py);
    }

    // tensor1 cannot be Mask or DiagonalTensor due to num_legs constraint
    if (is_Mask(tensor2)) {
        return py_compose_with_mask(tensor1, tensor2, t1_first).attr("set_labels")(res_labels_py);
    }
    if (is_DiagonalTensor(tensor2)) {
        return scale_axis_py(tensor1, tensor2, py::int_(t1_first))
          .attr("set_labels")(res_labels_py);
    }

    auto backend = get_same_backend({ tensor1, tensor2 });
    auto data = backend->partial_compose(tensor1.cast<SymmetricTensorCPtr>(),
                                         tensor2.cast<SymmetricTensorCPtr>(),
                                         t1_first,
                                         new_codomain.cast<TensorProduct::Ptr>(),
                                         new_domain.cast<TensorProduct::Ptr>());
    return make_python_symmetric_tensor(
      std::move(data), new_codomain, new_domain, backend, res_labels_py);
}

py::object
partial_trace_py(py::object tensor, std::vector<py::object> pairs, py::object levels)
{
    // --- hints from Python partial_trace ---
    // check legs are compatible
    // deal with other tensor types
    // only remaining option after input checks is the full trace.
    // charge leg is not traced and thus does not braid.
    // so its level is irrelevant. just make sure its not a duplicate
    // scalar result
    // ensure copy
    // should be a scalar
    // ---
    // check legs are compatible
    std::vector<std::pair<int64, int64>> parsed_pairs;
    parsed_pairs.reserve(pairs.size());
    py::list traced_idcs_list;
    for (auto const& pair : pairs) {
        py::object idcs = tensor.attr("get_leg_idcs")(pair);
        int64 i1 = idcs.attr("__getitem__")(0).cast<int64>();
        int64 i2 = idcs.attr("__getitem__")(1).cast<int64>();
        parsed_pairs.emplace_back(i1, i2);
        traced_idcs_list.append(i1);
        traced_idcs_list.append(i2);
    }
    py::object duplicates = duplicate_entries(traced_idcs_list);
    if (py::len(duplicates) > 0) {
        throw std::invalid_argument("Pairs may not contain duplicates.");
    }
    {
        std::vector<py::object> as_cod;
        std::vector<py::object> as_dom;
        for (auto const& [i1, i2] : parsed_pairs) {
            as_cod.push_back(tensor.attr("_as_codomain_leg")(i1));
            as_dom.push_back(tensor.attr("_as_domain_leg")(i2));
        }
        check_legs(as_cod, as_dom);
    }

    if (pairs.empty()) {
        return tensor;
    }
    // deal with other tensor types
    if (is_DiagonalTensor(tensor) || is_Mask(tensor)) {
        // only remaining option after input checks is the full trace.
        return trace_py(tensor);
    }
    if (is_ChargedTensor(tensor)) {
        if (!levels.is_none()) {
            // charge leg is not traced and thus does not braid.
            // so its level is irrelevant. just make sure its not a duplicate
            py::list levels_list = py::list(levels);
            py::object min_level = py::module_::import("builtins").attr("min")(levels_list);
            levels_list.append(min_level.attr("__sub__")(1));
            levels = levels_list;
        }
        // rebuild pairs as py objects for recursive call
        std::vector<py::object> pair_objs;
        for (auto const& [i1, i2] : parsed_pairs) {
            pair_objs.push_back(py::make_tuple(i1, i2));
        }
        py::object invariant_part =
          partial_trace_py(tensor.attr("invariant_part"), pair_objs, levels);
        if (invariant_part.attr("num_legs").cast<int64>() == 1) {
            // scalar result
            auto backend = tensor.attr("backend").cast<TensorBackend::Ptr>();
            auto bb = backend->block_backend;
            py::object inv_block =
              invariant_part.attr("to_dense_block")(py::arg("understood_braiding") = true);
            auto res = bb->tdot(inv_block.cast<BlockBackend::BlockPtr>(),
                                tensor.attr("charged_state").cast<BlockBackend::BlockPtr>(),
                                { 0 },
                                { 0 });
            return scalar_to_py(bb->item(res));
        }
        return make_python_charged_tensor(invariant_part, tensor.attr("charged_state"));
    }
    if (is_HiddenLegTensor(tensor)) {
        std::vector<int64> all_traced;
        for (auto const& [i1, i2] : parsed_pairs) {
            all_traced.push_back(i1);
            all_traced.push_back(i2);
        }
        reject_hidden_leg_arguments(tensor, all_traced, "partial_trace");
        // Fall through to SymmetricTensor path; leftover hidden legs stay open.
    }
    if (!is_SymmetricTensor(tensor)) {
        throw py::type_error(
          std::format("Unexpected tensor type: {}",
                      std::string(py::str(py::type::of(tensor).attr("__name__")))));
    }

    std::vector<std::optional<int64>> levels_vec;
    int64 num_legs = tensor.attr("num_legs").cast<int64>();
    if (levels.is_none()) {
        levels_vec.assign(static_cast<std::size_t>(num_legs), std::nullopt);
    } else {
        for (auto item : levels) {
            if (item.is_none()) {
                levels_vec.push_back(std::nullopt);
            } else {
                levels_vec.push_back(item.cast<int64>());
            }
        }
    }

    auto backend = tensor.attr("backend").cast<TensorBackend::Ptr>();
    TensorBackend::DataPtr data;
    TensorProduct::Ptr codomain;
    TensorProduct::Ptr domain;
    try {
        auto traced =
          backend->partial_trace(tensor.cast<SymmetricTensorCPtr>(), parsed_pairs, levels_vec);
        data = std::move(std::get<0>(traced));
        codomain = std::move(std::get<1>(traced));
        domain = std::move(std::get<2>(traced));
    } catch (...) {
        handle_permute_legs_symmetry_error();
    }

    if (num_legs == static_cast<int64>(py::len(traced_idcs_list))) {
        // should be a scalar — C++ backends return 0-d / single-block DataPtr
        return scalar_to_py(backend->data_item(std::move(data)));
    }
    std::unordered_set<int64> traced_set;
    for (auto const& [i1, i2] : parsed_pairs) {
        traced_set.insert(i1);
        traced_set.insert(i2);
    }
    LegLabels labels;
    LegLabels all_labels = leg_labels_from_py(tensor.attr("_labels"));
    for (std::size_t n = 0; n < all_labels.size(); ++n) {
        if (!traced_set.contains(static_cast<int64>(n))) {
            labels.push_back(all_labels[n]);
        }
    }
    return maybe_wrap_hidden(make_python_symmetric_tensor(
      std::move(data), py::cast(codomain), py::cast(domain), backend, labels_to_py(labels)),
                           is_HiddenLegTensor(tensor));
}

py::object
pinv_py(py::object tensor, float64 cutoff)
{
    if (is_Identity(tensor)) {
        return tensor;
    }
    if (is_DiagonalTensor(tensor)) {
        return py::cast(cutoff_inverse(tensor.cast<DiagonalTensorCPtr>(), cutoff));
    }
    auto [U, S, Vh, err, renormalize] = truncated_svd(tensor.cast<TensorCPtr>(),
                                                      /*new_labels=*/std::nullopt,
                                                      /*new_leg_dual=*/false,
                                                      /*charge_leg_top=*/true,
                                                      /*algorithm=*/std::nullopt,
                                                      /*normalize_to=*/std::nullopt,
                                                      /*chi_max=*/std::nullopt,
                                                      /*chi_min=*/1,
                                                      /*degeneracy_tol=*/0.,
                                                      /*trunc_cut=*/0.,
                                                      /*svd_min=*/cutoff);
    (void)err;
    (void)renormalize;
    return dagger_py(
      compose_py(compose_py(py::cast(U), py::cast(cutoff_inverse(S, cutoff))), py::cast(Vh)));
}

py::object
scalar_multiply_py(py::object a, py::object v)
{
    // --- hints from Python scalar_multiply ---
    // remaining case: SymmetricTensor
    // ---
    if (!is_Number_or_Scalar(a)) {
        throw py::type_error(std::format("unsupported scalar type: {}",
                                         std::string(py::str(py::type::of(a).attr("__name__")))));
    }
    py::object bb = v.attr("backend").attr("block_backend");
    a = bb.attr("as_scalar")(a);
    if (is_DiagonalTensor(v)) {
        auto a_sc = a.cast<BlockBackend::Scalar>();
        BlockUnaryFn func = [a_sc](BlockBackend::BlockPtr const& _v) { return a_sc * (*_v); };
        auto v_d = std::const_pointer_cast<DiagonalTensor>(v.cast<DiagonalTensorCPtr>());
        return py::cast(v_d->_elementwise_unary(std::move(func), /*maps_zero_to_zero=*/true));
    }
    if (is_Mask(v)) {
        char const* msg = "Converting to SymmetricTensor for scalar multiplication. "
                          "Use as_SymmetricTensor() explicitly to suppress the warning.";
        v = v.attr("as_SymmetricTensor")(py::arg("warning") = msg);
    }
    if (is_ChargedTensor(v)) {
        auto backend = v.attr("backend").cast<TensorBackend::Ptr>();
        auto charged_state = backend->block_backend->mul(
          a.cast<BlockBackend::Scalar>(), v.attr("charged_state").cast<BlockBackend::BlockPtr>());
        return make_python_charged_tensor(v.attr("invariant_part"), py::cast(charged_state));
    }
    if (is_HiddenLegTensor(v)) {
        auto backend = v.attr("backend").cast<TensorBackend::Ptr>();
        auto data = backend->mul(a.cast<BlockBackend::Scalar>(), v.cast<TensorCPtr>());
        return maybe_wrap_hidden(make_python_symmetric_tensor(
                                   std::move(data), v.attr("codomain"), v.attr("domain"), backend, v.attr("_labels")),
                               true);
    }
    // remaining case: SymmetricTensor
    auto backend = v.attr("backend").cast<TensorBackend::Ptr>();
    auto data = backend->mul(a.cast<BlockBackend::Scalar>(), v.cast<TensorCPtr>());
    return make_python_symmetric_tensor(
      std::move(data), v.attr("codomain"), v.attr("domain"), backend, v.attr("_labels"));
}

py::object
scale_axis_py(py::object tensor, py::object diag, py::object leg)
{
    (void)same_device2(tensor, diag);

    if (is_Identity(diag)) {
        return tensor;
    }

    // transpose if needed
    auto parsed = tensor.attr("_parse_leg_idx")(leg);
    bool in_domain = parsed.attr("__getitem__")(0).cast<bool>();
    int64 co_domain_idx = parsed.attr("__getitem__")(1).cast<int64>();
    int64 leg_idx = parsed.attr("__getitem__")(2).cast<int64>();
    py::object tens_leg = in_domain ? tensor.attr("domain").attr("__getitem__")(co_domain_idx)
                                    : tensor.attr("codomain").attr("__getitem__")(co_domain_idx);
    if (!tensor.attr("symmetry").attr("is_equivalent_to")(diag.attr("symmetry")).cast<bool>()) {
        throw SymmetryError("scale_axis requires equivalent symmetries");
    }
    if (py_eq(tens_leg, diag.attr("leg"))) {
        // pass
    } else if (py_eq(tens_leg, diag.attr("leg").attr("dual"))) {
        diag = transpose_py(diag);
    } else {
        throw std::invalid_argument("Incompatible legs");
    }

    if (is_DiagonalTensor(tensor)) {
        return tensor.attr("__mul__")(diag).attr("set_labels")(tensor.attr("labels"));
    }
    if (is_Mask(tensor)) {
        if (leg_idx == 0) {
            return compose_py(diag, tensor).attr("set_labels")(tensor.attr("labels"));
        }
        return compose_py(tensor, diag).attr("set_labels")(tensor.attr("labels"));
    }
    if (is_ChargedTensor(tensor)) {
        py::object inv_part =
          scale_axis_py(tensor.attr("invariant_part"), diag, py::int_(leg_idx));
        return make_python_charged_tensor(inv_part, tensor.attr("charged_state"));
    }
    auto backend = get_same_backend({ tensor, diag });
    auto data =
      backend->scale_axis(tensor.cast<TensorCPtr>(), diag.cast<DiagonalTensorCPtr>(), leg_idx);
    return make_python_symmetric_tensor(std::move(data),
                                        tensor.attr("codomain"),
                                        tensor.attr("domain"),
                                        backend,
                                        tensor.attr("_labels"));
}

py::object
tdot_py(py::object tensor1,
        py::object tensor2,
        py::object legs1,
        py::object legs2,
        std::optional<std::map<std::string, std::string>> relabel1,
        std::optional<std::map<std::string, std::string>> relabel2)
{
    // --- hints from Python tdot ---
    // parse legs to list[int] and check they are valid
    // deal with relabelling once using recursion.
    // This means we do not need to worry about labels in each of the many return sites below
    // Deal with Masks: either return or reduce to SymmetricTensor
    // move legs to tdot convention
    // contract the large leg first
    // then trace over the small leg
    // scalar result
    // Deal with DiagonalTensor: either return or reduce to SymmetricTensor
    // Identity is considered in this branch too
    // Deal with ChargedTensor
    // note: its important that we have already used get_leg_idcs
    // Remaining case: both are SymmetricTenor
    // OPTIMIZE actually, we only need to permute legs to *any* matching order.
    // could use ``legs1[perm]`` and ``legs2[perm]`` instead, if that means fewer braids.
    // ---
    (void)same_device2(tensor1, tensor2);

    // parse legs to list[int] and check they are valid
    py::object legs1_idcs = tensor1.attr("get_leg_idcs")(legs1);
    py::object legs2_idcs = tensor2.attr("get_leg_idcs")(legs2);
    std::vector<int64> legs1_v = legs1_idcs.cast<std::vector<int64>>();
    std::vector<int64> legs2_v = legs2_idcs.cast<std::vector<int64>>();
    if (py::len(duplicate_entries(legs1_idcs)) > 0 || py::len(duplicate_entries(legs2_idcs)) > 0) {
        throw std::invalid_argument("Duplicate leg entries.");
    }
    int64 num_contr = static_cast<int64>(legs1_v.size());
    if (static_cast<int64>(legs2_v.size()) != num_contr) {
        throw std::invalid_argument("legs1 and legs2 must have the same length");
    }
    {
        std::vector<py::object> as_dom;
        std::vector<py::object> as_cod;
        for (std::size_t i = 0; i < legs1_v.size(); ++i) {
            as_dom.push_back(tensor1.attr("_as_domain_leg")(legs1_v[i]));
            as_cod.push_back(tensor2.attr("_as_codomain_leg")(legs2_v[i]));
        }
        check_legs(as_dom, as_cod);
    }

    // deal with relabelling once using recursion.
    // This means we do not need to worry about labels in each of the many return sites below
    bool do_relabel = (relabel1.has_value() && !relabel1->empty()) ||
                      (relabel2.has_value() && !relabel2->empty());
    if (do_relabel) {
        // Implicit hidden duals are contracted even if not listed in legs1/legs2.
        auto hidden_pairs = implicit_hidden_contraction_pairs(tensor1, tensor2);
        std::unordered_set<int64> skip1;
        std::unordered_set<int64> skip2;
        for (auto i : legs1_v) {
            skip1.insert(i);
        }
        for (auto i : legs2_v) {
            skip2.insert(i);
        }
        for (auto const& [i1, i2] : hidden_pairs) {
            skip1.insert(i1);
            skip2.insert(i2);
        }
        LegLabels codomain_labels;
        LegLabels all1 = leg_labels_from_py(tensor1.attr("_labels"));
        for (std::size_t n = 0; n < all1.size(); ++n) {
            if (!skip1.contains(static_cast<int64>(n))) {
                codomain_labels.push_back(relabel_one(all1[n], relabel1));
            }
        }
        LegLabels domain_labels;
        LegLabels all2 = leg_labels_from_py(tensor2.attr("_labels"));
        for (std::size_t n = 0; n < all2.size(); ++n) {
            if (!skip2.contains(static_cast<int64>(n))) {
                domain_labels.push_back(relabel_one(all2[n], relabel2));
            }
        }
        py::object res = tdot_py(tensor1, tensor2, legs1_idcs, legs2_idcs);
        if (is_Number_or_Scalar(res)) {
            return res;
        }
        LegLabels flat = codomain_labels;
        flat.insert(flat.end(), domain_labels.begin(), domain_labels.end());
        res.attr("set_labels")(labels_to_py(flat));
        return res;
    }

    // Deal with Masks: either return or reduce to SymmetricTensor
    if (is_Mask(tensor1)) {
        if (num_contr == 0) {
            tensor1 = tensor1.attr("as_SymmetricTensor")();
        } else if (num_contr == 1) {
            bool t1_in_domain = legs1_v[0] == 1;
            bool t2_in_domain = legs2_v[0] >= tensor2.attr("num_codomain_legs").cast<int64>();
            py::object res;
            if (t2_in_domain == t1_in_domain) {
                res = py_compose_with_mask(tensor2, transpose_py(tensor1), legs2_v[0]);
            } else {
                res = py_compose_with_mask(tensor2, tensor1, legs2_v[0]);
            }
            res.attr("set_label")(legs2_v[0],
                                  tensor1.attr("labels").attr("__getitem__")(1 - legs1_v[0]));
            // move legs to tdot convention
            try {
                return tensors_mod().attr("permute_legs")(res, py::arg("codomain") = legs1_idcs);
            } catch (...) {
                handle_permute_legs_symmetry_error();
            }
        } else if (num_contr == 2) {
            // contract the large leg first
            bool is_proj = tensor1.attr("is_projection").cast<bool>();
            auto which_is_large = static_cast<std::size_t>(
              std::find(legs1_v.begin(), legs1_v.end(), is_proj ? 1 : 0) - legs1_v.begin());
            bool t1_in_domain = is_proj;
            bool t2_in_domain =
              legs2_v[which_is_large] >= tensor2.attr("num_codomain_legs").cast<int64>();
            py::object res;
            if (t1_in_domain == t2_in_domain) {
                res =
                  py_compose_with_mask(tensor2, transpose_py(tensor1), legs2_v[which_is_large]);
            } else {
                res = py_compose_with_mask(tensor2, tensor1, legs2_v[which_is_large]);
            }
            // then trace over the small leg
            res = partial_trace_py(res, { legs2_idcs });
            // move legs to tdot convention
            if (tensor2.attr("num_legs").cast<int64>() == 2) { // scalar result
                return res;
            }
            return tensors_mod().attr("bend_legs")(res, py::arg("num_codomain_legs") = 0);
        }
    }
    if (is_Mask(tensor2)) {
        if (num_contr == 0) {
            tensor2 = tensor2.attr("as_SymmetricTensor")();
        } else if (num_contr == 1) {
            bool t1_in_domain = legs1_v[0] >= tensor1.attr("num_codomain_legs").cast<int64>();
            bool t2_in_domain = legs2_v[0] == 1;
            py::object res;
            if (t1_in_domain == t2_in_domain) {
                res = py_compose_with_mask(tensor1, transpose_py(tensor2), legs1_v[0]);
            } else {
                res = py_compose_with_mask(tensor1, tensor2, legs1_v[0]);
            }
            res.attr("set_label")(legs1_v[0],
                                  tensor2.attr("labels").attr("__getitem__")(1 - legs2_v[0]));
            // move legs to tdot convention
            try {
                return tensors_mod().attr("permute_legs")(
                  res, py::arg("domain") = legs2_idcs, py::arg("bend_right") = py::none());
            } catch (...) {
                handle_permute_legs_symmetry_error();
            }
        } else if (num_contr == 2) {
            // contract the large leg first
            bool is_proj = tensor2.attr("is_projection").cast<bool>();
            auto which_is_large = static_cast<std::size_t>(
              std::find(legs2_v.begin(), legs2_v.end(), is_proj ? 1 : 0) - legs2_v.begin());
            bool t1_in_domain =
              legs1_v[which_is_large] >= tensor1.attr("num_codomain_legs").cast<int64>();
            bool t2_in_domain = is_proj;
            py::object res;
            if (t1_in_domain == t2_in_domain) {
                res =
                  py_compose_with_mask(tensor1, transpose_py(tensor2), legs1_v[which_is_large]);
            } else {
                res = py_compose_with_mask(tensor1, tensor2, legs1_v[which_is_large]);
            }
            // then trace over the small leg
            res = partial_trace_py(res, { legs1_idcs });
            // move legs to tdot convention
            if (tensor1.attr("num_legs").cast<int64>() == 2) { // scalar result
                return res;
            }
            return tensors_mod().attr("bend_legs")(res, py::arg("num_domain_legs") = 0);
        }
    }

    if (is_Identity(tensor1)) {
        if (num_contr == 1) {
            py::object res = tensors_mod().attr("permute_legs")(
              tensor2, py::arg("codomain") = legs2_idcs, py::arg("bend_right") = py::none());
            return res.attr("set_label")(
              0, tensor1.attr("labels").attr("__getitem__")(1 - legs1_v[0]));
        }
        if (num_contr == 2) {
            py::object res = partial_trace_py(tensor2, { legs2_idcs });
            return tensors_mod().attr("bend_legs")(res, py::arg("num_codomain_legs") = 0);
        }
        tensor1 = tensor1.attr("as_DiagonalTensor")();
    }

    if (is_Identity(tensor2)) {
        if (num_contr == 1) {
            // Match Python (computes res then ignores it):
            (void)tensors_mod().attr("permute_legs")(
              tensor1, py::arg("domain") = legs1_idcs, py::arg("bend_right") = py::none());
            return tensor1.attr("copy")(py::arg("deep") = false)
              .attr("set_label")(legs1_v[0],
                                 tensor2.attr("labels").attr("__getitem__")(1 - legs2_v[0]));
        }
        if (num_contr == 2) {
            py::object res = partial_trace_py(tensor1, { legs1_idcs });
            return tensors_mod().attr("bend_legs")(res, py::arg("num_domain_legs") = 0);
        }
        tensor2 = tensor2.attr("as_DiagonalTensor")();
    }

    // Deal with DiagonalTensor: either return or reduce to SymmetricTensor
    if (is_DiagonalTensor(tensor1)) {
        // Identity is considered in this branch too
        if (num_contr == 0) {
            tensor1 = tensor1.attr("as_SymmetricTensor")();
        } else if (num_contr == 1) {
            py::object res = scale_axis_py(tensor2, tensor1, py::int_(legs2_v[0]));
            res.attr("set_label")(legs2_v[0],
                                  tensor1.attr("labels").attr("__getitem__")(1 - legs1_v[0]));
            try {
                return tensors_mod().attr("permute_legs")(res, py::arg("codomain") = legs1_idcs);
            } catch (...) {
                handle_permute_legs_symmetry_error();
            }
        } else if (num_contr == 2) {
            py::object res = scale_axis_py(tensor2, tensor1, py::int_(legs2_v[0]));
            res = partial_trace_py(res, { legs2_idcs });
            if (tensor2.attr("num_legs").cast<int64>() == 2) { // scalar result
                return res;
            }
            return tensors_mod().attr("bend_legs")(res, py::arg("num_codomain_legs") = 0);
        }
    }
    if (is_DiagonalTensor(tensor2)) {
        if (num_contr == 0) {
            tensor2 = tensor2.attr("as_SymmetricTensor")();
        } else if (num_contr == 1) {
            py::object res = scale_axis_py(tensor1, tensor2, py::int_(legs1_v[0]));
            res.attr("set_label")(legs1_v[0],
                                  tensor2.attr("labels").attr("__getitem__")(1 - legs2_v[0]));
            try {
                return tensors_mod().attr("permute_legs")(res, py::arg("domain") = legs1_idcs);
            } catch (...) {
                handle_permute_legs_symmetry_error();
            }
        } else if (num_contr == 2) {
            py::object res = scale_axis_py(tensor1, tensor2, py::int_(legs1_v[0]));
            res = partial_trace_py(res, { legs1_idcs });
            if (tensor1.attr("num_legs").cast<int64>() == 2) { // scalar result
                return res;
            }
            return tensors_mod().attr("bend_legs")(res, py::arg("num_domain_legs") = 0);
        }
    }

    // Deal with ChargedTensor / HiddenLegTensor
    if ((is_ChargedTensor(tensor1) && is_HiddenLegTensor(tensor2)) ||
        (is_HiddenLegTensor(tensor1) && is_ChargedTensor(tensor2))) {
        throw std::invalid_argument(
          "Cannot tdot ChargedTensor with HiddenLegTensor. Unhide or convert first.");
    }
    if (is_HiddenLegTensor(tensor1) || is_HiddenLegTensor(tensor2)) {
        reject_hidden_leg_arguments(tensor1, legs1_v, "tdot");
        reject_hidden_leg_arguments(tensor2, legs2_v, "tdot");
        auto pairs = implicit_hidden_contraction_pairs(tensor1, tensor2);
        for (auto const& [i1, i2] : pairs) {
            legs1_v.push_back(i1);
            legs2_v.push_back(i2);
        }
        legs1_idcs = py::cast(legs1_v);
        legs2_idcs = py::cast(legs2_v);
        num_contr = static_cast<int64>(legs1_v.size());
        // Fall through to SymmetricTensor path (HiddenLegTensor subclasses SymmetricTensor).
    }
    if (is_ChargedTensor(tensor1) && is_ChargedTensor(tensor2)) {
        // note: its important that we have already used get_leg_idcs
        std::string c = charge_leg_label();
        std::string c1 = c + "1";
        std::string c2 = c + "2";
        py::object inv_part = tdot_py(tensor1.attr("invariant_part"),
                                      tensor2.attr("invariant_part"),
                                      legs1_idcs,
                                      legs2_idcs,
                                      std::map<std::string, std::string>{ { c, c1 } },
                                      std::map<std::string, std::string>{ { c, c2 } });
        inv_part = tensors_mod().attr("move_leg")(inv_part, c1, py::arg("domain_pos") = 0);
        return tensors_mod()
          .attr("ChargedTensor")
          .attr("from_two_charge_legs")(inv_part,
                                        py::arg("state1") = tensor1.attr("charged_state"),
                                        py::arg("state2") = tensor2.attr("charged_state"));
    }
    if (is_ChargedTensor(tensor1)) {
        py::object inv_part =
          tdot_py(tensor1.attr("invariant_part"), tensor2, legs1_idcs, legs2_idcs);
        inv_part =
          tensors_mod().attr("move_leg")(inv_part, charge_leg_label(), py::arg("domain_pos") = 0);
        return tensors_mod()
          .attr("ChargedTensor")
          .attr("from_invariant_part")(inv_part, tensor1.attr("charged_state"));
    }
    if (is_ChargedTensor(tensor2)) {
        py::object inv_part =
          tdot_py(tensor1, tensor2.attr("invariant_part"), legs1_idcs, legs2_idcs);
        return tensors_mod()
          .attr("ChargedTensor")
          .attr("from_invariant_part")(inv_part, tensor2.attr("charged_state"));
    }

    // Remaining case: both are SymmetricTensor (including HiddenLegTensor)

    // OPTIMIZE actually, we only need to permute legs to *any* matching order.
    //          could use ``legs1[perm]`` and ``legs2[perm]`` instead, if that means fewer braids.
    try {
        tensor1 = tensors_mod().attr("permute_legs")(
          tensor1, py::arg("domain") = legs1_idcs, py::arg("bend_right") = py::none());
        tensor2 = tensors_mod().attr("permute_legs")(
          tensor2, py::arg("codomain") = legs2_idcs, py::arg("bend_right") = py::none());
    } catch (...) {
        handle_permute_legs_symmetry_error();
    }
    return maybe_wrap_hidden(
      py_from_compose_sym(_compose_SymmetricTensors(tensor1.cast<SymmetricTensorCPtr>(),
                                                    tensor2.cast<SymmetricTensorCPtr>())),
      is_HiddenLegTensor(tensor1) || is_HiddenLegTensor(tensor2));
}

py::object
trace_py(py::object tensor)
{
    if (is_HiddenLegTensor(tensor)) {
        require_no_remaining_hidden(tensor, "trace");
    }
    check_spaces({ tensor.attr("domain") }, { tensor.attr("codomain") });
    if (is_Identity(tensor)) {
        return tensor.attr("leg").attr("dim");
    }
    if (is_DiagonalTensor(tensor)) {
        auto backend = tensor.attr("backend").cast<TensorBackend::Ptr>();
        return scalar_to_py(
          backend->diagonal_tensor_trace_full(tensor.cast<DiagonalTensorCPtr>()));
    }
    if (is_ChargedTensor(tensor)) {
        // OPTIMIZE can project to trivial sector on charge leg first
        int64 N = tensor.attr("num_legs").cast<int64>();
        int64 n_cod = tensor.attr("num_codomain_legs").cast<int64>();
        std::vector<py::object> pairs;
        pairs.reserve(static_cast<std::size_t>(n_cod));
        for (int64 n = 0; n < n_cod; ++n) {
            pairs.push_back(py::make_tuple(n, N - 1 - n));
        }
        py::object inv_block = partial_trace_py(tensor.attr("invariant_part"), pairs);
        inv_block = inv_block.attr("to_dense_block")(py::arg("understood_braiding") = true);
        auto backend = tensor.attr("backend").cast<TensorBackend::Ptr>();
        auto bb = backend->block_backend;
        auto res = bb->tdot(inv_block.cast<BlockBackend::BlockPtr>(),
                            tensor.attr("charged_state").cast<BlockBackend::BlockPtr>(),
                            { 0 },
                            { 0 });
        return scalar_to_py(bb->item(res));
    }
    auto backend = tensor.attr("backend").cast<TensorBackend::Ptr>();
    return scalar_to_py(backend->trace_full(tensor.cast<SymmetricTensorCPtr>(), {}, {}));
}

py::object
transpose_py(py::object tensor)
{
    LegLabels domain_labels = leg_labels_from_py(tensor.attr("domain_labels"));
    LegLabels codomain_labels = leg_labels_from_py(tensor.attr("codomain_labels"));
    LegLabels labels;
    for (auto it = domain_labels.rbegin(); it != domain_labels.rend(); ++it) {
        labels.push_back(*it);
    }
    labels.insert(labels.end(), codomain_labels.begin(), codomain_labels.end());
    py::object labels_py = labels_to_py(labels);

    if (is_Mask(tensor)) {
        auto backend = tensor.attr("backend").cast<TensorBackend::Ptr>();
        auto [space_in, space_out, data] = backend->mask_transpose(tensor.cast<MaskCPtr>());
        return make_python_mask(std::move(data),
                                py::cast(space_in),
                                py::cast(space_out),
                                !tensor.attr("is_projection").cast<bool>(),
                                backend,
                                labels_py);
    }
    if (is_Identity(tensor)) {
        auto backend = tensor.attr("backend").cast<TensorBackend::Ptr>();
        return make_python_identity(tensor.attr("leg").attr("dual"), backend, labels_py);
    }
    if (is_DiagonalTensor(tensor)) {
        auto backend = tensor.attr("backend").cast<TensorBackend::Ptr>();
        auto [dual_leg, data] = backend->diagonal_transpose(tensor.cast<DiagonalTensorCPtr>());
        return make_python_diagonal_tensor(
          std::move(data), py::cast(dual_leg), backend, labels_py);
    }
    if (is_SymmetricTensor(tensor)) {
        int64 n_cod = tensor.attr("num_codomain_legs").cast<int64>();
        int64 n_dom = tensor.attr("num_domain_legs").cast<int64>();
        int64 n_legs = tensor.attr("num_legs").cast<int64>();
        std::vector<int64> codomain;
        for (int64 i = n_cod; i < n_legs; ++i) {
            codomain.push_back(i);
        }
        std::vector<int64> domain;
        for (int64 i = n_cod - 1; i >= 0; --i) {
            domain.push_back(i);
        }
        std::vector<bool> bend_right(static_cast<std::size_t>(n_cod), false);
        bend_right.insert(bend_right.end(), static_cast<std::size_t>(n_dom), true);
        return tensors_mod().attr("permute_legs")(tensor,
                                                  py::cast(codomain),
                                                  py::cast(domain),
                                                  py::arg("bend_right") = py::cast(bend_right));
    }
    if (is_ChargedTensor(tensor)) {
        if (!tensor.attr("symmetry").attr("has_trivial_braid").cast<bool>()) {
            throw SymmetryError(
              "transpose is not defined for ChargedTensors with fermionic symmetries. "
              "This is because there is no way to recover the ChargedTensor format in such a "
              "way that transposing twice gives back the original tensor. "
              "Use permute_legs instead");
        }
        py::object inv_part = transpose_py(tensor.attr("invariant_part"));
        inv_part =
          tensors_mod().attr("move_leg")(inv_part, charge_leg_label(), py::arg("domain_pos") = 0);
        return make_python_charged_tensor(inv_part, tensor.attr("charged_state"));
    }
    throw py::type_error("Invalid type for tensor.");
}

namespace {

py::object
py_leg(LegRef const& leg)
{
    return std::visit([](auto const& x) -> py::object { return py::cast(x); }, leg);
}

py::list
py_legs(std::vector<LegRef> const& legs)
{
    py::list out;
    for (auto const& leg : legs) {
        out.append(py_leg(leg));
    }
    return out;
}

BlockBackend::Scalar
coerce_scalar(py::object o, TensorCPtr hint)
{
    try {
        return o.cast<BlockBackend::Scalar>();
    } catch (py::cast_error const&) {
    }
    return hint->backend->block_backend->as_scalar(o, hint->dtype);
}

std::variant<TensorPtr, BlockBackend::Scalar>
coerce_tensor_or_scalar(py::object o, TensorCPtr hint)
{
    if (py::isinstance<Tensor>(o)) {
        return o.cast<TensorPtr>();
    }
    return coerce_scalar(o, hint);
}

} // namespace

bool
almost_equal(TensorCPtr tensor_1,
             TensorCPtr tensor_2,
             float64 rtol,
             float64 atol,
             bool allow_different_types)
{
    return almost_equal_py(
      tensor_as_py(tensor_1), tensor_as_py(tensor_2), rtol, atol, allow_different_types);
}

TensorPtr
apply_mask(TensorCPtr tensor, MaskCPtr mask, LegRef leg)
{
    return apply_mask_py(tensor_as_py(tensor), py::cast(mask), py_leg(leg)).cast<TensorPtr>();
}

TensorPtr
enlarge_leg(TensorCPtr tensor, MaskCPtr mask, LegRef leg)
{
    return enlarge_leg_py(tensor_as_py(tensor), py::cast(mask), py_leg(leg)).cast<TensorPtr>();
}

TensorPtr
dagger(TensorCPtr tensor)
{
    return dagger_py(tensor_as_py(tensor)).cast<TensorPtr>();
}

std::variant<TensorPtr, BlockBackend::Scalar>
compose(TensorCPtr tensor1,
        TensorCPtr tensor2,
        std::optional<std::map<std::string, std::string>> relabel1,
        std::optional<std::map<std::string, std::string>> relabel2)
{
    return coerce_tensor_or_scalar(
      compose_py(tensor_as_py(tensor1), tensor_as_py(tensor2), relabel1, relabel2), tensor1);
}

BlockBackend::Scalar
inner(TensorCPtr A, TensorCPtr B, bool do_dagger)
{
    return coerce_scalar(inner_py(tensor_as_py(A), tensor_as_py(B), do_dagger), A);
}

BlockBackend::Scalar
inner(VectorLikeCPtr A, VectorLikeCPtr B, bool do_dagger)
{
    if (!A || !B) {
        throw std::invalid_argument("inner() requires non-null VectorLike arguments");
    }
    if (auto ta = std::dynamic_pointer_cast<Tensor const>(A)) {
        if (auto tb = std::dynamic_pointer_cast<Tensor const>(B)) {
            return coerce_scalar(
              inner_py(tensor_as_py(ta), tensor_as_py(tb), do_dagger),
              ta);
        }
    }
    return A->vector_inner(std::move(B), do_dagger);
}

bool
is_scalar(TensorCPtr obj)
{
    return is_scalar_py(py::cast(obj));
}

BlockBackend::Scalar
item(TensorCPtr tensor)
{
    return coerce_scalar(item_py(tensor_as_py(tensor)), tensor);
}

TensorPtr
linear_combination(BlockBackend::Scalar const& a,
                   TensorCPtr v,
                   BlockBackend::Scalar const& b,
                   TensorCPtr w)
{
    return linear_combination_py(
             py::cast(a), tensor_as_py(v), py::cast(b), tensor_as_py(w))
      .cast<TensorPtr>();
}

VectorLikePtr
linear_combination(BlockBackend::Scalar const& a,
                   VectorLikeCPtr v,
                   BlockBackend::Scalar const& b,
                   VectorLikeCPtr w)
{
    if (!v || !w) {
        throw std::invalid_argument("linear_combination() requires non-null VectorLike arguments");
    }
    if (auto tv = std::dynamic_pointer_cast<Tensor const>(v)) {
        auto tw = std::dynamic_pointer_cast<Tensor const>(w);
        if (!tw) {
            throw std::invalid_argument(
              "linear_combination: mixed Tensor / non-Tensor VectorLike arguments");
        }
        return linear_combination(a, std::move(tv), b, std::move(tw));
    }
    return v->axpy(a, w->scaled(b));
}

BlockBackend::Scalar
norm(TensorCPtr tensor)
{
    return coerce_scalar(norm_py(tensor_as_py(tensor)), tensor);
}

BlockBackend::Scalar
norm(VectorLikeCPtr vec)
{
    if (!vec) {
        throw std::invalid_argument("norm() requires a non-null VectorLike");
    }
    if (auto t = std::dynamic_pointer_cast<Tensor const>(vec)) {
        return coerce_scalar(norm_py(tensor_as_py(t)), t);
    }
    return vec->vector_norm();
}

TensorPtr
on_device(TensorCPtr tensor, std::string device, bool copy)
{
    return on_device_py(tensor_as_py(tensor), std::move(device), copy).cast<TensorPtr>();
}

TensorPtr
outer(TensorCPtr tensor1,
      TensorCPtr tensor2,
      std::optional<std::map<std::string, std::string>> relabel1,
      std::optional<std::map<std::string, std::string>> relabel2)
{
    return outer_py(tensor_as_py(tensor1), tensor_as_py(tensor2), relabel1, relabel2)
      .cast<TensorPtr>();
}

TensorPtr
partial_compose(TensorCPtr tensor1,
                TensorCPtr tensor2,
                LegRef tensor1_first_leg,
                std::optional<std::map<std::string, std::string>> relabel1,
                std::optional<std::map<std::string, std::string>> relabel2)
{
    return partial_compose_py(tensor_as_py(tensor1),
                              tensor_as_py(tensor2),
                              py_leg(tensor1_first_leg),
                              relabel1,
                              relabel2)
      .cast<TensorPtr>();
}

std::variant<TensorPtr, BlockBackend::Scalar>
partial_trace(TensorCPtr tensor,
              std::vector<std::vector<LegRef>> pairs,
              std::optional<LevelsSpec> levels)
{
    std::vector<py::object> py_pairs;
    py_pairs.reserve(pairs.size());
    for (auto const& pair : pairs) {
        py_pairs.push_back(py_legs(pair));
    }
    py::object levels_py = py::none();
    if (levels.has_value()) {
        py::list out;
        for (auto const& lv : *levels) {
            if (lv.has_value()) {
                out.append(*lv);
            } else {
                out.append(py::none());
            }
        }
        levels_py = out;
    }
    return coerce_tensor_or_scalar(
      partial_trace_py(tensor_as_py(tensor), std::move(py_pairs), levels_py), tensor);
}

TensorPtr
pinv(TensorCPtr tensor, float64 cutoff)
{
    return pinv_py(tensor_as_py(tensor), cutoff).cast<TensorPtr>();
}

TensorPtr
scalar_multiply(BlockBackend::Scalar const& a, TensorCPtr v)
{
    return scalar_multiply_py(py::cast(a), tensor_as_py(v)).cast<TensorPtr>();
}

VectorLikePtr
scalar_multiply(BlockBackend::Scalar const& a, VectorLikeCPtr v)
{
    if (!v) {
        throw std::invalid_argument("scalar_multiply() requires a non-null VectorLike");
    }
    if (auto t = std::dynamic_pointer_cast<Tensor const>(v)) {
        return scalar_multiply(a, std::move(t));
    }
    return v->scaled(a);
}

TensorPtr
scale_axis(TensorCPtr tensor, DiagonalTensorCPtr diag, LegRef leg)
{
    return scale_axis_py(tensor_as_py(tensor), py::cast(diag), py_leg(leg)).cast<TensorPtr>();
}

std::variant<TensorPtr, BlockBackend::Scalar>
tdot(TensorCPtr tensor1,
     TensorCPtr tensor2,
     std::vector<LegRef> legs1,
     std::vector<LegRef> legs2,
     std::optional<std::map<std::string, std::string>> relabel1,
     std::optional<std::map<std::string, std::string>> relabel2)
{
    return coerce_tensor_or_scalar(tdot_py(tensor_as_py(tensor1),
                                           tensor_as_py(tensor2),
                                           py_legs(legs1),
                                           py_legs(legs2),
                                           relabel1,
                                           relabel2),
                                   tensor1);
}

BlockBackend::Scalar
trace(TensorCPtr tensor)
{
    return coerce_scalar(trace_py(tensor_as_py(tensor)), tensor);
}

TensorPtr
transpose(TensorCPtr tensor)
{
    return transpose_py(tensor_as_py(tensor)).cast<TensorPtr>();
}

} // namespace cyten
