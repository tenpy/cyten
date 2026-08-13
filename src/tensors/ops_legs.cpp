#include <cyten/tensors/ops_legs.h>

#include <cyten/backends/no_symmetry.h>
#include <cyten/backends/tensor_backend.h>
#include <cyten/symmetries/exceptions.h>
#include <cyten/symmetries/spaces.h>
#include <cyten/tensors/charged_tensor.h>
#include <cyten/tensors/diagonal_tensor.h>
#include <cyten/tensors/labels.h>
#include <cyten/tensors/mask.h>
#include <cyten/tensors/ops_algebra.h>
#include <cyten/tensors/symmetric_tensor.h>
#include <cyten/tensors/tensor.h>
#include <cyten/tools.h>

#include <algorithm>
#include <cassert>
#include <format>
#include <map>
#include <stdexcept>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

namespace cyten {

namespace {

py::object
tensors_mod()
{
    return py::module_::import("cyten.tensors._tensors");
}

py::object
spaces_mod()
{
    return py::module_::import("cyten.symmetries.spaces");
}

py::object
misc_mod()
{
    return py::module_::import("cyten.tools.misc");
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
is_LegPipe(py::object obj)
{
    return py::isinstance(obj, spaces_mod().attr("LegPipe")) || py::isinstance<LegPipe>(obj);
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
make_python_charged_tensor(py::object invariant_part, py::object charged_state)
{
    return tensors_mod().attr("ChargedTensor")(invariant_part, charged_state);
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
nested_leg_labels_to_py(LegLabels const& codomain_labels, LegLabels const& domain_labels)
{
    return py::make_tuple(labels_to_py(codomain_labels), labels_to_py(domain_labels));
}

std::vector<int64>
as_int64_vector(py::object seq)
{
    std::vector<int64> out;
    for (auto item : py::reinterpret_borrow<py::iterable>(seq)) {
        out.push_back(item.cast<int64>());
    }
    return out;
}

std::vector<int64>
get_leg_idcs_py(py::object tensor, py::object which)
{
    return as_int64_vector(tensor.attr("get_leg_idcs")(which));
}

bool
contains_int(std::vector<int64> const& v, int64 x)
{
    return std::find(v.begin(), v.end(), x) != v.end();
}

std::vector<int64>
inverse_permutation_local(std::vector<int64> const& perm)
{
    std::vector<int64> inv(perm.size());
    for (std::size_t i = 0; i < perm.size(); ++i) {
        auto const idx = perm[i];
        assert(idx >= 0);
        assert(static_cast<std::size_t>(idx) < perm.size());
        inv[static_cast<std::size_t>(idx)] = static_cast<int64>(i);
    }
    return inv;
}

std::vector<Leg::Ptr>
legs_from_sequence(py::object seq)
{
    std::vector<Leg::Ptr> out;
    for (auto item : py::reinterpret_borrow<py::iterable>(seq)) {
        out.push_back(py::reinterpret_borrow<py::object>(item).cast<Leg::Ptr>());
    }
    return out;
}

bool
as_py_bool(py::object obj)
{
    // Accept Python / numpy bool scalars via the Python truth protocol.
    int r = PyObject_IsTrue(obj.ptr());
    if (r < 0) {
        throw py::error_already_set();
    }
    return r != 0;
}

bool
is_true_or_false(py::object obj)
{
    // Match Python ``x in [True, False]`` (also accepts numpy bool scalars).
    // Do not call this on multi-element arrays (ambiguous truth value).
    return py_eq(obj, py::bool_(true)) || py_eq(obj, py::bool_(false));
}

} // namespace

py::object bend_legs_py(py::object tensor, std::optional<int64> num_codomain_legs,
                        std::optional<int64> num_domain_legs);
void check_same_legs_py(py::object t1, py::object t2);
py::object permute_legs_py(py::object tensor, py::object codomain = py::none(),
                           py::object domain = py::none(), py::object levels = py::none(),
                           py::object bend_right = py::none());
py::object move_leg_py(py::object tensor, py::object which_leg,
                       std::optional<int64> codomain_pos, std::optional<int64> domain_pos,
                       py::object levels, py::object bend_right);
py::object combine_legs_py(py::object tensor, std::vector<py::object> which_legs,
                           py::object pipe_dualities = py::none(), py::object pipes = py::none(),
                           py::object levels = py::none());
py::object combine_to_matrix_py(py::object tensor, py::object codomain, py::object domain,
                                py::object levels);
py::object split_legs_py(py::object tensor, py::object legs = py::none());
py::object squeeze_legs_py(py::object tensor, py::object legs = py::none());

py::object
bend_legs_py(py::object tensor,
          std::optional<int64> num_codomain_legs,
          std::optional<int64> num_domain_legs)
{
    if (!num_codomain_legs.has_value() && !num_domain_legs.has_value()) {
        throw std::invalid_argument("Must specify either num_codomain_legs or num_domain_legs");
    }
    int64 num_legs = tensor.attr("num_legs").cast<int64>();
    int64 n_cod;
    int64 n_dom;
    if (!num_domain_legs.has_value()) {
        n_cod = *num_codomain_legs;
        n_dom = num_legs - n_cod;
    } else if (!num_codomain_legs.has_value()) {
        n_dom = *num_domain_legs;
        n_cod = num_legs - n_dom;
    } else {
        n_cod = *num_codomain_legs;
        n_dom = *num_domain_legs;
        assert(n_cod + n_dom == num_legs);
        (void)n_dom;
    }

    py::list codomain;
    for (int64 i = 0; i < n_cod; ++i) {
        codomain.append(i);
    }
    py::list domain;
    for (int64 i = num_legs - 1; i >= n_cod; --i) {
        domain.append(i);
    }
    return permute_legs_py(tensor, codomain, domain, py::none(), py::bool_(true));
}

void
check_same_legs_py(py::object t1, py::object t2)
{
    // --- hints from Python check_same_legs_py ---
    // either l1 is None or l1 not in l2.labels
    // ---
    if (!t1.attr("symmetry").attr("is_equivalent_to")(t2.attr("symmetry")).cast<bool>()) {
        throw std::invalid_argument("Incompatible symmetries");
    }
    bool incompatible_labels = false;
    py::object labels1 = t1.attr("_labels");
    py::object labelmap2 = t2.attr("_labelmap");
    int64 n = 0;
    for (auto l1 : py::reinterpret_borrow<py::iterable>(labels1)) {
        py::object n2 = labelmap2.attr("get")(l1, py::none());
        if (n2.is_none()) {
            // either l1 is None or l1 not in l2.labels
            ++n;
            continue;
        }
        if (n2.cast<int64>() != n) {
            incompatible_labels = true;
            break;
        }
        ++n;
    }
    bool same_legs = py_eq(t1.attr("domain"), t2.attr("domain")) &&
                     py_eq(t1.attr("codomain"), t2.attr("codomain"));
    if (!same_legs) {
        std::string msg = "Incompatible legs. ";
        if (incompatible_labels) {
            // Match Python f'{t1.labels=}  {t2.labels=}'
            msg += std::format("Should you permute_legs first? t1.labels={}  t2.labels={}",
                               std::string(py::repr(t1.attr("labels"))),
                               std::string(py::repr(t2.attr("labels"))));
        }
        throw std::invalid_argument(msg);
    }
    if (incompatible_labels) {
        auto logger = py::module_::import("logging").attr("getLogger")("cyten.tensors._tensors");
        logger.attr("warning")(
          "Compatible legs with permuted labels detected. Double check your leg order!",
          py::arg("stacklevel") = 3);
    }
}

py::object
permute_legs_py(py::object tensor,
             py::object codomain,
             py::object domain,
             py::object levels,
             py::object bend_right)
{
    // --- hints from Python permute_legs_py ---
    // Parse domain and codomain to list[int]. Get rid of duplicates.
    // to preserve order of Tensor.legs, need to put domain legs in descending order of their
    // leg_idx Special case: if no legs move parse levels to format list[int | None] parse
    // bend_right to format list[bool | None] default -> all undefined single bool applies to all
    // legs check if those that need to be specified are it doesnt matter which way. choose all
    // right Deal with other tensor types OPTIMIZE : else we have a twist in addition to the
    // transpose. we could exploit that structure for DiagonalTensor, to return another
    // DiagonalTensor. We can not preserve the Mask structure, since the twist (in general)
    // introduces phases. other cases involve two legs either in the domain or codomain. Cant be
    // done with Mask / DiagonalTensor assign level `None` to the charge leg. it does not braid, so
    // we dont need to define it. Build new codomain and domain (co)domain has the same factor as
    // before, only permuted -> can re-use sectors!
    // ---
    // Parse domain and codomain to list[int]. Get rid of duplicates.
    if (codomain.is_none() && domain.is_none()) {
        throw std::invalid_argument("Need to specify either domain or codomain.");
    }

    std::vector<int64> domain_v;
    std::vector<int64> codomain_v;
    int64 num_legs = tensor.attr("num_legs").cast<int64>();
    int64 num_codomain_legs = tensor.attr("num_codomain_legs").cast<int64>();

    if (codomain.is_none()) {
        domain_v = get_leg_idcs_py(tensor, domain);
        for (int64 n = 0; n < num_legs; ++n) {
            if (!contains_int(domain_v, n)) {
                codomain_v.push_back(n);
            }
        }
    } else if (domain.is_none()) {
        codomain_v = get_leg_idcs_py(tensor, codomain);
        // to preserve order of Tensor.legs, need to put domain legs in descending order
        for (int64 n = num_legs - 1; n >= 0; --n) {
            if (!contains_int(codomain_v, n)) {
                domain_v.push_back(n);
            }
        }
    } else {
        domain_v = get_leg_idcs_py(tensor, domain);
        codomain_v = get_leg_idcs_py(tensor, codomain);
        std::vector<int64> specified_legs = domain_v;
        specified_legs.insert(specified_legs.end(), codomain_v.begin(), codomain_v.end());
        py::object duplicates = misc_mod().attr("duplicate_entries")(py::cast(specified_legs));
        if (py::len(duplicates) > 0) {
            std::string joined;
            bool first = true;
            for (auto d : duplicates) {
                if (!first) {
                    joined += ", ";
                }
                first = false;
                joined += std::to_string(d.cast<int64>());
            }
            throw std::invalid_argument(
              std::format("Duplicate entries. By leg index: {}", joined));
        }
        std::vector<int64> missing;
        for (int64 n = 0; n < num_legs; ++n) {
            if (!contains_int(specified_legs, n)) {
                missing.push_back(n);
            }
        }
        if (!missing.empty()) {
            std::string joined;
            bool first = true;
            for (auto m : missing) {
                if (!first) {
                    joined += ", ";
                }
                first = false;
                joined += std::to_string(m);
            }
            throw std::invalid_argument(std::format("Missing legs. By leg index: {}", joined));
        }
    }

    // Special case: if no legs move
    bool unchanged = true;
    if (static_cast<int64>(codomain_v.size()) != num_codomain_legs) {
        unchanged = false;
    } else {
        for (int64 i = 0; i < num_codomain_legs; ++i) {
            if (codomain_v[static_cast<std::size_t>(i)] != i) {
                unchanged = false;
                break;
            }
        }
        if (unchanged) {
            int64 expect = num_legs - 1;
            for (auto d : domain_v) {
                if (d != expect) {
                    unchanged = false;
                    break;
                }
                --expect;
            }
        }
    }
    if (unchanged) {
        return tensor;
    }

    // parse levels to format list[int | None]
    std::vector<std::optional<int64>> levels_v;
    levels_v.reserve(static_cast<std::size_t>(num_legs));
    if (levels.is_none()) {
        levels_v.assign(static_cast<std::size_t>(num_legs), std::nullopt);
    } else if (py::isinstance<py::dict>(levels)) {
        levels_v.assign(static_cast<std::size_t>(num_legs), std::nullopt);
        py::dict levels_dict = py::reinterpret_borrow<py::dict>(levels);
        for (auto item : levels_dict) {
            py::object leg = py::reinterpret_borrow<py::object>(item.first);
            py::object level = py::reinterpret_borrow<py::object>(item.second);
            int64 idx = get_leg_idcs_py(tensor, leg)[0];
            if (levels_v[static_cast<std::size_t>(idx)].has_value()) {
                throw std::invalid_argument(std::format("Level for leg {} defined multiple times.",
                                                        std::string(py::str(leg))));
            }
            levels_v[static_cast<std::size_t>(idx)] = level.cast<int64>();
        }
    } else {
        levels_v = {};
        for (auto item : levels) {
            if (item.is_none()) {
                levels_v.push_back(std::nullopt);
            } else {
                levels_v.push_back(item.cast<int64>());
            }
        }
        assert(static_cast<int64>(levels_v.size()) == num_legs);
    }

    // parse bend_right to format list[bool | None]
    std::vector<int64> legs_bending_down;
    for (auto i : domain_v) {
        if (i < num_codomain_legs) {
            legs_bending_down.push_back(i);
        }
    }
    std::vector<int64> legs_bending_up;
    for (auto i : codomain_v) {
        if (i >= num_codomain_legs) {
            legs_bending_up.push_back(i);
        }
    }
    std::vector<int64> bending_legs = legs_bending_down;
    bending_legs.insert(bending_legs.end(), legs_bending_up.begin(), legs_bending_up.end());

    std::vector<std::optional<bool>> bend_right_v;
    bend_right_v.reserve(static_cast<std::size_t>(num_legs));
    if (py::isinstance<py::dict>(bend_right)) {
        bend_right_v.assign(static_cast<std::size_t>(num_legs), std::nullopt);
        py::dict bend_dict = py::reinterpret_borrow<py::dict>(bend_right);
        for (auto item : bend_dict) {
            py::object leg = py::reinterpret_borrow<py::object>(item.first);
            py::object b = py::reinterpret_borrow<py::object>(item.second);
            int64 idx = get_leg_idcs_py(tensor, leg)[0];
            if (b.is_none()) {
                bend_right_v[static_cast<std::size_t>(idx)] = std::nullopt;
            } else {
                bend_right_v[static_cast<std::size_t>(idx)] = b.cast<bool>();
            }
        }
    } else if (!bend_right.is_none() && is_iterable(bend_right)) {
        assert(static_cast<int64>(py::len(bend_right)) == num_legs);
        for (auto item : bend_right) {
            if (item.is_none()) {
                bend_right_v.push_back(std::nullopt);
            } else {
                bend_right_v.push_back(as_py_bool(py::reinterpret_borrow<py::object>(item)));
            }
        }
    } else if (bend_right.is_none()) {
        bend_right_v.assign(static_cast<std::size_t>(num_legs), std::nullopt);
    } else if (is_true_or_false(bend_right)) {
        bool b = as_py_bool(bend_right);
        bend_right_v.assign(static_cast<std::size_t>(num_legs), b);
    } else {
        throw std::invalid_argument("Invalid bend_right.");
    }

    // check if those that need to be specified are
    if (tensor.attr("symmetry").attr("has_trivial_braid").cast<bool>()) {
        // it doesnt matter which way. choose all right
        bend_right_v.assign(static_cast<std::size_t>(num_legs), true);
    } else {
        for (auto l : bending_legs) {
            if (!bend_right_v[static_cast<std::size_t>(l)].has_value()) {
                throw SymmetryError("Need to specify bend_right!");
            }
        }
    }

    // Deal with other tensor types
    if (is_DiagonalTensor(tensor) || is_Mask(tensor)) {
        if (codomain_v == std::vector<int64>{ 0 } && domain_v == std::vector<int64>{ 1 }) {
            return tensor;
        }
        if (codomain_v == std::vector<int64>{ 1 } && domain_v == std::vector<int64>{ 0 }) {
            bool trivial_braid = tensor.attr("symmetry").attr("has_trivial_braid").cast<bool>();
            bool opposite_bends = bend_right_v[0].has_value() && bend_right_v[1].has_value() &&
                                  (*bend_right_v[0] != *bend_right_v[1]);
            if (trivial_braid || opposite_bends) {
                return py::cast(transpose(tensor.cast<TensorCPtr>()));
            }
            // OPTIMIZE : else we have a twist in addition to the transpose.
        }
        // other cases involve two legs either in the domain or codomain.
        // Cant be done with Mask / DiagonalTensor
        char const* msg = "Converting to SymmetricTensor for permuting legs. "
                          "Use as_SymmetricTensor() explicitly to suppress the warning.";
        tensor = tensor.attr("as_SymmetricTensor")(py::arg("warning") = msg);
    }
    if (is_ChargedTensor(tensor)) {
        // assign level `None` to the charge leg. it does not braid, so we dont need to define it.
        py::list domain_with_charge;
        domain_with_charge.append(-1);
        for (auto d : domain_v) {
            domain_with_charge.append(d);
        }
        py::list levels_ext;
        for (auto const& lv : levels_v) {
            if (lv.has_value()) {
                levels_ext.append(*lv);
            } else {
                levels_ext.append(py::none());
            }
        }
        levels_ext.append(py::none());
        py::list bend_ext;
        for (auto const& b : bend_right_v) {
            if (b.has_value()) {
                bend_ext.append(*b);
            } else {
                bend_ext.append(py::none());
            }
        }
        bend_ext.append(py::none());
        py::object inv_part = permute_legs_py(tensor.attr("invariant_part"),
                                           py::cast(codomain_v),
                                           domain_with_charge,
                                           levels_ext,
                                           bend_ext);
        return make_python_charged_tensor(inv_part, tensor.attr("charged_state"));
    }

    // Build new codomain and domain
    TensorProduct::Ptr new_codomain;
    TensorProduct::Ptr new_domain;
    if (!bending_legs.empty()) {
        py::list cod_spaces;
        for (auto i : codomain_v) {
            cod_spaces.append(tensor.attr("_as_codomain_leg")(i));
        }
        py::list dom_spaces;
        for (auto i : domain_v) {
            dom_spaces.append(tensor.attr("_as_domain_leg")(i));
        }
        new_codomain = spaces_mod()
                         .attr("TensorProduct")(cod_spaces, tensor.attr("symmetry"))
                         .cast<TensorProduct::Ptr>();
        new_domain = spaces_mod()
                       .attr("TensorProduct")(dom_spaces, tensor.attr("symmetry"))
                       .cast<TensorProduct::Ptr>();
    } else {
        // (co)domain has the same factor as before, only permuted -> can re-use sectors!
        new_codomain = tensor.attr("codomain").cast<TensorProduct::Ptr>()->permuted(codomain_v);
        std::vector<int64> dom_perm;
        dom_perm.reserve(domain_v.size());
        for (auto i : domain_v) {
            dom_perm.push_back(num_legs - 1 - i);
        }
        new_domain = tensor.attr("domain").cast<TensorProduct::Ptr>()->permuted(dom_perm);
    }

    auto backend = tensor.attr("backend").cast<TensorBackend::Ptr>();
    auto data = backend->permute_legs(tensor.cast<TensorCPtr>(),
                                      codomain_v,
                                      domain_v,
                                      new_codomain,
                                      new_domain,
                                      /*mixes_codomain_domain=*/!bending_legs.empty(),
                                      levels_v,
                                      bend_right_v);

    LegLabels all_labels = leg_labels_from_py(tensor.attr("_labels"));
    LegLabels cod_labels;
    LegLabels dom_labels;
    for (auto n : codomain_v) {
        cod_labels.push_back(all_labels[static_cast<std::size_t>(n)]);
    }
    for (auto n : domain_v) {
        dom_labels.push_back(all_labels[static_cast<std::size_t>(n)]);
    }
    return make_python_symmetric_tensor(std::move(data),
                                        py::cast(new_codomain),
                                        py::cast(new_domain),
                                        backend,
                                        nested_leg_labels_to_py(cod_labels, dom_labels));
}

py::object
move_leg_py(py::object tensor,
         py::object which_leg,
         std::optional<int64> codomain_pos,
         std::optional<int64> domain_pos,
         py::object levels,
         py::object bend_right)
{
    py::object parsed = tensor.attr("_parse_leg_idx")(which_leg);
    bool from_domain = parsed.attr("__getitem__")(0).cast<bool>();
    int64 leg_idx = parsed.attr("__getitem__")(2).cast<int64>();
    int64 num_codomain_legs = tensor.attr("num_codomain_legs").cast<int64>();
    int64 num_legs = tensor.attr("num_legs").cast<int64>();

    std::vector<int64> new_codomain;
    std::vector<int64> new_domain;
    if (from_domain) {
        for (int64 n = 0; n < num_codomain_legs; ++n) {
            new_codomain.push_back(n);
        }
        for (int64 n = num_legs - 1; n >= num_codomain_legs; --n) {
            if (n != leg_idx) {
                new_domain.push_back(n);
            }
        }
    } else {
        for (int64 n = 0; n < num_codomain_legs; ++n) {
            if (n != leg_idx) {
                new_codomain.push_back(n);
            }
        }
        for (int64 n = num_legs - 1; n >= num_codomain_legs; --n) {
            new_domain.push_back(n);
        }
    }

    if (codomain_pos.has_value()) {
        if (domain_pos.has_value()) {
            throw std::invalid_argument("Can not specify both codomain_pos and domain_pos.");
        }
        int64 pos = to_valid_idx(*codomain_pos, static_cast<int64>(new_codomain.size()) + 1);
        new_codomain.insert(new_codomain.begin() + pos, leg_idx);
    } else if (domain_pos.has_value()) {
        int64 pos = to_valid_idx(*domain_pos, static_cast<int64>(new_domain.size()) + 1);
        new_domain.insert(new_domain.begin() + pos, leg_idx);
    } else {
        throw std::invalid_argument("Need to specify either codomain_pos or domain_pos.");
    }

    return permute_legs_py(tensor, py::cast(new_codomain), py::cast(new_domain), levels, bend_right);
}

py::object
combine_legs_py(py::object tensor,
             std::vector<py::object> which_legs,
             py::object pipe_dualities,
             py::object pipes,
             py::object levels)
{
    // --- hints from Python combine_legs_py ---
    // 1) Deal with different tensor types. Reduce everything to SymmetricTensor.
    // note: its important to parse negative integers before via tensor.get_leg_idcs, since
    // the invariant part has an additional leg.
    // charge leg is not combined with anything and thus does not braid.
    // so its level is irrelevant. just make sure its not a duplicate
    // 2) permute legs such that the groups are contiguous and fully in codomain or fully in domain
    // build indices for permute_legs_py
    // easier to build right-to-left.
    // note: the group is given in right-to-left convention, but this is what we expect.
    // n is one of the legs to be combined, but it is not the first of its group.
    // leg positions have changed, so we need to update the following lists/dicts:
    // 3) build new domain and codomain, labels
    // have already used pipes[:i]
    // Note: this is the result.domain[some_idx],  which has opposite duality from
    // result.legs[-1-some_idx], so we need to invert pipe_dualities[i]
    // n is part of a group, but not the *first* of its group
    // OPTIMIZE if no bending happened, we can re-use the (co)domain.sector_decomposition.
    // 4) Build the data / finish up
    // ---
    // 1) Deal with different tensor types. Reduce everything to SymmetricTensor.
    if (is_DiagonalTensor(tensor) || is_Mask(tensor)) {
        char const* msg = "Converting to SymmetricTensor for combine_legs. "
                          "Use as_SymmetricTensor() explicitly to suppress the warning.";
        tensor = tensor.attr("as_SymmetricTensor")(py::arg("warning") = msg);
    }

    std::vector<std::vector<int64>> which_legs_v;
    which_legs_v.reserve(which_legs.size());
    for (auto const& group : which_legs) {
        which_legs_v.push_back(get_leg_idcs_py(tensor, group));
    }

    if (is_ChargedTensor(tensor)) {
        // note: its important to parse negative integers before via tensor.get_leg_idcs, since
        //       the invariant part has an additional leg.
        py::object levels_for_inv = levels;
        if (!levels.is_none()) {
            // charge leg is not combined with anything and thus does not braid.
            // so its level is irrelevant. just make sure its not a duplicate
            py::list levels_list = py::list(levels);
            int64 min_level = levels_list[0].cast<int64>();
            for (auto item : levels_list) {
                min_level = std::min(min_level, item.cast<int64>());
            }
            levels_list.append(min_level - 1);
            levels_for_inv = levels_list;
        }
        std::vector<py::object> which_as_py;
        which_as_py.reserve(which_legs_v.size());
        for (auto const& g : which_legs_v) {
            which_as_py.push_back(py::cast(g));
        }
        py::object inv_part = combine_legs_py(
          tensor.attr("invariant_part"), which_as_py, pipe_dualities, pipes, levels_for_inv);
        return make_python_charged_tensor(inv_part, tensor.attr("charged_state"));
    }

    // 2) permute legs such that the groups are contiguous and fully in codomain or fully in domain
    int64 N = tensor.attr("num_legs").cast<int64>();
    int64 J = tensor.attr("num_codomain_legs").cast<int64>();
    std::vector<int64> to_combine;
    for (auto const& group : which_legs_v) {
        to_combine.insert(to_combine.end(), group.begin(), group.end());
    }
    if (py::len(misc_mod().attr("duplicate_entries")(py::cast(to_combine))) > 0) {
        throw std::invalid_argument("Groups may not contain duplicates.");
    }

    // build indices for permute_legs_py
    std::map<int64, std::vector<int64>> codomain_groups;
    std::map<int64, std::vector<int64>> domain_groups;
    for (auto const& group : which_legs_v) {
        if (group[0] < J) {
            codomain_groups[group[0]] = group;
        } else {
            domain_groups[group[0]] = group;
        }
    }
    std::vector<int64> codomain_idcs;
    std::vector<int64> domain_idcs_reversed; // easier to build right-to-left.
    for (int64 n = 0; n < N; ++n) {
        if (codomain_groups.contains(n)) {
            auto const& g = codomain_groups[n];
            codomain_idcs.insert(codomain_idcs.end(), g.begin(), g.end());
        } else if (domain_groups.contains(n)) {
            // note: the group is given in right-to-left convention, but this is what we expect.
            auto const& g = domain_groups[n];
            domain_idcs_reversed.insert(domain_idcs_reversed.end(), g.begin(), g.end());
        } else if (contains_int(to_combine, n)) {
            // n is one of the legs to be combined, but it is not the first of its group.
        } else if (n < J) {
            codomain_idcs.push_back(n);
        } else {
            domain_idcs_reversed.push_back(n);
        }
    }

    std::vector<int64> domain_idcs = domain_idcs_reversed;
    std::reverse(domain_idcs.begin(), domain_idcs.end());
    tensor = permute_legs_py(tensor, py::cast(codomain_idcs), py::cast(domain_idcs), levels);

    // leg positions have changed, so we need to update the following lists/dicts:
    std::vector<int64> full_perm = codomain_idcs;
    full_perm.insert(full_perm.end(), domain_idcs_reversed.begin(), domain_idcs_reversed.end());
    auto inv_perm = inverse_permutation_local(full_perm);
    for (auto& group : which_legs_v) {
        for (auto& l : group) {
            l = inv_perm[static_cast<std::size_t>(l)];
        }
    }
    to_combine.clear();
    for (auto const& group : which_legs_v) {
        to_combine.insert(to_combine.end(), group.begin(), group.end());
    }
    J = tensor.attr("num_codomain_legs").cast<int64>();
    codomain_groups.clear();
    domain_groups.clear();
    for (auto const& group : which_legs_v) {
        if (group[0] < J) {
            codomain_groups[group[0]] = group;
        } else {
            domain_groups[group[0]] = group;
        }
    }

    // 3) build new domain and codomain, labels
    py::list pipes_list;
    if (pipes.is_none()) {
        for (std::size_t k = 0; k < which_legs_v.size(); ++k) {
            pipes_list.append(py::none());
        }
    } else {
        pipes_list = py::list(pipes);
    }
    std::vector<bool> pipe_dualities_v;
    if (pipe_dualities.is_none()) {
        pipe_dualities_v.assign(which_legs_v.size(), false);
    } else if (is_iterable(pipe_dualities)) {
        assert(static_cast<std::size_t>(py::len(pipe_dualities)) == which_legs_v.size());
        for (auto item : pipe_dualities) {
            pipe_dualities_v.push_back(as_py_bool(py::reinterpret_borrow<py::object>(item)));
        }
    } else {
        pipe_dualities_v.assign(which_legs_v.size(), as_py_bool(pipe_dualities));
    }

    auto backend = tensor.attr("backend").cast<TensorBackend::Ptr>();
    std::vector<py::object> codomain_spaces;
    std::vector<LegLabel> codomain_labels;
    std::vector<LegLabel> domain_labels_reversed;
    std::vector<py::object> domain_spaces_reversed;
    std::size_t i = 0; // have already used pipes[:i]
    int64 label_offset = 0;
    LegLabels all_labels = leg_labels_from_py(tensor.attr("labels"));

    for (int64 n = 0; n < N; ++n) {
        if (codomain_groups.contains(n)) {
            auto const& group = codomain_groups[n];
            py::object spaces_to_combine =
              tensor.attr("codomain")
                .attr("__getitem__")(py::slice(static_cast<py::ssize_t>(group.front()),
                                               static_cast<py::ssize_t>(group.back() + 1),
                                               1));
            LegPipe::Ptr pipe_arg =
              pipes_list[static_cast<py::ssize_t>(i)].is_none()
                ? nullptr
                : pipes_list[static_cast<py::ssize_t>(i)].cast<LegPipe::Ptr>();
            auto combined = backend->make_pipe(
              legs_from_sequence(spaces_to_combine), pipe_dualities_v[i], pipe_arg);
            pipes_list[static_cast<py::ssize_t>(i)] = py::cast(combined);
            codomain_spaces.push_back(py::cast(combined));
            LegLabels group_labels(all_labels.begin() + group.front(),
                                   all_labels.begin() + group.back() + 1);
            codomain_labels.push_back(_combine_leg_labels(group_labels, label_offset));
            ++i;
            int64 none_count = 0;
            for (auto l : group) {
                if (!all_labels[static_cast<std::size_t>(l)].has_value()) {
                    ++none_count;
                }
            }
            label_offset += none_count;
        } else if (domain_groups.contains(n)) {
            auto const& group = domain_groups[n];
            int64 domain_idx1 = N - 1 - group.front();
            int64 codomain_idx2 = N - 1 - group.back();
            py::object spaces_to_combine = tensor.attr("domain").attr("__getitem__")(
              py::slice(static_cast<py::ssize_t>(codomain_idx2),
                        static_cast<py::ssize_t>(domain_idx1 + 1),
                        1));
            // Note: this is the result.domain[some_idx],  which has opposite duality from
            //       result.legs[-1-some_idx], so we need to invert pipe_dualities[i]
            LegPipe::Ptr pipe_arg =
              pipes_list[static_cast<py::ssize_t>(i)].is_none()
                ? nullptr
                : pipes_list[static_cast<py::ssize_t>(i)].cast<LegPipe::Ptr>();
            auto combined = backend->make_pipe(
              legs_from_sequence(spaces_to_combine), !pipe_dualities_v[i], pipe_arg);
            pipes_list[static_cast<py::ssize_t>(i)] = py::cast(combined);
            domain_spaces_reversed.push_back(py::cast(combined));
            LegLabels group_labels(all_labels.begin() + group.front(),
                                   all_labels.begin() + group.back() + 1);
            domain_labels_reversed.push_back(_combine_leg_labels(group_labels, label_offset));
            ++i;
            int64 none_count = 0;
            for (auto l : group) {
                if (!all_labels[static_cast<std::size_t>(l)].has_value()) {
                    ++none_count;
                }
            }
            label_offset += none_count;
        } else if (contains_int(to_combine, n)) {
            // n is part of a group, but not the *first* of its group
        } else if (n < J) {
            codomain_spaces.push_back(tensor.attr("codomain").attr("__getitem__")(n));
            codomain_labels.push_back(all_labels[static_cast<std::size_t>(n)]);
        } else {
            domain_spaces_reversed.push_back(tensor.attr("domain").attr("__getitem__")(N - 1 - n));
            domain_labels_reversed.push_back(all_labels[static_cast<std::size_t>(n)]);
        }
    }

    // OPTIMIZE if no bending happened, we can re-use the (co)domain.sector_decomposition.
    py::list domain_spaces;
    for (auto it = domain_spaces_reversed.rbegin(); it != domain_spaces_reversed.rend(); ++it) {
        domain_spaces.append(*it);
    }
    py::object codomain =
      spaces_mod().attr("TensorProduct")(py::cast(codomain_spaces), tensor.attr("symmetry"));
    py::object domain = spaces_mod().attr("TensorProduct")(domain_spaces, tensor.attr("symmetry"));

    // 4) Build the data / finish up
    std::sort(which_legs_v.begin(), which_legs_v.end());
    std::vector<LegPipe::Ptr> pipes_ptr;
    pipes_ptr.reserve(which_legs_v.size());
    for (std::size_t k = 0; k < which_legs_v.size(); ++k) {
        pipes_ptr.push_back(pipes_list[static_cast<py::ssize_t>(k)].cast<LegPipe::Ptr>());
    }
    auto data = backend->combine_legs(tensor.cast<TensorCPtr>(),
                                      which_legs_v,
                                      pipes_ptr,
                                      codomain.cast<TensorProduct::Ptr>(),
                                      domain.cast<TensorProduct::Ptr>());

    LegLabels res_labels = codomain_labels;
    // domain_labels_reversed is already in legs order for the domain part of tensor.legs
    // (right-to-left build), matching Python [*codomain_labels, *domain_labels_reversed]
    res_labels.insert(
      res_labels.end(), domain_labels_reversed.begin(), domain_labels_reversed.end());
    return make_python_symmetric_tensor(
      std::move(data), codomain, domain, backend, labels_to_py(res_labels));
}

py::object
combine_to_matrix_py(py::object tensor, py::object codomain, py::object domain, py::object levels)
{
    py::object res = permute_legs_py(tensor, codomain, domain, levels);
    int64 n_cod = res.attr("num_codomain_legs").cast<int64>();
    int64 n_legs = res.attr("num_legs").cast<int64>();
    py::list cod_range;
    for (int64 i = 0; i < n_cod; ++i) {
        cod_range.append(i);
    }
    py::list dom_range;
    for (int64 i = n_cod; i < n_legs; ++i) {
        dom_range.append(i);
    }
    return combine_legs_py(res, { cod_range, dom_range });
}

py::object
split_legs_py(py::object tensor, py::object legs)
{
    // --- hints from Python split_legs ---
    // Deal with different tensor types. Reduce everything to SymmetricTensor.
    // parse indices
    // build new (co)domain
    // we only split, i.e. remove parentheses in tensor products, so sectors dont change
    // build labels
    // ---
    // Deal with different tensor types. Reduce everything to SymmetricTensor.
    if (is_DiagonalTensor(tensor) || is_Mask(tensor)) {
        char const* msg = "Converting to SymmetricTensor for split_legs. Use as_SymmetricTensor() "
                          "explicitly to suppress the warning.";
        tensor = tensor.attr("as_SymmetricTensor")(py::arg("warning") = msg);
    }
    if (is_ChargedTensor(tensor)) {
        py::object legs_for_inv = legs;
        if (!legs.is_none()) {
            legs_for_inv = py::cast(get_leg_idcs_py(tensor, legs));
        }
        return make_python_charged_tensor(split_legs_py(tensor.attr("invariant_part"), legs_for_inv),
                                          tensor.attr("charged_state"));
    }

    // parse indices
    std::vector<int64> leg_idcs;
    std::vector<int64> codomain_split;
    std::vector<int64> domain_split;
    int64 num_legs = tensor.attr("num_legs").cast<int64>();

    if (legs.is_none()) {
        int64 n = 0;
        for (auto l : tensor.attr("codomain")) {
            if (is_LegPipe(py::reinterpret_borrow<py::object>(l))) {
                codomain_split.push_back(n);
            }
            ++n;
        }
        n = 0;
        for (auto l : tensor.attr("domain")) {
            if (is_LegPipe(py::reinterpret_borrow<py::object>(l))) {
                domain_split.push_back(n);
            }
            ++n;
        }
        leg_idcs = codomain_split;
        for (auto it = domain_split.rbegin(); it != domain_split.rend(); ++it) {
            leg_idcs.push_back(num_legs - 1 - *it);
        }
    } else {
        auto sorted = get_leg_idcs_py(tensor, legs);
        std::sort(sorted.begin(), sorted.end());
        for (auto l : sorted) {
            py::object parsed = tensor.attr("_parse_leg_idx")(l);
            bool in_domain = parsed.attr("__getitem__")(0).cast<bool>();
            int64 co_domain_idx = parsed.attr("__getitem__")(1).cast<int64>();
            int64 leg_idx = parsed.attr("__getitem__")(2).cast<int64>();
            leg_idcs.push_back(leg_idx);
            if (in_domain) {
                domain_split.push_back(co_domain_idx);
            } else {
                codomain_split.push_back(co_domain_idx);
            }
            if (!is_LegPipe(tensor.attr("get_leg_co_domain")(leg_idx))) {
                throw std::invalid_argument("Not a LegPipe.");
            }
        }
    }

    // build new (co)domain
    py::list codomain_spaces;
    int64 n = 0;
    for (auto l : tensor.attr("codomain")) {
        py::object lo = py::reinterpret_borrow<py::object>(l);
        if (contains_int(codomain_split, n)) {
            for (auto sub : lo.attr("legs")) {
                codomain_spaces.append(sub);
            }
        } else {
            codomain_spaces.append(lo);
        }
        ++n;
    }
    py::list domain_spaces;
    n = 0;
    for (auto l : tensor.attr("domain")) {
        py::object lo = py::reinterpret_borrow<py::object>(l);
        if (contains_int(domain_split, n)) {
            for (auto sub : lo.attr("legs")) {
                domain_spaces.append(sub);
            }
        } else {
            domain_spaces.append(lo);
        }
        ++n;
    }

    // we only split, i.e. remove parentheses in tensor products, so sectors dont change
    py::object codomain = spaces_mod().attr("TensorProduct")(
      codomain_spaces,
      tensor.attr("symmetry"),
      py::arg("_sector_decomposition") = tensor.attr("codomain").attr("sector_decomposition"),
      py::arg("_multiplicities") = tensor.attr("codomain").attr("multiplicities"));
    py::object domain = spaces_mod().attr("TensorProduct")(
      domain_spaces,
      tensor.attr("symmetry"),
      py::arg("_sector_decomposition") = tensor.attr("domain").attr("sector_decomposition"),
      py::arg("_multiplicities") = tensor.attr("domain").attr("multiplicities"));

    // build labels
    LegLabels all_labels = leg_labels_from_py(tensor.attr("labels"));
    LegLabels labels;
    std::unordered_set<int64> leg_idcs_set(leg_idcs.begin(), leg_idcs.end());
    for (int64 idx = 0; idx < static_cast<int64>(all_labels.size()); ++idx) {
        if (leg_idcs_set.contains(idx)) {
            int64 num = tensor.attr("get_leg_co_domain")(idx).attr("num_legs").cast<int64>();
            auto split = _split_leg_label(all_labels[static_cast<std::size_t>(idx)], num);
            labels.insert(labels.end(), split.begin(), split.end());
        } else {
            labels.push_back(all_labels[static_cast<std::size_t>(idx)]);
        }
    }

    std::sort(leg_idcs.begin(), leg_idcs.end());
    auto backend = tensor.attr("backend").cast<TensorBackend::Ptr>();
    auto data = backend->split_legs(tensor.cast<TensorCPtr>(),
                                    leg_idcs,
                                    codomain.cast<TensorProduct::Ptr>(),
                                    domain.cast<TensorProduct::Ptr>());
    return make_python_symmetric_tensor(
      std::move(data), codomain, domain, backend, labels_to_py(labels));
}

py::object
squeeze_legs_py(py::object tensor, py::object legs)
{
    // --- hints from Python squeeze_legs ---
    // Remaining case: SymmetricTensor
    // the fusion with the trivial legs was trivial, so removing it doesnt change the sectors
    // ---
    std::vector<int64> legs_v;
    if (legs.is_none()) {
        int64 n = 0;
        for (auto l : tensors_mod().attr("conventional_leg_order")(tensor)) {
            if (py::reinterpret_borrow<py::object>(l).attr("is_trivial").cast<bool>()) {
                legs_v.push_back(n);
            }
            ++n;
        }
    } else {
        legs_v = get_leg_idcs_py(tensor, legs);
        for (auto n : legs_v) {
            if (!tensor.attr("get_leg_co_domain")(n).attr("is_trivial").cast<bool>()) {
                throw std::invalid_argument("Can only squeeze trivial legs");
            }
        }
    }
    if (legs_v.empty()) {
        return tensor;
    }
    if (is_DiagonalTensor(tensor) || is_Mask(tensor)) {
        char const* msg = "Converting to SymmetricTensor for squeeze_legs. "
                          "Use as_SymmetricTensor() explicitly to suppress the warning.";
        tensor = tensor.attr("as_SymmetricTensor")(py::arg("warning") = msg);
    }
    if (is_ChargedTensor(tensor)) {
        return make_python_charged_tensor(
          squeeze_legs_py(tensor.attr("invariant_part"), py::cast(legs_v)),
          tensor.attr("charged_state"));
    }
    // Remaining case: SymmetricTensor
    int64 num_legs = tensor.attr("num_legs").cast<int64>();
    int64 num_codomain_legs = tensor.attr("num_codomain_legs").cast<int64>();
    int64 num_domain_legs = tensor.attr("num_domain_legs").cast<int64>();
    std::unordered_set<int64> legs_set(legs_v.begin(), legs_v.end());
    std::vector<int64> remaining;
    for (int64 n = 0; n < num_legs; ++n) {
        if (!legs_set.contains(n)) {
            remaining.push_back(n);
        }
    }

    auto backend = tensor.attr("backend").cast<TensorBackend::Ptr>();
    auto data = backend->squeeze_legs(tensor.cast<TensorCPtr>(), legs_v);

    // the fusion with the trivial legs was trivial, so removing it doesnt change the sectors
    py::list cod_spaces;
    for (int64 n = 0; n < num_codomain_legs; ++n) {
        if (!legs_set.contains(n)) {
            cod_spaces.append(tensor.attr("codomain").attr("__getitem__")(n));
        }
    }
    py::list dom_spaces;
    for (int64 n = 0; n < num_domain_legs; ++n) {
        if (!legs_set.contains(num_legs - 1 - n)) {
            dom_spaces.append(tensor.attr("domain").attr("__getitem__")(n));
        }
    }
    py::object codomain = spaces_mod().attr("TensorProduct")(
      cod_spaces,
      tensor.attr("symmetry"),
      py::arg("_sector_decomposition") = tensor.attr("codomain").attr("sector_decomposition"),
      py::arg("_multiplicities") = tensor.attr("codomain").attr("multiplicities"));
    py::object domain = spaces_mod().attr("TensorProduct")(
      dom_spaces,
      tensor.attr("symmetry"),
      py::arg("_sector_decomposition") = tensor.attr("domain").attr("sector_decomposition"),
      py::arg("_multiplicities") = tensor.attr("domain").attr("multiplicities"));

    LegLabels all_labels = leg_labels_from_py(tensor.attr("_labels"));
    LegLabels labels;
    for (auto n : remaining) {
        labels.push_back(all_labels[static_cast<std::size_t>(n)]);
    }
    return make_python_symmetric_tensor(
      std::move(data), codomain, domain, backend, labels_to_py(labels));
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

py::object
py_opt_legs(std::optional<std::vector<LegRef>> const& legs)
{
    if (!legs.has_value()) {
        return py::none();
    }
    return py_legs(*legs);
}

py::object
py_levels(std::optional<LevelsSpec> const& levels)
{
    if (!levels.has_value()) {
        return py::none();
    }
    py::list out;
    for (auto const& lv : *levels) {
        if (lv.has_value()) {
            out.append(*lv);
        } else {
            out.append(py::none());
        }
    }
    return out;
}

py::object
py_bend_right(std::optional<BendRight> const& bend_right)
{
    if (!bend_right.has_value()) {
        return py::none();
    }
    return std::visit(
      [](auto const& spec) -> py::object {
          using T = std::decay_t<decltype(spec)>;
          if constexpr (std::is_same_v<T, bool>) {
              return py::bool_(spec);
          } else {
              py::list out;
              for (auto const& b : spec) {
                  if (b.has_value()) {
                      out.append(*b);
                  } else {
                      out.append(py::none());
                  }
              }
              return out;
          }
      },
      *bend_right);
}

py::object
py_pipe_dualities(std::optional<PipeDualities> const& pipe_dualities)
{
    if (!pipe_dualities.has_value()) {
        return py::none();
    }
    return std::visit(
      [](auto const& spec) -> py::object {
          using T = std::decay_t<decltype(spec)>;
          if constexpr (std::is_same_v<T, bool>) {
              return py::bool_(spec);
          } else {
              py::list out;
              for (bool b : spec) {
                  out.append(b);
              }
              return out;
          }
      },
      *pipe_dualities);
}

py::object
py_pipes(std::optional<std::vector<Leg::Ptr>> const& pipes)
{
    if (!pipes.has_value()) {
        return py::none();
    }
    py::list out;
    for (auto const& p : *pipes) {
        out.append(p);
    }
    return out;
}

std::vector<py::object>
py_which_legs(std::vector<std::vector<LegRef>> const& which_legs)
{
    std::vector<py::object> out;
    out.reserve(which_legs.size());
    for (auto const& group : which_legs) {
        out.push_back(py_legs(group));
    }
    return out;
}

} // namespace

TensorPtr
bend_legs(TensorCPtr tensor,
          std::optional<int64> num_codomain_legs,
          std::optional<int64> num_domain_legs)
{
    return bend_legs_py(py::cast(tensor), num_codomain_legs, num_domain_legs).cast<TensorPtr>();
}

void
check_same_legs(TensorCPtr t1, TensorCPtr t2)
{
    check_same_legs_py(py::cast(t1), py::cast(t2));
}

TensorPtr
combine_legs(TensorCPtr tensor,
             std::vector<std::vector<LegRef>> which_legs,
             std::optional<PipeDualities> pipe_dualities,
             std::optional<std::vector<Leg::Ptr>> pipes,
             std::optional<LevelsSpec> levels)
{
    return combine_legs_py(py::cast(tensor),
                           py_which_legs(which_legs),
                           py_pipe_dualities(pipe_dualities),
                           py_pipes(pipes),
                           py_levels(levels))
      .cast<TensorPtr>();
}

TensorPtr
combine_to_matrix(TensorCPtr tensor,
                  std::optional<std::vector<LegRef>> codomain,
                  std::optional<std::vector<LegRef>> domain,
                  std::optional<LevelsSpec> levels)
{
    return combine_to_matrix_py(
             py::cast(tensor), py_opt_legs(codomain), py_opt_legs(domain), py_levels(levels))
      .cast<TensorPtr>();
}

TensorPtr
move_leg(TensorCPtr tensor,
         LegRef which_leg,
         std::optional<int64> codomain_pos,
         std::optional<int64> domain_pos,
         std::optional<LevelsSpec> levels,
         std::optional<BendRight> bend_right)
{
    return move_leg_py(py::cast(tensor),
                       py_leg(which_leg),
                       codomain_pos,
                       domain_pos,
                       py_levels(levels),
                       py_bend_right(bend_right))
      .cast<TensorPtr>();
}

TensorPtr
permute_legs(TensorCPtr tensor,
             std::optional<std::vector<LegRef>> codomain,
             std::optional<std::vector<LegRef>> domain,
             std::optional<LevelsSpec> levels,
             std::optional<BendRight> bend_right)
{
    return permute_legs_py(py::cast(tensor),
                           py_opt_legs(codomain),
                           py_opt_legs(domain),
                           py_levels(levels),
                           py_bend_right(bend_right))
      .cast<TensorPtr>();
}

TensorPtr
split_legs(TensorCPtr tensor, std::optional<std::vector<LegRef>> legs)
{
    return split_legs_py(py::cast(tensor), py_opt_legs(legs)).cast<TensorPtr>();
}

TensorPtr
squeeze_legs(TensorCPtr tensor, std::optional<std::vector<LegRef>> legs)
{
    return squeeze_legs_py(py::cast(tensor), py_opt_legs(legs)).cast<TensorPtr>();
}

} // namespace cyten
