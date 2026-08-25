#include <cyten/tensors/planar.h>

#include <cyten/backends/fusion_tree_backend.h>
#include <cyten/symmetries/spaces.h>
#include <cyten/tensors/charged_tensor.h>
#include <cyten/tensors/symmetric_tensor.h>
#include <cyten/tools.h>

#include <algorithm>
#include <format>
#include <numeric>
#include <ranges>
#include <set>
#include <stdexcept>
#include <utility>

namespace cyten {

namespace {

constexpr char const* kChargeLeg = ChargedTensor::_CHARGE_LEG_LABEL;

[[nodiscard]] std::vector<LegRef>
as_leg_refs(std::vector<int64> const& idcs)
{
    std::vector<LegRef> out;
    out.reserve(idcs.size());
    for (auto i : idcs) {
        out.emplace_back(i);
    }
    return out;
}

[[nodiscard]] std::vector<LegRef>
as_leg_refs(std::vector<std::string> const& labels)
{
    std::vector<LegRef> out;
    out.reserve(labels.size());
    for (auto const& l : labels) {
        out.emplace_back(l);
    }
    return out;
}

template<typename Range, typename T>
[[nodiscard]] bool
contains(Range const& range, T const& value)
{
    return std::ranges::find(range, value) != std::ranges::end(range);
}

template<typename T>
[[nodiscard]] int64
index_of(std::vector<T> const& v, T const& value)
{
    auto it = std::ranges::find(v, value);
    if (it == v.end()) {
        throw std::invalid_argument("value not in sequence");
    }
    return static_cast<int64>(it - v.begin());
}

[[nodiscard]] TensorPtr
require_tensor(PlanarResult const& res)
{
    if (std::holds_alternative<BlockBackend::Scalar>(res)) {
        throw std::runtime_error("Expected a tensor result, got a scalar");
    }
    return std::get<TensorPtr>(res);
}

[[nodiscard]] std::vector<std::string>
as_strings(LegLabels const& labels)
{
    std::vector<std::string> out;
    out.reserve(labels.size());
    for (auto const& l : labels) {
        out.push_back(l.value_or("None"));
    }
    return out;
}

[[nodiscard]] int64
py_mod(int64 idx, int64 length)
{
    if (length == 0) {
        throw std::invalid_argument("modulo with length 0");
    }
    idx %= length;
    return to_valid_idx(idx, length);
}

void
warn(std::string const& msg, int stacklevel = 1)
{
    auto warnings = py::module_::import("warnings");
    if (stacklevel == 1) {
        warnings.attr("warn")(msg);
    } else {
        warnings.attr("warn")(
          msg, py::module_::import("builtins").attr("UserWarning"), stacklevel);
    }
}

[[nodiscard]] std::string
format_str_list(std::vector<std::string> const& v)
{
    std::string out = "[";
    for (std::size_t i = 0; i < v.size(); ++i) {
        if (i > 0) {
            out += ", ";
        }
        out += '\'';
        out += v[i];
        out += '\'';
    }
    out += ']';
    return out;
}

[[nodiscard]] std::vector<int64>
range_of(int64 start, int64 stop)
{
    std::vector<int64> out;
    if (stop > start) {
        out.reserve(static_cast<std::size_t>(stop - start));
    }
    for (int64 i = start; i < stop; ++i) {
        out.push_back(i);
    }
    return out;
}

[[nodiscard]] std::vector<int64>
range_of(int64 n)
{
    return range_of(0, n);
}

[[nodiscard]] std::vector<int64>
reversed_range(int64 start, int64 stop)
{
    auto v = range_of(start, stop);
    std::ranges::reverse(v);
    return v;
}

template<typename T>
[[nodiscard]] std::vector<T>
reversed_copy(std::vector<T> v)
{
    std::ranges::reverse(v);
    return v;
}

[[nodiscard]] std::vector<std::string>
split_str(std::string const& s, char delim)
{
    std::vector<std::string> parts;
    std::string cur;
    for (char c : s) {
        if (c == delim) {
            parts.push_back(std::move(cur));
            cur.clear();
        } else {
            cur += c;
        }
    }
    parts.push_back(std::move(cur));
    return parts;
}

[[nodiscard]] std::vector<std::string>
split_str(std::string const& s, std::string const& delim)
{
    std::vector<std::string> parts;
    std::size_t start = 0;
    while (true) {
        auto pos = s.find(delim, start);
        if (pos == std::string::npos) {
            parts.push_back(s.substr(start));
            break;
        }
        parts.push_back(s.substr(start, pos - start));
        start = pos + delim.size();
    }
    return parts;
}

[[nodiscard]] std::string
strip_ws(std::string s)
{
    auto a = s.find_first_not_of(" \t\n\r");
    if (a == std::string::npos) {
        return {};
    }
    auto b = s.find_last_not_of(" \t\n\r");
    return s.substr(a, b - a + 1);
}

[[nodiscard]] LegLabels
labels_from_strings(std::vector<std::string> const& labels)
{
    LegLabels out;
    out.reserve(labels.size());
    for (auto const& l : labels) {
        out.emplace_back(l);
    }
    return out;
}

[[nodiscard]] bool
map_keys_equal(auto const& a, auto const& b)
{
    if (a.size() != b.size()) {
        return false;
    }
    for (auto const& [k, _] : a) {
        if (!b.contains(k)) {
            return false;
        }
    }
    return true;
}

[[nodiscard]] std::string
join_lines(std::vector<std::string> const& lines)
{
    std::string out;
    for (std::size_t i = 0; i < lines.size(); ++i) {
        if (i > 0) {
            out += '\n';
        }
        out += lines[i];
    }
    return out;
}

enum class PlanarDecompWhich
{
    qr,
    lq,
    eigh,
    eig,
    eigvals,
    svd,
    truncated_svd
};

struct PlanarDecompResult
{
    TensorPtr A;
    TensorPtr B;
    DiagonalTensorPtr S;
    float64 err = 0.;
    float64 renormalize = 0.;
};

PlanarDecompResult
planar_decomposition(TensorCPtr tensor,
                     int64 codomain_cut,
                     int64 domain_cut,
                     PlanarDecompWhich which,
                     std::optional<LegLabels> new_labels,
                     bool new_leg_dual,
                     std::optional<std::string> sort = std::nullopt,
                     std::optional<std::string> algorithm = std::nullopt,
                     std::optional<float64> normalize_to = std::nullopt,
                     std::optional<int64> chi_max = std::nullopt,
                     int64 chi_min = 1,
                     float64 degeneracy_tol = 0.,
                     float64 trunc_cut = 0.,
                     float64 svd_min = 0.)
{
    // OPTIMIZE for fusion tree backend, can probably work something better out with explicit
    // trees?
    if (!(0 <= codomain_cut && codomain_cut <= tensor->num_codomain_legs())) {
        throw std::invalid_argument("codomain_cut out of range");
    }
    if (!(0 <= domain_cut && domain_cut <= tensor->num_domain_legs())) {
        throw std::invalid_argument("domain_cut out of range");
    }
    if (codomain_cut == 0 && domain_cut == 0) {
        throw std::invalid_argument("Nothing to do");
    }
    if (codomain_cut == tensor->num_codomain_legs() && domain_cut == tensor->num_domain_legs()) {
        throw std::invalid_argument("Nothing to do");
    }

    auto codom = range_of(tensor->num_legs - domain_cut, tensor->num_legs);
    auto codom_rest = range_of(codomain_cut);
    codom.insert(codom.end(), codom_rest.begin(), codom_rest.end());
    auto dom = reversed_range(codomain_cut, tensor->num_legs - domain_cut);
    auto to_decompose = planar_permute_legs(tensor, as_leg_refs(codom), as_leg_refs(dom));

    TensorPtr A;
    TensorPtr B;
    DiagonalTensorPtr S;
    float64 err = 0.;
    float64 renormalize = 0.;

    if (which == PlanarDecompWhich::qr) {
        auto [q, r] = qr(to_decompose, new_labels, new_leg_dual);
        A = std::move(q);
        B = std::move(r);
    } else if (which == PlanarDecompWhich::lq) {
        auto [l, q] = lq(to_decompose, new_labels, new_leg_dual);
        A = std::move(l);
        B = std::move(q);
    } else if (which == PlanarDecompWhich::eigh) {
        // eigh returns W, V, where V is the unitary -> permute legs as, e.g., Q in QR
        auto [W, V] = eigh(to_decompose, new_labels.value_or(LegLabels{}), new_leg_dual, sort);
        B = std::move(W);
        A = std::move(V);
    } else if (which == PlanarDecompWhich::eig) {
        auto [W, V] = eig(to_decompose, new_labels.value_or(LegLabels{}), new_leg_dual, sort);
        B = std::move(W);
        A = std::move(V);
    } else if (which == PlanarDecompWhich::eigvals) {
        auto W = eigvals(to_decompose, new_labels.value_or(LegLabels{}), new_leg_dual, sort);
        B = std::move(W);
    } else if (which == PlanarDecompWhich::svd) {
        auto [u, s, vh] = svd(to_decompose, new_labels, new_leg_dual, true, algorithm);
        A = std::move(u);
        S = std::move(s);
        B = std::move(vh);
    } else if (which == PlanarDecompWhich::truncated_svd) {
        auto [u, s, vh, e, ren] = truncated_svd(to_decompose,
                                                new_labels,
                                                new_leg_dual,
                                                true,
                                                algorithm,
                                                normalize_to,
                                                chi_max,
                                                chi_min,
                                                degeneracy_tol,
                                                trunc_cut,
                                                svd_min);
        A = std::move(u);
        S = std::move(s);
        B = std::move(vh);
        err = e;
        renormalize = ren;
    } else {
        throw std::invalid_argument("Invalid decomposition");
    }

    if (which != PlanarDecompWhich::eigh && which != PlanarDecompWhich::eig &&
        which != PlanarDecompWhich::eigvals) {
        // B contains the eigenvalues for eigh / eig / eigvals
        auto B_codom = range_of(tensor->num_codomain_legs() - codomain_cut + 1);
        auto B_dom = reversed_range(tensor->num_codomain_legs() - codomain_cut + 1, B->num_legs);
        B = planar_permute_legs(B, as_leg_refs(B_codom), as_leg_refs(B_dom));
    }
    if (A) {
        auto A_codom = range_of(domain_cut, A->num_codomain_legs());
        auto A_dom = reversed_range(0, domain_cut);
        A_dom.push_back(A->num_codomain_legs());
        A = planar_permute_legs(A, as_leg_refs(A_codom), as_leg_refs(A_dom));
    }

    return { std::move(A), std::move(B), std::move(S), err, renormalize };
}

} // namespace

BigOPolynomial
product_of(std::vector<BigOPolynomial> const& polys)
{
    if (polys.empty()) {
        return BigOPolynomial();
    }
    return polys.front().prod(std::vector<BigOPolynomial>(polys.begin() + 1, polys.end()));
}

std::string
_as_valid_name(std::string name)
{
    name = strip_ws(std::move(name));
    if (!is_valid_leg_label(LegLabel{ name })) {
        throw std::invalid_argument(std::format("Invalid name or leg label: {}", name));
    }
    return name;
}

bool
_is_charge_leg_label(LegLabel const& label)
{
    if (!label) {
        return false;
    }
    auto const& s = *label;
    return s == kChargeLeg || s.ends_with(std::string(":") + kChargeLeg) ||
           s.starts_with(kChargeLeg);
}

void
_assert_cyclic_labels(std::string const& name,
                      std::vector<std::string> const& expected,
                      std::vector<std::string> const& actual)
{
    if (actual == expected) {
        return;
    }
    if (actual.empty()) {
        throw std::invalid_argument(std::format(
          "Mismatching labels on \"{}\". Expected {} up to cyclical permutation. Got {}",
          name,
          format_str_list(expected),
          format_str_list(actual)));
    }
    auto it = std::ranges::find(expected, actual[0]);
    if (it == expected.end()) {
        throw std::invalid_argument(std::format(
          "Mismatching labels on \"{}\". Expected {} up to cyclical permutation. Got {}",
          name,
          format_str_list(expected),
          format_str_list(actual)));
    }
    auto roll = static_cast<std::size_t>(it - expected.begin());
    std::vector<std::string> expect_rolled;
    expect_rolled.insert(
      expect_rolled.end(), expected.begin() + static_cast<std::ptrdiff_t>(roll), expected.end());
    expect_rolled.insert(
      expect_rolled.end(), expected.begin(), expected.begin() + static_cast<std::ptrdiff_t>(roll));
    if (actual != expect_rolled) {
        throw std::invalid_argument(
          std::format("Mismatching labels on \"{}\". Expected {}. Got {}",
                      name,
                      format_str_list(expect_rolled),
                      format_str_list(actual)));
    }
}

std::vector<std::pair<std::string, std::vector<std::string>>>
_split_tensor_text(std::string const& text)
{
    // A[a, b, c], B[a, s, x], ...
    std::vector<std::pair<std::string, std::vector<std::string>>> res;
    int64 done = -1; // should only consider the rest of text[done + 1:]
    bool broken = false;
    for (int _i = 0; _i < 10000; ++_i) { // should be broken, number is just to avoid infinite loop
        auto i = text.find('[', static_cast<std::size_t>(done + 1));
        auto j = text.find(']', static_cast<std::size_t>(done + 1));
        if (i == std::string::npos) {
            throw std::invalid_argument("Invalid syntax");
        }
        if (j == std::string::npos) {
            throw std::invalid_argument("Bracket opened but not closed.");
        }
        auto tensor_name = _as_valid_name(strip_ws(text.substr(
          static_cast<std::size_t>(done + 1), i - static_cast<std::size_t>(done + 1))));
        std::vector<std::string> legs;
        for (auto const& part : split_str(text.substr(i + 1, j - (i + 1)), ',')) {
            legs.push_back(_as_valid_name(part));
        }
        res.emplace_back(std::move(tensor_name), std::move(legs));
        auto next_comma = text.find(',', j + 1);
        if (next_comma == std::string::npos) {
            done = static_cast<int64>(j);
            broken = true;
            break;
        }
        done = static_cast<int64>(next_comma);
    }
    (void)broken;
    if (!strip_ws(text.substr(static_cast<std::size_t>(done + 1))).empty()) {
        throw std::invalid_argument("Invalid syntax");
    }
    return res;
}

TensorPlaceholder::TensorPlaceholder(std::vector<std::string> labels,
                                     std::vector<BigOPolynomial> dims,
                                     BigOPolynomial cost_to_make)
  : LabelledLegs(labels_from_strings(labels))
  , cost_to_make(std::move(cost_to_make))
{
    if (dims.empty()) {
        this->dims.assign(labels.size(), BigOPolynomial::from_str("None"));
    } else {
        if (dims.size() != labels.size()) {
            throw std::invalid_argument("dims must have the same length as labels");
        }
        this->dims = std::move(dims);
    }
}

TensorPlaceholder
TensorPlaceholder::copy(bool /*deep*/) const
{
    // note: accessing the self.labels property already makes a copy of the list.
    return TensorPlaceholder(string_labels(), dims, cost_to_make);
}

std::vector<std::string>
TensorPlaceholder::string_labels() const
{
    return as_strings(labels());
}

std::string
TensorPlaceholder::__repr__() const
{
    std::string dim_s;
    for (std::size_t i = 0; i < dims.size(); ++i) {
        if (i > 0) {
            dim_s += ", ";
        }
        dim_s += dims[i].str();
    }
    return std::format(
      "TensorPlaceholder({}, dims=[{}])", format_str_list(string_labels()), dim_s);
}

std::vector<std::string>
_expected_labels(std::vector<std::string> const& ph_labels, TensorCPtr tensor)
{
    if (std::dynamic_pointer_cast<ChargedTensor const>(tensor)) {
        return ph_labels;
    }
    std::vector<std::string> out;
    for (auto const& l : ph_labels) {
        if (l != kChargeLeg) {
            out.push_back(l);
        }
    }
    return out;
}

std::vector<std::string>
_expected_labels(std::vector<std::string> const& ph_labels, TensorPlaceholder const&)
{
    return ph_labels;
}

TensorPlaceholder
_combine_placeholder_charge_legs(TensorPlaceholder const& ph)
{
    auto labs = ph.string_labels();
    std::vector<int64> charge_idcs;
    for (int64 i = 0; i < static_cast<int64>(labs.size()); ++i) {
        if (_is_charge_leg_label(LegLabel{ labs[static_cast<std::size_t>(i)] })) {
            charge_idcs.push_back(i);
        }
    }
    if (charge_idcs.empty()) {
        return ph;
    }
    if (charge_idcs.size() == 1) {
        labs[static_cast<std::size_t>(charge_idcs[0])] = kChargeLeg;
        return TensorPlaceholder(labs, ph.dims, ph.cost_to_make);
    }
    std::vector<int64> other;
    try {
        std::tie(std::ignore, other) = parse_leg_bipartition(charge_idcs, ph.num_legs);
    } catch (std::invalid_argument const&) {
        throw std::invalid_argument("Open charge legs are not contiguous");
    }
    std::vector<std::string> labels;
    std::vector<BigOPolynomial> charge_dims;
    std::vector<BigOPolynomial> dims;
    for (auto i : other) {
        labels.push_back(labs[static_cast<std::size_t>(i)]);
        dims.push_back(ph.dims[static_cast<std::size_t>(i)]);
    }
    labels.emplace_back(kChargeLeg);
    for (auto i : charge_idcs) {
        charge_dims.push_back(ph.dims[static_cast<std::size_t>(i)]);
    }
    auto combined_dim = product_of(charge_dims);
    dims.push_back(std::move(combined_dim));
    return TensorPlaceholder(std::move(labels), std::move(dims), ph.cost_to_make);
}

PlanarResult
_wrap_open_charge_legs(TensorPtr tens,
                       std::map<std::string, BlockBackend::BlockPtr> const& charged_states)
{
    std::vector<int64> charge_idcs;
    auto labs = tens->labels();
    for (int64 i = 0; i < tens->num_legs; ++i) {
        if (_is_charge_leg_label(labs[static_cast<std::size_t>(i)])) {
            charge_idcs.push_back(i);
        }
    }
    if (charge_idcs.empty()) {
        return tens;
    }
    std::vector<int64> sorted_charge;
    std::vector<int64> other;
    try {
        std::tie(sorted_charge, other) = parse_leg_bipartition(charge_idcs, tens->num_legs);
    } catch (std::invalid_argument const&) {
        throw std::invalid_argument("Open charge legs are not contiguous");
    }

    auto domain_idcs = reversed_copy(sorted_charge);
    if (!other.empty()) {
        tens = planar_permute_legs(tens, as_leg_refs(other), as_leg_refs(domain_idcs));
    } else {
        tens = planar_permute_legs(tens, std::nullopt, as_leg_refs(domain_idcs));
    }

    auto n_charge = static_cast<int64>(sorted_charge.size());
    labs = tens->labels();
    std::vector<BlockBackend::BlockPtr> states;
    for (int64 i = tens->num_legs - n_charge; i < tens->num_legs; ++i) {
        auto const& lab = labs[static_cast<std::size_t>(i)];
        if (lab) {
            auto it = charged_states.find(*lab);
            states.push_back(it == charged_states.end() ? nullptr : it->second);
        } else {
            states.push_back(nullptr);
        }
    }
    while (n_charge >= 2) {
        tens = combine_legs(tens, { { LegRef{ int64(-2) }, LegRef{ int64(-1) } } });
        auto s1 = states[states.size() - 2];
        auto s2 = states[states.size() - 1];
        BlockBackend::BlockPtr combined;
        if (!s1 && !s2) {
            combined = nullptr;
        } else if (!s1 || !s2) {
            throw std::invalid_argument("Must specify either both or none of the states");
        } else {
            auto pipe = std::dynamic_pointer_cast<LegPipe>((*tens->domain)[0]);
            if (!pipe) {
                throw std::runtime_error(
                  "Expected a LegPipe on the domain after combining charge legs");
            }
            auto obj = tens->backend->state_tensor_product(s1, s2, pipe);
            if (obj.is_none()) {
                combined = nullptr;
            } else {
                combined = obj.cast<BlockBackend::BlockPtr>();
            }
        }
        states.pop_back();
        states.back() = std::move(combined);
        n_charge -= 1;
    }
    tens->set_label(-1, LegLabel{ std::string(kChargeLeg) });
    auto sym = std::dynamic_pointer_cast<SymmetricTensor>(tens);
    if (!sym) {
        throw std::runtime_error(
          "Expected SymmetricTensor invariant part when wrapping charge legs");
    }
    auto wrapped = ChargedTensor::from_invariant_part(std::move(sym), states[0]);
    if (std::holds_alternative<BlockBackend::Scalar>(wrapped)) {
        return std::get<BlockBackend::Scalar>(std::move(wrapped));
    }
    return std::static_pointer_cast<Tensor>(std::get<ChargedTensor::Ptr>(std::move(wrapped)));
}

TensorPlaceholder
_finalize_charge_legs(TensorPlaceholder const& tens,
                      std::map<std::string, BlockBackend::BlockPtr> const&)
{
    return _combine_placeholder_charge_legs(tens);
}

PlanarResult
_finalize_charge_legs(TensorPtr tens,
                      std::map<std::string, BlockBackend::BlockPtr> const& charged_states)
{
    return _wrap_open_charge_legs(std::move(tens), charged_states);
}

PlanarResult
_finalize_charge_legs(PlanarResult tens,
                      std::map<std::string, BlockBackend::BlockPtr> const& charged_states)
{
    if (std::holds_alternative<BlockBackend::Scalar>(tens)) {
        return tens;
    }
    return _finalize_charge_legs(std::get<TensorPtr>(std::move(tens)), charged_states);
}

ContractionTreeNode::ContractionTreeNode(Ptr parent,
                                         Ptr left_child,
                                         Ptr right_child,
                                         std::optional<std::string> value)
  : parent(std::move(parent))
  , left_child(std::move(left_child))
  , right_child(std::move(right_child))
  , value(std::move(value))
{
    if (!this->left_child && !this->right_child && !this->value) {
        throw std::invalid_argument("Node must be nontrivial, i.e., either have two child nodes "
                                    "or a value different from `None`");
    }
    if (bool(this->left_child) != bool(this->right_child)) {
        throw std::invalid_argument("Must have either none or two child nodes");
    }
}

void
ContractionTreeNode::test_sanity() const
{
    if (left_child == nullptr && right_child == nullptr) {
        if (!value) {
            throw std::invalid_argument("leaf node must have a value");
        }
    } else if (left_child != nullptr && right_child != nullptr) {
        left_child->test_sanity();
        right_child->test_sanity();
    } else {
        throw std::invalid_argument("Must have either none or two child nodes");
    }
}

bool
ContractionTreeNode::is_leaf() const
{
    return left_child == nullptr && right_child == nullptr;
}

ContractionTreeNode::Ptr
ContractionTreeNode::copy(Ptr new_parent) const
{
    Ptr left;
    Ptr right;
    if (left_child) {
        left = left_child->copy(nullptr);
    }
    if (right_child) {
        right = right_child->copy(nullptr);
    }
    auto node = std::make_shared<ContractionTreeNode>(std::move(new_parent), left, right, value);
    if (left) {
        left->parent = node;
    }
    if (right) {
        right->parent = node;
    }
    return node;
}

std::pair<std::vector<std::string>, int64>
ContractionTreeNode::get_leaves() const
{
    if (is_leaf()) {
        return { { value.value_or("None") }, 0 };
    }
    auto [leaves_L, num_L] = left_child->get_leaves();
    auto [leaves_R, num_R] = right_child->get_leaves();
    leaves_L.insert(leaves_L.end(), leaves_R.begin(), leaves_R.end());
    return { std::move(leaves_L), 2 + num_L + num_R };
}

std::pair<std::optional<std::string>, std::optional<std::string>>
ContractionTreeNode::remove_children()
{
    if (is_leaf()) {
        throw std::invalid_argument("Cannot remove children from a leaf");
    }
    auto a = left_child->value;
    auto b = right_child->value;
    left_child->parent.reset();
    right_child->parent.reset();
    left_child.reset();
    right_child.reset();
    return { std::move(a), std::move(b) };
}

std::tuple<std::optional<std::string>, std::string, std::string, std::string>
ContractionTreeNode::pop_contraction()
{
    if (is_leaf()) {
        throw std::invalid_argument("Can not pop a contraction from a single leaf");
    }
    if (!left_child->is_leaf()) {
        return left_child->pop_contraction();
    }
    if (!right_child->is_leaf()) {
        return right_child->pop_contraction();
    }
    // remaining case: both children are leaves
    auto X = value;
    auto [a, b] = remove_children();
    auto a_s = a.value_or("None");
    auto b_s = b.value_or("None");
    auto new_value = std::format("{} @ {}", a_s, b_s);
    value = new_value;
    return { std::move(X), std::move(a_s), std::move(b_s), std::move(new_value) };
}

std::vector<std::string>
ContractionTreeNode::_str_lines(std::string const& prefix_0, std::string const& prefix) const
{
    if (is_leaf()) {
        return { prefix_0 + value.value_or("None") };
    }
    // `(prefix_0 + "┓") if value is None else str(value)` — when value is set, prefix_0 is NOT
    // prepended.
    std::string first = value ? *value : prefix_0 + "┓";
    auto left_lines = left_child->_str_lines(prefix + "┣━", prefix + "┃ ");
    auto right_lines = right_child->_str_lines(prefix + "┗━", prefix + "  ");
    std::vector<std::string> out;
    out.push_back(std::move(first));
    out.insert(out.end(), left_lines.begin(), left_lines.end());
    out.insert(out.end(), right_lines.begin(), right_lines.end());
    return out;
}

std::string
ContractionTreeNode::show_whole_tree() const
{
    auto root = std::const_pointer_cast<ContractionTreeNode>(shared_from_this());
    while (true) {
        auto p = root->parent.lock();
        if (!p) {
            break;
        }
        root = std::move(p);
    }
    return join_lines(root->_str_lines());
}

ContractionTree::ContractionTree(ContractionTreeNode::Ptr root)
  : root(std::move(root))
{
}

void
ContractionTree::test_sanity() const
{
    if (root->parent.lock()) {
        throw std::invalid_argument("The root of a contraction tree cannot have a parent.");
    }
    root->test_sanity();
}

std::vector<std::string>
ContractionTree::leaves() const
{
    auto [lvs, _] = root->get_leaves();
    return lvs;
}

int64
ContractionTree::num_leaves() const
{
    return static_cast<int64>(leaves().size());
}

int64
ContractionTree::num_nodes() const
{
    auto [_, num_nodes_below] = root->get_leaves();
    return 1 + num_nodes_below;
}

int64
ContractionTree::num_inner_nodes() const
{
    auto [lvs, num_nodes_below] = root->get_leaves();
    return 1 + num_nodes_below - static_cast<int64>(lvs.size());
}

ContractionTree
ContractionTree::from_contraction_order(
  std::vector<std::pair<std::string, std::string>> const& order)
{
    if (order.empty()) {
        throw std::invalid_argument("Can not be empty");
    }
    struct Component
    {
        ContractionTree tree;
        std::vector<std::string> lst;
    };
    std::vector<Component> contracted;
    for (auto const& [t1, t2] : order) {
        if (t1 == t2) {
            // partial trace
            continue;
        }
        std::vector<std::size_t> t1_matches;
        std::vector<std::size_t> t2_matches;
        for (std::size_t n = 0; n < contracted.size(); ++n) {
            if (contains(contracted[n].lst, t1)) {
                t1_matches.push_back(n);
            }
            if (contains(contracted[n].lst, t2)) {
                t2_matches.push_back(n);
            }
        }
        if (t1_matches.size() > 1 || t2_matches.size() > 1) {
            throw std::runtime_error("");
        }
        if (t1_matches.empty() && t2_matches.empty()) { // dont have either tensor yet
            auto left = from_single_node(t1);
            auto right = from_single_node(t2);
            contracted.push_back({ left.fuse(right), { t1, t2 } });
        } else if (t1_matches.empty()) { // have t2 but not t1
            auto n2 = t2_matches[0];
            auto left = from_single_node(t1);
            auto fused = left.fuse(contracted[n2].tree);
            std::vector<std::string> lst{ t1 };
            lst.insert(lst.end(), contracted[n2].lst.begin(), contracted[n2].lst.end());
            contracted[n2] = { std::move(fused), std::move(lst) };
        } else if (t2_matches.empty()) { // have t1 but not t2
            auto n1 = t1_matches[0];
            auto right = from_single_node(t2);
            auto fused = contracted[n1].tree.fuse(right);
            auto lst = contracted[n1].lst;
            lst.push_back(t2);
            contracted[n1] = { std::move(fused), std::move(lst) };
        } else if (t1_matches == t2_matches) { // have already "contracted" them
            continue;
        } else { // already have both, but not contracted yet
            auto n1 = t1_matches[0];
            auto n2 = t2_matches[0];
            auto fused = contracted[n1].tree.fuse(contracted[n2].tree);
            auto lst = contracted[n1].lst;
            lst.insert(lst.end(), contracted[n2].lst.begin(), contracted[n2].lst.end());
            contracted[n1] = { std::move(fused), std::move(lst) };
            contracted.erase(contracted.begin() + static_cast<std::ptrdiff_t>(n2));
        }
    }
    if (contracted.size() != 1) {
        throw std::invalid_argument("The planar diagram is disconnected");
    }
    return contracted[0].tree;
}

ContractionTree
ContractionTree::from_single_node(std::string const& node)
{
    auto root = std::make_shared<ContractionTreeNode>(nullptr, nullptr, nullptr, node);
    return ContractionTree(std::move(root));
}

ContractionTree
ContractionTree::copy() const
{
    return ContractionTree(root->copy());
}

ContractionTree
ContractionTree::fuse(ContractionTree& other, std::optional<std::string> value)
{
    auto a = root;
    auto b = other.root;
    auto new_root = std::make_shared<ContractionTreeNode>(nullptr, a, b, std::move(value));
    a->parent = new_root;
    b->parent = new_root;
    return ContractionTree(new_root);
}

std::tuple<std::optional<std::string>, std::string, std::string, std::string>
ContractionTree::pop_contraction()
{
    auto res = root->pop_contraction();
    root->test_sanity(); // OPTIMIZE rm
    return res;
}

std::string
ContractionTree::str() const
{
    return join_lines(root->_str_lines());
}

PlanarDiagram::PlanarDiagram(TensorPlaceholderMap tensors,
                             std::vector<std::string> tensor_names,
                             std::vector<DiagramInstruction> definition,
                             ContractionTree order,
                             bool allow_multiple_charged_tensors)
  : tensors(std::move(tensors))
  , tensor_names_(std::move(tensor_names))
  , definition(std::move(definition))
  , order(std::move(order))
  , allow_multiple_charged_tensors(allow_multiple_charged_tensors)
{
    if (this->order.num_leaves() != static_cast<int64>(this->tensors.size())) {
        throw std::invalid_argument("The planar diagram is disconnected");
    }
    auto [ol, cost] = verify_diagram();
    open_legs = std::move(ol);
    contraction_cost = std::move(cost);
}

PlanarDiagram::PlanarDiagram(TensorPlaceholderMap tensors,
                             std::vector<std::string> tensor_names,
                             std::vector<DiagramInstruction> definition,
                             std::string const& order_str,
                             bool allow_multiple_charged_tensors)
  : tensors(std::move(tensors))
  , tensor_names_(std::move(tensor_names))
  , definition(std::move(definition))
  , order(ContractionTree::from_single_node(tensor_names_.empty() ? std::string("_")
                                                                  : tensor_names_.front()))
  , allow_multiple_charged_tensors(allow_multiple_charged_tensors)
{
    order = parse_order(order_str);
    if (this->order.num_leaves() != static_cast<int64>(this->tensors.size())) {
        throw std::invalid_argument("The planar diagram is disconnected");
    }
    auto [ol, cost] = verify_diagram();
    open_legs = std::move(ol);
    contraction_cost = std::move(cost);
}

PlanarDiagram
PlanarDiagram::add_tensor(TensorPlaceholderMap extra_tensors,
                          std::vector<DiagramInstruction> extra_definition,
                          std::string const& order) const
{
    extra_tensors = parse_tensors(std::move(extra_tensors));
    if (extra_tensors.size() != 1) {
        throw std::invalid_argument("Expected a single extra tensor");
    }
    auto new_name = extra_tensors.begin()->first;
    if (tensors.contains(new_name)) {
        throw std::invalid_argument("There already is a tensor with that name");
    }
    auto new_tensors = tensors;
    new_tensors.insert(extra_tensors.begin(), extra_tensors.end());
    auto new_names = tensor_names_;
    new_names.push_back(new_name);

    std::vector<int64>
      outdated; // collect indices of the old_definitions that are outdated in the new
    extra_definition = parse_definition(std::move(extra_definition));
    for (auto const& [t1, l1, t2, l2] : extra_definition) {
        if (!t2) {
            continue; // new open leg: nothing to do
        }
        if (t1 == new_name && *t2 == new_name) {
            continue; // trace on the new tensor: nothing to do
        }
        std::string new_tens_leg;
        std::string other_tens;
        std::string other_tens_leg;
        if (t1 == new_name) {
            new_tens_leg = l1;
            other_tens = *t2;
            other_tens_leg = l2;
        } else if (*t2 == new_name) {
            new_tens_leg = l2;
            other_tens = t1;
            other_tens_leg = l1;
        } else {
            throw std::invalid_argument(
              "Invalid extra_definition. Must reference the new tensor!");
        }
        auto n = _find_open_leg_definition(other_tens, other_tens_leg);
        if (!n) {
            throw std::invalid_argument(
              std::format("Invalid extra_definition. Attempted to contract "
                          "{}:{} @ {}:{}, but the latter "
                          "is not an open leg of the existing diagram",
                          new_name,
                          new_tens_leg,
                          other_tens,
                          other_tens_leg));
        }
        outdated.push_back(*n);
    }
    std::vector<DiagramInstruction> new_definition;
    for (std::size_t n = 0; n < definition.size(); ++n) {
        if (!contains(outdated, static_cast<int64>(n))) {
            new_definition.push_back(definition[n]);
        }
    }
    new_definition.insert(new_definition.end(), extra_definition.begin(), extra_definition.end());
    return PlanarDiagram(std::move(new_tensors),
                         std::move(new_names),
                         std::move(new_definition),
                         order,
                         allow_multiple_charged_tensors);
}

PlanarResult
PlanarDiagram::evaluate(std::map<std::string, TensorPtr> tensors) const
{
    if (!map_keys_equal(tensors, this->tensors)) {
        throw std::invalid_argument("Invalid tensor names (keys)");
    }
    std::string charge = kChargeLeg;
    std::map<std::string, BlockBackend::BlockPtr> charged_states;
    std::map<std::string, TensorPtr> prepared;
    for (auto const& [name, t] : tensors) {
        auto const& ph = this->tensors.at(name);
        auto charged = std::dynamic_pointer_cast<ChargedTensor>(t);
        std::vector<std::string> actual;
        if (charged) {
            actual = as_strings(charged->invariant_part->labels());
        } else {
            actual = as_strings(t->labels());
        }
        auto expected = _expected_labels(ph.string_labels(), t);
        _assert_cyclic_labels(name, expected, actual);

        if (charged) {
            prepared[name] = charged->invariant_part->copy();
            charged_states[std::format("{}:{}", name, charge)] = charged->charged_state;
        } else {
            prepared[name] = t->copy();
        }
    }

    // relabel such that labels are globally unique
    // (prepend the name of the tensor it was originally on)
    std::map<std::string, PlanarResult> working;
    for (auto const& [name, t] : prepared) {
        std::map<std::string, std::string> mapping;
        for (auto const& l : t->labels()) {
            if (l) {
                mapping[*l] = std::format("{}:{}", name, *l);
            }
        }
        t->relabel(mapping);
        working.emplace(name, t);
    }
    std::vector<std::tuple<std::string, std::string, std::string>> traces;
    std::vector<std::tuple<std::string, std::string, std::string, std::string>> contractions;
    std::vector<std::pair<std::string, std::string>> open_legs;
    for (auto const& [t1, l1, t2, l2] : definition) {
        auto rel_l1 = std::format("{}:{}", t1, l1);
        auto t1_tens = require_tensor(working.at(t1));
        bool t1_has = t1_tens->has_label(rel_l1);
        if (!t2) {
            if (!t1_has) {
                if (l1 == charge) {
                    continue;
                }
                throw std::invalid_argument(std::format("Missing open leg {}:{}", t1, l1));
            }
            open_legs.emplace_back(rel_l1, l2);
        } else if (t1 == *t2) {
            auto rel_l2 = std::format("{}:{}", t1, l2);
            bool t2_has = t1_tens->has_label(rel_l2);
            if (!t1_has || !t2_has) {
                if (l1 == charge || l2 == charge) {
                    continue;
                }
                throw std::invalid_argument(
                  std::format("Missing trace legs {}:{}, {}:{}", t1, l1, t1, l2));
            }
            traces.emplace_back(t1, rel_l1, rel_l2);
        } else {
            auto rel_l2 = std::format("{}:{}", *t2, l2);
            auto t2_tens = require_tensor(working.at(*t2));
            bool t2_has = t2_tens->has_label(rel_l2);
            if (!t1_has || !t2_has) {
                if (l1 == charge || l2 == charge) {
                    continue;
                }
                throw std::invalid_argument(
                  std::format("Missing contraction legs {}:{}, {}:{}", t1, l1, *t2, l2));
            }
            contractions.emplace_back(t1, rel_l1, *t2, rel_l2);
        }
    }

    _do_traces(working, traces);
    _do_contractions(working, contractions, order);
    if (working.size() != 1) {
        throw std::invalid_argument("Expected a single contraction result");
    }
    auto res_name = working.begin()->first;
    working[res_name] = _finalize_charge_legs(working[res_name], charged_states);
    return _extract_result(working, open_legs);
}

TensorPlaceholder
PlanarDiagram::evaluate(std::map<std::string, TensorPlaceholder> tensors) const
{
    if (!map_keys_equal(tensors, this->tensors)) {
        throw std::invalid_argument("Invalid tensor names (keys)");
    }
    std::string charge = kChargeLeg;
    std::map<std::string, BlockBackend::BlockPtr> charged_states;
    std::map<std::string, TensorPlaceholder> prepared;
    for (auto const& [name, t] : tensors) {
        auto const& ph = this->tensors.at(name);
        auto actual = t.string_labels();
        auto expected = _expected_labels(ph.string_labels(), t);
        _assert_cyclic_labels(name, expected, actual);
        prepared.emplace(name, t.copy());
    }

    // relabel such that labels are globally unique
    // (prepend the name of the tensor it was originally on)
    for (auto& [name, t] : prepared) {
        std::map<std::string, std::string> mapping;
        for (auto const& l : t.string_labels()) {
            mapping[l] = std::format("{}:{}", name, l);
        }
        t.relabel(mapping);
    }
    std::vector<std::tuple<std::string, std::string, std::string>> traces;
    std::vector<std::tuple<std::string, std::string, std::string, std::string>> contractions;
    std::vector<std::pair<std::string, std::string>> open_legs;
    for (auto const& [t1, l1, t2, l2] : definition) {
        auto rel_l1 = std::format("{}:{}", t1, l1);
        bool t1_has = prepared.at(t1).has_label(rel_l1);
        if (!t2) {
            if (!t1_has) {
                if (l1 == charge) {
                    continue;
                }
                throw std::invalid_argument(std::format("Missing open leg {}:{}", t1, l1));
            }
            open_legs.emplace_back(rel_l1, l2);
        } else if (t1 == *t2) {
            auto rel_l2 = std::format("{}:{}", t1, l2);
            bool t2_has = prepared.at(t1).has_label(rel_l2);
            if (!t1_has || !t2_has) {
                if (l1 == charge || l2 == charge) {
                    continue;
                }
                throw std::invalid_argument(
                  std::format("Missing trace legs {}:{}, {}:{}", t1, l1, t1, l2));
            }
            traces.emplace_back(t1, rel_l1, rel_l2);
        } else {
            auto rel_l2 = std::format("{}:{}", *t2, l2);
            bool t2_has = prepared.at(*t2).has_label(rel_l2);
            if (!t1_has || !t2_has) {
                if (l1 == charge || l2 == charge) {
                    continue;
                }
                throw std::invalid_argument(
                  std::format("Missing contraction legs {}:{}, {}:{}", t1, l1, *t2, l2));
            }
            contractions.emplace_back(t1, rel_l1, *t2, rel_l2);
        }
    }

    _do_traces(prepared, traces);
    _do_contractions(prepared, contractions, order);
    if (prepared.size() != 1) {
        throw std::invalid_argument("Expected a single contraction result");
    }
    auto res_name = prepared.begin()->first;
    prepared.at(res_name) = _finalize_charge_legs(prepared.at(res_name), charged_states);
    return _extract_result(prepared, open_legs);
}

ContractionTree
PlanarDiagram::optimize_order(std::string const& strategy) const
{
    if (strategy == "greedy") {
        // falling back on order "by definition" as a very greedy optimization as a temp solution
        return parse_order("definition");
    }
    throw NotImplemented("Optimization of contraction order is not supported yet");
}

std::vector<DiagramInstruction>
PlanarDiagram::parse_definition(std::string const& definition)
{
    std::vector<DiagramInstruction> res;
    for (auto i : split_str(definition, ',')) {
        i = strip_ws(std::move(i));
        if (i.find(CONTRACT_SYMBOL) != std::string::npos) {
            res.push_back(_parse_contract_instruction(i));
        } else if (std::string(definition).find(OPEN_LEG_SYMBOL) != std::string::npos) {
            res.push_back(_parse_open_leg_instruction(i));
        } else {
            throw std::invalid_argument(std::format("Invalid syntax: \"{}\"", i));
        }
    }
    return res;
}

std::vector<DiagramInstruction>
PlanarDiagram::parse_definition(std::vector<DiagramInstruction> definition)
{
    for (auto const& x : definition) {
        auto const& [t1, l1, t2, l2] = x;
        if (t1 != _as_valid_name(t1)) {
            throw std::invalid_argument(std::format("Invalid tensor name: {}", t1));
        }
        if (l1 != _as_valid_name(l1)) {
            throw std::invalid_argument(std::format("Invalid leg label: {}", l1));
        }
        if (t2 && *t2 != _as_valid_name(*t2)) {
            throw std::invalid_argument(std::format("Invalid tensor name: {}", *t2));
        }
        if (l2 != _as_valid_name(l2)) {
            throw std::invalid_argument(std::format("Invalid leg label: {}", l2));
        }
    }
    return definition;
}

ContractionTree
PlanarDiagram::parse_order(std::string const& order) const
{
    if (tensors.size() == 1) {
        auto name = tensor_names_.empty() ? tensors.begin()->first : tensor_names_.front();
        return ContractionTree::from_single_node(name);
    }
    if (order == "definition") {
        std::vector<std::pair<std::string, std::string>> pairs;
        for (auto const& [t1, l1, t2, l2] : definition) {
            (void)l1;
            (void)l2;
            if (t2) {
                pairs.emplace_back(t1, *t2);
            }
        }
        return ContractionTree::from_contraction_order(pairs);
    }
    if (order == "greedy" || order == "optimal") {
        return optimize_order(order);
    }
    std::vector<std::pair<std::string, std::string>> contraction_order;
    for (auto const& i : split_str(order, ',')) {
        auto parts = split_str(i, CONTRACT_SYMBOL);
        if (parts.size() != 2) {
            throw std::invalid_argument(std::format("Invalid syntax for order: {}", i));
        }
        contraction_order.emplace_back(_as_valid_name(parts[0]), _as_valid_name(parts[1]));
    }
    return ContractionTree::from_contraction_order(contraction_order);
}

ContractionTree
PlanarDiagram::parse_order(ContractionTree const& order) const
{
    if (tensors.size() == 1) {
        auto name = tensor_names_.empty() ? tensors.begin()->first : tensor_names_.front();
        return ContractionTree::from_single_node(name);
    }
    if (order.num_leaves() != static_cast<int64>(tensors.size())) {
        throw std::invalid_argument("order.num_leaves must equal the number of tensors");
    }
    return order;
}

TensorPlaceholderMap
PlanarDiagram::parse_tensors(
  std::string const& tensors,
  std::optional<std::map<std::string, std::vector<std::string>>> const& dims,
  std::vector<std::string>* name_order)
{
    auto parsed = _split_tensor_text(tensors);
    if (name_order) {
        name_order->clear();
        for (auto const& [name, _] : parsed) {
            name_order->push_back(name);
        }
    }

    std::map<std::string, std::string> leg_label_to_dim;
    if (dims) {
        for (auto const& [dim, labels] : *dims) {
            for (auto const& l : labels) {
                leg_label_to_dim[l] = dim;
            }
        }
        std::vector<std::string> all_leg_labels;
        for (auto const& [_, legs] : parsed) {
            all_leg_labels.insert(all_leg_labels.end(), legs.begin(), legs.end());
        }
        std::vector<std::string> defined;
        for (auto const& [l, _] : leg_label_to_dim) {
            defined.push_back(l);
        }
        std::vector<std::string> undefined;
        for (auto const& l : all_leg_labels) {
            if (!leg_label_to_dim.contains(l) && l != kChargeLeg) {
                undefined.push_back(l);
            }
        }
        std::vector<std::string> unused;
        for (auto const& l : defined) {
            if (!contains(all_leg_labels, l)) {
                unused.push_back(l);
            }
        }
        if (!undefined.empty()) {
            std::string joined;
            for (std::size_t i = 0; i < undefined.size(); ++i) {
                if (i > 0) {
                    joined += ", ";
                }
                joined += undefined[i];
            }
            throw std::invalid_argument(
              std::format("If dims are specified, all must be specified. Missing: {}", joined));
        }
        bool any_missing = false;
        for (auto const& l : all_leg_labels) {
            if (l != kChargeLeg && !leg_label_to_dim.contains(l)) {
                any_missing = true;
                break;
            }
        }
        if (any_missing) {
            std::string joined;
            for (std::size_t i = 0; i < unused.size(); ++i) {
                if (i > 0) {
                    joined += ", ";
                }
                joined += unused[i];
            }
            warn(std::format("The following leg labels were given in dims, but do not exist: {}",
                             joined),
                 3);
        }
    }

    TensorPlaceholderMap res;
    for (auto const& [name, legs] : parsed) {
        std::vector<BigOPolynomial> tdims;
        tdims.reserve(legs.size());
        for (auto const& l : legs) {
            auto it = leg_label_to_dim.find(l);
            std::string dim_str;
            if (it != leg_label_to_dim.end()) {
                dim_str = it->second;
            } else {
                dim_str = (l == kChargeLeg) ? "1" : "?";
            }
            tdims.push_back(BigOPolynomial::from_str(dim_str));
        }
        res.emplace(name, TensorPlaceholder(legs, std::move(tdims), BigOPolynomial()));
    }
    return res;
}

TensorPlaceholderMap
PlanarDiagram::parse_tensors(
  TensorPlaceholderMap tensors,
  std::optional<std::map<std::string, std::vector<std::string>>> const& dims,
  std::vector<std::string>* name_order)
{
    if (dims) {
        warn("dims are ignored if tensors is given as a dict");
    }
    if (name_order) {
        name_order->clear();
        for (auto const& [name, _] : tensors) {
            name_order->push_back(name);
        }
    }
    return tensors;
}

PlanarDiagram
PlanarDiagram::remove_tensor(std::string const& name,
                             std::vector<DiagramInstruction> extra_definition,
                             std::string const& order) const
{
    if (!tensors.contains(name)) {
        throw std::invalid_argument(std::format("Tensor does not exist: {}", name));
    }
    TensorPlaceholderMap new_tensors;
    std::vector<std::string> new_names;
    for (auto const& n : tensor_names_) {
        if (n != name) {
            new_tensors.emplace(n, tensors.at(n));
            new_names.push_back(n);
        }
    }
    std::vector<DiagramInstruction> new_definition;
    std::vector<std::pair<std::string, std::string>> new_open_legs;
    for (auto const& [t1, l1, t2, l2] : definition) {
        if ((t1 == name && t2 && *t2 == name) || (t1 == name && !t2)) {
            // partial trace or open leg of removed tensor
        } else if (t1 == name) {
            new_open_legs.emplace_back(*t2, l2);
        } else if (t2 && *t2 == name) {
            new_open_legs.emplace_back(t1, l1);
        } else {
            new_definition.emplace_back(t1, l1, t2, l2);
        }
    }
    extra_definition = parse_definition(std::move(extra_definition));
    for (auto const& [t1, l1, t2, l2] : extra_definition) {
        if (t2) {
            throw std::invalid_argument("extra_definition may only contain open legs");
        }
        auto it = std::ranges::find(new_open_legs, std::pair<std::string, std::string>{ t1, l1 });
        if (it != new_open_legs.end()) {
            new_open_legs.erase(it);
            new_definition.emplace_back(t1, l1, t2, l2);
        } else {
            throw std::invalid_argument("extra_definition may only refer to legs previously "
                                        "contracted with the removed tensor.");
        }
    }
    for (auto const& [t1, l1] : new_open_legs) {
        // unspecified open legs, just keep their label
        new_definition.emplace_back(t1, l1, std::nullopt, l1);
    }
    return PlanarDiagram(std::move(new_tensors),
                         std::move(new_names),
                         std::move(new_definition),
                         order,
                         allow_multiple_charged_tensors);
}

std::pair<std::vector<std::string>, BigOPolynomial>
PlanarDiagram::verify_diagram()
{
    int64 num_legs = 0;
    for (auto const& [t1, l1, t2, l2] : definition) {
        check(tensors.contains(t1), std::format("No tensor with name {}", t1));
        check(tensors.at(t1).has_label(l1), std::format("Tensor {} has no leg {}", t1, l1));
        num_legs += 1;
        if (!t2) {
            check(is_valid_leg_label(LegLabel{ l2 }), std::format("Invalid leg label {}", l2));
        } else {
            check(tensors.contains(*t2), std::format("No tensor with name {}", *t2));
            check(tensors.at(*t2).has_label(l2), std::format("Tensor {} has no leg {}", *t2, l2));
            num_legs += 1;
        }
    }
    int64 total_legs = 0;
    for (auto const& [_, tensor] : tensors) {
        total_legs += tensor.num_legs;
    }
    if (total_legs != num_legs) {
        throw std::invalid_argument(
          "Number of contracted and open legs does not match the total number of legs");
    }

    int64 n_charged = 0;
    for (auto const& [_, ph] : tensors) {
        if (ph.has_label(kChargeLeg)) {
            ++n_charged;
        }
    }
    if (n_charged > 1 && !allow_multiple_charged_tensors) {
        throw std::invalid_argument(
          "Multiple ChargedTensor placeholders require allow_multiple_charged_tensors=True");
    }

    // run the contraction with placeholders.
    // - verifies if the contractions actually are planar
    // - figures out the open_legs
    // - figures out the cost
    auto res = evaluate(tensors);
    std::vector<std::string> ol;
    for (auto const& l : res.labels()) {
        if (!_is_charge_leg_label(l)) {
            ol.push_back(l.value_or("None"));
        }
    }
    return { std::move(ol), res.cost_to_make };
}

std::map<std::string, PlanarResult>&
PlanarDiagram::_do_contractions(
  std::map<std::string, PlanarResult>& tensors,
  std::vector<std::tuple<std::string, std::string, std::string, std::string>> contractions,
  ContractionTree order)
{
    order = order.copy();
    while (tensors.size() > 1) {
        auto [_, t_a, t_b, res_name] = order.pop_contraction();
        std::vector<std::string> legs_a;
        std::vector<std::string> legs_b;
        std::vector<std::size_t> contractions_done;
        for (std::size_t n = 0; n < contractions.size(); ++n) {
            auto const& [t1, l1, t2, l2] = contractions[n];
            if (t1 == t_a && t2 == t_b) {
                legs_a.push_back(l1);
                legs_b.push_back(l2);
                contractions_done.push_back(n);
            } else if (t1 == t_b && t2 == t_a) {
                legs_a.push_back(l2);
                legs_b.push_back(l1);
                contractions_done.push_back(n);
            }
        }

        // put contraction result as t_a, delete t_b
        auto ta = require_tensor(tensors.at(t_a));
        auto tb = require_tensor(tensors.at(t_b));
        tensors[res_name] = planar_contraction(ta, tb, as_leg_refs(legs_a), as_leg_refs(legs_b));
        tensors.erase(t_a);
        tensors.erase(t_b);
        // remove the used contractions
        std::vector<std::tuple<std::string, std::string, std::string, std::string>> remaining;
        for (std::size_t n = 0; n < contractions.size(); ++n) {
            if (!contains(contractions_done, n)) {
                remaining.push_back(contractions[n]);
            }
        }
        contractions = std::move(remaining);
        // contractions involving t_a, t_b now need to reference res_name instead
        for (auto& [t1, l1, t2, l2] : contractions) {
            if (t1 == t_a || t1 == t_b) {
                t1 = res_name;
            }
            if (t2 == t_a || t2 == t_b) {
                t2 = res_name;
            }
        }
    }
    return tensors;
}

std::map<std::string, TensorPlaceholder>&
PlanarDiagram::_do_contractions(
  std::map<std::string, TensorPlaceholder>& tensors,
  std::vector<std::tuple<std::string, std::string, std::string, std::string>> contractions,
  ContractionTree order)
{
    order = order.copy();
    while (tensors.size() > 1) {
        auto [_, t_a, t_b, res_name] = order.pop_contraction();
        std::vector<std::string> legs_a;
        std::vector<std::string> legs_b;
        std::vector<std::size_t> contractions_done;
        for (std::size_t n = 0; n < contractions.size(); ++n) {
            auto const& [t1, l1, t2, l2] = contractions[n];
            if (t1 == t_a && t2 == t_b) {
                legs_a.push_back(l1);
                legs_b.push_back(l2);
                contractions_done.push_back(n);
            } else if (t1 == t_b && t2 == t_a) {
                legs_a.push_back(l2);
                legs_b.push_back(l1);
                contractions_done.push_back(n);
            }
        }

        tensors.insert_or_assign(
          res_name,
          planar_contraction(
            tensors.at(t_a), tensors.at(t_b), as_leg_refs(legs_a), as_leg_refs(legs_b)));
        tensors.erase(t_a);
        tensors.erase(t_b);
        std::vector<std::tuple<std::string, std::string, std::string, std::string>> remaining;
        for (std::size_t n = 0; n < contractions.size(); ++n) {
            if (!contains(contractions_done, n)) {
                remaining.push_back(contractions[n]);
            }
        }
        contractions = std::move(remaining);
        for (auto& [t1, l1, t2, l2] : contractions) {
            if (t1 == t_a || t1 == t_b) {
                t1 = res_name;
            }
            if (t2 == t_a || t2 == t_b) {
                t2 = res_name;
            }
        }
    }
    return tensors;
}

void
PlanarDiagram::_do_traces(
  std::map<std::string, PlanarResult>& tensors,
  std::vector<std::tuple<std::string, std::string, std::string>> const& traces)
{
    std::map<std::string, std::vector<std::vector<LegRef>>> combined_traces;
    for (auto const& [name, l1, l2] : traces) {
        combined_traces[name].push_back({ LegRef{ l1 }, LegRef{ l2 } });
    }
    for (auto const& [name, pairs] : combined_traces) {
        tensors[name] = planar_partial_trace(require_tensor(tensors.at(name)), pairs);
    }
}

void
PlanarDiagram::_do_traces(
  std::map<std::string, TensorPlaceholder>& tensors,
  std::vector<std::tuple<std::string, std::string, std::string>> const& traces)
{
    std::map<std::string, std::vector<std::vector<LegRef>>> combined_traces;
    for (auto const& [name, l1, l2] : traces) {
        combined_traces[name].push_back({ LegRef{ l1 }, LegRef{ l2 } });
    }
    for (auto const& [name, pairs] : combined_traces) {
        tensors.insert_or_assign(name, planar_partial_trace(tensors.at(name), pairs));
    }
}

PlanarResult
PlanarDiagram::_extract_result(std::map<std::string, PlanarResult> const& tensors,
                               std::vector<std::pair<std::string, std::string>> const& open_legs)
{
    if (tensors.size() != 1) {
        throw std::invalid_argument("Expected a single tensor");
    }
    auto tens = tensors.begin()->second;
    std::vector<std::pair<std::string, std::string>> visible_open_legs;
    for (auto const& [old, neu] : open_legs) {
        if (!_is_charge_leg_label(LegLabel{ old })) {
            visible_open_legs.emplace_back(old, neu);
        }
    }
    if (std::holds_alternative<BlockBackend::Scalar>(tens)) {
        // result is a number
        if (!visible_open_legs.empty()) {
            throw std::invalid_argument(
              "Number of expected open legs inconsistent with planar diagram");
        }
        return tens;
    }
    auto t = std::get<TensorPtr>(tens);
    std::vector<std::string> visible_labels;
    for (auto const& l : t->labels()) {
        if (!_is_charge_leg_label(l)) {
            visible_labels.push_back(l.value_or("None"));
        }
    }
    if (visible_open_legs.empty()) {
        // result is a number, or a ChargedTensor / placeholder with only a charge leg
        // TODO this may change, see Issue 13 on Github
        if (!visible_labels.empty()) {
            throw std::invalid_argument(
              "Number of expected open legs inconsistent with planar diagram");
        }
        return t;
    }
    if (visible_open_legs.size() != visible_labels.size()) {
        throw std::invalid_argument(
          "Number of expected open legs inconsistent with planar diagram");
    }
    std::set<std::string> visible_set(visible_labels.begin(), visible_labels.end());
    std::set<std::string> old_set;
    for (auto const& [old, _] : visible_open_legs) {
        old_set.insert(old);
    }
    if (visible_set != old_set) {
        throw std::invalid_argument("Inconsistent open legs");
    }
    std::map<std::string, std::string> mapping;
    for (auto const& [old, neu] : visible_open_legs) {
        mapping[old] = neu;
    }
    t->relabel(mapping);
    return t;
}

TensorPlaceholder
PlanarDiagram::_extract_result(std::map<std::string, TensorPlaceholder> const& tensors,
                               std::vector<std::pair<std::string, std::string>> const& open_legs)
{
    if (tensors.size() != 1) {
        throw std::invalid_argument("Expected a single tensor");
    }
    auto tens = tensors.begin()->second;
    std::vector<std::pair<std::string, std::string>> visible_open_legs;
    for (auto const& [old, neu] : open_legs) {
        if (!_is_charge_leg_label(LegLabel{ old })) {
            visible_open_legs.emplace_back(old, neu);
        }
    }
    std::vector<std::string> visible_labels;
    for (auto const& l : tens.labels()) {
        if (!_is_charge_leg_label(l)) {
            visible_labels.push_back(l.value_or("None"));
        }
    }
    if (visible_open_legs.empty()) {
        if (!visible_labels.empty()) {
            throw std::invalid_argument(
              "Number of expected open legs inconsistent with planar diagram");
        }
        return tens;
    }
    if (visible_open_legs.size() != visible_labels.size()) {
        throw std::invalid_argument(
          "Number of expected open legs inconsistent with planar diagram");
    }
    std::set<std::string> visible_set(visible_labels.begin(), visible_labels.end());
    std::set<std::string> old_set;
    for (auto const& [old, _] : visible_open_legs) {
        old_set.insert(old);
    }
    if (visible_set != old_set) {
        throw std::invalid_argument("Inconsistent open legs");
    }
    std::map<std::string, std::string> mapping;
    for (auto const& [old, neu] : visible_open_legs) {
        mapping[old] = neu;
    }
    tens.relabel(mapping);
    return tens;
}

DiagramInstruction
PlanarDiagram::_parse_contract_instruction(std::string const& i)
{
    auto parts = split_str(i, CONTRACT_SYMBOL);
    if (parts.size() < 2) {
        throw std::invalid_argument(i);
    }
    auto const& left = parts[0];
    auto const& right = parts[1];
    auto more = parts.size() > 2;
    auto left_parts = split_str(left, LEG_SELECT_SYMBOL);
    auto right_parts = split_str(right, LEG_SELECT_SYMBOL);
    if (more || left_parts.size() != 2 || right_parts.size() != 2) {
        throw std::invalid_argument(i);
    }
    auto t1 = _as_valid_name(left_parts[0]);
    auto l1 = _as_valid_name(left_parts[1]);
    auto t2 = _as_valid_name(right_parts[0]);
    auto l2 = _as_valid_name(right_parts[1]);
    return { t1, l1, t2, l2 };
}

DiagramInstruction
PlanarDiagram::_parse_open_leg_instruction(std::string const& i)
{
    auto parts = split_str(i, std::string(OPEN_LEG_SYMBOL));
    if (parts.size() < 2) {
        throw std::invalid_argument(i);
    }
    auto const& left = parts[0];
    auto const& right = parts[1];
    auto more = parts.size() > 2;
    auto left_parts = split_str(left, LEG_SELECT_SYMBOL);
    if (more || left_parts.size() != 2) {
        throw std::invalid_argument(i);
    }
    auto t1 = _as_valid_name(left_parts[0]);
    auto l1 = _as_valid_name(left_parts[1]);
    auto l2 = _as_valid_name(right);
    return { t1, l1, std::nullopt, l2 };
}

std::optional<int64>
PlanarDiagram::_find_open_leg_definition(std::string const& name, std::string const& leg) const
{
    for (std::size_t n = 0; n < definition.size(); ++n) {
        auto const& [t1, l1, t2, _] = definition[n];
        if (!t2 && t1 == name && l1 == leg) {
            return static_cast<int64>(n);
        }
    }
    return std::nullopt;
}

PlanarLinearOperator::PlanarLinearOperator(PlanarDiagram const& op_diagram,
                                           PlanarDiagram const& matvec_diagram,
                                           std::map<std::string, TensorPtr> op_tensors,
                                           std::string vec_name)
  : LinearOperator({}, Dtype::Float64, std::nullopt)
  , op_diagram(op_diagram)
  , matvec_diagram(matvec_diagram)
  , op_tensors(std::move(op_tensors))
  , vec_name(std::move(vec_name))
{
    if (this->op_tensors.empty()) {
        throw std::invalid_argument("PlanarLinearOperator requires at least one operator tensor");
    }
    dtype = this->op_tensors.begin()->second->dtype;
    std::set<std::string> matvec_names(this->matvec_diagram.tensor_names().begin(),
                                       this->matvec_diagram.tensor_names().end());
    std::set<std::string> expected(this->op_diagram.tensor_names().begin(),
                                   this->op_diagram.tensor_names().end());
    expected.insert(this->vec_name);
    if (matvec_names != expected) {
        throw std::invalid_argument(std::format(
          "Inconsistent tensor names. The matvec_diagram must have the tensor names from "
          "the op_diagram, in addition to the single name {} of the vector.",
          this->vec_name));
    }
}

VectorLike::Ptr
PlanarLinearOperator::matvec(VectorLike::CPtr vec)
{
    auto tvec = std::dynamic_pointer_cast<Tensor const>(vec);
    if (!tvec) {
        throw std::invalid_argument("PlanarLinearOperator.matvec expects a Tensor input");
    }
    auto tensors = op_tensors;
    tensors[vec_name] = std::const_pointer_cast<Tensor>(tvec);
    auto res = matvec_diagram.evaluate(std::move(tensors));
    if (std::holds_alternative<BlockBackend::Scalar>(res)) {
        throw std::runtime_error("PlanarLinearOperator.matvec unexpectedly returned a scalar");
    }
    return std::get<TensorPtr>(std::move(res));
}

TensorPtr
PlanarLinearOperator::to_tensor(TensorBackend::Ptr)
{
    auto res = op_diagram.evaluate(op_tensors);
    if (std::holds_alternative<BlockBackend::Scalar>(res)) {
        throw std::runtime_error("PlanarLinearOperator.to_tensor unexpectedly returned a scalar");
    }
    return std::get<TensorPtr>(std::move(res));
}

std::tuple<TensorPtr, TensorPtr>
horizontal_factorization(TensorCPtr tensor,
                         int64 codomain_cut,
                         int64 domain_cut,
                         std::optional<LegLabels> new_labels,
                         std::optional<float64> cutoff_singular_values)
{
    if (!cutoff_singular_values) {
        return planar_qr(tensor, codomain_cut, domain_cut, new_labels);
    }
    auto [A, S, Vh, e, r] = planar_truncated_svd(tensor,
                                                 codomain_cut,
                                                 domain_cut,
                                                 new_labels,
                                                 false,
                                                 std::nullopt,
                                                 std::nullopt,
                                                 std::nullopt,
                                                 1,
                                                 0.,
                                                 0.,
                                                 *cutoff_singular_values);
    (void)e;
    (void)r;
    return { std::move(A), scale_axis(Vh, S, LegRef{ int64(0) }) };
}

bool
planar_almost_equal(TensorCPtr tensor_1, TensorCPtr tensor_2, float64 rtol, float64 atol)
{
    auto labs1 = tensor_1->labels();
    auto labs2 = tensor_2->labels();
    if (contains(labs1, LegLabel{}) || contains(labs2, LegLabel{})) {
        throw std::invalid_argument("Can only compare tensors for which each leg has a label");
    }
    std::set<std::string> s1;
    std::set<std::string> s2;
    for (auto const& l : labs1) {
        s1.insert(*l);
    }
    for (auto const& l : labs2) {
        s2.insert(*l);
    }
    if (s1 != s2) {
        throw std::invalid_argument("Both tensors need to have the same leg labels");
    }
    std::vector<std::string> codomain;
    std::vector<std::string> domain;
    for (int64 i = 0; i < tensor_2->num_codomain_legs(); ++i) {
        codomain.push_back(*labs2[static_cast<std::size_t>(i)]);
    }
    for (int64 i = tensor_2->num_legs - 1; i >= tensor_2->num_codomain_legs(); --i) {
        domain.push_back(*labs2[static_cast<std::size_t>(i)]);
    }
    auto t1 = planar_permute_legs(tensor_1, as_leg_refs(codomain), as_leg_refs(domain));
    return almost_equal(t1, tensor_2, rtol, atol, true);
}

TensorPtr
planar_combine_legs(TensorCPtr T,
                    std::vector<std::vector<LegRef>> which_legs,
                    std::optional<PipeDualities> pipe_dualities,
                    std::optional<std::vector<Leg::Ptr>> pipes)
{
    std::vector<std::vector<int64>> which;
    which.reserve(which_legs.size());
    for (auto const& group : which_legs) {
        which.push_back(T->get_leg_idcs(group));
    }
    // identify if is there is a group on the left / right with legs that need
    // to be bent and where to combine them
    std::optional<std::size_t> right_group_idx;
    std::optional<std::size_t> left_group_idx;
    bool right_group_in_domain = false;
    bool left_group_in_domain = false;
    for (std::size_t idx = 0; idx < which.size(); ++idx) {
        auto const& group = which[idx];
        if (contains(group, T->num_codomain_legs() - 1) &&
            contains(group, T->num_codomain_legs())) {
            right_group_idx = idx;
            right_group_in_domain = group[0] >= T->num_codomain_legs();
        } else if (contains(group, int64(0)) && contains(group, T->num_legs - 1)) {
            left_group_idx = idx;
            left_group_in_domain = group[0] >= T->num_codomain_legs();
        }
    }

    for (auto& group : which) {
        group = parse_leg_bipartition(group, T->num_legs).first;
    }

    // get new codomain and domain for planar_permute_legs, update which_legs for left bends
    auto new_codomain = range_of(T->num_codomain_legs());
    auto new_domain = reversed_range(T->num_codomain_legs(), T->num_legs);
    if (right_group_idx) {
        auto const& right_group = which[*right_group_idx];
        // number group legs in codomain
        auto num = index_of(right_group, T->num_codomain_legs() - 1) + 1;
        if (right_group_in_domain) {
            auto tail = std::vector<int64>(new_codomain.end() - num, new_codomain.end());
            std::ranges::reverse(tail);
            new_domain.insert(new_domain.end(), tail.begin(), tail.end());
            new_codomain.erase(new_codomain.end() - num, new_codomain.end());
        } else {
            num = static_cast<int64>(right_group.size()) - num;
            auto tail = std::vector<int64>(new_domain.end() - num, new_domain.end());
            std::ranges::reverse(tail);
            new_codomain.insert(new_codomain.end(), tail.begin(), tail.end());
            new_domain.erase(new_domain.end() - num, new_domain.end());
        }
    }
    if (left_group_idx) {
        auto const& left_group = which[*left_group_idx];
        // number group legs in domain
        auto num = index_of(left_group, T->num_legs - 1) + 1;
        if (left_group_in_domain) {
            num = static_cast<int64>(left_group.size()) - num;
            auto head = std::vector<int64>(new_codomain.begin(), new_codomain.begin() + num);
            std::ranges::reverse(head);
            new_codomain.erase(new_codomain.begin(), new_codomain.begin() + num);
            new_domain.insert(new_domain.begin(), head.begin(), head.end());
            for (auto& group : which) {
                for (auto& leg : group) {
                    leg = py_mod(leg - num, T->num_legs);
                }
            }
        } else {
            auto head = std::vector<int64>(new_domain.begin(), new_domain.begin() + num);
            std::ranges::reverse(head);
            new_domain.erase(new_domain.begin(), new_domain.begin() + num);
            new_codomain.insert(new_codomain.begin(), head.begin(), head.end());
            for (auto& group : which) {
                for (auto& leg : group) {
                    leg = py_mod(leg + num, T->num_legs);
                }
            }
        }
    }

    auto T2 = planar_permute_legs(T, as_leg_refs(new_codomain), as_leg_refs(new_domain));
    std::vector<std::vector<LegRef>> which_refs;
    which_refs.reserve(which.size());
    for (auto const& group : which) {
        which_refs.push_back(as_leg_refs(group));
    }
    if (!pipe_dualities) {
        pipe_dualities = PipeDualities{ false };
    }
    return combine_legs(T2, which_refs, pipe_dualities, pipes);
}

PlanarResult
planar_contraction(TensorCPtr tensor1,
                   TensorCPtr tensor2,
                   std::vector<LegRef> legs1,
                   std::vector<LegRef> legs2,
                   std::map<std::string, std::string> relabel1,
                   std::map<std::string, std::string> relabel2)
{
    auto legs1_idcs = tensor1->get_leg_idcs(legs1);
    auto legs2_idcs = tensor2->get_leg_idcs(legs2);
    auto num_contr = static_cast<int64>(legs1_idcs.size());
    if (static_cast<int64>(legs2_idcs.size()) != num_contr) {
        throw std::invalid_argument("legs1 and legs2 must have the same length");
    }

    // check if the contraction actually is planar
    // 1) check if the legs on each tensor are divided into two contiguous subsets
    auto [contr1, open1] = parse_leg_bipartition(legs1_idcs, tensor1->num_legs);
    auto [_, open2] = parse_leg_bipartition(legs2_idcs, tensor2->num_legs);
    (void)_;
    // 2) check that the contracted legs connect without braids:
    //    as contr1 goes around tensor1 counter-clockwise, their connection targets must go around
    //    tensor2 clockwise
    std::vector<int64> contr2;
    contr2.reserve(contr1.size());
    for (auto c1 : contr1) {
        contr2.push_back(legs2_idcs[static_cast<std::size_t>(index_of(legs1_idcs, c1))]);
    }
    for (std::size_t n = 0; n + 1 < contr2.size(); ++n) {
        auto n1 = contr2[n];
        auto n2 = contr2[n + 1];
        if (n2 != py_mod(n1 - 1, tensor2->num_legs)) {
            throw std::invalid_argument("Not a planar contraction");
        }
    }

    // find out how we can have the least number of bends before compose / partial_compose
    // Step 1: determine if it is cheaper to contract codomain of tensor1 with domain of
    //         tensor2 or vice versa
    // Step 2: determine if we can use partial_compose (ignores legs that are not
    //         contracted; can only ignore uncontracted legs for one of the tensors)
    int64 tensor1_bend_up = 0;
    for (auto l : contr1) {
        if (l < tensor1->num_codomain_legs()) {
            ++tensor1_bend_up;
        }
    }
    auto tensor1_bend_down = num_contr - tensor1_bend_up;
    int64 tensor2_bend_up = 0;
    for (auto l : contr2) {
        if (l < tensor2->num_codomain_legs()) {
            ++tensor2_bend_up;
        }
    }
    auto tensor2_bend_down = num_contr - tensor2_bend_up;
    if (tensor1_bend_up + tensor2_bend_down < tensor1_bend_down + tensor2_bend_up) {
        // contracted legs up for tensor1, down for tensor2

        // partial_compose requires all legs to be contracted of one tensor to
        // be in domain or codomain -> find out which tensor needs less bends
        auto tensor1_bend_away = tensor1->num_domain_legs() + tensor1_bend_up - num_contr;
        auto tensor2_bend_away = tensor2->num_codomain_legs() + tensor2_bend_down - num_contr;
        if (tensor2_bend_away < tensor1_bend_away) {
            auto [t1, partial_compose_leg] = _planar_contraction_helper(tensor1, contr1, true);
            auto t2 = planar_permute_legs(
              tensor2, as_leg_refs(reversed_copy(contr2)), as_leg_refs(reversed_copy(open2)));
            if (t1->num_domain_legs() > num_contr) {
                return partial_compose(t1, t2, LegRef{ *partial_compose_leg }, relabel1, relabel2);
            }
            return compose(t1, t2, relabel1, relabel2);
        } else {
            auto [t2, partial_compose_leg] = _planar_contraction_helper(tensor2, contr2, false);
            auto t1 =
              planar_permute_legs(tensor1, as_leg_refs(open1), as_leg_refs(reversed_copy(contr1)));
            if (t2->num_codomain_legs() > num_contr) {
                return partial_compose(t2, t1, LegRef{ *partial_compose_leg }, relabel2, relabel1);
            }
            return compose(t1, t2, relabel1, relabel2);
        }
    } else {
        // contracted legs down for tensor1, up for tensor2
        auto tensor1_bend_away = tensor1->num_codomain_legs() + tensor1_bend_down - num_contr;
        auto tensor2_bend_away = tensor2->num_domain_legs() + tensor2_bend_up - num_contr;
        if (tensor2_bend_away < tensor1_bend_away) {
            auto [t1, partial_compose_leg] = _planar_contraction_helper(tensor1, contr1, false);
            auto t2 = planar_permute_legs(tensor2, as_leg_refs(open2), as_leg_refs(contr2));
            if (t1->num_codomain_legs() > num_contr) {
                return partial_compose(t1, t2, LegRef{ *partial_compose_leg }, relabel1, relabel2);
            }
            return compose(t2, t1, relabel2, relabel1);
        } else {
            auto [t2, partial_compose_leg] = _planar_contraction_helper(tensor2, contr2, true);
            auto t1 =
              planar_permute_legs(tensor1, as_leg_refs(contr1), as_leg_refs(reversed_copy(open1)));
            if (t2->num_domain_legs() > num_contr) {
                return partial_compose(t2, t1, LegRef{ *partial_compose_leg }, relabel2, relabel1);
            }
            return compose(t2, t1, relabel2, relabel1);
        }
    }
}

TensorPlaceholder
planar_contraction(TensorPlaceholder const& tensor1,
                   TensorPlaceholder const& tensor2,
                   std::vector<LegRef> legs1,
                   std::vector<LegRef> legs2)
{
    auto legs1_idcs = tensor1.get_leg_idcs(legs1);
    auto legs2_idcs = tensor2.get_leg_idcs(legs2);
    auto num_contr = static_cast<int64>(legs1_idcs.size());
    if (static_cast<int64>(legs2_idcs.size()) != num_contr) {
        throw std::invalid_argument("legs1 and legs2 must have the same length");
    }

    auto [contr1, open1] = parse_leg_bipartition(legs1_idcs, tensor1.num_legs);
    auto [_, open2] = parse_leg_bipartition(legs2_idcs, tensor2.num_legs);
    (void)_;
    std::vector<int64> contr2;
    for (auto c1 : contr1) {
        contr2.push_back(legs2_idcs[static_cast<std::size_t>(index_of(legs1_idcs, c1))]);
    }
    for (std::size_t n = 0; n + 1 < contr2.size(); ++n) {
        auto n1 = contr2[n];
        auto n2 = contr2[n + 1];
        if (n2 != py_mod(n1 - 1, tensor2.num_legs)) {
            throw std::invalid_argument("Not a planar contraction");
        }
    }

    auto labs1 = tensor1.string_labels();
    auto labs2 = tensor2.string_labels();
    std::vector<std::string> labels;
    std::vector<BigOPolynomial> dims;
    for (auto n : open1) {
        labels.push_back(labs1[static_cast<std::size_t>(n)]);
        dims.push_back(tensor1.dims[static_cast<std::size_t>(n)]);
    }
    for (auto n : open2) {
        labels.push_back(labs2[static_cast<std::size_t>(n)]);
        dims.push_back(tensor2.dims[static_cast<std::size_t>(n)]);
    }
    std::vector<BigOPolynomial> contr_dims1;
    for (auto n : contr1) {
        contr_dims1.push_back(tensor1.dims[static_cast<std::size_t>(n)]);
    }
    std::vector<BigOPolynomial> contr_dims2;
    for (auto n : contr2) {
        contr_dims2.push_back(tensor2.dims[static_cast<std::size_t>(n)]);
    }
    auto contr_dims = product_of(contr_dims1);
    // TODO this may actually happen when forgetting to specify the dims for one tensor...
    if (contr_dims != product_of(contr_dims2)) {
        throw std::invalid_argument("contracted dimensions do not match");
    }
    auto cost_dims = dims;
    cost_dims.push_back(contr_dims);
    auto cost = tensor1.cost_to_make + tensor2.cost_to_make + product_of(cost_dims);
    return TensorPlaceholder(std::move(labels), std::move(dims), std::move(cost));
}

std::tuple<DiagonalTensorPtr, TensorPtr>
planar_eigh(TensorCPtr tensor,
            int64 codomain_cut,
            int64 domain_cut,
            std::optional<LegLabels> new_labels,
            bool new_leg_dual,
            std::optional<std::string> sort)
{
    auto r = planar_decomposition(tensor,
                                  codomain_cut,
                                  domain_cut,
                                  PlanarDecompWhich::eigh,
                                  std::move(new_labels),
                                  new_leg_dual,
                                  sort);
    auto W = std::dynamic_pointer_cast<DiagonalTensor>(r.B);
    if (!W) {
        throw std::runtime_error("planar_eigh expected DiagonalTensor eigenvalues");
    }
    return { std::move(W), std::move(r.A) };
}

std::tuple<DiagonalTensorPtr, TensorPtr>
planar_eig(TensorCPtr tensor,
           int64 codomain_cut,
           int64 domain_cut,
           std::optional<LegLabels> new_labels,
           bool new_leg_dual,
           std::optional<std::string> sort)
{
    auto r = planar_decomposition(tensor,
                                  codomain_cut,
                                  domain_cut,
                                  PlanarDecompWhich::eig,
                                  std::move(new_labels),
                                  new_leg_dual,
                                  sort);
    auto W = std::dynamic_pointer_cast<DiagonalTensor>(r.B);
    if (!W) {
        throw std::runtime_error("planar_eig expected DiagonalTensor eigenvalues");
    }
    return { std::move(W), std::move(r.A) };
}

DiagonalTensorPtr
planar_eigvals(TensorCPtr tensor,
               int64 codomain_cut,
               int64 domain_cut,
               std::optional<LegLabels> new_labels,
               bool new_leg_dual,
               std::optional<std::string> sort)
{
    auto r = planar_decomposition(tensor,
                                  codomain_cut,
                                  domain_cut,
                                  PlanarDecompWhich::eigvals,
                                  std::move(new_labels),
                                  new_leg_dual,
                                  sort);
    auto W = std::dynamic_pointer_cast<DiagonalTensor>(r.B);
    if (!W) {
        throw std::runtime_error("planar_eigvals expected DiagonalTensor eigenvalues");
    }
    return W;
}

std::tuple<TensorPtr, TensorPtr>
planar_lq(TensorCPtr tensor,
          int64 codomain_cut,
          int64 domain_cut,
          std::optional<LegLabels> new_labels,
          bool new_leg_dual)
{
    auto r = planar_decomposition(tensor,
                                  codomain_cut,
                                  domain_cut,
                                  PlanarDecompWhich::lq,
                                  std::move(new_labels),
                                  new_leg_dual);
    return { std::move(r.A), std::move(r.B) };
}

PlanarResult
planar_partial_trace(TensorCPtr tensor, std::vector<std::vector<LegRef>> pairs)
{
    std::vector<std::vector<int64>> pair_idcs;
    pair_idcs.reserve(pairs.size());
    for (auto const& p : pairs) {
        pair_idcs.push_back(tensor->get_leg_idcs(p));
    }
    std::vector<int64> traced_legs;
    for (auto const& p : pair_idcs) {
        traced_legs.insert(traced_legs.end(), p.begin(), p.end());
    }
    for (auto pair : pair_idcs) {
        if (pair.size() != 2) {
            throw std::invalid_argument("each trace pair must have two legs");
        }
        auto l1 = pair[0];
        auto l2 = pair[1];
        // sort s.t. l1 < l2
        if (l1 > l2) {
            std::swap(l1, l2);
        }
        if (l1 == l2) {
            throw std::invalid_argument("trace pair legs must be distinct");
        }
        // living on a circle, there are two different regions "between" l1 and l2.
        // at least one of them may contain only traced legs
        bool first_half_only_traces = true;
        bool second_half_only_traces = true;
        for (int64 l = l1 + 1; l < l2; ++l) { // first half
            if (contains(traced_legs, l)) {
                // must connect to another leg *in the same half*, otherwise there are braids
                std::vector<int64> other_ls;
                for (auto const& pr : pair_idcs) {
                    if (pr[1] == l) {
                        other_ls.push_back(pr[0]);
                    }
                    if (pr[0] == l) {
                        other_ls.push_back(pr[1]);
                    }
                }
                if (other_ls.size() != 1) {
                    throw std::invalid_argument("each traced leg must appear in exactly one pair");
                }
                if (!(l1 < other_ls[0] && other_ls[0] < l2)) {
                    throw std::invalid_argument("Not a planar trace");
                }
            } else {
                first_half_only_traces = false;
            }
        }
        for (auto l : range_of(l2 + 1, tensor->num_legs)) { // second half
            if (contains(traced_legs, l)) {
                std::vector<int64> other_ls;
                for (auto const& pr : pair_idcs) {
                    if (pr[1] == l) {
                        other_ls.push_back(pr[0]);
                    }
                    if (pr[0] == l) {
                        other_ls.push_back(pr[1]);
                    }
                }
                if (other_ls.size() != 1) {
                    throw std::invalid_argument("each traced leg must appear in exactly one pair");
                }
                if (l1 < other_ls[0] && other_ls[0] < l2) {
                    throw std::invalid_argument("Not a planar trace");
                }
            } else {
                second_half_only_traces = false;
            }
        }
        for (int64 l = 0; l < l1; ++l) {
            if (contains(traced_legs, l)) {
                std::vector<int64> other_ls;
                for (auto const& pr : pair_idcs) {
                    if (pr[1] == l) {
                        other_ls.push_back(pr[0]);
                    }
                    if (pr[0] == l) {
                        other_ls.push_back(pr[1]);
                    }
                }
                if (other_ls.size() != 1) {
                    throw std::invalid_argument("each traced leg must appear in exactly one pair");
                }
                if (l1 < other_ls[0] && other_ls[0] < l2) {
                    throw std::invalid_argument("Not a planar trace");
                }
            } else {
                second_half_only_traces = false;
            }
        }
        if (!(first_half_only_traces || second_half_only_traces)) {
            throw std::invalid_argument("Not a planar trace");
        }
    }

    LevelsSpec levels(static_cast<std::size_t>(tensor->num_legs), std::nullopt);

    // OPTIMIZE
    // fusion tree backend requires legs that are traced over to be next to each other
    if (std::dynamic_pointer_cast<FusionTreeBackend>(tensor->backend)) {
        // check how many legs need to be bent up or down and choose the way with the lower number
        int64 num_up_bends = 0;
        int64 num_down_bends = 0;
        for (auto pair : pair_idcs) {
            std::ranges::sort(pair);
            auto legs_between = range_of(pair[0] + 1, pair[1]);
            // all legs between are traced over -> trace over the right side allowed
            bool all_traced = true;
            for (auto leg : legs_between) {
                if (!contains(traced_legs, leg)) {
                    all_traced = false;
                    break;
                }
            }
            if (all_traced) {
                continue;
            }
            // remaining case: must trace over the left side
            num_up_bends = std::max(num_up_bends, pair[0] + 1);
            num_down_bends = std::max(num_down_bends, tensor->num_legs - pair[1]);
        }

        if (num_down_bends == 0) {
            if (num_up_bends != 0) {
                throw std::invalid_argument("inconsistent bend counts");
            }
        } else if (num_up_bends > num_down_bends) {
            // bend legs down
            // number of legs to be bent twice
            int64 num_legs_in_codom = 0;
            if (tensor->num_domain_legs() <= num_down_bends) {
                num_legs_in_codom = num_down_bends - tensor->num_domain_legs();
            }
            auto codomain = range_of(tensor->num_legs - num_down_bends, tensor->num_legs);
            auto extra = range_of(tensor->num_codomain_legs() - num_legs_in_codom);
            codomain.insert(codomain.end(), extra.begin(), extra.end());
            auto domain =
              reversed_range(tensor->num_codomain_legs(), tensor->num_legs - num_down_bends);
            tensor = planar_permute_legs(tensor, as_leg_refs(codomain), as_leg_refs(domain));
            // update pairs
            for (auto& pair : pair_idcs) {
                for (auto& idx : pair) {
                    idx = py_mod(idx + num_down_bends, tensor->num_legs);
                }
            }
        } else {
            // bend legs up
            int64 num_legs_in_dom = 0;
            if (tensor->num_codomain_legs() <= num_up_bends) {
                num_legs_in_dom = num_up_bends - tensor->num_codomain_legs();
            }
            auto codomain = range_of(num_up_bends, tensor->num_codomain_legs());
            auto domain = reversed_range(0, num_up_bends);
            auto extra =
              reversed_range(tensor->num_codomain_legs() + num_legs_in_dom, tensor->num_legs);
            domain.insert(domain.end(), extra.begin(), extra.end());
            tensor = planar_permute_legs(tensor, as_leg_refs(codomain), as_leg_refs(domain));
            // update pairs
            for (auto& pair : pair_idcs) {
                for (auto& idx : pair) {
                    idx = py_mod(idx - num_up_bends, tensor->num_legs);
                }
            }
        }

        // give the traced legs levels in case they are not next to each other
        // and one leg needs to braid with another leg pair
        for (int64 i = 0; i < static_cast<int64>(pair_idcs.size()); ++i) {
            levels[static_cast<std::size_t>(pair_idcs[static_cast<std::size_t>(i)][0])] = i;
            levels[static_cast<std::size_t>(pair_idcs[static_cast<std::size_t>(i)][1])] = i;
        }
    }

    std::vector<std::vector<LegRef>> pair_refs;
    pair_refs.reserve(pair_idcs.size());
    for (auto const& p : pair_idcs) {
        pair_refs.push_back(as_leg_refs(p));
    }
    return partial_trace(tensor, pair_refs, levels);
}

TensorPlaceholder
planar_partial_trace(TensorPlaceholder const& tensor, std::vector<std::vector<LegRef>> pairs)
{
    std::vector<std::vector<int64>> pair_idcs;
    pair_idcs.reserve(pairs.size());
    for (auto const& p : pairs) {
        pair_idcs.push_back(tensor.get_leg_idcs(p));
    }
    std::vector<int64> traced_legs;
    for (auto const& p : pair_idcs) {
        traced_legs.insert(traced_legs.end(), p.begin(), p.end());
    }
    for (auto pair : pair_idcs) {
        if (pair.size() != 2) {
            throw std::invalid_argument("each trace pair must have two legs");
        }
        auto l1 = pair[0];
        auto l2 = pair[1];
        if (l1 > l2) {
            std::swap(l1, l2);
        }
        if (l1 == l2) {
            throw std::invalid_argument("trace pair legs must be distinct");
        }
        bool first_half_only_traces = true;
        bool second_half_only_traces = true;
        for (int64 l = l1 + 1; l < l2; ++l) {
            if (contains(traced_legs, l)) {
                std::vector<int64> other_ls;
                for (auto const& pr : pair_idcs) {
                    if (pr[1] == l) {
                        other_ls.push_back(pr[0]);
                    }
                    if (pr[0] == l) {
                        other_ls.push_back(pr[1]);
                    }
                }
                if (other_ls.size() != 1) {
                    throw std::invalid_argument("each traced leg must appear in exactly one pair");
                }
                if (!(l1 < other_ls[0] && other_ls[0] < l2)) {
                    throw std::invalid_argument("Not a planar trace");
                }
            } else {
                first_half_only_traces = false;
            }
        }
        for (auto l : range_of(l2 + 1, tensor.num_legs)) {
            if (contains(traced_legs, l)) {
                std::vector<int64> other_ls;
                for (auto const& pr : pair_idcs) {
                    if (pr[1] == l) {
                        other_ls.push_back(pr[0]);
                    }
                    if (pr[0] == l) {
                        other_ls.push_back(pr[1]);
                    }
                }
                if (other_ls.size() != 1) {
                    throw std::invalid_argument("each traced leg must appear in exactly one pair");
                }
                if (l1 < other_ls[0] && other_ls[0] < l2) {
                    throw std::invalid_argument("Not a planar trace");
                }
            } else {
                second_half_only_traces = false;
            }
        }
        for (int64 l = 0; l < l1; ++l) {
            if (contains(traced_legs, l)) {
                std::vector<int64> other_ls;
                for (auto const& pr : pair_idcs) {
                    if (pr[1] == l) {
                        other_ls.push_back(pr[0]);
                    }
                    if (pr[0] == l) {
                        other_ls.push_back(pr[1]);
                    }
                }
                if (other_ls.size() != 1) {
                    throw std::invalid_argument("each traced leg must appear in exactly one pair");
                }
                if (l1 < other_ls[0] && other_ls[0] < l2) {
                    throw std::invalid_argument("Not a planar trace");
                }
            } else {
                second_half_only_traces = false;
            }
        }
        if (!(first_half_only_traces || second_half_only_traces)) {
            throw std::invalid_argument("Not a planar trace");
        }
    }

    std::vector<BigOPolynomial> contr_dims;
    for (auto const& p : pair_idcs) {
        contr_dims.push_back(tensor.dims[static_cast<std::size_t>(p[0])]);
    }
    std::vector<BigOPolynomial> contr_dims_other;
    for (auto const& p : pair_idcs) {
        contr_dims_other.push_back(tensor.dims[static_cast<std::size_t>(p[1])]);
    }
    if (contr_dims != contr_dims_other) {
        throw std::invalid_argument("traced dimensions do not match");
    }
    std::vector<BigOPolynomial> open_dims;
    std::vector<std::string> labels;
    auto labs = tensor.string_labels();
    for (int64 l = 0; l < tensor.num_legs; ++l) {
        if (!contains(traced_legs, l)) {
            open_dims.push_back(tensor.dims[static_cast<std::size_t>(l)]);
            labels.push_back(labs[static_cast<std::size_t>(l)]);
        }
    }
    auto cost_dims = open_dims;
    cost_dims.insert(cost_dims.end(), contr_dims.begin(), contr_dims.end());
    auto cost = tensor.cost_to_make + product_of(cost_dims);
    return TensorPlaceholder(std::move(labels), std::move(open_dims), std::move(cost));
}

TensorPtr
planar_permute_legs(TensorCPtr T,
                    std::optional<std::vector<LegRef>> codomain,
                    std::optional<std::vector<LegRef>> domain)
{
    // Note: parse_leg_bipartition cannot easily be used in this function due to how it interacts
    // with empty (co)domains

    if (!codomain && !domain) {
        throw std::invalid_argument("Need to specify either codomain or domain that is non-empty");
    }
    if ((!codomain && domain && domain->empty()) || (!domain && codomain && codomain->empty())) {
        throw std::invalid_argument("Specified codomain or domain is empty");
    }

    // do this for both before potentially comparing (avoid comparing to labels)
    std::optional<std::vector<int64>> domain_idcs;
    std::optional<std::vector<int64>> codomain_idcs;
    if (domain) {
        domain_idcs = T->get_leg_idcs(*domain);
    }
    if (codomain) {
        codomain_idcs = T->get_leg_idcs(*codomain);
    }

    if (domain_idcs && !domain_idcs->empty()) {
        std::vector<int64> expect;
        expect.reserve(domain_idcs->size());
        for (int64 i = 0; i < static_cast<int64>(domain_idcs->size()); ++i) {
            expect.push_back(py_mod((*domain_idcs)[domain_idcs->size() - 1] + i, T->num_legs));
        }
        std::ranges::reverse(expect);
        if (*domain_idcs != expect) {
            throw std::invalid_argument("The given domain is a non-planar permutation");
        }
        auto num_codom_legs = T->num_legs - static_cast<int64>(domain_idcs->size());
        std::vector<int64> codomain2;
        for (int64 i = 0; i < num_codom_legs; ++i) {
            codomain2.push_back(py_mod((*domain_idcs)[0] + 1 + i, T->num_legs));
        }
        if (!codomain_idcs) {
            codomain_idcs = std::move(codomain2);
        } else if (*codomain_idcs != codomain2) {
            throw std::invalid_argument("The given codomain and domain are inconsistent!");
        }
    }
    if (codomain_idcs && !codomain_idcs->empty()) {
        std::vector<int64> expect;
        expect.reserve(codomain_idcs->size());
        for (int64 i = 0; i < static_cast<int64>(codomain_idcs->size()); ++i) {
            expect.push_back(py_mod((*codomain_idcs)[0] + i, T->num_legs));
        }
        if (*codomain_idcs != expect) {
            throw std::invalid_argument("The given codomain is a non-planar permutation");
        }
        auto num_dom_legs = T->num_legs - static_cast<int64>(codomain_idcs->size());
        std::vector<int64> reverse_domain;
        for (int64 i = 0; i < num_dom_legs; ++i) {
            reverse_domain.push_back(py_mod(codomain_idcs->back() + 1 + i, T->num_legs));
        }
        auto domain2 = reversed_copy(reverse_domain);
        if (!domain_idcs) {
            domain_idcs = std::move(domain2);
        } else if (*domain_idcs != domain2) {
            throw std::invalid_argument("The given codomain and domain are inconsistent!");
        }
    }

    auto const& co = *codomain_idcs;
    auto const& dom = *domain_idcs;

    // figure out if legs need to bend right or left of the tensor.
    std::vector<int64> codomain_staying;
    for (int64 n = 0; n < T->num_codomain_legs(); ++n) {
        if (contains(co, n)) {
            codomain_staying.push_back(n);
        }
    }
    std::vector<int64> domain_staying;
    for (int64 n = 0; n < T->num_domain_legs(); ++n) {
        if (contains(dom, T->num_legs - 1 - n)) {
            domain_staying.push_back(n);
        }
    }

    // requires two bends of at least one leg
    bool codomain_winding = false;
    if (!codomain_staying.empty() && contains(co, int64(0)) &&
        contains(co, T->num_codomain_legs() - 1)) {
        codomain_winding = index_of(co, T->num_codomain_legs() - 1) < index_of(co, int64(0));
    }
    bool domain_winding = false;
    if (!domain_staying.empty() && contains(dom, T->num_codomain_legs()) &&
        contains(dom, T->num_legs - 1)) {
        domain_winding = index_of(dom, T->num_codomain_legs()) < index_of(dom, T->num_legs - 1);
    }
    // one at most can be True
    if (codomain_winding && domain_winding) {
        throw std::invalid_argument("codomain and domain cannot both wind");
    }

    std::vector<std::optional<bool>> bend_right_list;
    bool have_bend_list = false;

    if (codomain_staying.empty() && domain_staying.empty() && !co.empty() && !dom.empty()) {
        // they swap places completely -> choose the direction such that we have less left bends
        // than right bends
        if (T->num_codomain_legs() < T->num_domain_legs()) {
            bend_right_list.assign(static_cast<std::size_t>(T->num_codomain_legs()), false);
            bend_right_list.insert(
              bend_right_list.end(), static_cast<std::size_t>(T->num_domain_legs()), true);
        } else {
            bend_right_list.assign(static_cast<std::size_t>(T->num_codomain_legs()), true);
            bend_right_list.insert(
              bend_right_list.end(), static_cast<std::size_t>(T->num_domain_legs()), false);
        }
        have_bend_list = true;
    } else if (codomain_winding) {
        // special case where the group of legs that stay in the codomain "wraps around",
        // i.e. surrounds the ones that should go to the domain on both sides
        // three groups: stay, bend up, bend twice ("around")
        // this is an arbitrary choice of orientation "counter-clockwise"
        auto bend_up = T->num_codomain_legs() - static_cast<int64>(codomain_staying.size());
        auto dont_bend = co.back() + 1;
        auto bend_twice = index_of(co, T->num_codomain_legs() - 1) + 1;
        if (dont_bend + bend_up + bend_twice != T->num_codomain_legs()) {
            throw std::invalid_argument("inconsistent codomain winding partition");
        }
        // OPTIMIZE achieve it in a single backend function? also in a similar branch below
        // OPTIMIZE we go around counter-clockwise, clockwise could be more efficient in some cases
        auto res = permute_legs(T,
                                as_leg_refs(range_of(dont_bend)),
                                as_leg_refs(reversed_range(dont_bend, T->num_legs)),
                                std::nullopt,
                                BendRight{ true });
        auto co2 = range_of(dont_bend + bend_up, T->num_legs);
        auto co2b = range_of(dont_bend);
        co2.insert(co2.end(), co2b.begin(), co2b.end());
        res =
          permute_legs(res,
                       as_leg_refs(co2),
                       as_leg_refs(reversed_range(dont_bend, T->num_codomain_legs() - bend_twice)),
                       std::nullopt,
                       BendRight{ false });
        return res;
    } else if (domain_winding) {
        // special case where the group of legs that stay in the domain "wraps around",
        // i.e. surrounds the ones that should go to the codomain on both sides
        // three groups (in leg order): stay, bend down, bend twice ("around")
        // this is an arbitrary choice of orientation "counter-clockwise"
        auto bend_down = T->num_domain_legs() - static_cast<int64>(domain_staying.size());
        auto dont_bend = dom[0] + 1 - T->num_codomain_legs();
        auto bend_twice = static_cast<int64>(dom.size()) - index_of(dom, T->num_legs - 1);
        if (bend_twice + bend_down + dont_bend != T->num_domain_legs()) {
            throw std::invalid_argument("inconsistent domain winding partition");
        }
        auto co1 = range_of(T->num_codomain_legs() + dont_bend, T->num_legs);
        auto co1b = range_of(T->num_codomain_legs());
        co1.insert(co1.end(), co1b.begin(), co1b.end());
        auto res = permute_legs(
          T,
          as_leg_refs(co1),
          as_leg_refs(reversed_range(T->num_codomain_legs(), T->num_codomain_legs() + dont_bend)),
          std::nullopt,
          BendRight{ false });
        res = permute_legs(res,
                           as_leg_refs(range_of(bend_down)),
                           as_leg_refs(reversed_range(bend_down, T->num_legs)),
                           std::nullopt,
                           BendRight{ true });
        return res;
    } else if (static_cast<int64>(codomain_staying.size()) == T->num_codomain_legs() &&
               static_cast<int64>(domain_staying.size()) == T->num_domain_legs()) {
        // nothing to do
        // the number of entries in codomain_staying is only sufficient to detect this case after
        // considering the winding cases (codomain or domain could be empty)
        return std::const_pointer_cast<Tensor>(T);
    } else if (T->num_codomain_legs() == 0) {
        // split into three groups: bending down right, staying, bending down left
        // note that one of the outer groups (but not both) may be empty
        int64 left_bending = 0;
        if (contains(co, T->num_legs - 1) && T->num_legs != 1) {
            // for a one-leg tensor, bend right
            left_bending = index_of(co, T->num_legs - 1) + 1;
        }
        auto dont_bend = static_cast<int64>(domain_staying.size());
        int64 right_bending = 0;
        if (contains(co, int64(0))) {
            right_bending = static_cast<int64>(co.size()) - index_of(co, int64(0));
        }
        if (dont_bend == 0 && left_bending == T->num_legs && right_bending == T->num_legs) {
            // special case when all legs need to be bent down to the right or left
            left_bending = 0;
        }
        if (left_bending + dont_bend + right_bending != T->num_legs) {
            throw std::invalid_argument("inconsistent empty-codomain partition");
        }
        bend_right_list.assign(static_cast<std::size_t>(right_bending), true);
        bend_right_list.insert(
          bend_right_list.end(), static_cast<std::size_t>(dont_bend), std::nullopt);
        bend_right_list.insert(
          bend_right_list.end(), static_cast<std::size_t>(left_bending), false);
        have_bend_list = true;
    } else if (T->num_domain_legs() == 0) {
        // split into three groups: bending up left, staying, bending up right
        // note that one of the outer groups (but not both) may be empty
        int64 left_bending = 0;
        if (contains(dom, int64(0)) && T->num_legs != 1) {
            // for a one-leg tensor, bend right
            left_bending = index_of(dom, int64(0)) + 1;
        }
        auto dont_bend = static_cast<int64>(codomain_staying.size());
        int64 right_bending = 0;
        if (contains(dom, T->num_legs - 1)) {
            right_bending = static_cast<int64>(dom.size()) - index_of(dom, T->num_legs - 1);
        }
        if (dont_bend == 0 && left_bending == T->num_legs && right_bending == T->num_legs) {
            // special case when all legs need to be bent up to the right or left
            left_bending = 0;
        }
        if (left_bending + dont_bend + right_bending != T->num_legs) {
            throw std::invalid_argument("inconsistent empty-domain partition");
        }
        bend_right_list.assign(static_cast<std::size_t>(left_bending), false);
        bend_right_list.insert(
          bend_right_list.end(), static_cast<std::size_t>(dont_bend), std::nullopt);
        bend_right_list.insert(
          bend_right_list.end(), static_cast<std::size_t>(right_bending), true);
        have_bend_list = true;
    } else if (codomain_staying.empty()) {
        // codomain goes up as a whole, either right or left
        auto domain_bend_left = domain_staying[0];
        auto domain_bend_right = T->num_domain_legs() - 1 - domain_staying.back();
        if (domain_bend_left + static_cast<int64>(domain_staying.size()) + domain_bend_right !=
            T->num_domain_legs()) {
            throw std::invalid_argument("inconsistent domain staying partition");
        }
        if (static_cast<int64>(domain_staying.size()) == T->num_domain_legs()) {
            // domain stays, codomain is divided and bent right and left
            auto num_bend_left = index_of(dom, T->num_legs - 1);
            bend_right_list.assign(static_cast<std::size_t>(num_bend_left), false);
            bend_right_list.insert(
              bend_right_list.end(),
              static_cast<std::size_t>(T->num_codomain_legs() - num_bend_left),
              true);
            bend_right_list.insert(
              bend_right_list.end(), static_cast<std::size_t>(T->num_domain_legs()), std::nullopt);
        } else if (domain_bend_left == 0) {
            // bend the codomain up to the left
            bend_right_list.assign(static_cast<std::size_t>(T->num_codomain_legs()), false);
            bend_right_list.insert(
              bend_right_list.end(), static_cast<std::size_t>(domain_bend_right), true);
            bend_right_list.insert(bend_right_list.end(), domain_staying.size(), std::nullopt);
        } else if (domain_bend_right == 0) {
            // bend the codomain up to the right
            bend_right_list.assign(static_cast<std::size_t>(T->num_codomain_legs()), true);
            bend_right_list.insert(bend_right_list.end(), domain_staying.size(), std::nullopt);
            bend_right_list.insert(
              bend_right_list.end(), static_cast<std::size_t>(domain_bend_left), false);
        } else {
            throw std::runtime_error("Not planar, but that should have been detected earlier?");
        }
        have_bend_list = true;
    } else if (domain_staying.empty()) {
        // domain goes down as a whole, either right or left
        auto codomain_bend_left = codomain_staying[0];
        auto codomain_bend_right = T->num_codomain_legs() - 1 - codomain_staying.back();
        if (codomain_bend_left + static_cast<int64>(codomain_staying.size()) +
              codomain_bend_right !=
            T->num_codomain_legs()) {
            throw std::invalid_argument("inconsistent codomain staying partition");
        }
        if (static_cast<int64>(codomain_staying.size()) == T->num_codomain_legs()) {
            auto num_bend_left = index_of(co, int64(0));
            bend_right_list.assign(static_cast<std::size_t>(T->num_codomain_legs()), std::nullopt);
            bend_right_list.insert(bend_right_list.end(),
                                   static_cast<std::size_t>(T->num_domain_legs() - num_bend_left),
                                   true);
            bend_right_list.insert(
              bend_right_list.end(), static_cast<std::size_t>(num_bend_left), false);
        } else if (codomain_bend_left == 0) {
            // bend the domain down to the left
            bend_right_list.assign(codomain_staying.size(), std::nullopt);
            bend_right_list.insert(
              bend_right_list.end(), static_cast<std::size_t>(codomain_bend_right), true);
            bend_right_list.insert(
              bend_right_list.end(), static_cast<std::size_t>(T->num_domain_legs()), false);
        } else if (codomain_bend_right == 0) {
            // bend the domain down to the right
            bend_right_list.assign(static_cast<std::size_t>(codomain_bend_left), false);
            bend_right_list.insert(bend_right_list.end(), codomain_staying.size(), std::nullopt);
            bend_right_list.insert(
              bend_right_list.end(), static_cast<std::size_t>(T->num_domain_legs()), true);
        } else {
            throw std::runtime_error("Not planar, but that should have been detected earlier?");
        }
        have_bend_list = true;
    } else {
        auto codomain_bend_left = codomain_staying[0];
        auto codomain_bend_right = T->num_codomain_legs() - 1 - codomain_staying.back();
        auto domain_bend_left = domain_staying[0];
        auto domain_bend_right = T->num_domain_legs() - 1 - domain_staying.back();
        if (!(codomain_bend_left == 0 || domain_bend_left == 0)) {
            throw std::invalid_argument("not planar permute");
        }
        if (!(codomain_bend_right == 0 || domain_bend_right == 0)) {
            throw std::invalid_argument("not planar permute");
        }
        bend_right_list.assign(static_cast<std::size_t>(codomain_bend_left), false);
        bend_right_list.insert(bend_right_list.end(), codomain_staying.size(), std::nullopt);
        bend_right_list.insert(
          bend_right_list.end(), static_cast<std::size_t>(codomain_bend_right), true);
        bend_right_list.insert(
          bend_right_list.end(), static_cast<std::size_t>(domain_bend_right), true);
        bend_right_list.insert(bend_right_list.end(), domain_staying.size(), std::nullopt);
        bend_right_list.insert(
          bend_right_list.end(), static_cast<std::size_t>(domain_bend_left), false);
        have_bend_list = true;
    }

    (void)have_bend_list;
    return permute_legs(
      T, as_leg_refs(co), as_leg_refs(dom), std::nullopt, BendRight{ std::move(bend_right_list) });
}

std::tuple<TensorPtr, TensorPtr>
planar_qr(TensorCPtr tensor,
          int64 codomain_cut,
          int64 domain_cut,
          std::optional<LegLabels> new_labels,
          bool new_leg_dual)
{
    auto r = planar_decomposition(tensor,
                                  codomain_cut,
                                  domain_cut,
                                  PlanarDecompWhich::qr,
                                  std::move(new_labels),
                                  new_leg_dual);
    return { std::move(r.A), std::move(r.B) };
}

std::tuple<TensorPtr, DiagonalTensorPtr, TensorPtr>
planar_svd(TensorCPtr tensor,
           int64 codomain_cut,
           int64 domain_cut,
           std::optional<LegLabels> new_labels,
           bool new_leg_dual,
           std::optional<std::string> algorithm)
{
    auto r = planar_decomposition(tensor,
                                  codomain_cut,
                                  domain_cut,
                                  PlanarDecompWhich::svd,
                                  std::move(new_labels),
                                  new_leg_dual,
                                  std::nullopt,
                                  algorithm);
    return { std::move(r.A), std::move(r.S), std::move(r.B) };
}

std::tuple<TensorPtr, DiagonalTensorPtr, TensorPtr, float64, float64>
planar_truncated_svd(TensorCPtr tensor,
                     int64 codomain_cut,
                     int64 domain_cut,
                     std::optional<LegLabels> new_labels,
                     bool new_leg_dual,
                     std::optional<std::string> algorithm,
                     std::optional<float64> normalize_to,
                     std::optional<int64> chi_max,
                     int64 chi_min,
                     float64 degeneracy_tol,
                     float64 trunc_cut,
                     float64 svd_min)
{
    auto r = planar_decomposition(tensor,
                                  codomain_cut,
                                  domain_cut,
                                  PlanarDecompWhich::truncated_svd,
                                  std::move(new_labels),
                                  new_leg_dual,
                                  std::nullopt,
                                  algorithm,
                                  normalize_to,
                                  chi_max,
                                  chi_min,
                                  degeneracy_tol,
                                  trunc_cut,
                                  svd_min);
    return { std::move(r.A), std::move(r.S), std::move(r.B), r.err, r.renormalize };
}

std::pair<std::vector<int64>, std::vector<int64>>
parse_leg_bipartition(std::vector<int64> const& legs, int64 num_legs)
{
    {
        std::set<int64> seen;
        for (auto l : legs) {
            check(seen.insert(l).second, "duplicate legs");
            check(0 <= l && l < num_legs, "leg index out of range");
        }
    }
    // special cases
    if (legs.empty()) {
        return { {}, range_of(num_legs) };
    }
    if (static_cast<int64>(legs.size()) == num_legs) {
        return { range_of(num_legs), {} };
    }

    auto sorted_legs = legs;
    std::ranges::sort(sorted_legs);
    std::vector<int64> jumps;
    for (std::size_t i = 0; i + 1 < sorted_legs.size(); ++i) {
        if (sorted_legs[i + 1] != sorted_legs[i] + 1) {
            jumps.push_back(static_cast<int64>(i));
        }
    }
    if (jumps.empty()) {
        // legs is contiguous even on a line -> other subset wraps around the circle
        auto other_legs = range_of(sorted_legs.back() + 1, num_legs);
        auto head = range_of(sorted_legs.front());
        other_legs.insert(other_legs.end(), head.begin(), head.end());
        return { sorted_legs, std::move(other_legs) };
    }
    if (jumps.size() == 1 && sorted_legs.front() == 0 && sorted_legs.back() == num_legs - 1) {
        // a single jump is ok, but only if the legs "wrap around", i.e. contain 0 and L-1
        // legs "wraps" around the circle -> other subset is contiguous even on the line
        auto last = sorted_legs[static_cast<std::size_t>(jumps[0])];
        auto first = sorted_legs[static_cast<std::size_t>(jumps[0] + 1)];
        auto res_legs = range_of(first, num_legs);
        auto tail = range_of(last + 1);
        res_legs.insert(res_legs.end(), tail.begin(), tail.end());
        auto other_legs = range_of(last + 1, first);
        return { std::move(res_legs), std::move(other_legs) };
    }
    throw std::invalid_argument("Not a planar bipartition");
}

std::pair<TensorPtr, std::optional<int64>>
_planar_contraction_helper(TensorCPtr tensor, std::vector<int64> const& contr, bool domain)
{
    // case 1: no legs are contracted, compose used in planar_contraction,
    //         do appropriate bends here, resulting leg order does not matter
    if (contr.empty()) {
        std::optional<std::vector<LegRef>> new_codom =
          domain ? std::optional{ as_leg_refs(range_of(tensor->num_legs)) } : std::nullopt;
        std::optional<std::vector<LegRef>> new_dom =
          domain ? std::nullopt
                 : std::optional{ as_leg_refs(reversed_range(0, tensor->num_legs)) };
        return { planar_permute_legs(tensor, new_codom, new_dom), std::nullopt };
    }

    // case 2: all legs are contracted, compose used in planar_contraction,
    //         it is possible that the legs need to be cyclically permuted
    //         -> we cannot just work with a number of bends to perform
    if (static_cast<int64>(contr.size()) == tensor->num_legs) {
        std::optional<std::vector<LegRef>> new_codom =
          domain ? std::nullopt : std::optional{ as_leg_refs(contr) };
        std::optional<std::vector<LegRef>> new_dom =
          domain ? std::optional{ as_leg_refs(contr) } : std::nullopt;
        return { planar_permute_legs(tensor, new_codom, new_dom), std::nullopt };
    }

    // case 3: legs are contracted, there is at least one uncontracted leg left,
    //         we can in general work with either left or right bends and do not need both
    //         -> knowing whether to bend right or left and the number of bends is sufficient
    bool bend_right = true;
    if (contains(contr, tensor->num_codomain_legs() - 1) &&
        contains(contr, tensor->num_codomain_legs())) {
        bend_right = true;
    } else if (contains(contr, int64(0)) && contains(contr, tensor->num_legs - 1)) {
        bend_right = false;
    }

    int64 num_bends = 0;
    if (bend_right && domain) {
        // bend right and up
        num_bends = tensor->num_codomain_legs() - *std::ranges::min_element(contr);
    } else if (bend_right) {
        // bend right and down
        num_bends = *std::ranges::max_element(contr) + 1 - tensor->num_codomain_legs();
    } else if (domain) {
        // bend left and up
        std::vector<int64> legs_in_codom;
        for (auto l : contr) {
            if (l < tensor->num_codomain_legs()) {
                legs_in_codom.push_back(l);
            }
        }
        if (legs_in_codom.empty()) {
            return { std::const_pointer_cast<Tensor>(tensor), *std::ranges::min_element(contr) };
        }
        num_bends = *std::ranges::max_element(legs_in_codom) + 1;
    } else {
        // bend left and down
        std::vector<int64> legs_in_dom;
        for (auto l : contr) {
            if (l >= tensor->num_codomain_legs()) {
                legs_in_dom.push_back(l);
            }
        }
        if (legs_in_dom.empty()) {
            return { std::const_pointer_cast<Tensor>(tensor), *std::ranges::min_element(contr) };
        }
        num_bends = tensor->num_legs - *std::ranges::min_element(legs_in_dom);
    }

    if (num_bends > 0) {
        std::optional<int64> partial_compose_leg;
        std::vector<int64> new_codom;
        std::vector<int64> new_dom;
        if (bend_right) {
            partial_compose_leg = *std::ranges::min_element(contr);
            if (domain) {
                new_codom = range_of(tensor->num_codomain_legs() - num_bends);
                new_dom = range_of(tensor->num_codomain_legs() - num_bends, tensor->num_legs);
            } else {
                new_codom = range_of(tensor->num_codomain_legs() + num_bends);
                new_dom = range_of(tensor->num_codomain_legs() + num_bends, tensor->num_legs);
            }
        } else {
            if (domain) {
                partial_compose_leg = tensor->num_legs - static_cast<int64>(contr.size());
                new_codom = range_of(num_bends, tensor->num_codomain_legs());
                new_dom = range_of(tensor->num_codomain_legs(), tensor->num_legs);
                auto extra = range_of(num_bends);
                new_dom.insert(new_dom.end(), extra.begin(), extra.end());
            } else {
                partial_compose_leg = 0;
                new_codom = range_of(tensor->num_legs - num_bends, tensor->num_legs);
                auto extra = range_of(tensor->num_codomain_legs());
                new_codom.insert(new_codom.end(), extra.begin(), extra.end());
                new_dom = range_of(tensor->num_codomain_legs(), tensor->num_legs - num_bends);
            }
        }
        std::ranges::reverse(new_dom);
        return { planar_permute_legs(tensor, as_leg_refs(new_codom), as_leg_refs(new_dom)),
                 partial_compose_leg };
    }
    return { std::const_pointer_cast<Tensor>(tensor), *std::ranges::min_element(contr) };
}

} // namespace cyten
