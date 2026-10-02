#include <cyten/tensors/labels.h>
#include <iostream>

#include <cassert>
#include <format>
#include <ranges>
#include <set>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

namespace cyten {

namespace {

/// Duplicate entries in ``seq``, ignoring unset (``nullopt``) labels.
[[nodiscard]] std::unordered_set<std::string>
duplicate_leg_labels(OptionalLabels const& seq)
{
    std::unordered_set<std::string> seen;
    std::unordered_set<std::string> dups;
    for (auto const& ele : seq) {
        if (!ele) {
            continue;
        }
        if (!seen.insert(*ele).second) {
            dups.insert(*ele);
        }
    }
    return dups;
}

[[nodiscard]] bool
leg_labels_contain(OptionalLabels const& labels, OptionalLabel const& label)
{
    return std::ranges::find(labels, label) != labels.end();
}

[[nodiscard]] std::string
format_leg_labels(OptionalLabels const& labels)
{
    std::ostringstream out;
    out << '[';
    bool first = true;
    for (auto const& l : labels) {
        if (!first) {
            out << ", ";
        }
        first = false;
        if (l) {
            out << *l;
        } else {
            out << "None";
        }
    }
    out << ']';
    return out.str();
}

[[nodiscard]] std::unordered_map<std::string, int64>
build_labelmap(OptionalLabels const& labels)
{
    std::unordered_map<std::string, int64> map;
    for (int64 i = 0; i < static_cast<int64>(labels.size()); ++i) {
        if (labels[static_cast<std::size_t>(i)]) {
            map[*labels[static_cast<std::size_t>(i)]] = i;
        }
    }
    return map;
}

} // namespace

bool
is_valid_leg_label(OptionalLabel const& label)
{
    if (!label) {
        return true;
    }
    // TODO extend: check for valid syntax of combined / conjugated labels?
    for (char const* forbidden : FORBIDDEN_LEG_LABEL_CHARS) {
        if (label->find(forbidden) != std::string::npos) {
            return false;
        }
    }
    return true;
}

std::string
_combine_leg_labels(OptionalLabels const& labels, int64 offset)
{
    std::string joined;
    for (std::size_t n = 0; n < labels.size(); ++n) {
        if (n > 0) {
            joined += '.';
        }
        if (!labels[n]) {
            joined += std::format("?{}", static_cast<int64>(n) + offset);
        } else {
            joined += *labels[n];
        }
    }
    return std::format("({})", joined);
}

OptionalLabels
_split_leg_label(OptionalLabel const& label, std::optional<int64> num)
{
    if (!label) {
        if (!num.has_value()) {
            throw std::invalid_argument("num is required when splitting an unset combined label");
        }
        return OptionalLabels(static_cast<std::size_t>(*num), std::nullopt);
    }
    if (label->starts_with('(') && label->ends_with(')')) {
        std::string const inner = label->substr(1, label->size() - 2);
        OptionalLabels labels;
        // Match Python ``str.split('.')`` (keeps empty segments).
        std::size_t start = 0;
        while (true) {
            std::size_t const dot = inner.find('.', start);
            std::string part =
              (dot == std::string::npos) ? inner.substr(start) : inner.substr(start, dot - start);
            if (part.starts_with('?')) {
                labels.push_back(std::nullopt);
            } else {
                labels.emplace_back(std::move(part));
            }
            if (dot == std::string::npos) {
                break;
            }
            start = dot + 1;
        }
        if (num.has_value() && static_cast<int64>(labels.size()) != *num) {
            throw std::invalid_argument(std::format(
              "expected {} components in combined label, got {}", *num, labels.size()));
        }
        return labels;
    }
    throw std::invalid_argument("Invalid format for a combined label");
}

OptionalLabels
_dual_label_list(OptionalLabels const& labels)
{
    OptionalLabels out;
    out.reserve(labels.size());
    for (auto const& l : labels | std::views::reverse) {
        out.push_back(_dual_leg_label(l));
    }
    return out;
}

OptionalLabel
_dual_leg_label(OptionalLabel const& label)
{
    if (!label) {
        return std::nullopt;
    }
    if (label->starts_with('(') && label->ends_with(')')) {
        return _combine_leg_labels(_dual_label_list(_split_leg_label(label)));
    }
    if (label->ends_with('*')) {
        return label->substr(0, label->size() - 1);
    }
    return *label + '*';
}

OptionalLabels
_get_matching_labels(OptionalLabels const& labels1, OptionalLabels const& labels2)
{
    OptionalLabels labels;
    std::vector<int64> conflicts;
    auto const n = std::min(labels1.size(), labels2.size());
    labels.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        auto const& l1 = labels1[i];
        auto const& l2 = labels2[i];
        if (!l1) {
            labels.push_back(l2);
        } else if (!l2 || l1 == l2) {
            labels.push_back(l1);
        } else {
            conflicts.push_back(static_cast<int64>(i));
            labels.push_back(std::nullopt);
        }
    }
    if (!conflicts.empty() && Py_IsInitialized()) {
        std::string positions;
        for (std::size_t i = 0; i < conflicts.size(); ++i) {
            if (i > 0) {
                positions += ", ";
            }
            positions += std::to_string(conflicts[i]);
        }
        auto const msg =
          std::format("Conflicting labels at positions {} are dropped. labels1={}, labels2={}.",
                      positions,
                      format_leg_labels(labels1),
                      format_leg_labels(labels2));
        // C++ debug log (was Python logging.getLogger(...).debug).
        std::cerr << "cyten.tensors._tensors DEBUG: " << msg << '\n';
    }
    return labels;
}

LabelledLegs::LabelledLegs(OptionalLabels labels)
{
    auto const dup = duplicate_leg_labels(labels);
    if (!dup.empty()) {
        std::string joined;
        bool first = true;
        for (auto const& d : dup) {
            if (!first) {
                joined += ", ";
            }
            first = false;
            joined += d;
        }
        throw std::invalid_argument(std::format("Duplicate leg labels: {}", joined));
    }
    num_legs = static_cast<int64>(labels.size());
    _labelmap = build_labelmap(labels);
    _labels = std::move(labels);
}

void
LabelledLegs::test_sanity() const
{
    assert(std::ranges::all_of(_labels, [](auto const& l) { return is_valid_leg_label(l); }));
    assert(duplicate_leg_labels(_labels).empty());
    std::unordered_set<int64> values;
    for (auto const& [key, v] : _labelmap) {
        (void)key;
        assert(values.insert(v).second);
    }
}

bool
LabelledLegs::is_fully_labelled() const
{
    return std::ranges::all_of(_labels, [](OptionalLabel const& l) { return l.has_value(); });
}

OptionalLabels
LabelledLegs::labels() const
{
    return _labels;
}

std::vector<int64>
LabelledLegs::get_leg_idcs(int64 idx) const
{
    return { to_valid_idx(idx, num_legs) };
}

std::vector<int64>
LabelledLegs::get_leg_idcs(std::string const& label) const
{
    auto it = _labelmap.find(label);
    if (it == _labelmap.end()) {
        throw std::invalid_argument(
          std::format("No leg with label {}. Labels are {}", label, format_leg_labels(_labels)));
    }
    return { it->second };
}

std::vector<int64>
LabelledLegs::get_leg_idcs(std::vector<std::variant<int64, std::string>> const& idcs) const
{
    std::vector<int64> res;
    res.reserve(idcs.size());
    for (auto const& idx : idcs) {
        if (std::holds_alternative<std::string>(idx)) {
            auto const& label = std::get<std::string>(idx);
            auto it = _labelmap.find(label);
            if (it == _labelmap.end()) {
                throw std::invalid_argument(std::format(
                  "No leg with label {}. Labels are {}", label, format_leg_labels(_labels)));
            }
            res.push_back(it->second);
        } else {
            res.push_back(to_valid_idx(std::get<int64>(idx), num_legs));
        }
    }
    return res;
}

bool
LabelledLegs::has_label(std::string const& label) const
{
    return leg_labels_contain(_labels, OptionalLabel{ label });
}

bool
LabelledLegs::has_label(std::vector<std::string> const& more) const
{
    return std::ranges::all_of(more, [this](std::string const& l) {
        return leg_labels_contain(_labels, OptionalLabel{ l });
    });
}

bool
LabelledLegs::labels_are(std::vector<std::string> const& want, bool planar) const
{
    if (!is_fully_labelled()) {
        return false;
    }
    if (static_cast<int64>(want.size()) != num_legs) {
        return false;
    }
    if (planar) {
        if (want.empty()) {
            return true;
        }
        auto const start = _labelmap.find(want.front());
        if (start == _labelmap.end()) {
            return false;
        }
        for (std::size_t i = 1; i < want.size(); ++i) {
            auto const idx = (static_cast<std::size_t>(start->second) + i) % want.size();
            if (*_labels[idx] != want[i]) {
                return false;
            }
        }
        return true;
    }

    // Equal-length lists with unique labels can be compared as sets.
    std::set<std::string> a(want.begin(), want.end());
    std::set<std::string> b;
    for (auto const& l : _labels) {
        b.insert(*l);
    }
    return a == b;
}

LabelledLegs&
LabelledLegs::relabel(std::map<std::string, std::string> const& mapping)
{
    OptionalLabels next;
    next.reserve(_labels.size());
    for (auto const& l : _labels) {
        if (l) {
            auto it = mapping.find(*l);
            next.push_back(it == mapping.end() ? l : OptionalLabel{ it->second });
        } else {
            next.push_back(std::nullopt);
        }
    }
    return set_labels(std::move(next));
}

LabelledLegs&
LabelledLegs::set_label(int64 pos, OptionalLabel label)
{
    pos = to_valid_idx(pos, num_legs);
    auto const p = static_cast<std::size_t>(pos);
    for (std::size_t i = 0; i < p; ++i) {
        if (_labels[i] == label) {
            throw std::invalid_argument("Duplicate label");
        }
    }
    for (std::size_t i = p + 1; i < _labels.size(); ++i) {
        if (_labels[i] == label) {
            throw std::invalid_argument("Duplicate label");
        }
    }
    if (_labels[p]) {
        _labelmap.erase(*_labels[p]);
    }
    _labels[p] = label;
    if (label) {
        _labelmap[*label] = pos;
    }
    return *this;
}

LabelledLegs&
LabelledLegs::set_labels(OptionalLabels labels)
{
    auto const dups = duplicate_leg_labels(labels);
    if (!dups.empty()) {
        std::string joined;
        bool first = true;
        for (auto const& d : dups) {
            if (!first) {
                joined += ", ";
            }
            first = false;
            joined += d;
        }
        throw std::invalid_argument(std::format("Duplicate leg labels: {}", joined));
    }
    if (static_cast<int64>(labels.size()) != num_legs) {
        throw std::invalid_argument(
          std::format("expected {} labels, got {}", num_legs, labels.size()));
    }
    _labels = std::move(labels);
    _labelmap = build_labelmap(_labels);
    return *this;
}

} // namespace cyten

// =============================================================================
// ORPHANED PYTHON COMMENT HINTS (no matching C++ function body found)
// =============================================================================
// --- DiagonalTensor.__rsub__ ---
// other - self
// --- DiagonalTensor.__sub__ ---
// other - self
// --- Mask.__and__ ---
// ``self & other``
// --- Mask.__eq__ ---
// ``self == other``
// --- Mask.__invert__ ---
// ``~self``
// --- Mask.__ne__ ---
// ``self != other``
// --- Mask.__or__ ---
// ``self | other``
// --- Mask.__rand__ ---
// ``other & self``
// --- Mask.__ror__ ---
// ``other | self``
// --- Mask.__rxor__ ---
// ``other ^ self``
// --- Mask.__xor__ ---
// ``self ^ other``
// --- _elementwise_function ---
// kwargs take precedence over func_kwargs
// =============================================================================
