#pragma once

#include <cyten/cyten.h>
#include <cyten/tools.h>

#include <map>
#include <optional>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace cyten {

/// Optional string label for a tensor leg (`nullopt` when unlabeled).
using LegLabel = std::optional<std::string>;
/// Flat list of leg labels in `legs` order.
using LegLabels = std::vector<LegLabel>;

/// Reserved character to indicate contractions in `cyten.planar` diagrams.
inline constexpr char CONTRACT_SYMBOL = '@';
/// Reserved character to select a leg of a tensor in `cyten.planar` diagrams.
inline constexpr char LEG_SELECT_SYMBOL = ':';
/// Reserved characters to indicate an open leg in `cyten.planar` diagrams.
inline constexpr char const* OPEN_LEG_SYMBOL = "->";

/// Characters that are forbidden in leg labels.
inline constexpr char const* FORBIDDEN_LEG_LABEL_CHARS[] = {
    " ", "\t", "\n", "@", ":", "-", ">",
};

/// If the given string is a valid leg label.
[[nodiscard]] bool is_valid_leg_label(LegLabel const& label);

/// The label that a combined leg should have.
///
/// @param labels Labels of the legs being combined.
/// @param offset Added to the index used for anonymous `?n` placeholders.
///     Defaults to `0` (needed by `_dual_leg_label` for nested combined labels).
[[nodiscard]] std::string _combine_leg_labels(LegLabels const& labels, int64 offset = 0);

/// Undo `_combine_leg_labels`, i.e. recover the original labels.
///
/// @param label Combined label, or `nullopt` for an anonymous group.
/// @param num Required length when `label` is `nullopt`; otherwise optional.
[[nodiscard]] LegLabels _split_leg_label(LegLabel const& label,
                                         std::optional<int64> num = std::nullopt);

/// The label that a leg should have after conjugation.
[[nodiscard]] LegLabel _dual_leg_label(LegLabel const& label);

/// Dual labels in reversed order (helper for conjugated combined labels).
[[nodiscard]] LegLabels _dual_label_list(LegLabels const& labels);

/// Utility function to combine two lists of labels that should match.
///
/// Per pair of labels:
/// - If one is unset (`nullopt`), use the other.
/// - If they are equal, use that label.
/// - If they are different, emit a DEBUG message to the logger and choose
///   `nullopt`.
///
/// @param labels1,labels2 Label lists that should match pairwise.
/// @returns Combined labels (length of the shorter input).
[[nodiscard]] LegLabels _get_matching_labels(LegLabels const& labels1, LegLabels const& labels2);

/// Base class that implements handling of labelled legs.
class LabelledLegs
{
  public:
    using Ptr = std::shared_ptr<LabelledLegs>;

    /// Number of legs (== `labels().size()`).
    int64 num_legs = 0;

    explicit LabelledLegs(LegLabels labels);
    virtual ~LabelledLegs() = default;

    LabelledLegs(LabelledLegs const&) = default;
    LabelledLegs(LabelledLegs&&) = default;
    LabelledLegs& operator=(LabelledLegs const&) = default;
    LabelledLegs& operator=(LabelledLegs&&) = default;

    /// Perform sanity checks.
    virtual void test_sanity() const;

    /// Whether every leg has a non-`nullopt` label.
    [[nodiscard]] bool is_fully_labelled() const;

    /// The labels that refer to the legs.
    ///
    /// Thus, `labels[:K]` are the `codomain_labels` and `labels[K:][::-1]` are the
    /// `domain_labels` where `K == num_codomain_legs`.
    ///
    /// @returns A copy of the internal label list.
    [[nodiscard]] LegLabels labels() const;

    /// Parse leg indices or leg labels to leg indices (indices of the legs).
    [[nodiscard]] std::vector<int64> get_leg_idcs(int64 idx) const;
    [[nodiscard]] std::vector<int64> get_leg_idcs(std::string const& label) const;
    [[nodiscard]] std::vector<int64> get_leg_idcs(
      std::vector<std::variant<int64, std::string>> const& idcs) const;

    /// True if all given labels are present.
    [[nodiscard]] bool has_label(std::string const& label) const;
    [[nodiscard]] bool has_label(std::vector<std::string> const& labels) const;

    /// If the given labels and `labels()` are the same, up to permutation.
    [[nodiscard]] bool labels_are(std::vector<std::string> const& labels) const;

    /// Apply mapping to labels. In-place. Returns `*this`.
    LabelledLegs& relabel(std::map<std::string, std::string> const& mapping);

    /// Set a single label at given position, in-place. Return the modified instance.
    virtual LabelledLegs& set_label(int64 pos, LegLabel label);

    /// Set the given labels, in-place. Return the modified instance.
    virtual LabelledLegs& set_labels(LegLabels labels);

    /// Label → leg-index map (excludes `nullopt` labels).
    [[nodiscard]] std::unordered_map<std::string, int64> const& labelmap() const
    {
        return _labelmap;
    }

  protected:
    LegLabels _labels;
    std::unordered_map<std::string, int64> _labelmap;
};

} // namespace cyten
