#pragma once

#include <cyten/block_backend/block_backend.h>
#include <cyten/cyten.h>
#include <cyten/tensors/charged_tensor.h>
#include <cyten/tensors/decompositions.h>
#include <cyten/tensors/diagonal_tensor.h>
#include <cyten/tensors/labels.h>
#include <cyten/tensors/ops_algebra.h>
#include <cyten/tensors/ops_legs.h>
#include <cyten/tensors/sparse.h>
#include <cyten/tensors/tensor.h>
#include <cyten/tools/cost_polynomials.h>

#include <map>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

namespace cyten {

/// One planar-diagram instruction: contraction ``(t1, l1, t2, l2)`` or open leg
/// ``(t1, l1, nullopt, new_label)``.
using DiagramInstruction =
  std::tuple<std::string, std::string, std::optional<std::string>, std::string>;

/// Result of a planar contraction that may fully reduce to a scalar.
using PlanarResult = std::variant<TensorPtr, BlockBackend::Scalar>;

/// Product of zero or more polynomials (empty product is the empty polynomial).
[[nodiscard]] BigOPolynomial product_of(std::vector<BigOPolynomial> const& polys);

/// Strip whitespace and check that `name` is valid as a tensor name or leg label.
[[nodiscard]] std::string _as_valid_name(std::string name);

/// Whether a (possibly relabelled) label refers to a ChargedTensor charge leg.
[[nodiscard]] bool _is_charge_leg_label(LegLabel const& label);

/// Raise if `actual` is not a cyclic permutation of `expected`.
void _assert_cyclic_labels(std::string const& name,
                           std::vector<std::string> const& expected,
                           std::vector<std::string> const& actual);

/// Split up text that appears as the `tensors` input to :class:`PlanarDiagram`.
[[nodiscard]] std::vector<std::pair<std::string, std::vector<std::string>>> _split_tensor_text(
  std::string const& text);

/// Parse a planar bipartition of legs into two subsets.
[[nodiscard]] std::pair<std::vector<int64>, std::vector<int64>> parse_leg_bipartition(
  std::vector<int64> const& legs,
  int64 num_legs);

/// Placeholder for a tensor used to define :class:`PlanarDiagram` s.
class TensorPlaceholder : public LabelledLegs
{
  public:
    std::vector<BigOPolynomial> dims;
    BigOPolynomial cost_to_make;

    explicit TensorPlaceholder(std::vector<std::string> labels,
                               std::vector<BigOPolynomial> dims = {},
                               BigOPolynomial cost_to_make = BigOPolynomial());

    [[nodiscard]] TensorPlaceholder copy(bool deep = true) const;
    [[nodiscard]] std::vector<std::string> string_labels() const;
    [[nodiscard]] std::string __repr__() const;
};

/// Named tensors in a :class:`PlanarDiagram` (insertion order is :attr:`tensor_names`).
using TensorPlaceholderMap = std::map<std::string, TensorPlaceholder>;

/// Placeholder labels that `tensor` must match, up to cyclic permutation.
[[nodiscard]] std::vector<std::string> _expected_labels(std::vector<std::string> const& ph_labels,
                                                        TensorCPtr tensor);
[[nodiscard]] std::vector<std::string> _expected_labels(std::vector<std::string> const& ph_labels,
                                                        TensorPlaceholder const& tensor);

/// Combine leftover charge-like labels on a placeholder into a single ``'!'``.
[[nodiscard]] TensorPlaceholder _combine_placeholder_charge_legs(TensorPlaceholder const& ph);

/// Move leftover charge legs to the domain, combine them, and wrap as a ChargedTensor.
[[nodiscard]] PlanarResult _wrap_open_charge_legs(
  TensorPtr tens,
  std::map<std::string, BlockBackend::BlockPtr> const& charged_states);

/// Combine leftover charge legs after a planar contraction.
[[nodiscard]] TensorPlaceholder _finalize_charge_legs(
  TensorPlaceholder const& tens,
  std::map<std::string, BlockBackend::BlockPtr> const& charged_states);
[[nodiscard]] PlanarResult _finalize_charge_legs(
  TensorPtr tens,
  std::map<std::string, BlockBackend::BlockPtr> const& charged_states);
[[nodiscard]] PlanarResult _finalize_charge_legs(
  PlanarResult tens,
  std::map<std::string, BlockBackend::BlockPtr> const& charged_states);

/// Node in a :class:`ContractionTree`.
class ContractionTreeNode : public std::enable_shared_from_this<ContractionTreeNode>
{
  public:
    using Ptr = std::shared_ptr<ContractionTreeNode>;

    std::weak_ptr<ContractionTreeNode> parent;
    Ptr left_child;
    Ptr right_child;
    std::optional<std::string> value;

    ContractionTreeNode(Ptr parent,
                        Ptr left_child,
                        Ptr right_child,
                        std::optional<std::string> value);

    void test_sanity() const;
    [[nodiscard]] bool is_leaf() const;
    [[nodiscard]] Ptr copy(Ptr parent = nullptr) const;
    [[nodiscard]] std::pair<std::vector<std::string>, int64> get_leaves() const;
    std::pair<std::optional<std::string>, std::optional<std::string>> remove_children();
    std::tuple<std::optional<std::string>, std::string, std::string, std::string>
    pop_contraction();
    [[nodiscard]] std::vector<std::string> _str_lines(std::string const& prefix_0 = "",
                                                      std::string const& prefix = "") const;
    [[nodiscard]] std::string show_whole_tree() const;
};

/// Representation of the contraction order in a :class:`PlanarDiagram` as a tree structure.
class ContractionTree
{
  public:
    ContractionTreeNode::Ptr root;

    explicit ContractionTree(ContractionTreeNode::Ptr root);

    void test_sanity() const;
    [[nodiscard]] std::vector<std::string> leaves() const;
    [[nodiscard]] int64 num_leaves() const;
    [[nodiscard]] int64 num_nodes() const;
    [[nodiscard]] int64 num_inner_nodes() const;

    [[nodiscard]] static ContractionTree from_contraction_order(
      std::vector<std::pair<std::string, std::string>> const& order);
    [[nodiscard]] static ContractionTree from_single_node(std::string const& node);
    [[nodiscard]] ContractionTree copy() const;
    /// Fuse two trees. In-place on both trees. `self` becomes the left child.
    [[nodiscard]] ContractionTree fuse(ContractionTree& other,
                                       std::optional<std::string> value = std::nullopt);
    std::tuple<std::optional<std::string>, std::string, std::string, std::string>
    pop_contraction();
    [[nodiscard]] std::string str() const;
};

/// Abstract representation for the contraction of multiple tensors without any braids.
class PlanarDiagram
{
  public:
    TensorPlaceholderMap tensors;
    std::vector<std::string> tensor_names_;
    std::vector<DiagramInstruction> definition;
    ContractionTree order;
    std::vector<std::string> open_legs;
    BigOPolynomial contraction_cost;
    bool allow_multiple_charged_tensors = false;

    PlanarDiagram(TensorPlaceholderMap tensors,
                  std::vector<std::string> tensor_names,
                  std::vector<DiagramInstruction> definition,
                  ContractionTree order,
                  bool allow_multiple_charged_tensors = false);

    PlanarDiagram(TensorPlaceholderMap tensors,
                  std::vector<std::string> tensor_names,
                  std::vector<DiagramInstruction> definition,
                  std::string const& order,
                  bool allow_multiple_charged_tensors = false);

    [[nodiscard]] std::vector<std::string> const& tensor_names() const { return tensor_names_; }

    [[nodiscard]] PlanarDiagram add_tensor(TensorPlaceholderMap extra_tensors,
                                           std::vector<DiagramInstruction> extra_definition,
                                           std::string const& order = "definition") const;

    [[nodiscard]] PlanarResult evaluate(std::map<std::string, TensorPtr> tensors) const;
    [[nodiscard]] TensorPlaceholder evaluate(
      std::map<std::string, TensorPlaceholder> tensors) const;

    [[nodiscard]] ContractionTree optimize_order(std::string const& strategy) const;

    [[nodiscard]] static std::vector<DiagramInstruction> parse_definition(
      std::string const& definition);
    [[nodiscard]] static std::vector<DiagramInstruction> parse_definition(
      std::vector<DiagramInstruction> definition);

    [[nodiscard]] ContractionTree parse_order(std::string const& order) const;
    [[nodiscard]] ContractionTree parse_order(ContractionTree const& order) const;

    [[nodiscard]] static TensorPlaceholderMap parse_tensors(
      std::string const& tensors,
      std::optional<std::map<std::string, std::vector<std::string>>> const& dims = std::nullopt,
      std::vector<std::string>* name_order = nullptr);
    [[nodiscard]] static TensorPlaceholderMap parse_tensors(
      TensorPlaceholderMap tensors,
      std::optional<std::map<std::string, std::vector<std::string>>> const& dims = std::nullopt,
      std::vector<std::string>* name_order = nullptr);

    [[nodiscard]] PlanarDiagram remove_tensor(
      std::string const& name,
      std::vector<DiagramInstruction> extra_definition = {},
      std::string const& order = "greedy") const;

    [[nodiscard]] std::pair<std::vector<std::string>, BigOPolynomial> verify_diagram();

    static std::map<std::string, PlanarResult>& _do_contractions(
      std::map<std::string, PlanarResult>& tensors,
      std::vector<std::tuple<std::string, std::string, std::string, std::string>> contractions,
      ContractionTree order);
    static std::map<std::string, TensorPlaceholder>& _do_contractions(
      std::map<std::string, TensorPlaceholder>& tensors,
      std::vector<std::tuple<std::string, std::string, std::string, std::string>> contractions,
      ContractionTree order);

    static void _do_traces(
      std::map<std::string, PlanarResult>& tensors,
      std::vector<std::tuple<std::string, std::string, std::string>> const& traces);
    static void _do_traces(
      std::map<std::string, TensorPlaceholder>& tensors,
      std::vector<std::tuple<std::string, std::string, std::string>> const& traces);

    [[nodiscard]] static PlanarResult _extract_result(
      std::map<std::string, PlanarResult> const& tensors,
      std::vector<std::pair<std::string, std::string>> const& open_legs);
    [[nodiscard]] static TensorPlaceholder _extract_result(
      std::map<std::string, TensorPlaceholder> const& tensors,
      std::vector<std::pair<std::string, std::string>> const& open_legs);

    [[nodiscard]] static DiagramInstruction _parse_contract_instruction(std::string const& i);
    [[nodiscard]] static DiagramInstruction _parse_open_leg_instruction(std::string const& i);
    [[nodiscard]] std::optional<int64> _find_open_leg_definition(std::string const& name,
                                                                 std::string const& leg) const;
};

/// Base class for :class:`LinearOperator` s defined in terms of :class:`PlanarDiagram` s.
class PlanarLinearOperator : public LinearOperator
{
  public:
    using Ptr = std::shared_ptr<PlanarLinearOperator>;

    PlanarDiagram op_diagram;
    PlanarDiagram matvec_diagram;
    std::map<std::string, TensorPtr> op_tensors;
    std::string vec_name;

    PlanarLinearOperator(PlanarDiagram const& op_diagram,
                         PlanarDiagram const& matvec_diagram,
                         std::map<std::string, TensorPtr> op_tensors,
                         std::string vec_name);

    [[nodiscard]] VectorLike::Ptr matvec(VectorLike::CPtr vec) override;
    [[nodiscard]] TensorPtr to_tensor(TensorBackend::Ptr backend = nullptr) override;
};

[[nodiscard]] std::tuple<TensorPtr, TensorPtr> horizontal_factorization(
  TensorCPtr tensor,
  int64 codomain_cut,
  int64 domain_cut,
  std::optional<LegLabels> new_labels = std::nullopt,
  std::optional<float64> cutoff_singular_values = std::nullopt);

[[nodiscard]] bool planar_almost_equal(TensorCPtr tensor_1,
                                       TensorCPtr tensor_2,
                                       float64 rtol = 1e-5,
                                       float64 atol = 1e-8);

[[nodiscard]] TensorPtr planar_combine_legs(
  TensorCPtr T,
  std::vector<std::vector<LegRef>> which_legs,
  std::optional<PipeDualities> pipe_dualities = std::nullopt,
  std::optional<std::vector<Leg::Ptr>> pipes = std::nullopt);

[[nodiscard]] PlanarResult planar_contraction(TensorCPtr tensor1,
                                              TensorCPtr tensor2,
                                              std::vector<LegRef> legs1,
                                              std::vector<LegRef> legs2,
                                              std::map<std::string, std::string> relabel1 = {},
                                              std::map<std::string, std::string> relabel2 = {});
[[nodiscard]] TensorPlaceholder planar_contraction(TensorPlaceholder const& tensor1,
                                                   TensorPlaceholder const& tensor2,
                                                   std::vector<LegRef> legs1,
                                                   std::vector<LegRef> legs2);

[[nodiscard]] std::tuple<DiagonalTensorPtr, TensorPtr> planar_eigh(
  TensorCPtr tensor,
  int64 codomain_cut,
  int64 domain_cut,
  std::optional<LegLabels> new_labels = std::nullopt,
  bool new_leg_dual = false,
  std::optional<std::string> sort = std::nullopt);

[[nodiscard]] std::tuple<TensorPtr, TensorPtr> planar_lq(
  TensorCPtr tensor,
  int64 codomain_cut,
  int64 domain_cut,
  std::optional<LegLabels> new_labels = std::nullopt,
  bool new_leg_dual = false);

[[nodiscard]] PlanarResult planar_partial_trace(TensorCPtr tensor,
                                                std::vector<std::vector<LegRef>> pairs);
[[nodiscard]] TensorPlaceholder planar_partial_trace(TensorPlaceholder const& tensor,
                                                     std::vector<std::vector<LegRef>> pairs);

[[nodiscard]] TensorPtr planar_permute_legs(
  TensorCPtr T,
  std::optional<std::vector<LegRef>> codomain = std::nullopt,
  std::optional<std::vector<LegRef>> domain = std::nullopt);

[[nodiscard]] std::tuple<TensorPtr, TensorPtr> planar_qr(
  TensorCPtr tensor,
  int64 codomain_cut,
  int64 domain_cut,
  std::optional<LegLabels> new_labels = std::nullopt,
  bool new_leg_dual = false);

[[nodiscard]] std::tuple<TensorPtr, DiagonalTensorPtr, TensorPtr> planar_svd(
  TensorCPtr tensor,
  int64 codomain_cut,
  int64 domain_cut,
  std::optional<LegLabels> new_labels = std::nullopt,
  bool new_leg_dual = false,
  std::optional<std::string> algorithm = std::nullopt);

[[nodiscard]] std::tuple<TensorPtr, DiagonalTensorPtr, TensorPtr, float64, float64>
planar_truncated_svd(TensorCPtr tensor,
                     int64 codomain_cut,
                     int64 domain_cut,
                     std::optional<LegLabels> new_labels = std::nullopt,
                     bool new_leg_dual = false,
                     std::optional<std::string> algorithm = std::nullopt,
                     std::optional<float64> normalize_to = std::nullopt,
                     std::optional<int64> chi_max = std::nullopt,
                     int64 chi_min = 1,
                     float64 degeneracy_tol = 0.,
                     float64 trunc_cut = 0.,
                     float64 svd_min = 0.);

[[nodiscard]] std::pair<TensorPtr, std::optional<int64>>
_planar_contraction_helper(TensorCPtr tensor, std::vector<int64> const& contr, bool domain);

} // namespace cyten
