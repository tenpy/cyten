#include <cyten/tensors/planar.h>

#include <cyten/tensors/ops_legs.h>
#include <cyten/tensors/sparse.h>
#include <cyten/tensors/tensor.h>
#include <cyten/tools.h>
#include <cyten/tools/cost_polynomials.h>

#include "../py_cyten_pybind11.h"

#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace cyten {

namespace {

class PyPlanarLinearOperator
  : public PlanarLinearOperator
  , public py::trampoline_self_life_support
{
  public:
    using PlanarLinearOperator::PlanarLinearOperator;

    VectorLike::Ptr matvec(VectorLike::CPtr vec) override
    {
        PYBIND11_OVERRIDE(VectorLike::Ptr, PlanarLinearOperator, matvec, vec);
    }

    TensorPtr to_tensor(TensorBackend::Ptr backend) override
    {
        PYBIND11_OVERRIDE(TensorPtr, PlanarLinearOperator, to_tensor, backend);
    }
};

LegRef
py_as_leg_ref(py::object obj)
{
    if (py::isinstance<py::str>(obj)) {
        return obj.cast<std::string>();
    }
    return obj.cast<int64>();
}

std::vector<LegRef>
py_as_leg_refs(py::object obj)
{
    std::vector<LegRef> out;
    if (py::isinstance<py::str>(obj) || !py::isinstance<py::iterable>(obj) ||
        py::isinstance<py::dict>(obj)) {
        out.push_back(py_as_leg_ref(obj));
        return out;
    }
    for (auto item : py::reinterpret_borrow<py::iterable>(obj)) {
        out.push_back(py_as_leg_ref(py::reinterpret_borrow<py::object>(item)));
    }
    return out;
}

std::optional<std::vector<LegRef>>
py_opt_leg_refs(py::object obj)
{
    if (obj.is_none()) {
        return std::nullopt;
    }
    return py_as_leg_refs(obj);
}

py::object
py_from_tensor_or_scalar(std::variant<TensorPtr, BlockBackend::Scalar> const& v)
{
    return std::visit([](auto const& x) -> py::object { return py::cast(x); }, v);
}

py::object
eval_result_to_py(PlanarResult const& v)
{
    return py_from_tensor_or_scalar(v);
}

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

std::optional<float64>
py_opt_float(py::object obj)
{
    if (obj.is_none()) {
        return std::nullopt;
    }
    return obj.cast<float64>();
}

std::optional<int64>
py_opt_int(py::object obj)
{
    if (obj.is_none()) {
        return std::nullopt;
    }
    return obj.cast<int64>();
}

std::map<std::string, std::string>
py_relabel_map(py::object obj)
{
    if (obj.is_none()) {
        return {};
    }
    return obj.cast<std::map<std::string, std::string>>();
}

BigOPolynomial
as_polynomial(py::object obj)
{
    if (obj.is_none()) {
        return BigOPolynomial::from_str("None");
    }
    if (py::isinstance<BigOPolynomial>(obj)) {
        return obj.cast<BigOPolynomial>();
    }
    return BigOPolynomial::from_str(std::string(py::str(obj)));
}

std::vector<BigOPolynomial>
parse_dims_list(py::object dims, std::size_t /*n_labels*/)
{
    if (dims.is_none()) {
        return {};
    }
    std::vector<BigOPolynomial> out;
    for (auto item : dims) {
        out.push_back(as_polynomial(py::reinterpret_borrow<py::object>(item)));
    }
    return out;
}

std::optional<std::map<std::string, std::vector<std::string>>>
parse_dims_dict(py::object dims)
{
    if (dims.is_none()) {
        return std::nullopt;
    }
    std::map<std::string, std::vector<std::string>> out;
    for (auto item : dims.cast<py::dict>()) {
        std::vector<std::string> labels;
        for (auto lab : py::reinterpret_borrow<py::iterable>(
               py::reinterpret_borrow<py::object>(item.second))) {
            labels.push_back(py::reinterpret_borrow<py::object>(lab).cast<std::string>());
        }
        out.emplace(item.first.cast<std::string>(), std::move(labels));
    }
    return out;
}

std::vector<DiagramInstruction>
parse_definition_py(py::object obj)
{
    if (obj.is_none()) {
        return {};
    }
    if (py::isinstance<py::str>(obj)) {
        auto s = obj.cast<std::string>();
        if (s.empty()) {
            return {};
        }
        return PlanarDiagram::parse_definition(std::move(s));
    }
    std::vector<DiagramInstruction> out;
    for (auto item : obj) {
        auto seq = py::reinterpret_borrow<py::sequence>(item);
        if (seq.size() != 4) {
            throw py::value_error("Each definition entry must be a 4-tuple");
        }
        auto t1 = seq[0].cast<std::string>();
        auto l1 = seq[1].cast<std::string>();
        std::optional<std::string> t2;
        py::object t2obj = seq[2];
        if (!t2obj.is_none()) {
            t2 = t2obj.cast<std::string>();
        }
        auto l2 = seq[3].cast<std::string>();
        out.emplace_back(std::move(t1), std::move(l1), std::move(t2), std::move(l2));
    }
    return PlanarDiagram::parse_definition(std::move(out));
}

TensorPlaceholderMap
parse_tensors_py(py::object tensors, py::object dims, std::vector<std::string>* names)
{
    auto dims_opt = parse_dims_dict(dims);
    if (py::isinstance<py::dict>(tensors)) {
        TensorPlaceholderMap map;
        if (names) {
            names->clear();
        }
        for (auto item : tensors.cast<py::dict>()) {
            auto key = item.first.cast<std::string>();
            if (names) {
                names->push_back(key);
            }
            map.emplace(key, item.second.cast<TensorPlaceholder>());
        }
        return PlanarDiagram::parse_tensors(std::move(map), dims_opt, nullptr);
    }
    if (py::isinstance<py::str>(tensors)) {
        return PlanarDiagram::parse_tensors(tensors.cast<std::string>(), dims_opt, names);
    }
    throw py::type_error("Expected dict or str for tensors");
}

ContractionTree
from_nested_containers_py(py::object tree)
{
    if (!py::isinstance<py::tuple>(tree) && !py::isinstance<py::list>(tree)) {
        return ContractionTree::from_single_node(tree.cast<std::string>());
    }
    auto seq = py::reinterpret_borrow<py::sequence>(tree);
    if (seq.size() != 2) {
        throw py::value_error("Nested contraction-order containers must have length 2");
    }
    auto left = from_nested_containers_py(seq[0]);
    auto right = from_nested_containers_py(seq[1]);
    return left.fuse(right, std::nullopt);
}

ContractionTree
parse_order_py(PlanarDiagram const& self, py::object order)
{
    if (py::isinstance<py::str>(order)) {
        return self.parse_order(order.cast<std::string>());
    }
    if (py::isinstance<ContractionTree>(order)) {
        return self.parse_order(order.cast<ContractionTree>());
    }
    if (self.tensors.size() == 1) {
        auto name =
          self.tensor_names_.empty() ? self.tensors.begin()->first : self.tensor_names_.front();
        return ContractionTree::from_single_node(name);
    }
    return from_nested_containers_py(order);
}

PlanarDiagram
make_planar_diagram(py::object tensors,
                    py::object definition,
                    py::object dims,
                    py::object order,
                    bool allow)
{
    std::vector<std::string> names;
    auto tens = parse_tensors_py(tensors, dims, &names);
    auto defn = parse_definition_py(definition);
    if (tens.size() <= 1 || py::isinstance<py::str>(order)) {
        std::string order_str =
          py::isinstance<py::str>(order) ? order.cast<std::string>() : std::string("definition");
        return PlanarDiagram(std::move(tens), std::move(names), std::move(defn), order_str, allow);
    }
    if (py::isinstance<ContractionTree>(order)) {
        return PlanarDiagram(std::move(tens),
                             std::move(names),
                             std::move(defn),
                             order.cast<ContractionTree>(),
                             allow);
    }
    auto tree = from_nested_containers_py(order);
    return PlanarDiagram(
      std::move(tens), std::move(names), std::move(defn), std::move(tree), allow);
}

PlanarDiagram
with_order(PlanarDiagram diag, py::object order)
{
    if (py::isinstance<py::str>(order)) {
        return PlanarDiagram(std::move(diag.tensors),
                             std::move(diag.tensor_names_),
                             std::move(diag.definition),
                             order.cast<std::string>(),
                             diag.allow_multiple_charged_tensors);
    }
    auto tree = parse_order_py(diag, order);
    return PlanarDiagram(std::move(diag.tensors),
                         std::move(diag.tensor_names_),
                         std::move(diag.definition),
                         std::move(tree),
                         diag.allow_multiple_charged_tensors);
}

py::dict
placeholder_map_to_dict(PlanarDiagram const& self)
{
    py::dict out;
    auto names = self.tensor_names();
    if (names.empty()) {
        for (auto const& [k, v] : self.tensors) {
            out[py::cast(k)] = py::cast(v);
        }
        return out;
    }
    for (auto const& n : names) {
        out[py::cast(n)] = py::cast(self.tensors.at(n));
    }
    return out;
}

py::list
definition_to_py(std::vector<DiagramInstruction> const& defn)
{
    py::list out;
    for (auto const& [t1, l1, t2, l2] : defn) {
        py::object t2py = t2 ? py::cast(*t2) : py::none();
        out.append(py::make_tuple(t1, l1, t2py, l2));
    }
    return out;
}

py::object
evaluate_diagram(PlanarDiagram const& self, py::object tensors_obj)
{
    py::dict tensors = py::dict(std::move(tensors_obj));
    if (tensors.size() == 0) {
        return eval_result_to_py(self.evaluate(std::map<std::string, TensorPtr>{}));
    }
    py::object first;
    for (auto item : tensors) {
        first = py::reinterpret_borrow<py::object>(item.second);
        break;
    }
    if (py::isinstance<TensorPlaceholder>(first)) {
        std::map<std::string, TensorPlaceholder> map;
        for (auto item : tensors) {
            map.emplace(item.first.cast<std::string>(), item.second.cast<TensorPlaceholder>());
        }
        return py::cast(self.evaluate(std::move(map)));
    }
    std::map<std::string, TensorPtr> map;
    for (auto item : tensors) {
        map.emplace(item.first.cast<std::string>(), item.second.cast<TensorPtr>());
    }
    return eval_result_to_py(self.evaluate(std::move(map)));
}

std::optional<PipeDualities>
py_as_pipe_dualities(py::object obj)
{
    if (obj.is_none()) {
        return std::nullopt;
    }
    if (py::isinstance<py::bool_>(obj) ||
        py::isinstance(obj, py::module_::import("numpy").attr("bool_")) ||
        !py::isinstance<py::iterable>(obj) || py::isinstance<py::str>(obj)) {
        return obj.cast<bool>();
    }
    std::vector<bool> out;
    for (auto item : obj) {
        out.push_back(py::reinterpret_borrow<py::object>(item).cast<bool>());
    }
    return out;
}

std::optional<std::vector<Leg::Ptr>>
py_as_pipes(py::object obj)
{
    if (obj.is_none()) {
        return std::nullopt;
    }
    std::vector<Leg::Ptr> out;
    for (auto item : obj) {
        py::object o = py::reinterpret_borrow<py::object>(item);
        if (o.is_none()) {
            out.push_back(nullptr);
        } else {
            out.push_back(o.cast<Leg::Ptr>());
        }
    }
    return out;
}

std::vector<std::string>
labels_as_strings(py::object labels)
{
    std::vector<std::string> out;
    for (auto item : labels) {
        out.push_back(py::reinterpret_borrow<py::object>(item).cast<std::string>());
    }
    return out;
}

} // namespace

void
bind_tensors_planar(py::module_& m)
{
    py::class_<TensorPlaceholder, LabelledLegs, py::smart_holder> tensor_placeholder(
      m, "TensorPlaceholder");
    tensor_placeholder.doc() = R"pydoc(
Placeholder for a tensor used to define :class:`PlanarDiagram` s.

Attributes
----------
labels : list of str
    The labels of the tensor (up to cyclic permutation). This means that as long as we go
    clockwise around the shape, any starting point can be chosen for the labels.
dims : list of (str | None)
    For each of the legs, an optional symbol to represent its dimension.
cost_to_make : :class:`BigOPolynomial`
    Algorithmic cost of creating the tensor.
)pydoc";

    tensor_placeholder
      .def(py::init([](py::object labels, py::object dims, py::object cost_to_make) {
               auto labs = labels_as_strings(labels);
               auto d = parse_dims_list(dims, labs.size());
               BigOPolynomial cost;
               if (!cost_to_make.is_none()) {
                   cost = as_polynomial(cost_to_make);
               }
               return TensorPlaceholder(std::move(labs), std::move(d), std::move(cost));
           }),
           py::arg("labels"),
           py::arg("dims") = py::none(),
           py::arg("cost_to_make") = py::none())
      .def_readwrite("dims", &TensorPlaceholder::dims)
      .def_readwrite("cost_to_make", &TensorPlaceholder::cost_to_make)
      .def("copy", &TensorPlaceholder::copy, py::arg("deep") = true)
      .def("__repr__", &TensorPlaceholder::__repr__);

    py::class_<ContractionTreeNode, py::smart_holder> contraction_tree_node(m,
                                                                            "ContractionTreeNode");
    contraction_tree_node.doc() = R"pydoc(
Node in a :class:`ContractionTree`.

Represents a single tensor contraction in a contraction tree, where the left and
right child (if not `None`) may correspond a single tensor or contractions of
multiple tensors. The result of the represented tensor contraction can be part of
subsequent contractions represented by the parent (if not `None`).
If both children are `None`, the node only represents a tensor.

A node must not be trivial, that is, it must either represent a tensor contraction
(i.e., have a left and right child; value is optional) or have a value different
from `None` when representing a tensor.

Graphically::

    |            parent
    |              │                   parent━value
    |            value            ==            ┣━left_child
    |       ┏━━━━━━┷━━━━━━┓                     ┗━right_child
    |   left_child   right_child

The RHS above corresponds to the graphic representation of the node in the full
contraction tree, as constructed in :meth:`show_whole_tree`.

Parameters
----------
parent : :class:`ContractionTreeNode` or None
    Node representing a subsequent tensor contraction for which the result of the
    contraction represented by `self` is a left or right child.
left_child : :class:`ContractionTreeNode` or None
    Represents the left tensor to be contracted.
    May itself be the result of a tensor contraction.
    May be `None` if `self` represents a single tensor rather than a tensor
    contraction. In such a case, `right_child` must also be `None`.
right_child : :class:`ContractionTreeNode` or None
    Represents the right tensor to be contracted.
    May itself be the result of a tensor contraction.
    May be `None` if `self` represents a single tensor rather than a tensor
    contraction. In such a case, `left_child` must also be `None`.
value : str or None
    Value describing the contraction tree node.
)pydoc";

    contraction_tree_node
      .def(
        py::init(
          [](py::object parent, py::object left_child, py::object right_child, py::object value) {
              ContractionTreeNode::Ptr p;
              ContractionTreeNode::Ptr left;
              ContractionTreeNode::Ptr right;
              if (!parent.is_none()) {
                  p = parent.cast<ContractionTreeNode::Ptr>();
              }
              if (!left_child.is_none()) {
                  left = left_child.cast<ContractionTreeNode::Ptr>();
              }
              if (!right_child.is_none()) {
                  right = right_child.cast<ContractionTreeNode::Ptr>();
              }
              std::optional<std::string> val;
              if (!value.is_none()) {
                  val = value.cast<std::string>();
              }
              return std::make_shared<ContractionTreeNode>(
                std::move(p), std::move(left), std::move(right), std::move(val));
          }),
        py::arg("parent"),
        py::arg("left_child"),
        py::arg("right_child"),
        py::arg("value"))
      .def_property(
        "parent",
        [](ContractionTreeNode const& self) -> py::object {
            auto p = self.parent.lock();
            if (!p) {
                return py::none();
            }
            return py::cast(p);
        },
        [](ContractionTreeNode& self, py::object parent) {
            if (parent.is_none()) {
                self.parent.reset();
            } else {
                self.parent = parent.cast<ContractionTreeNode::Ptr>();
            }
        })
      .def_readwrite("left_child", &ContractionTreeNode::left_child)
      .def_readwrite("right_child", &ContractionTreeNode::right_child)
      .def_readwrite("value", &ContractionTreeNode::value)
      .def_property_readonly("is_leaf", &ContractionTreeNode::is_leaf)
      .def("test_sanity", &ContractionTreeNode::test_sanity, "Perform sanity checks.")
      .def("copy",
           &ContractionTreeNode::copy,
           py::arg("parent") = nullptr,
           "Implement :meth:`ContractionTree.copy` recursively.")
      .def("get_leaves", &ContractionTreeNode::get_leaves, "Returns ``leaves, num_nodes_below``.")
      .def("remove_children",
           &ContractionTreeNode::remove_children,
           "Remove both children and return their values.")
      .def("pop_contraction",
           &ContractionTreeNode::pop_contraction,
           "Implement :meth:`ContractionTree.pop_contraction` recursively.")
      .def("show_whole_tree",
           &ContractionTreeNode::show_whole_tree,
           "Return a graphic representation of the full contraction tree.");

    py::class_<ContractionTree, py::smart_holder> contraction_tree(m, "ContractionTree");
    contraction_tree.doc() = R"pydoc(
Representation of the contraction order in a :class:`PlanarDiagram` as a tree structure.

The leaf nodes represent the tensor names in a diagram and the tree structure indicates an
order of pairwise contractions.

The values of non-leaf nodes currently have no meaning and are always set to ``None``,
but may cary extra information about leg handling during a pairwise contraction in the future.

Parameters
----------
root : :class:`ContractionTreeNode`
    Node representing the root of the contraction tree, i.e., the upper-most node that does not
    have a parent.
)pydoc";

    contraction_tree.def(py::init<ContractionTreeNode::Ptr>(), py::arg("root"))
      .def_readwrite("root", &ContractionTree::root)
      .def_property_readonly("leaves", &ContractionTree::leaves)
      .def_property_readonly("num_leaves", &ContractionTree::num_leaves)
      .def_property_readonly("num_nodes", &ContractionTree::num_nodes)
      .def_property_readonly("num_inner_nodes", &ContractionTree::num_inner_nodes)
      .def("test_sanity", &ContractionTree::test_sanity, "Perform sanity checks.")
      .def("copy", &ContractionTree::copy)
      .def(
        "fuse",
        [](ContractionTree& self, ContractionTree& other, py::object value) {
            std::optional<std::string> v;
            if (!value.is_none()) {
                v = value.cast<std::string>();
            }
            return self.fuse(other, std::move(v));
        },
        py::arg("other"),
        py::arg("value") = py::none(),
        R"pydoc(
        Fuse two trees. In-place on both trees.

        Graphically::

            |                                        value
            |                                       /     \
            |       a             b                a        b
            |      / \     ,     / \      ->      / \      / \
            |    ... ...       ... ...          ... ...  ... ...

        Parameters
        ----------
        other : :class:ContractionTree
            The contraction tree that will become the right child of the resulting
            combined contraction tree; `self` becomes the left child.
        value : str or None
            The value of the new root node at which `self` and `other` are fused.
        )pydoc")
      .def("pop_contraction", &ContractionTree::pop_contraction, R"pydoc(
          Replace a bottom node (where both children are leaves) with a single leaf, in-place.

          Graphically::

              |    ...              ...
              |     |                |
              |     X       ->    new_value
              |    / \
              |   a   b

          Returns
          -------
          X : str or None
              The value at the non-leaf node that is replaced
          a, b : str or None
              The values of the leaf nodes that are removed
          new_value : str
              The value of the new leaf, conventionally ``'a @ b'``.
          )pydoc")
      .def("__str__", &ContractionTree::str)
      .def_static(
        "from_contraction_order", &ContractionTree::from_contraction_order, py::arg("order"))
      .def_static("from_single_node",
                  &ContractionTree::from_single_node,
                  py::arg("node"),
                  "Contraction tree from a single node, i.e., without any child and parent nodes.")
      .def_static(
        "from_nested_containers",
        [](py::object tree) { return from_nested_containers_py(tree); },
        py::arg("tree"));

    py::class_<PlanarDiagram, py::smart_holder> planar_diagram(m, "PlanarDiagram");
    planar_diagram.doc() = R"pydoc(
Abstract representation for the contraction of multiple tensors without any braids.

The tensors in a planar diagram are represented using placeholders that have the same leg
labels as the actual tensors for which the diagram is to be evaluated. The tensor contractions
of the planar diagram, as well as its open (non-contracted) legs, are specified using only the
leg labels of the tensors / placeholders. The full diagram must be both planar and connected,
meaning that the tensor legs must not braid with each other and that the full diagram is
contractible to a single tensor. Disconnected tensors can nevertheless be considered by
combining them with another tensor using :func:`outer` before adding them to the diagram.
When specifying the leg labels of the tensors, their order must coincide with the conventional
(counter-clockwise) leg ordering of tensors. The leg labels may however be cyclically permuted
since such permutations are planar. It is further irrelevant how the legs are distributed among
the codomain and domain (as long as the order is correct).

The contractions specified by a planar diagram can be performed for a concrete set of tensors
by using :meth:`evaluate` or directly calling the planar diagram instance with the
corresponding tensors as argument. The result is only specified up to cyclic leg permutations.

In general, optimizing the contraction order of the tensors is expensive and should be done
once during development and then hard-coded. Alternatively, a greedy optimization can be run
during the instantiation. The intended use case is to create instances of planar diagrams on
module level, such that the instantiation happens at import time.

It is possible to use a planar diagram for creating a new one by adding or removing a tensor,
see :meth:`add_tensor` and :meth:`remove_tensor`, respectively.

It is possible to create planar diagrams that contract `ChargedTensor` by adding the
corresponding charge leg labels (`'!'`) to the tensor placeholders, where a ChargedTensor
is allowed. Still, plain `SymmetricTensor` are accepted for such placeholders during evaluation,
in which case the charge leg label is ignored.
The result of a planar diagram containing open charge legs is always a `ChargedTensor`,
and any remaining open charge legs need to be contiguous after the contractions.

If multiple ChargedTensor placeholders with `'!'` label are specified
(and `allow_multiple_charged_tensors` is set to True),
one can also specify contractions between the charge legs, as is done for
regular legs, just using the `'!'` leg label.
Again, these contractions are ignored during evaluation if the
corresponding tensors are `SymmetricTensor` without charge and corresponding charge legs.
If both tensors are `ChargedTensor`, the charge leg will be contracted (and has to match!),
potentially resulting in a SymmetricTensor for the result.
This is useful e.g. for infinite MPS with non-zero charge in the unit cell,
where we contract the charge legs when applying the transfer matrix.


Parameters
----------
tensors : str or {str: TensorPlaceholder}
    Specifies the tensors in the planar diagram, each with leg labels and a unique name.
    Syntax for string input: a comma (`,`) separated list of entries, each for one tensor.
    The entry for a tensor is its name, followed by comma separated leg labels enclosed in
    brackets. Example: ``'theta[vL, p0, p1, vR], U[p0, p1, p1*, p0*]'``.
    The same format as the attribute :attr:`tensors` (dict) is accepted as well.
definition : str or list of (str, str, str | None, str)
    Specifies the planar diagram, i.e., how the `tensors` are contracted.
    Syntax for string input: a comma (`,`) separated list of instructions, each either
    a contraction or an open leg.
    Contractions are of the form ``'{tensorA}:{legA} @ {tensorB}:{legB}'``.
    Open legs are of the form ``'{tensorA}:{legA} -> {new_label}``.
    The same format as the attribute :attr:`definition` (list of tuples) is accepted as well.
dims : {str: list of str}, optional
    Specifies a symbol for the dimension of each leg, used to show or optimize the contraction
    cost in terms of a :class:`BigOPolynomial`.
    A dictionary with pairs ``{dim: labels}`` indicating that the legs with ``labels`` have
    a dimension represented by the symbol ``dim``. If given, *all* labels in the diagram should
    be assigned to a symbol. Legs with the same label must have the same dimension.
order : 'greedy' | 'optimal' | 'definition' | str | nested tuples of str | ContractionTree
    Specifies the contraction order, or how to determine it.
    If ``'greedy'`` (default) or ``'optimal'``, it is optimized via :meth:`optimize_order`.
    If ``'definition'``, it is taken from the order of the `definition`, with minimal extra
    optimizations (always do traces first and when contracting two tensors, contract all shared
    legs at once).
    If a single string, expect a comma separated list of instructions
    ``'{tensorA} @ {tensorB}'`` which indicate the order of pairwise contractions.
    If nested tuples of strings, interpret those strings as tensor names, and interpret
    the bracketing as the order of pairwise contractions, contracting innermost tuples first.
    The same format as the attribute :attr:`order` (``ContractionTree``) is accepted as well.
allow_multiple_charged_tensors : bool
    Whether multiple `ChargedTensor` are allowed to be part of the planar diagram.
    When there are multiple open charge legs, they must be contiguous after the contractions,
    such that the individual charge legs can be combined to a single one.
    When there is a specified contraction between two charge legs, this contraction must also
    be planar.
    It is allowed to evaluate a planar diagram containing tensor placeholders for
    `ChargedTensor` (placeholders containing the label `'!'`) with `SymmetricTensor`. In this
    case, the `SymmetricTensor` must have the same leg labels except for the charge leg label.
    The contraction between the charge legs is then ignored.

Attributes
----------
tensors : {str: TensorPlaceholder}
    The tensors in the planar diagram, as a dictionary from name to its placeholder, which
    stores leg labels and dims.
definition : list of (str, str, str | None, str)
    Defines the contractions in the planar diagram.
    An entry ``(t1, l1, t2, l2)`` indicates to contract leg ``l1`` of ``tensors[t1]`` with
    leg ``l2`` of ``tensors[t2]``.
    An entry ``(t1, l1, None, new_l)`` indicates that leg ``l1`` of ``tensors[t1]`` is an open
    leg of the planar diagram and should have label ``new_l`` in the result.
order : ContractionTree
    Specifies the order for the tensor contractions during :meth:`evaluate`.
open_legs : list of str
    The open legs of the planar diagram, up to cyclical permutation.
    This is such that the result of :meth:`evaluate` has these leg labels (up to cycl. perm.).
    Charge legs (``'!'``) are not included; remaining open charge legs make the result a
    :class:`~cyten.tensors.ChargedTensor`.
allow_multiple_charged_tensors : bool
    Whether multiple `ChargedTensor` are allowed to be part of the planar diagram.

Examples
--------
1. For a local two-site MPS tensor `theta` with legs ``vL, p0, p1, vR`` and a two-site operator
`op` with legs ``p0, p1, p1*, p0*``, the expectation value of `op` can be expressed as the
following planar diagram::

    exp_val_diagram = PlanarDiagram(
        tensors='theta[vL, p0, p1, vR], theta_hc[vR*, p1*, p0*, vL*], op[p0, p1, p1*, p0*]',
        definition='theta:p0 @ op:p0*, theta:p1 @ op:p1*, '
        'theta:vL @ theta_hc:vL*, theta:vR @ theta_hc:vR*, '
        'op:p0 @ theta_hc:p0*, op:p1 @ theta_hc:p1*',
        dims=dict(chi=['vR', 'vR*', 'vL', 'vL*'], d=['p0', 'p0*', 'p1', 'p1*']),
    )
    exp_val = exp_val_diagram.evaluate(dict(theta=theta, theta_hc=theta.hc, op=op))

2. For a local two-site MPS tensor `theta` with legs ``vL, p0, p1, vR`` and a two-site unitary
operator `U` with legs ``p0, p1, p1*, p0*`` that is applied to `theta` (as done in TEBD), the
updated tensor expressed as the following planar diagram::

    TEBD_diagram = PlanarDiagram(
        tensors='theta[vL, p0, p1, vR], U[p0, p1, p1*, p0*]',
        definition='theta:p0 @ U:p0*, theta:p1 @ U:p1*, theta:vL -> vL, theta:vR -> vR, U:p0 -> p0, U:p1 -> p1',
        dims=dict(chi=['vR', 'vL'], d=['p0', 'p0*', 'p1', 'p1*']),
    )
    theta_updated = TEBD_diagram.evaluate(dict(theta=theta, U=U))

3. The two examples above (`exp_val_diagram` and `TEBD_diagram`) can be related using
:meth:`add_tensor` and :meth:`remove_tensor` (note the correspondence between `op` and `U`)
as::

    TEBD_diagram2 = exp_val_diagram.remove_tensor(
        name='theta_hc',
        extra_definition='theta:vL -> vL, theta:vR -> vR, '
        'op:p0 -> p0, op:p1 -> p1',
    )
    theta_updated2 = TEBD_diagram2.evaluate(dict(theta=theta, op=U))
    assert planar_almost_equal(theta_updated, theta_updated2)

    exp_val_diagram2 = TEBD_diagram.add_tensor(
        tensor='theta_hc[vR*, p1*, p0*, vL*]'
        extra_definition='theta:vL @ theta_hc:vL*, theta:vR @ theta_hc:vR*, '
        'U:p0 @ theta_hc:p0*, U:p1 @ theta_hc:p1*',
        extra_dims='dict(chi=['vR*', 'vL*'], d=['p0*', 'p1*'])'
    )
    exp_val2 = exp_val_diagram2.evaluate(dict(theta=theta, theta_hc=theta.hc, U=op))
    assert np.isclose(exp_val, exp_val2)  # number, not a tensor

4. Contraction of a left MPS environment with the transfer matrix, where the MPS tensors may
have a charge leg::

    TM_diagram = PlanarDiagram(
        tensors='LP[vR*, vR], ket[vL, p, vR, !], bra[vR*, p*, vL*, !]',
        definition='LP:vR @ ket:vL, ket:p @ bra:p*, LP:vR* @ bra:vL*, ket:! @ bra:!, ket:vR -> vR, bra:vR* -> vR*',
        dims=dict(chi=['vR', 'vL', 'vR*', 'vL*'], d=['p', 'p*']),
        allow_multiple_charged_tensors=True,
    )
    LP = TM_diagram.evaluate(dict(LP=LP, ket=ket, bra=bra))
)pydoc";

    planar_diagram
      .def(py::init([](py::object tensors,
                       py::object definition,
                       py::object dims,
                       py::object order,
                       bool allow_multiple_charged_tensors) {
               return make_planar_diagram(
                 tensors, definition, dims, order, allow_multiple_charged_tensors);
           }),
           py::arg("tensors"),
           py::arg("definition"),
           py::arg("dims") = py::none(),
           py::arg("order") = "definition",
           py::arg("allow_multiple_charged_tensors") = false)
      .def_property(
        "tensors",
        [](PlanarDiagram const& self) { return placeholder_map_to_dict(self); },
        [](PlanarDiagram& self, py::object obj) {
            std::vector<std::string> names;
            self.tensors = parse_tensors_py(obj, py::none(), &names);
            if (!names.empty()) {
                self.tensor_names_ = std::move(names);
            }
        })
      .def_property(
        "definition",
        [](PlanarDiagram const& self) { return definition_to_py(self.definition); },
        [](PlanarDiagram& self, py::object obj) { self.definition = parse_definition_py(obj); })
      .def_readwrite("order", &PlanarDiagram::order)
      .def_readwrite("open_legs", &PlanarDiagram::open_legs)
      .def_readwrite("contraction_cost", &PlanarDiagram::contraction_cost)
      .def_property_readonly("tensor_names", &PlanarDiagram::tensor_names)
      .def_readwrite("allow_multiple_charged_tensors",
                     &PlanarDiagram::allow_multiple_charged_tensors)
      .def(
        "add_tensor",
        [](PlanarDiagram const& self,
           py::object tensor,
           py::object extra_definition,
           py::object extra_dims,
           py::object order) {
            std::vector<std::string> names;
            auto extra = parse_tensors_py(tensor, extra_dims, &names);
            auto extra_def = parse_definition_py(extra_definition);
            if (py::isinstance<py::str>(order)) {
                return self.add_tensor(
                  std::move(extra), std::move(extra_def), order.cast<std::string>());
            }
            auto tmp = self.add_tensor(std::move(extra), std::move(extra_def), "definition");
            return with_order(std::move(tmp), order);
        },
        py::arg("tensor"),
        py::arg("extra_definition"),
        py::arg("extra_dims") = py::none(),
        py::arg("order") = "definition",
        R"pydoc(
        Create a new planar diagram with an additional tensor.

        The new planar diagram arises from the old one by adding a single tensor and contracting
        (some of) its legs with open legs of the old planar diagram. It is in particular not
        possible to change tensor contractions involving two tensors of the old planar diagram.

        TODO should we allow to reference the existing diagram as a whole, instead of its
             individual tensors?

        Parameters
        ----------
        tensor : str (or {str: TensorPlaceholder})
            Same as the parameter to :class:`PlanarDiagram`, but expect only a single tensor
            to be added to the diagram.
        extra_definition : str (or list of (str, str, str | None, str))
            Same as the parameter to :class:`PlanarDiagram`.
            Should define for each leg of the new tensor whether it is an open leg or contracted
            with another leg.
            The new :attr:`definition` is given by this extra definition together with the old
            definition, except for entries that correspond to legs that were open in the original
            diagram and are now contracted with the new tensor.
        extra_dims : {str: list of str}, optional
            Same as the parameter to :class:`PlanarDiagram`, but applies only to the new `tensor`.
        order : 'greedy' | 'optimal' | 'definition' | str | nested tuples of str
            Same as the parameter to :class:`PlanarDiagram`, applies to the entire new diagram.
        )pydoc")
      .def(
        "remove_tensor",
        [](PlanarDiagram const& self,
           std::string name,
           py::object extra_definition,
           py::object order) {
            auto extra_def = parse_definition_py(extra_definition);
            if (py::isinstance<py::str>(order)) {
                return self.remove_tensor(name, std::move(extra_def), order.cast<std::string>());
            }
            auto tmp = self.remove_tensor(name, extra_def, "definition");
            return with_order(std::move(tmp), order);
        },
        py::arg("name"),
        py::arg("extra_definition") = py::list(),
        py::arg("order") = "greedy",
        R"pydoc(
        Create a new planar diagram by removing one tensor.

        The new planar diagram arises from the old one by removing a single tensor and leaving the
        legs that were previously contracted with this tensor open. It is in particular not
        possible to change any tensor contractions in the planar diagram.

        Parameters
        ----------
        name : str
            The name of the tensor to be removed.
        extra_definition : str (or list of (str, str, None, str))
            Extra instructions to be added to the :attr:`definition`. Expected to only contain
            instructions for the legs that were contracted with `name` in the old planar diagram
            and are now open legs.
            Same format as the `definition` parameter to :class:`PlanarDiagram`.
        order : 'greedy' | 'optimal' | 'definition' | str | nested tuples of str
            Same as the parameter to :class:`PlanarDiagram`, applies to the entire new diagram.
        )pydoc")
      .def(
        "evaluate",
        [](PlanarDiagram const& self, py::object tensors) {
            return evaluate_diagram(self, tensors);
        },
        py::arg("tensors"),
        "Do the contractions defined by the planar diagram for given concrete `tensors`.")
      .def("__call__",
           [](PlanarDiagram const& self, py::kwargs kwargs) {
               return evaluate_diagram(self, kwargs);
           })
      .def("optimize_order",
           &PlanarDiagram::optimize_order,
           py::arg("strategy"),
           R"pydoc(
           Find the optimal contraction order for the given planar diagram.

           TODO make it easy to print what you need to hard-code.
           TODO allow relations like ``d < w < chi``, or ``d^2 < chi`` to simplify the polynomials.
           TODO support cost as polynomials or with concrete numbers
           )pydoc")
      .def_static(
        "parse_definition",
        [](py::object definition) { return definition_to_py(parse_definition_py(definition)); },
        py::arg("definition"),
        "Parse the input format for the ``definition`` arg to :class:`PlanarDiagram`.")
      .def(
        "parse_order",
        [](PlanarDiagram const& self, py::object order) { return parse_order_py(self, order); },
        py::arg("order"),
        "Parse the input format for the ``order`` arg to :class:`PlanarDiagram`.")
      .def_static(
        "parse_tensors",
        [](py::object tensors, py::object dims) {
            std::vector<std::string> names;
            auto map = parse_tensors_py(tensors, dims, &names);
            py::dict out;
            if (names.empty()) {
                for (auto const& [k, v] : map) {
                    out[py::cast(k)] = py::cast(v);
                }
            } else {
                for (auto const& n : names) {
                    out[py::cast(n)] = py::cast(map.at(n));
                }
            }
            return out;
        },
        py::arg("tensors"),
        py::arg("dims") = py::none(),
        "Parse the input format for the ``tensors`` arg to :class:`PlanarDiagram`.")
      .def("verify_diagram", &PlanarDiagram::verify_diagram, R"pydoc(
          Verify the definition of the planar diagram. Returns the :attr:`open_legs`.

          Returns
          -------
          open_legs : list of str
              The leg labels of a result of :meth:`evaluate`.
          cost : BigOPolynomial
              The cost to contract the diagram, as a polynomial in terms of the dims.
          )pydoc");

    // `py::dynamic_attr` plus no data descriptors for `op_diagram` / `matvec_diagram`: subclasses
    // are documented to store those as *class* variables, and `self.op_diagram` in an
    // uninitialized C++ instance would otherwise hit `def_readwrite` and crash.
    py::class_<PlanarLinearOperator, LinearOperator, PyPlanarLinearOperator, py::smart_holder>
      planar_linear_operator(m, "PlanarLinearOperator", py::dynamic_attr());
    planar_linear_operator.doc() = R"pydoc(
Base class for :class:`LinearOperator`\ s defined in terms of :class:`PlanarDiagram`\ s.

.. warning ::
    Instantiating a :class:`PlanarDiagram` may be expensive if the order is optimized.
    Make sure to either hard-code the order, or make the planar diagram instance as early as
    possible, e.g., as a *class* variable of the parent class instead of during its
    ``__init__``.

Parameters
----------
op_diagram : :class:`PlanarDiagram`
    The diagram that defines the operator (without acting on a vector).
matvec_diagram : :class:`PlanarDiagram`
    The diagram that defines the action of the operator on a vector.
    Must have the same tensor names as the `op_diagram` in addition to a single tensor
    with `vec_name`.
op_tensors : {str : :class:`Tensor`}
    The concrete tensors that define the operator, see `op_diagram`.
vec_name : str
    The name of the "vector", i.e., the tensor that the linear operator acts on in the
    `matvec_diagram`.
)pydoc";

    planar_linear_operator
      .def(py::init<PlanarDiagram const&,
                    PlanarDiagram const&,
                    std::map<std::string, TensorPtr>,
                    std::string>(),
           py::arg("op_diagram"),
           py::arg("matvec_diagram"),
           py::arg("op_tensors"),
           py::arg("vec_name"))
      .def_readwrite("op_tensors", &PlanarLinearOperator::op_tensors)
      .def_readwrite("vec_name", &PlanarLinearOperator::vec_name)
      .def("matvec", &PlanarLinearOperator::matvec, py::arg("vec"))
      .def("to_tensor", &PlanarLinearOperator::to_tensor, py::arg("backend") = nullptr);

    m.def("parse_leg_bipartition",
          &parse_leg_bipartition,
          py::arg("legs"),
          py::arg("num_legs"),
          R"pydoc(
          Parse a planar bipartition of legs into two subsets.

          We view the indices on a circle with length `num_legs`, i.e., ``0`` comes after ``num_legs - 1``.
          We verify that the ``legs`` form a single contiguous subset on that circle.
          Note that "on the circle" means that it may "wrap around", e.g., ``[7, 8, 0, 1, 2]`` is
          contiguous if ``num_legs=9``.

          Parameters
          ----------
          legs : list of int
              A subset of legs, in any order. Is explicitly checked to be contiguous on the circle.
          num_legs : int
              The total number of legs, such that we look at subsets of ``range(num_legs)``.

          Returns
          -------
          legs : list of int
              The `legs`, sorted in order around the circle.
              Note that this may include a jump, e.g., ``[7, 8, 0, 1, 2]`` is sorted if ``num_legs=9``.
          other_legs : list of int
              The complementary subset, sorted in order around the circle.
              Note that this may include a jump, e.g., ``[7, 8, 0, 1, 2]`` is sorted if ``num_legs=9``.
          )pydoc");

    m.def(
      "planar_contraction",
      [](py::object tensor1,
         py::object tensor2,
         py::object legs1,
         py::object legs2,
         py::object relabel1,
         py::object relabel2) -> py::object {
          auto l1 = py_as_leg_refs(legs1);
          auto l2 = py_as_leg_refs(legs2);
          if (py::isinstance<TensorPlaceholder>(tensor1) ||
              py::isinstance<TensorPlaceholder>(tensor2)) {
              return py::cast(planar_contraction(tensor1.cast<TensorPlaceholder>(),
                                                 tensor2.cast<TensorPlaceholder>(),
                                                 std::move(l1),
                                                 std::move(l2)));
          }
          return eval_result_to_py(planar_contraction(tensor1.cast<TensorCPtr>(),
                                                      tensor2.cast<TensorCPtr>(),
                                                      std::move(l1),
                                                      std::move(l2),
                                                      py_relabel_map(relabel1),
                                                      py_relabel_map(relabel2)));
      },
      py::arg("tensor1"),
      py::arg("tensor2"),
      py::arg("legs1"),
      py::arg("legs2"),
      py::arg("relabel1") = std::map<std::string, std::string>{},
      py::arg("relabel2") = std::map<std::string, std::string>{},
      R"pydoc(
      Planar version of :func:`~cyten.tensors.tdot`.

      Here, planar means that the contraction diagram can be drawn in a plane without any braids.

      We do not make assumptions about the leg arrangement of the result.
      It is constrained by the planar requirement, but otherwise arbitrary.
      That is, it is the leg arrangement of the result of :func:`~cyten.tensors.tdot` up to
      braid-free :func:`~cyten.tensors.permute_legs`, i.e., up to arbitrary leg bendings.

      For example::

          |    ╭───╮   │   │
          |    │   4   3   2
          |    │  ┏┷━━━┷━━━┷┓
          |    │  ┃    B    ┃
          |    │  ┗━━┯━━━┯━━┛
          |    │     0   1
          |    │     │   ╰─────╮
          |    │     ╰───╮     │       ==    planar_contraction(A, B, [2, 3, 4], [1, 0, 4])
          |    ╰─────╮   │     │
          |          4   3     │
          |       ┏━━┷━━━┷━━┓  │
          |       ┃    A    ┃  │
          |       ┗┯━━━┯━━━┯┛  │
          |        0   1   2   │
          |        │   │   ╰───╯

      Parameters
      ----------
      tensor1, tensor2 : :class:`Tensor`
          The two tensors to contract.
      legs1, legs2
          Which legs to contract: ``legs1[n]`` on `tensor1` is contracted with ``legs2[n]`` on
          `tensor2`.
      relabel1, relabel2 : dict[str, str], optional
          A mapping of labels for each of the tensors. The result has labels as if the
          input tensors were relabelled accordingly before contraction.

      Returns
      -------
      Tensor given by the contraction whose legs may be cyclically permuted.

      See Also
      --------
      tdot, compose, partial_compose, apply_mask, scale_axis
      )pydoc");

    m.def(
      "planar_partial_trace",
      [](py::object tensor, py::args pairs) -> py::object {
          std::vector<std::vector<LegRef>> ps;
          ps.reserve(static_cast<std::size_t>(pairs.size()));
          for (auto item : pairs) {
              ps.push_back(py_as_leg_refs(py::reinterpret_borrow<py::object>(item)));
          }
          if (py::isinstance<TensorPlaceholder>(tensor)) {
              return py::cast(
                planar_partial_trace(tensor.cast<TensorPlaceholder>(), std::move(ps)));
          }
          return eval_result_to_py(planar_partial_trace(tensor.cast<TensorCPtr>(), std::move(ps)));
      },
      py::arg("tensor"),
      R"pydoc(
      Planar version of :func:`~cyten.tensors.partial_trace`.

      Here, planar means that the trace can be drawn as a diagram in a plane, without any braids.

      For example::

          |    ╭───╮   │   │   ╭───╮
          |    │   7   6   5   4   │
          |    │  ┏┷━━━┷━━━┷━━━┷┓  │
          |    │  ┃      A      ┃  │    ==   planar_partial_trace(A, (0, 1), (2, -1), (3, 4))
          |    │  ┗┯━━━┯━━━┯━━━┯┛  │
          |    │   0   1   2   3   │
          |    │   ╰───╯   │   ╰───╯
          |    ╰───────────╯

      Parameters
      ----------
      tensor : :class:`Tensor`
          The tensor to act on.
      *pairs : list of str or int
          A number of pairs, each describing two legs via index or via label.
          Each pair is connected, realizing a partial trace.
          By definition, we create loops between the legs in a planar way by connecting them over
          the left or right side of the tensor. If both a connecting loop over the left and the
          right side are planar, the result is independent of this choice.
          Must be compatible ``tensor.get_leg(pair[0]) == tensor.get_leg(pair[1]).dual``.

      Returns
      -------
      If all legs are traced, a python scalar.
      If legs are left open, a tensor with the same type as `tensor`.

      See Also
      --------
      partial_trace
          Non-planar partial trace which may include braiding of legs with specified levels.
      )pydoc");

    m.def(
      "planar_permute_legs",
      [](TensorCPtr T, py::object codomain, py::object domain) {
          return planar_permute_legs(
            std::move(T), py_opt_leg_refs(codomain), py_opt_leg_refs(domain));
      },
      py::arg("T"),
      py::kw_only(),
      py::arg("codomain") = py::none(),
      py::arg("domain") = py::none(),
      R"pydoc(
      Planar special case of :func:`~cyten.permute_legs`, without braids.

      It permutes the :attr:`Tensor.legs` only cyclically, and bends them to the proper codomain / domain.

      A planar permutation consists only of leg bends, either to the left or right of the tensor.
      It leaves the :attr:`cyten.Tensor.legs` unchanged up to cyclical permutation.
      It is fully specified by assigning each leg to either the new codomain or the new domain.

      Parameters
      ----------
      tensor : :class:`Tensor`
          The tensor whose legs are to be permuted.
      codomain, domain : list of {str | int}
          The legs that should be in the new (co)domain, in the correct order.
          Only one of `codomain`, `domain` is required when the other can be unambiguously inferred.
          This is the case when the specified `codomain` or `domain` contains at least one leg.
      )pydoc");

    m.def(
      "planar_combine_legs",
      [](TensorCPtr T, py::args which_legs, py::object pipe_dualities, py::object pipes) {
          std::vector<std::vector<LegRef>> groups;
          groups.reserve(static_cast<std::size_t>(which_legs.size()));
          for (auto item : which_legs) {
              groups.push_back(py_as_leg_refs(py::reinterpret_borrow<py::object>(item)));
          }
          return planar_combine_legs(std::move(T),
                                     std::move(groups),
                                     py_as_pipe_dualities(pipe_dualities),
                                     py_as_pipes(pipes));
      },
      py::arg("T"),
      py::kw_only(),
      py::arg("pipe_dualities") = false,
      py::arg("pipes") = py::none(),
      R"pydoc(
      Planar special case of :func:`~cyten.combine_legs`, without braids.

      The legs to be combined must be contiguous, but they do not need to be ordered within each of
      the groups. In the general case, the legs are bent up / down before combining. The combined leg
      is the codomain (domain) if the first leg of the group is in the codomain (domain).

      For example::

          |       ║       ║    │
          |    ╭──╨╮   ╭──╨╮   │   ╭───╮
          |    │   9   8   7   6   5   │
          |    │  ┏┷━━━┷━━━┷━━━┷━━━┷┓  │
          |    │  ┃        T        ┃  │    ==   planar_combine_legs(T, [-1, 0], [3, 4, 5], [7, 8])
          |    │  ┗┯━━━┯━━━┯━━━┯━━━┯┛  │
          |    │   0   1   2   3   4   │
          |    ╰───╯   │   │   ╰╥──┴───╯
          |            │   │    ║

      In the above example, choosing the group ``[-1, 0]`` means that the combined leg is in the
      domain, whereas it would end up in the codomain when specifying ``[0, -1]`` instead.
      Similarly, the combined leg corresponding to the group ``[3, 4, 5]`` would be in the domain
      when specifying this group as ``[5, 3, 4]`` or ``[5, 4, 3]``.

      Parameters
      ----------
      T : :class:`Tensor`
          The tensor whose legs should be combined.
      *which_legs : list of {int | str}
          One or more groups of legs to combine.
      pipe_dualities : list of bool, optional
          Can optionally specify the :attr:`LegPipe.is_dual` attribute of each resulting pipe.
          This is an arbitrary choice for each pipe.
          The pipes are formed such that ``result.legs.[pipe_idx].is_dual == pipe_dualities[i]``.
          Defaults to all ``False``.
      pipes : list of {LegPipe | None}, optional
          For each ``group = which_legs[i]`` of legs, the resulting pipe can be passed to
          avoid recomputation. If we group to the codomain (``group[0] < tensor.num_codomain_legs``),
          we expect ``LegPipe([tensor._as_codomain_leg(i) for i in group])``.
          Otherwise we expect ``LegPipe([tensor._as_domain_leg(i) for i in reversed(group)])``.
          Note the reverse order in the latter case!
          In the intended use case, when another tensor with the same legs has already been combined,
          obtain those pipes simply via :meth:`Tensor.get_leg_co_domain`.
          It is possible to pass only some of the pipes, use ``None`` as filler.

      See Also
      --------
      combine_legs
          Non-planar version that automatically braids legs in order to combine them.
      )pydoc");

    m.def(
      "planar_eigh",
      [](TensorCPtr tensor,
         int64 codomain_cut,
         int64 domain_cut,
         py::object new_labels,
         bool new_leg_dual,
         py::object sort) {
          return planar_eigh(std::move(tensor),
                             codomain_cut,
                             domain_cut,
                             py_opt_labels(new_labels),
                             new_leg_dual,
                             py_opt_string(sort));
      },
      py::arg("tensor"),
      py::arg("codomain_cut"),
      py::arg("domain_cut"),
      py::arg("new_labels") = py::none(),
      py::arg("new_leg_dual") = false,
      py::arg("sort") = py::none(),
      R"pydoc(
      Planar eigen-decomposition of a hermitian tensor.

      A :ref:`tensor decomposition <decompositions>` ``tensor ~ V @ W @ dagger(V)`` with
      the following properties:

      - ``V`` is unitary.
      - ``W`` is a :class:`DiagonalTensor` with the real eigenvalues of ``tensor``.

      This planar decomposition differs from :func:`~cyten.tensors.eigh` in the sense that
      it decomposes a tensor into more general left and right parts rather than into codomain
      and domain.

      *Assumes* that `tensor` is hermitian with respect to the legs specified by
      `codomain_cut` and `domain_cut`. If `T` is obtained from `tensor` by bending legs
      s.t. all legs on the left (right) are in the codomain (domain), or, equivalently,
      ``T = planar_permute_legs(tensor, domain=[*range(codomain_cut, tensor.num_legs - domain_cut))][::-1])``,
      then ``dagger(T) ~ T``, which requires in particular that ``T.domain == T.codomain``.

      Graphically, here with ``codomain_cut=3, domain_cut=1``::

          |                                  │    │   │   │
          |                                  │   ┏┷━━━┷━━━┷┓
          |                                  │   ┃dagger(V)┃
          |        │   │   │   │             │   ┗━┯━━━━━┯━┛
          |       ┏┷━━━┷━━━┷━━━┷┓            │   ┏━┷━┓   │
          |       ┃   tensor    ┃    ==      │   ┃ W ┃   │
          |       ┗┯━━━┯━━━┯━━━┯┛            │   ┗━┯━┛   │
          |        │   │   │   │           ┏━┷━━━━━┷━┓   │
          |                                ┃    V    ┃   │
          |                                ┗┯━━━┯━━━┯┛   │
          |                                 │   │   │    │

      Parameters
      ----------
      tensor: :class:`Tensor`
          The hermitian tensor to decompose.
      codomain_cut: int
          The first `codomain_cut` legs from the codomain end up in the codomain of `V`,
          the rest of the codomain ends up in the codomain of `dagger(V)`.
      domain_cut: int
          The first `domain_cut` legs from the domain end up in the domain of `V`, the rest
          of the domain ends up in the domain of `dagger(V)`.
      new_labels: (list of) str, optional
          The labels for the new legs can be specified in the following three ways;
          Three labels ``[a, b, c]`` result in ``V.labels[-1 - domain_cut] == a`` and
          ``W.labels == [b, c]``.
          Two labels ``[a, b]`` are equivalent to ``[a, b, a]``.
          A single label ``a`` is equivalent to ``[a, a*, a]``.
          The new legs are unlabelled by default.
      new_leg_dual: bool
          If the new leg should be a ket space (``False``) or bra space (``True``).
      sort: {'m>', 'm<', '>', '<', 'LI', 'SI', ``None``}
          How the eigenvalues should are sorted *within* each charge block.
          Defaults to ``None``, which is same as '<'. See :meth:`BlockBackend.argsort` for
          details.

      Returns
      -------
      W: :class:`DiagonalTensor`
          The real eigenvalues.
      V: :class:`SymmetricTensor`
          The orthonormal eigenvectors.

      See Also
      --------
      eigh
          Eigen decomposition with respect to codomain and domain. Corresponds to this
          function with parameters ``codomain_cut=tensor.num_codomain_legs``,
          ``domain_cut=0``.
      )pydoc");

    m.def(
      "planar_lq",
      [](TensorCPtr tensor,
         int64 codomain_cut,
         int64 domain_cut,
         py::object new_labels,
         bool new_leg_dual) {
          return planar_lq(
            std::move(tensor), codomain_cut, domain_cut, py_opt_labels(new_labels), new_leg_dual);
      },
      py::arg("tensor"),
      py::arg("codomain_cut"),
      py::arg("domain_cut"),
      py::arg("new_labels") = py::none(),
      py::arg("new_leg_dual") = false,
      R"pydoc(
      Planar LQ decomposition of a tensor.

      A :ref:`tensor decomposition <decompositions>` ``tensor ~ L @ Q`` with the following
      properties:

      - ``L`` has a lower triangular structure *in the coupled basis*.
      - ``Q`` is an isometry.

      This planar decomposition differs from :func:`~cyten.tensors.lq` in the sense that it
      decomposes a tensor into more general left and right parts rather than into codomain
      and domain.

      Graphically, here with ``codomain_cut=2, domain_cut=1``::

          |                                  │  │  │  │
          |                                  │ ┏┷━━┷━━┷┓
          |        │   │   │   │             │ ┃   Q   ┃
          |       ┏┷━━━┷━━━┷━━━┷┓            │ ┗━┯━━━┯━┛
          |       ┃   tensor    ┃    ==      │   │   │
          |       ┗━━┯━━━┯━━━┯━━┛          ┏━┷━━━┷━┓ │
          |          │   │   │             ┃   L   ┃ │
          |                                ┗━┯━━━┯━┛ │
          |                                  │   │   │

      We always compute the "reduced", a.k.a. "economic" version.

      Parameters
      ----------
      tensor: :class:`Tensor`
          The tensor to decompose.
      codomain_cut: int
          The first `codomain_cut` legs from the codomain end up in the codomain of `L`,
          the rest of the codomain ends up in the codomain of `Q`.
      domain_cut: int
          The first `domain_cut` legs from the domain end up in the domain of `L`, the rest
          of the domain ends up in the domain of `Q`.
      new_labels: (list of) str
          Labels for the new legs. Either two legs ``[a, b]`` s.t. ``L.labels[-1 - domain_cut] == a``
          and ``Q.labels[0] == b``. A single label ``a`` is equivalent to ``[a, a*]``.
      new_leg_dual: bool
          If the new leg should be a ket space (``False``) or bra space (``True``).

      See Also
      --------
      lq
          LQ decomposition with respect to codomain and domain. Corresponds to this
          function with parameters ``codomain_cut=tensor.num_codomain_legs``,
          ``domain_cut=0``.
      )pydoc");

    m.def(
      "planar_qr",
      [](TensorCPtr tensor,
         int64 codomain_cut,
         int64 domain_cut,
         py::object new_labels,
         bool new_leg_dual) {
          return planar_qr(
            std::move(tensor), codomain_cut, domain_cut, py_opt_labels(new_labels), new_leg_dual);
      },
      py::arg("tensor"),
      py::arg("codomain_cut"),
      py::arg("domain_cut"),
      py::arg("new_labels") = py::none(),
      py::arg("new_leg_dual") = false,
      R"pydoc(
      Planar QR decomposition of a tensor.

      A :ref:`tensor decomposition <decompositions>` ``tensor ~ Q @ R`` with the following
      properties:

      - ``Q`` is an isometry.
      - ``R`` has an upper triangular structure *in the coupled basis*.

      This planar decomposition differs from :func:`~cyten.tensors.qr` in the sense that it
      decomposes a tensor into more general left and right parts rather than into codomain
      and domain.

      Graphically, here with ``codomain_cut=2, domain_cut=1``::

          |                                  │  │  │  │
          |                                  │ ┏┷━━┷━━┷┓
          |        │   │   │   │             │ ┃   R   ┃
          |       ┏┷━━━┷━━━┷━━━┷┓            │ ┗━┯━━━┯━┛
          |       ┃   tensor    ┃    ==      │   │   │
          |       ┗━━┯━━━┯━━━┯━━┛          ┏━┷━━━┷━┓ │
          |          │   │   │             ┃   Q   ┃ │
          |                                ┗━┯━━━┯━┛ │
          |                                  │   │   │

      We always compute the "reduced", a.k.a. "economic" version.

      Parameters
      ----------
      tensor: :class:`Tensor`
          The tensor to decompose.
      codomain_cut: int
          The first `codomain_cut` legs from the codomain end up in the codomain of `Q`,
          the rest of the codomain ends up in the codomain of `R`.
      domain_cut: int
          The first `domain_cut` legs from the domain end up in the domain of `Q`, the rest
          of the domain ends up in the domain of `R`.
      new_labels: (list of) str
          Labels for the new legs. Either two legs ``[a, b]`` s.t. ``Q.labels[-1 - domain_cut] == a``
          and ``R.labels[0] == b``. A single label ``a`` is equivalent to ``[a, a*]``.
      new_leg_dual: bool
          If the new leg should be a ket space (``False``) or bra space (``True``).

      See Also
      --------
      qr
          QR decomposition with respect to codomain and domain. Corresponds to this
          function with parameters ``codomain_cut=tensor.num_codomain_legs``,
          ``domain_cut=0``.
      )pydoc");

    m.def(
      "planar_svd",
      [](TensorCPtr tensor,
         int64 codomain_cut,
         int64 domain_cut,
         py::object new_labels,
         bool new_leg_dual,
         py::object algorithm) {
          return planar_svd(std::move(tensor),
                            codomain_cut,
                            domain_cut,
                            py_opt_labels(new_labels),
                            new_leg_dual,
                            py_opt_string(algorithm));
      },
      py::arg("tensor"),
      py::arg("codomain_cut"),
      py::arg("domain_cut"),
      py::arg("new_labels") = py::none(),
      py::arg("new_leg_dual") = false,
      py::arg("algorithm") = py::none(),
      R"pydoc(
      Planar singular value decomposition (SVD) of a tensor.

      A :ref:`tensor decomposition <decompositions>` ``tensor ~ U @ S @ Vh`` with the following
      properties:

      - ``Vh`` and ``U`` are isometries.
      - ``S`` is a :class:`DiagonalTensor` with real, non-negative entries.
      - If `tensor` is a matrix (i.e. if it has exactly one leg each in domain and codomain), it
        reproduces the usual matrix SVD.

      .. note ::
          The basis for the newly generated leg is chosen arbitrarily, and in particular, unlike,
          e.g., :func:`numpy.linalg.svd`, it is not guaranteed that ``S.diag_numpy`` is sorted.

      This planar decomposition differs from :func:`~cyten.tensors.svd` in the sense that it
      decomposes a tensor into more general left and right parts rather than into codomain and
      domain.

      Graphically, here with ``codomain_cut=2, domain_cut=1``::

          |                                  │    │   │   │
          |                                  │   ┏┷━━━┷━━━┷┓
          |                                  │   ┃   Vh    ┃
          |        │   │   │   │             │   ┗━┯━━━━━┯━┛
          |       ┏┷━━━┷━━━┷━━━┷┓            │   ┏━┷━┓   │
          |       ┃   tensor    ┃    ==      │   ┃ S ┃   │
          |       ┗━━┯━━━┯━━━┯━━┛            │   ┗━┯━┛   │
          |          │   │   │             ┏━┷━━━━━┷━┓   │
          |                                ┃    U    ┃   │
          |                                ┗━┯━━━━━┯━┛   │
          |                                  │     │     │

      We always compute the "reduced", a.k.a. "economic" version of SVD, where the
      isometries are (in general) not full unitaries.

      Parameters
      ----------
      tensor: :class:`Tensor`
          The tensor to decompose.
      codomain_cut: int
          The first `codomain_cut` legs from the codomain end up in the codomain of `U`,
          the rest of the codomain ends up in the codomain of `Vh`.
      domain_cut: int
          The first `domain_cut` legs from the domain end up in the domain of `U`, the rest
          of the domain ends up in the domain of `Vh`.
      new_labels: (list of) str, optional
          The labels for the new legs can be specified in the following three ways;
          Four labels ``[a, b, c, d]`` result in ``U.labels[-1 - domain_cut] == a``,
          ``S.labels == [b, c]`` and ``Vh.labels[0] == d``.
          Two labels ``[a, b]`` are equivalent to ``[a, b, a, b]``.
          A single label ``a`` is equivalent to ``[a, a*, a, a*]``.
          The new legs are unlabelled by default.
      new_leg_dual: bool
          If the new leg should be a ket space (``False``) or bra space (``True``).
      algorithm: str, optional
          The algorithm (a.k.a. "driver") for the block-wise svd. Choices are backend-specific.
          See :meth:`~cyten.block_backends.BlockBackend.possible_svd_algorithms`.

      Returns
      -------
      U: SymmetricTensor
      S: DiagonalTensor
      Vh: SymmetricTensor

      See Also
      --------
      svd
          SVD decomposition with respect to codomain and domain. Corresponds to this
          function with parameters ``codomain_cut=tensor.num_codomain_legs``,
          ``domain_cut=0``.
      )pydoc");

    m.def(
      "planar_truncated_svd",
      [](TensorCPtr tensor,
         int64 codomain_cut,
         int64 domain_cut,
         py::object new_labels,
         bool new_leg_dual,
         py::object algorithm,
         py::object normalize_to,
         py::object chi_max,
         int64 chi_min,
         float64 degeneracy_tol,
         float64 trunc_cut,
         float64 svd_min) {
          return planar_truncated_svd(std::move(tensor),
                                      codomain_cut,
                                      domain_cut,
                                      py_opt_labels(new_labels),
                                      new_leg_dual,
                                      py_opt_string(algorithm),
                                      py_opt_float(normalize_to),
                                      py_opt_int(chi_max),
                                      chi_min,
                                      degeneracy_tol,
                                      trunc_cut,
                                      svd_min);
      },
      py::arg("tensor"),
      py::arg("codomain_cut"),
      py::arg("domain_cut"),
      py::arg("new_labels") = py::none(),
      py::arg("new_leg_dual") = false,
      py::arg("algorithm") = py::none(),
      py::arg("normalize_to") = py::none(),
      py::arg("chi_max") = py::none(),
      py::arg("chi_min") = 1,
      py::arg("degeneracy_tol") = 0.,
      py::arg("trunc_cut") = 0.,
      py::arg("svd_min") = 0.,
      R"pydoc(
      Truncated version of :func:`planar_svd`.

      Parameters
      ----------
      tensor, codomain_cut, domain_cut, new_labels, new_leg_dual, algorithm
          Same as for the non-truncated :func:`planar_svd`.
      normalize_to: float or None
          If ``None`` (default), the resulting singular values are not renormalized,
          resulting in an approximation in terms of ``U, S, Vh`` which has smaller norm than `a`.
          If a ``float``, the singular values are scaled such that ``norm(S) == normalize_to``.
      chi_max, chi_min, degeneracy_tol, trunc_cut, svd_min
          Options for truncations, see documentation of :func:`tensors.truncate_singular_values`.

      Returns
      -------
      U, S, Vh
          The tensors U, S, Vh that form the truncated SVD, such that
          ``U @ S @ Vh`` is *approximately* equal to `a`.
      err : float
          The relative 2-norm truncation error ``norm(a - U_S_Vh) / norm(a)``.
          This is the (relative) 2-norm weight of the discarded singular values.
      renormalize : float
          Factor, by which `S` was renormalized, i.e., ``norm(S) / norm(a)``, such that
          ``U @ S @ Vh / renormalize`` has the same norm as `a`.

      See Also
      --------
      planar_svd
          Planar SVD decomposition without truncation.
      truncated_svd
          Truncated SVD decomposition with respect to codomain and domain. Corresponds to
          this function with parameters ``codomain_cut=tensor.num_codomain_legs``,
          ``domain_cut=0``.
      )pydoc");

    m.def("planar_almost_equal",
          &planar_almost_equal,
          py::arg("tensor_1"),
          py::arg("tensor_2"),
          py::arg("rtol") = 1e-5,
          py::arg("atol") = 1e-8,
          R"pydoc(
          Checks if two tensors are equal up to numerical tolerance and planar permutation.

          We first permute the legs of `tensor_1` to the configuration of `tensor_2` and then
          compare the blocks, i.e., the free parameters of the tensors.
          The tensors count as almost equal if all block entries, i.e., all their free parameters
          individually fulfill ``abs(a1 - a2) <= atol + rtol * abs(a1)``.
          Note that this is a basis-dependent and backend-dependent notion of distance, which does
          not come from a norm in the strict mathematical sense.

          Parameters
          ----------
          tensor_1, tensor_2 : :class:`Tensor`
              The tensors to compare. The legs of both tensors need to be labelled with the same
              leg labels in order to find the planar permutation between them.
          atol, rtol : float
              Absolute and relative tolerance, see above.

          Notes
          -----
          Unlike `almost_equal`, this function does not have the argument `allow_different_types`
          since permuting legs may change the tensor type.

          See Also
          --------
          almost_equal
              Comparison between two tensors without planar permutations.
          )pydoc");

    m.def(
      "horizontal_factorization",
      [](TensorCPtr tensor,
         int64 codomain_cut,
         int64 domain_cut,
         py::object new_labels,
         py::object cutoff_singular_values) {
          return horizontal_factorization(std::move(tensor),
                                          codomain_cut,
                                          domain_cut,
                                          py_opt_labels(new_labels),
                                          py_opt_float(cutoff_singular_values));
      },
      py::arg("tensor"),
      py::arg("codomain_cut"),
      py::arg("domain_cut"),
      py::arg("new_labels") = py::none(),
      py::arg("cutoff_singular_values") = py::none(),
      R"pydoc(
      Factorize a tensor into left and right parts.

      Graphically, here with ``codomain_cut=3, domain_cut=1``::

          |      │   │   │               │           │   │             │   ╭──────╮    │   │
          |   ┏━━┷━━━┷━━━┷━━┓         ┏━━┷━━━━━━┓   ┏┷━━━┷┓         ┏━━┷━━━┷━━┓   │   ┏┷━━━┷┓
          |   ┃   tensor    ┃    =    ┃    A    ┠───┨  B  ┃   :=    ┃    A    ┃   │   ┃  B  ┃
          |   ┗┯━━━┯━━━┯━━━┯┛         ┗┯━━━┯━━━┯┛   ┗━━━━┯┛         ┗┯━━━┯━━━┯┛   │   ┗┯━━━┯┛
          |    │   │   │   │           │   │   │         │           │   │   │    ╰────╯   │

      Parameters
      ----------
      tensor: Tensor
          The tensor to factorize
      codomain_cut: int
          The first `codomain_cut` legs from the codomain end up in the codomain of `A`, the rest
          of the codomain ends up in the codomain of `B`.
      domain_cut: int
          The first `domain_cut` legs from the domain end up in the domain of `A`, the rest
          of the domain ends up in the domain of `B`.
      new_labels: (list of) str
          The labels for the new legs.
          Two entries ``[a, b]`` result in ``A.labels[-1 - domain_cut] == a`` and ``B.labels[0] == b``
          and a single entry ``a`` is equivalent to ``[a, a*]``.
      cutoff_singular_values: float, optional
          If ``None`` (default), we factorize using :func:`qr` without truncation. If given, we use a
          truncated SVD and truncate by discarding singular values below this threshold.

      Returns
      -------
      A, B: Tensor
          A factorization of the `tensor`, such that ``tdot(A, B, -1 - domain_cut, 1)`` reproduces
          the `tensor`, up to bending and possibly up to truncation if `cutoff_singular_values` is
          given.

      Notes
      -----
      This is achieved by bending legs such that we can do the factorization as a QR or SVD,
      then bend back, that is for the example case depicted above::

          |                                             │    │   │   ╭────╮         │   │   │
          |             │           │   │    ╭──╮       │ ┏━━┷━━━┷━━━┷━━┓ │         │  ┏┷━━━┷┓
          |             │  ╭────╮   │   │    │  │       │ ┃      B'     ┃ │         │  ┃  B  ┃
          |             │  │ ┏━━┷━━━┷━━━┷━━┓ │  │       │ ┗━━━━━━┯━━━━━━┛ │         │  ┗┯━━━┯┛
          |   LHS   =   │  │ ┃   tensor    ┃ │  │   =   │        │        │   =     │   │   │   =  RHS
          |             │  │ ┗┯━━━┯━━━┯━━━┯┛ │  │       │ ┏━━━━━━┷━━━━━━┓ │      ┏━━┷━━━┷━━┓│
          |             │  │  │   │   │   ╰──╯  │       │ ┃      A'     ┃ │      ┃    A    ┃│
          |             ╰──╯  │   │   │         │       │ ┗┯━━━┯━━━┯━━━┯┛ │      ┗┯━━━┯━━━┯┛│
          |                                             ╰──╯   │   │   │  │       │   │   │ │


      Note how we bend some legs to the left, to avoid any braids, such that the operation does not
      need to specify any braid chiralities.
      )pydoc");

    m.def(
      "planar_decomposition",
      [](TensorCPtr tensor,
         int64 codomain_cut,
         int64 domain_cut,
         std::string which,
         py::object new_labels,
         bool new_leg_dual,
         py::kwargs kwargs) -> py::object {
          auto labs = py_opt_labels(new_labels);
          auto kw = [&](char const* key) -> py::object {
              return kwargs.contains(key) ? py::object(kwargs[key]) : py::none();
          };
          if (which == "eigh") {
              return py::cast(planar_eigh(
                tensor, codomain_cut, domain_cut, labs, new_leg_dual, py_opt_string(kw("sort"))));
          }
          if (which == "lq") {
              return py::cast(planar_lq(tensor, codomain_cut, domain_cut, labs, new_leg_dual));
          }
          if (which == "qr") {
              return py::cast(planar_qr(tensor, codomain_cut, domain_cut, labs, new_leg_dual));
          }
          if (which == "svd") {
              return py::cast(planar_svd(tensor,
                                         codomain_cut,
                                         domain_cut,
                                         labs,
                                         new_leg_dual,
                                         py_opt_string(kw("algorithm"))));
          }
          if (which == "truncated_svd") {
              int64 chi_min = 1;
              if (!kw("chi_min").is_none()) {
                  chi_min = kw("chi_min").cast<int64>();
              }
              float64 degeneracy_tol = 0.;
              if (!kw("degeneracy_tol").is_none()) {
                  degeneracy_tol = kw("degeneracy_tol").cast<float64>();
              }
              float64 trunc_cut = 0.;
              if (!kw("trunc_cut").is_none()) {
                  trunc_cut = kw("trunc_cut").cast<float64>();
              }
              float64 svd_min = 0.;
              if (!kw("svd_min").is_none()) {
                  svd_min = kw("svd_min").cast<float64>();
              }
              return py::cast(planar_truncated_svd(tensor,
                                                   codomain_cut,
                                                   domain_cut,
                                                   labs,
                                                   new_leg_dual,
                                                   py_opt_string(kw("algorithm")),
                                                   py_opt_float(kw("normalize_to")),
                                                   py_opt_int(kw("chi_max")),
                                                   chi_min,
                                                   degeneracy_tol,
                                                   trunc_cut,
                                                   svd_min));
          }
          throw py::value_error(std::string("Invalid decomposition \"") + which + '"');
      },
      py::arg("tensor"),
      py::arg("codomain_cut"),
      py::arg("domain_cut"),
      py::arg("which"),
      py::arg("new_labels") = py::none(),
      py::arg("new_leg_dual") = false,
      R"pydoc(Planar generalization of eigen, QR, LQ, SV, and truncated SV decompositions.

See the respective docstrings of :func:`planar_eigh`, :func:`planar_qr`, :func:`planar_lq`,
:func:`planar_svd`, and :func:`planar_truncated_svd` for more details.
)pydoc");
}

} // namespace cyten
