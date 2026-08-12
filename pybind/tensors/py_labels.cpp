#include <cyten/tensors/labels.h>

#include "../py_cyten_pybind11.h"

#include <variant>
#include <vector>

namespace cyten {

namespace {

LegLabel
as_leg_label(py::handle obj)
{
    if (obj.is_none()) {
        return std::nullopt;
    }
    return obj.cast<std::string>();
}

std::vector<int64>
py_get_leg_idcs(LabelledLegs const& self, py::object idcs)
{
    if (py::isinstance<py::str>(idcs)) {
        return self.get_leg_idcs(idcs.cast<std::string>());
    }
    if (py::isinstance<py::int_>(idcs)) {
        return self.get_leg_idcs(idcs.cast<int64>());
    }
    std::vector<std::variant<int64, std::string>> parsed;
    for (auto item : py::reinterpret_borrow<py::iterable>(to_iterable(idcs))) {
        if (py::isinstance<py::str>(item)) {
            parsed.emplace_back(item.cast<std::string>());
        } else {
            parsed.emplace_back(item.cast<int64>());
        }
    }
    return self.get_leg_idcs(parsed);
}

std::vector<std::string>
args_as_strings(py::args const& args)
{
    std::vector<std::string> out;
    out.reserve(args.size());
    for (auto a : args) {
        out.push_back(a.cast<std::string>());
    }
    return out;
}

} // namespace

void
bind_tensors_labels(py::module_& m)
{
    m.attr("CONTRACT_SYMBOL") = CONTRACT_SYMBOL;
    m.attr("LEG_SELECT_SYMBOL") = LEG_SELECT_SYMBOL;
    m.attr("OPEN_LEG_SYMBOL") = OPEN_LEG_SYMBOL;
    {
        py::list forbidden;
        for (char const* c : FORBIDDEN_LEG_LABEL_CHARS) {
            forbidden.append(c);
        }
        m.attr("FORBIDDEN_LEG_LABEL_CHARS") = forbidden;
    }

    m.def(
      "is_valid_leg_label",
      [](py::object label) {
          if (label.is_none()) {
              return true;
          }
          if (!py::isinstance<py::str>(label)) {
              return false;
          }
          return is_valid_leg_label(LegLabel{ label.cast<std::string>() });
      },
      py::arg("label"),
      R"pydoc(If the given string is a valid leg label.)pydoc");

    m.def("_combine_leg_labels",
          &_combine_leg_labels,
          py::arg("labels"),
          py::arg("offset") = 0,
          R"pydoc(The label that a combined leg should have)pydoc");

    m.def(
      "_split_leg_label",
      [](py::object label, py::object num) {
          std::optional<int64> n;
          if (!num.is_none()) {
              n = num.cast<int64>();
          }
          return _split_leg_label(as_leg_label(label), n);
      },
      py::arg("label"),
      py::arg("num") = py::none(),
      R"pydoc(Undo _combine_leg_labels, i.e. recover the original labels)pydoc");

    m.def(
      "_dual_leg_label",
      [](py::object label) { return _dual_leg_label(as_leg_label(label)); },
      py::arg("label"),
      R"pydoc(The label that a leg should have after conjugation)pydoc");

    m.def("_dual_label_list",
          &_dual_label_list,
          py::arg("labels"),
          R"pydoc(Dual labels in reversed order.)pydoc");

    m.def(
      "_get_matching_labels",
      [](LegLabels const& labels1, LegLabels const& labels2, int64 /*stacklevel*/) {
          return _get_matching_labels(labels1, labels2);
      },
      py::arg("labels1"),
      py::arg("labels2"),
      py::arg("stacklevel") = 1,
      R"pydoc(
Utility function to combine two lists of labels that should match.

Per pair of labels::
    - If one is ``None``, use the other.
    - If they are equal, use that label.
    - If they are different, emit DEBUG message to the logger and choose ``None``.
      ``stacklevel=1`` refers to the line that calls this function. Increment to skip to
      higher frames.
)pydoc");

    py::class_<LabelledLegs, py::smart_holder> labelled_legs(m, "LabelledLegs");
    labelled_legs.doc() = R"pydoc(Base class that implements handling of labelled legs.)pydoc";

    labelled_legs.def(py::init<LegLabels>(), py::arg("labels"))
      .def_readwrite("num_legs", &LabelledLegs::num_legs)
      .def_property_readonly("is_fully_labelled", &LabelledLegs::is_fully_labelled)
      .def_property(
        "labels",
        &LabelledLegs::labels,
        [](LabelledLegs& self, LegLabels labels) { self.set_labels(std::move(labels)); },
        R"pydoc(
The labels that refer to the :attr:`legs`.

Thus, ``labels[:K]`` are the ``codomain_labels`` and ``labels[K:][::-1]`` are the
``domain_labels`` where ``K == num_codomain_legs``.
)pydoc")
      // Python free functions often access ``_labels``; keep as alias until those are converted.
      .def_property(
        "_labels",
        &LabelledLegs::labels,
        [](LabelledLegs& self, LegLabels labels) { self.set_labels(std::move(labels)); })
      .def_property_readonly("_labelmap",
                             [](LabelledLegs const& self) {
                                 // Expose the C++ label→index map to Python (used by free
                                 // functions / tests).
                                 py::dict out;
                                 for (auto const& [lab, idx] : self.labelmap()) {
                                     out[py::cast(lab)] = idx;
                                 }
                                 return out;
                             })
      .def("test_sanity", &LabelledLegs::test_sanity, R"pydoc(Perform sanity checks.)pydoc")
      .def("get_leg_idcs",
           &py_get_leg_idcs,
           py::arg("idcs"),
           R"pydoc(Parse leg-idcs of leg-labels to leg-idcs (i.e. indices of :attr:`legs`).)pydoc")
      .def(
        "has_label",
        [](LabelledLegs const& self, py::args args) {
            if (args.size() == 0) {
                throw py::value_error("has_label() requires at least one label");
            }
            return self.has_label(args_as_strings(args));
        },
        R"pydoc(True if all given labels are present.)pydoc")
      .def(
        "labels_are",
        [](LabelledLegs const& self, py::args args) {
            return self.labels_are(args_as_strings(args));
        },
        R"pydoc(If the given labels and the :attr:`labels` are the same, up to permutation.)pydoc")
      .def(
        "relabel",
        [](LabelledLegs& self, std::map<std::string, std::string> const& mapping)
          -> LabelledLegs& { return self.relabel(mapping); },
        py::arg("mapping"),
        py::return_value_policy::reference,
        R"pydoc(Apply mapping to labels. In-place.)pydoc")
      .def(
        "set_label",
        [](LabelledLegs& self, int64 pos, py::object label) -> LabelledLegs& {
            return self.set_label(pos, as_leg_label(label));
        },
        py::arg("pos"),
        py::arg("label"),
        py::return_value_policy::reference,
        R"pydoc(Set a single label at given position, in-place. Return the modified instance.)pydoc")
      .def(
        "set_labels",
        [](LabelledLegs& self, LegLabels labels) -> LabelledLegs& {
            return self.set_labels(std::move(labels));
        },
        py::arg("labels"),
        py::return_value_policy::reference,
        R"pydoc(Set the given labels, in-place. Return the modified instance.)pydoc");
}

} // namespace cyten
