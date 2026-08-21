#include <cyten/tensors/labels.h>

#include "../doc_plus.h"
#include "../py_cyten_pybind11.h"

#include "docstrings/tensors/labels.h"

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
      DOC(cyten, is_valid_leg_label));

    m.def("_combine_leg_labels",
          &_combine_leg_labels,
          py::arg("labels"),
          py::arg("offset") = 0,
          DOC(cyten, _combine_leg_labels));

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
      DOC(cyten, _split_leg_label));

    m.def(
      "_dual_leg_label",
      [](py::object label) { return _dual_leg_label(as_leg_label(label)); },
      py::arg("label"),
      DOC(cyten, _dual_leg_label));

    m.def("_dual_label_list",
          &_dual_label_list,
          py::arg("labels"),
          DOC(cyten, _dual_label_list));

    m.def(
      "_get_matching_labels",
      [](LegLabels const& labels1, LegLabels const& labels2, int64 /*stacklevel*/) {
          return _get_matching_labels(labels1, labels2);
      },
      py::arg("labels1"),
      py::arg("labels2"),
      py::arg("stacklevel") = 1,
      doc_plus(DOC(cyten, _get_matching_labels),
               R"pydoc(
The ``stacklevel`` argument is Python-only (accepted for API compatibility; unused).
In Python, ``None`` labels correspond to C++ ``nullopt``.
)pydoc"));

    py::class_<LabelledLegs, py::smart_holder> labelled_legs(m, "LabelledLegs");
    labelled_legs.doc() = DOC(cyten, LabelledLegs);

    labelled_legs.def(py::init<LegLabels>(), py::arg("labels"))
      .def_readwrite("num_legs", &LabelledLegs::num_legs)
      .def_property_readonly("is_fully_labelled",
                             &LabelledLegs::is_fully_labelled,
                             DOC(cyten, LabelledLegs, is_fully_labelled))
      .def_property(
        "labels",
        &LabelledLegs::labels,
        [](LabelledLegs& self, LegLabels labels) { self.set_labels(std::move(labels)); },
        DOC(cyten, LabelledLegs, labels))
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
      .def("test_sanity", &LabelledLegs::test_sanity, DOC(cyten, LabelledLegs, test_sanity))
      .def("get_leg_idcs",
           &py_get_leg_idcs,
           py::arg("idcs"),
           DOC(cyten, LabelledLegs, get_leg_idcs))
      .def(
        "has_label",
        [](LabelledLegs const& self, py::args args) {
            if (args.size() == 0) {
                throw py::value_error("has_label() requires at least one label");
            }
            return self.has_label(args_as_strings(args));
        },
        DOC(cyten, LabelledLegs, has_label))
      .def(
        "labels_are",
        [](LabelledLegs const& self, py::args args) {
            return self.labels_are(args_as_strings(args));
        },
        DOC(cyten, LabelledLegs, labels_are))
      .def(
        "relabel",
        [](LabelledLegs& self, std::map<std::string, std::string> const& mapping)
          -> LabelledLegs& { return self.relabel(mapping); },
        py::arg("mapping"),
        py::return_value_policy::reference,
        DOC(cyten, LabelledLegs, relabel))
      .def(
        "set_label",
        [](LabelledLegs& self, int64 pos, py::object label) -> LabelledLegs& {
            return self.set_label(pos, as_leg_label(label));
        },
        py::arg("pos"),
        py::arg("label"),
        py::return_value_policy::reference,
        DOC(cyten, LabelledLegs, set_label))
      .def(
        "set_labels",
        [](LabelledLegs& self, LegLabels labels) -> LabelledLegs& {
            return self.set_labels(std::move(labels));
        },
        py::arg("labels"),
        py::return_value_policy::reference,
        DOC(cyten, LabelledLegs, set_labels));
}

} // namespace cyten
