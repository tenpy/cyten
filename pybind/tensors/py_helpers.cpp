#include <cyten/tensors/helpers.h>

#include "../py_cyten_pybind11.h"

#include <map>
#include <optional>
#include <string>

namespace cyten {

void
bind_tensors_helpers(py::module_& m)
{
    m.def(
      "_check_compatible_legs",
      &_check_compatible_legs,
      py::arg("legs1"),
      py::arg("legs2"),
      py::arg("expect_equal") = true,
      R"pydoc(Check if legs are compatible (equal if `expect_equal`, otherwise mutually dual).)pydoc");

    m.def("_compose_with_Mask",
          &_compose_with_Mask,
          py::arg("tensor"),
          py::arg("mask"),
          py::arg("leg_idx"),
          R"pydoc(
Compose `tensor` with a mask, preserving the leg order of `tensor`

We expect ``tensor.codomain[leg_idx] == mask.domain[0]`` if `leg_idx` is in the codomain, or
``tensor.domain[co_domain_idx] == mask.codomain[0]`` otherwise.

That is we have::

    |      │   │   │            │   │  ┏┷┓
    |     ┏┷━━━┷━━━┷┓           │   │  ┃M┃
    |     ┃ tensor  ┃           │   │  ┗┯┛
    |     ┗┯━━━┯━━━┯┛   OR     ┏┷━━━┷━━━┷┓
    |      │  ┏┷┓  │           ┃ tensor  ┃
    |      │  ┃M┃  │           ┗┯━━━┯━━━┯┛
    |      │  ┗┯┛  │            │   │   │

Note that the resulting leg may be smaller than before (for a projection mask in the codomain
or an inclusion mask in the domain) or larger (otherwise).

The result hast the same leg order and labels as `tensor`.
)pydoc");

    m.def(
      "_compose_SymmetricTensors",
      [](py::object tensor1, py::object tensor2, py::object relabel1, py::object relabel2) {
          std::optional<std::map<std::string, std::string>> r1;
          std::optional<std::map<std::string, std::string>> r2;
          if (!relabel1.is_none()) {
              r1 = relabel1.cast<std::map<std::string, std::string>>();
          }
          if (!relabel2.is_none()) {
              r2 = relabel2.cast<std::map<std::string, std::string>>();
          }
          return _compose_SymmetricTensors(std::move(tensor1), std::move(tensor2), r1, r2);
      },
      py::arg("tensor1"),
      py::arg("tensor2"),
      py::arg("relabel1") = py::none(),
      py::arg("relabel2") = py::none(),
      R"pydoc(
Restricted case of :func:`compose` where we assume that both tensors are SymmetricTensor.

Is used by both compose and tdot.
)pydoc");

    m.def("_convert_abelian_to_FT",
          &_convert_abelian_to_FT,
          py::arg("tensor"),
          py::arg("backend"),
          py::arg("dtype"),
          py::arg("device"),
          R"pydoc(
Convert tensor from abelian backend to FT backend. Return the data

Same idea as :func:`_convert_FT_to_abelian`, see its docstring.
)pydoc");

    m.def("_convert_FT_to_abelian",
          &_convert_FT_to_abelian,
          py::arg("tensor"),
          py::arg("backend"),
          py::arg("dtype"),
          py::arg("device"),
          R"pydoc(
Convert tensor from abelian backend to FT backend. Return the data

Notes
-----
- For abelian symmetries, a fusion tree is completely determined by its uncoupled sectors
- This means that each forest blocks consists of a single tree block
- The blocks of the abelian backend correspond one-to-one to tree blocks in the FT backend,
  up to reshaping and transposing
- All that remains is to make sure we loop over all of them in an efficient manner.
- It is convenient to do the outer loops over combinations of uncoupled sectors
    - This way, we have the abelian block_inds by construction
    - we need to compute the coupled sectors to check for valid fusion channels anyway,
      which gives us the FT block inds with one additional lookup
    - While we jump back-and-forth between different coupled sectors, and thus different FT block
      while iterating, we know that we visit the tree-blocks within each FT block *in order*,
      and we can thus keep track of where we are within each FT block easily.
)pydoc");

    m.def("_decomposition_prepare",
          &_decomposition_prepare,
          py::arg("tensor"),
          py::arg("new_leg_dual"),
          R"pydoc(Common steps to prepare a SymmetricTensor before a decomposition)pydoc");

    m.def("_decomposition_labels", &_decomposition_labels, py::arg("new_labels"));

    m.def("_svd_new_labels",
          &_svd_new_labels,
          py::arg("new_labels"),
          R"pydoc(Parse label for :func:`svd`.)pydoc");
}

} // namespace cyten
