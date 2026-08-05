#include "py_cyten_pybind11.h"

#include "symmetries/casters.hpp"
#include "symmetries/py_trampolines.hpp"

#include <cyten/symmetries/symmetry.h>
#include <cyten/symmetries/symmetry_factor.h>

#include <memory>
#include <vector>

namespace cyten {

namespace {

/// Flatten a Python sequence of SymmetryFactor / Symmetry into SymmetryFactor::Ptr list.
std::vector<SymmetryFactor::Ptr>
flatten_factors_from_python(py::sequence seq)
{
    std::vector<SymmetryFactor::Ptr> out;
    for (auto item : seq) {
        if (py::isinstance<Symmetry>(item)) {
            auto const& sym = item.cast<Symmetry const&>();
            out.insert(out.end(), sym.factors.begin(), sym.factors.end());
        } else {
            out.push_back(item.cast<SymmetryFactor::Ptr>());
        }
    }
    return out;
}

} // namespace

void
bind_symmetry(py::module_& m)
{
    py::class_<Symmetry, BaseSymmetry, py::smart_holder> cls(m,
                                                             "Symmetry",
                                                             R"pydoc(
                                                             Describes a symmetry of a space or tensor.

                                                             A symmetry consists of several :attr:`factors`. For consistency, we always use this product structure,
                                                             even if there are no factors at all (trivial symmetry), or just a single factor.

                                                             The prototypical example of a symmetry comes from the (representation of) a :class:`Group`
                                                             and leads to conserved quantities. For a concrete example, we could have a :class:`U1`
                                                             that represents the :math:`S^z` conservation of a spin chain.
                                                             The framework of symmetries, however, is more general and extends to fermionic or anyonic
                                                             grading, see e.g. :class:`FermionParity` or :class:`FibonacciAnyonCategory`.

                                                             Attributes
                                                             ----------
                                                             factors : list of :class:`SymmetryFactor`
                                                                 The individual symmetries. We do not allow nesting, i.e. the `factors` can not
                                                                 be :class:`Symmetry`\ s themselves.
                                                             sector_slices : 1D ndarray
                                                                 Describes how the sectors of the `factors` are embedded in a sector of the product.
                                                                 Indicates that the slice ``sector_slices[i]:sector_slices[i + 1]`` of a sector of the
                                                                 product symmetry contains the entries of a sector of ``factors[i]``.

                                                             Parameters
                                                             ----------
                                                             factors : list of :class:`SymmetryFactor`
                                                                 The factors that comprise this symmetry. If any are already :class:`Symmetry`s, the
                                                                 nesting is flattened, i.e. ``[*others, symm]`` is translated to ``[*others, *symm.factors]``.
                                                             )pydoc");

    cls.def(py::init([](py::sequence factors) {
                return std::make_shared<Symmetry>(flatten_factors_from_python(factors));
            }),
            py::arg("factors"));

    cls
      .def_property_readonly("factors",
                             [](Symmetry const& self) {
                                 py::list out;
                                 for (auto const& f : self.factors) {
                                     out.append(py::cast(f));
                                 }
                                 return out;
                             })
      .def_property_readonly("sector_slices",
                             [](Symmetry const& self) {
                                 auto np = py::module_::import("numpy");
                                 return np.attr("array")(self.sector_slices,
                                                         py::arg("dtype") = np.attr("int64"));
                             })
      .def_readwrite("fusion_tensor_dtype", &Symmetry::fusion_tensor_dtype)
      .def_property_readonly("num_factors", &Symmetry::num_factors);

    cls.def("as_Symmetry", [](py::object self) { return self; })
      .def(
        "is_valid_sector",
        [](Symmetry const& self, py::object a) {
            if (py::isinstance<Sector>(a)) {
                return self.is_valid_sector(a.cast<Sector>());
            }
            if (!py::isinstance<py::array>(a)) {
                return false;
            }
            try {
                return self.is_valid_sector(sector_from_numpy(a));
            } catch (...) {
                return false;
            }
        },
        py::arg("a"),
        R"pydoc(
        Check if `a` is a valid sector.

        For a :class:`Symmetry`, the valid sectors are 1D integer arrays, which are "stacks" of
        valid sectors for each of the :attr:`factors`, see :attr:`sector_slices`.
        )pydoc")
      .def(
        "are_valid_sectors",
        [](Symmetry const& self, py::object sectors) {
            if (py::isinstance<SectorArray>(sectors)) {
                return self.are_valid_sectors(sectors.cast<SectorArray>());
            }
            if (!py::isinstance<py::array>(sectors)) {
                return false;
            }
            try {
                return self.are_valid_sectors(sector_array_from_numpy(sectors));
            } catch (...) {
                return false;
            }
        },
        py::arg("sectors"))
      .def("fusion_outcomes", &Symmetry::fusion_outcomes, py::arg("a"), py::arg("b"))
      .def("fusion_outcomes_broadcast",
           &Symmetry::fusion_outcomes_broadcast,
           py::arg("a"),
           py::arg("b"))
      .def(
        "has_factor",
        [](Symmetry const& self, py::object other) {
            if (py::isinstance<SymmetryFactor>(other)) {
                return self.has_factor(other.cast<SymmetryFactor const&>());
            }
            // type check: any factor isinstance of the given type
            if (py::isinstance<py::type>(other)) {
                for (auto const& f : self.factors) {
                    if (py::isinstance(py::cast(f), other)) {
                        return true;
                    }
                }
                return false;
            }
            throw py::type_error("Expected instance or subclass of SymmetryFactor.");
        },
        py::arg("other"))
      .def("dual_sector", &Symmetry::dual_sector, py::arg("a"))
      .def("dual_sectors", &Symmetry::dual_sectors, py::arg("sectors"))
      .def("_n_symbol", &Symmetry::_n_symbol, py::arg("a"), py::arg("b"), py::arg("c"))
      .def("all_sectors", &Symmetry::all_sectors)
      .def("factor_where",
           &Symmetry::factor_where,
           py::arg("descriptive_name"),
           R"pydoc(
           Return the index of the first factor with that name. Raises if not found.
           )pydoc")
      .def("qdim", &Symmetry::qdim, py::arg("a"))
      .def("sector_dim", &Symmetry::sector_dim, py::arg("a"))
      .def("batch_sector_dim", &Symmetry::batch_sector_dim, py::arg("a"))
      .def("batch_qdim", &Symmetry::batch_qdim, py::arg("a"))
      .def("sector_str", &Symmetry::sector_str, py::arg("a"))
      .def("_f_symbol",
           &Symmetry::_f_symbol,
           py::arg("a"),
           py::arg("b"),
           py::arg("c"),
           py::arg("d"),
           py::arg("e"),
           py::arg("f"))
      .def("_r_symbol", &Symmetry::_r_symbol, py::arg("a"), py::arg("b"), py::arg("c"))
      .def("_fusion_tensor",
           &Symmetry::_fusion_tensor,
           py::arg("a"),
           py::arg("b"),
           py::arg("c"),
           py::arg("Z_a") = false,
           py::arg("Z_b") = false)
      .def("swap_gate", &Symmetry::swap_gate, py::arg("a"), py::arg("b"))
      .def("Z_iso", &Symmetry::Z_iso, py::arg("a"))
      .def(
        "is_equivalent_to",
        [](Symmetry const& self, py::object other, bool strict_ordering) {
            Symmetry::Ptr other_sym;
            if (py::isinstance<Symmetry>(other)) {
                other_sym = other.cast<Symmetry::Ptr>();
            } else if (py::isinstance<SymmetryFactor>(other)) {
                other_sym = std::make_shared<Symmetry>(
                  std::vector<SymmetryFactor::Ptr>{ other.cast<SymmetryFactor::Ptr>() });
            } else {
                throw py::type_error("Expected Symmetry or SymmetryFactor");
            }
            return self.is_equivalent_to(*other_sym, strict_ordering);
        },
        py::arg("other"),
        py::arg("strict_ordering") = false,
        R"pydoc(
        If two symmetries are equivalent.

        Equivalence ignores the :attr:`SymmetryFactor.descriptive_name` of the factors.
        Ordering of the :attr:`factors` is also ignored, unless ``strict_ordering=True``.
        )pydoc")
      .def("__repr__", &Symmetry::repr)
      .def("__str__", &Symmetry::str)
      .def("__eq__",
           [](Symmetry const& self, py::object other) {
               if (!py::isinstance<Symmetry>(other)) {
                   return false;
               }
               return self.equals(other.cast<Symmetry const&>());
           })
      .def("__mul__",
           [](Symmetry const& self, py::object other) -> py::object {
               if (py::isinstance<Symmetry>(other)) {
                   return py::cast(self.mul(other.cast<Symmetry const&>()));
               }
               if (py::isinstance<SymmetryFactor>(other)) {
                   return py::cast(self.mul(other.cast<SymmetryFactor::Ptr>()));
               }
               return py::reinterpret_borrow<py::object>(py::handle(Py_NotImplemented));
           })
      .def("save_hdf5",
           &Symmetry::save_hdf5,
           py::arg("hdf5_saver"),
           py::arg("h5gr"),
           py::arg("subpath"))
      .def_static("from_hdf5",
                  &Symmetry::from_hdf5,
                  py::arg("hdf5_loader"),
                  py::arg("h5gr"),
                  py::arg("subpath"));
}

} // namespace cyten
