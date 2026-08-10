#include "py_cyten_pybind11.h"

#include "symmetries/casters.hpp"
#include "symmetries/py_trampolines.hpp"

#include <cyten/symmetries/symmetry.h>
#include <cyten/symmetries/symmetry_factor.h>

#include <optional>
#include <string>

namespace cyten {

void
bind_symmetry_factor(py::module_& m)
{
    py::class_<SymmetryFactor, BaseSymmetry, PySymmetryFactor, py::smart_holder> cls(
      m,
      "SymmetryFactor",
      R"pydoc(
      Base class for symmetries that impose a block-structure on tensors

      Attributes
      ----------
      can_be_dropped: bool
          If the symmetry could be dropped to :class:`NoSymmetry` while preserving the structure.
          This is e.g. the case for group symmetries.
          This means that there is a well-defined notion of a basis of graded vector spaces and of
          dense array representations of symmetric Tensor. See notes below.
      trivial_sector: Sector
          The trivial sector of the symmetry.
          For a group this is the "symmetric" sector, where the group acts trivially.
          For a general category, this is the monoidal unit.
      group_name: str
          A readable name for the symmetry, purely as a mathematical structure, e.g. ``'U(1)'``.
      descriptive_name: str | None
          Optionally, an additional name for the group, indicating e.g. how it arises.
          Could be e.g. ``'Sz'`` for the U(1) symmetry that conserves magnetization.
      num_sectors: int | float
          The number of sectors of the symmetry. An integer if finite, otherwise ``float('inf')``.
      sector_ind_len : int
          Valid sectors are numpy arrays with shape ``(sector_ind_len,)``.
      empty_sector_array : 2D ndarray
          A SectorArray with no sectors, shape ``(0, sector_ind_len)``.
      has_complex_topological_data : bool
          If any of the topological data (F, R, C, B symbols, twist) for any sectors is complex.
          If so, tensors with that symmetry must have a complex dtype (except DiagonalTensor or Mask),
          since real blocks become complex under leg manipulations.
          Note: for a group (and for fermions), the topo data must be real if the fusion tensors
          are real. This is because the associator, the braid, and the cup are all real for groups.

      Notes
      -----
      Some symmetries can be dropped to :class:`NoSymmetry`, see :attr:`can_be_dropped`.
      It implies that all operations that may be carried out on symmetric objects have a corresponding
      operation on a non-symmetric counterpart. For example, a symmetric space :math:`A` has a
      corresponding space :math:`\mathbb{C}^n_A`, without further structure.
      It "corresponds" to :math:`A` in the sense that it has the same properties, e.g. same dimension,
      and that there are compatible operations (tensor product, direct sum, ...) such that::

          symmetric :math:`A`  -------- (operation) --->   symmetric :math:`B`
                  |                                                 |
               (drop symm)                                       (drop symm)
                  |                                                 |
                  v                                                 v
          :math:`\mathbb{C}^{n_A}`  --- (operation) --->   :math:`\mathbb{C}^{n_B}`

      commutes.
      The same goes for tensors, i.e. for symmetric tensors there are corresponding non-symmetric
      tensors which we may manipulate instead. This means that if *and only if* the symmetry has this
      property does it make sense to convert between symmetric tensors and e.g. numpy arrays, which we can
      think of as tensors with :class:`NoSymmetry`. Additionally, the concept of a basis only makes
      sense in exactly these cases.
      )pydoc");

    cls.def(py::init<FusionStyle,
                     BraidingStyle,
                     Sector,
                     std::string,
                     float64,
                     bool,
                     std::optional<std::string>,
                     bool>(),
            py::arg("fusion_style"),
            py::arg("braiding_style"),
            py::arg("trivial_sector"),
            py::arg("group_name"),
            py::arg("num_sectors"),
            py::arg("has_complex_topological_data"),
            py::arg("descriptive_name") = py::none(),
            py::arg("trivial_shift") = true);

    cls.def_readwrite("group_name", &SymmetryFactor::group_name)
      .def_readwrite("descriptive_name", &SymmetryFactor::descriptive_name)
      .def_readwrite("fusion_tensor_dtype", &SymmetryFactor::fusion_tensor_dtype);

    cls
      .def(
        "is_valid_sector",
        [](SymmetryFactor const& self, py::object a) {
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
        Whether `a` is a valid sector of this symmetry
        )pydoc")
      .def(
        "are_valid_sectors",
        [](SymmetryFactor const& self, py::object sectors) {
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
      .def("fusion_outcomes",
           &SymmetryFactor::fusion_outcomes,
           py::arg("a"),
           py::arg("b"),
           R"pydoc(
           Returns all outcomes for the fusion of sectors

           Each sector appears only once, regardless of its multiplicity (given by n_symbol) in the fusion
           )pydoc")
      .def("__repr__", &SymmetryFactor::repr)
      .def(
        "is_equivalent_to",
        [](SymmetryFactor& self, py::object other) {
            if (py::isinstance<Symmetry>(other)) {
                return other.attr("is_equivalent_to")(py::cast(self)).cast<bool>();
            }
            return self._is_equivalent_factor(other.cast<SymmetryFactor const&>());
        },
        py::arg("other"))
      .def("_is_equivalent_factor",
           &SymmetryFactor::_is_equivalent_factor,
           py::arg("other"),
           R"pydoc(
           Whether self and other describe the same mathematical structure.

           In particular, :attr:`descriptive_name` is ignored.
           )pydoc")
      .def(
        "as_Symmetry",
        [](py::object self) {
            // Use the Python-held shared_ptr; shared_from_this fails with smart_holder
            // trampolines.
            auto ptr = self.cast<SymmetryFactor::Ptr>();
            return py::cast(std::make_shared<Symmetry>(std::vector<SymmetryFactor::Ptr>{ ptr }));
        },
        R"pydoc(
        Convert any :class:`SymmetryFactor` to a :class:`Symmetry` with that single factor.
        )pydoc")
      .def("__str__", &SymmetryFactor::str)
      .def("__mul__",
           [](py::object self, py::object other) -> py::object {
               auto self_ptr = self.cast<SymmetryFactor::Ptr>();
               if (py::isinstance<SymmetryFactor>(other)) {
                   return py::cast(std::make_shared<Symmetry>(std::vector<SymmetryFactor::Ptr>{
                     self_ptr, other.cast<SymmetryFactor::Ptr>() }));
               }
               if (py::isinstance<Symmetry>(other)) {
                   auto const& sym = other.cast<Symmetry const&>();
                   std::vector<SymmetryFactor::Ptr> factors;
                   factors.reserve(1 + sym.factors.size());
                   factors.push_back(self_ptr);
                   factors.insert(factors.end(), sym.factors.begin(), sym.factors.end());
                   return py::cast(std::make_shared<Symmetry>(std::move(factors)));
               }
               return py::reinterpret_borrow<py::object>(py::handle(Py_NotImplemented));
           })
      .def("__eq__",
           [](SymmetryFactor const& self, py::object other) {
               if (!py::isinstance<SymmetryFactor>(other)) {
                   return false;
               }
               return self.equals(other.cast<SymmetryFactor const&>());
           })
      .def(
        "save_hdf5",
        [](SymmetryFactor const& self,
           py::object saver,
           py::object h5gr,
           std::string const& subpath) {
            self.save_hdf5(saver, h5gr, subpath); // virtual dispatch
        },
        py::arg("hdf5_saver"),
        py::arg("h5gr"),
        py::arg("subpath"));
}

} // namespace cyten
