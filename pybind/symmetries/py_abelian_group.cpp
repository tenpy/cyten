#include "py_cyten_pybind11.h"

#include "symmetries/casters.hpp"
#include "symmetries/py_trampolines.hpp"

#include <cyten/symmetries/abelian_group.h>

#include <optional>
#include <string>

namespace cyten {

void
bind_abelian_group(py::module_& m)
{
    py::class_<AbelianGroup, Group, PyAbelianGroup, py::smart_holder> cls(m,
                                                                          "AbelianGroup",
                                                                          R"pydoc(
                                                                          Base-class for abelian symmetry groups.
                                                                          )pydoc");

    cls.def(py::init<Sector, std::string, float64, std::optional<std::string>, bool>(),
            py::arg("trivial_sector"),
            py::arg("group_name"),
            py::arg("num_sectors"),
            py::arg("descriptive_name") = py::none(),
            py::arg("trivial_shift") = true);

    // Abelian defaults (subclasses may override via trampoline).
    cls.def("sector_str", &AbelianGroup::sector_str, py::arg("a"),
    R"pydoc(
    Short and readable string for the sector. Is used in __str__ of symmetry-related objects.
    )pydoc")
      .def("sector_dim", &AbelianGroup::sector_dim, py::arg("a"),
      R"pydoc(
      The dimension of a sector, as an unstructured space (i.e. if we drop the symmetry).
      
      For bosonic braiding style, e.g. for group symmetries, this coincides with the quantum
      dimension computed by :meth:`qdim`.
      For other braiding styles,
      
      See Also
      --------
      :func:`cyten.swap_gate`
          Similar method for braiding general spaces, not just single sectors.
      )pydoc")
      .def(
        "batch_sector_dim",
        [](AbelianGroup const& self, SectorArray const& a) {
            return vector_i64_to_numpy(self.batch_sector_dim(a));
        },
        py::arg("a"),
        R"pydoc(
        sector_dim of every sector (row) in a
        )pydoc")
      .def("_n_symbol", &AbelianGroup::_n_symbol, py::arg("a"), py::arg("b"), py::arg("c"),
      R"pydoc(
      Optimized version of self.n_symbol that assumes that c is a valid fusion outcome.
      
      If it is not, the results may be nonsensical. We do this for optimization purposes
      )pydoc")
      .def("_f_symbol",
           &AbelianGroup::_f_symbol,
           py::arg("a"),
           py::arg("b"),
           py::arg("c"),
           py::arg("d"),
           py::arg("e"),
           py::arg("f"),
           R"pydoc(
           Internal implementation of :meth:`f_symbol`. Can assume that inputs are valid.
           )pydoc")
      .def("frobenius_schur", &AbelianGroup::frobenius_schur, py::arg("a"),
      R"pydoc(
      The Frobenius Schur indicator of a sector.
      )pydoc")
      .def("qdim", &AbelianGroup::qdim, py::arg("a"),
      R"pydoc(
      The quantum dimension ``Tr(id_a)`` of a sector
      )pydoc")
      .def("sqrt_qdim", &AbelianGroup::sqrt_qdim, py::arg("a"),
      R"pydoc(
      The square root of the quantum dimension.
      )pydoc")
      .def("inv_sqrt_qdim", &AbelianGroup::inv_sqrt_qdim, py::arg("a"),
      R"pydoc(
      The inverse square root of the quantum dimension.
      )pydoc")
      .def("_b_symbol", &AbelianGroup::_b_symbol, py::arg("a"), py::arg("b"), py::arg("c"),
      R"pydoc(
      Internal implementation of :meth:`b_symbol`. Can assume that inputs are valid.
      )pydoc")
      .def("_r_symbol", &AbelianGroup::_r_symbol, py::arg("a"), py::arg("b"), py::arg("c"),
      R"pydoc(
      Internal implementation of :meth:`r_symbol`. Can assume that inputs are valid.
      )pydoc")
      .def("_c_symbol",
           &AbelianGroup::_c_symbol,
           py::arg("a"),
           py::arg("b"),
           py::arg("c"),
           py::arg("d"),
           py::arg("e"),
           py::arg("f"),
           R"pydoc(
           Internal implementation of :meth:`c_symbol`. Can assume that inputs are valid.
           )pydoc")
      .def("_fusion_tensor",
           &AbelianGroup::_fusion_tensor,
           py::arg("a"),
           py::arg("b"),
           py::arg("c"),
           py::arg("Z_a"),
           py::arg("Z_b"),
           R"pydoc(
           Internal implementation of :meth:`fusion_tensor`. Can assume that inputs are valid.
           )pydoc")
      .def("Z_iso", &AbelianGroup::Z_iso, py::arg("a"),
      R"pydoc(
      The Z isomorphism :math:`Z_{\bar{a}} : \bar{a}^* \to a`.
      
      The dual :math:`a^*` of a sector :math:`a` is another irreducible space.
      However, it may not be itself a sector. It must be isomorphic to one of the sector
      representatives though, which we call :math:`\bar{a}`.
      The Z isomorphism :math:`Z_a : a^* \to \bar{a}` is that isomorphism.
      
      We return the matrix elements
      
      .. math ::
          (Z_{\bar{a}})_{mn} = \langle m \vert Z_{\bar{a}}(\langle n \vert)
      
      where :math:`m` goes over a (dual) basis of :math:`\bar{a}` and :math:`n` over a basis of
      :math:`a`.
      
      Parameters
      ----------
      a : Sector
          Note that this is the target sector of the map, not its subscript!
      
      Returns
      -------
      The matrix elements as a [d_a, d_a] numpy array.
      )pydoc");
}

} // namespace cyten
