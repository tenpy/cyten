#include "py_cyten_pybind11.h"

#include "symmetries/casters.hpp"
#include "symmetries/py_trampolines.hpp"

#include <cyten/symmetries/base_symmetry.h>

#include <cmath>
#include <vector>

namespace cyten {

void
bind_base_symmetry(py::module_& m)
{
    py::class_<BaseSymmetry, PyBaseSymmetry, py::smart_holder> cls(m,
                                                                   "BaseSymmetry",
                                                                   R"pydoc(
                                                                   Common method implementations for both :class:`SymmetryFactor` and :class:`Symmetry`.

                                                                   This contains the fallback implementations of e.g. :meth:`qdim` in terms of F symbols.
                                                                   )pydoc");

    cls.def(py::init<FusionStyle, BraidingStyle, Sector, float64, bool, bool>(),
            py::arg("fusion_style"),
            py::arg("braiding_style"),
            py::arg("trivial_sector"),
            py::arg("num_sectors"),
            py::arg("has_complex_topological_data"),
            py::arg("trivial_shift"));

    cls.def_readonly("fusion_style", &BaseSymmetry::fusion_style)
      .def_readonly("braiding_style", &BaseSymmetry::braiding_style)
      .def_property_readonly("trivial_sector",
                             [](BaseSymmetry const& self) { return self.trivial_sector; })
      .def_property_readonly("num_sectors",
                             [](BaseSymmetry const& self) -> py::object {
                                 // Match Python ``int | float``: finite counts as int, else
                                 // float('inf').
                                 if (std::isfinite(self.num_sectors)) {
                                     return py::int_(static_cast<long long>(self.num_sectors));
                                 }
                                 return py::float_(self.num_sectors);
                             })
      .def_readonly("sector_ind_len", &BaseSymmetry::sector_ind_len)
      .def_property_readonly("empty_sector_array",
                             [](BaseSymmetry const& self) { return self.empty_sector_array; })
      .def_readonly("has_complex_topological_data", &BaseSymmetry::has_complex_topological_data)
      .def_readonly("trivial_shift", &BaseSymmetry::trivial_shift);

    cls
      .def_property_readonly("can_be_dropped",
                             &BaseSymmetry::can_be_dropped,
                             R"pydoc(
                             If the symmetry supports converting tensors to/from numpy.
                             )pydoc")
      .def_property_readonly("has_symmetric_braid", &BaseSymmetry::has_symmetric_braid)
      .def_property_readonly("has_trivial_braid", &BaseSymmetry::has_trivial_braid)
      .def_property_readonly("is_abelian",
                             &BaseSymmetry::is_abelian,
                             R"pydoc(
                             If the symmetry is Abelian.

                             An Abelian symmetry is characterized by ``FusionStyle.single``, which implies that all
                             sectors are one-dimensional.
                             Note that this does *not* imply that it is a group, as the braiding may not be bosonic!
                             )pydoc")
      .def_property_readonly("has_unique_fusion",
                             &BaseSymmetry::has_unique_fusion,
                             R"pydoc(
                             If the symmetry always has unique fusion channels, i.e. if N symbols are 0 or 1.
                             )pydoc");

    cls
      .def("dual_sector",
           &BaseSymmetry::dual_sector,
           py::arg("a"),
           R"pydoc(
           The sector dual to a, such that N^{a,dual(a)}_u = 1.

           Note that the dual space :math:`a^\star` to a sector :math:`a` may not itself be one of
           the sectors, but it must be isomorphic to one of the sectors. This method returns that
           representative :math:`\bar{a}` of the equivalence class.
           )pydoc")
      .def("_n_symbol",
           &BaseSymmetry::_n_symbol,
           py::arg("a"),
           py::arg("b"),
           py::arg("c"),
           R"pydoc(
           Optimized version of self.n_symbol that assumes that c is a valid fusion outcome.

           If it is not, the results may be nonsensical. We do this for optimization purposes
           )pydoc")
      .def("_f_symbol",
           &BaseSymmetry::_f_symbol,
           py::arg("a"),
           py::arg("b"),
           py::arg("c"),
           py::arg("d"),
           py::arg("e"),
           py::arg("f"),
           R"pydoc(
           Internal implementation of :meth:`f_symbol`. Can assume that inputs are valid.
           )pydoc")
      .def("_r_symbol",
           &BaseSymmetry::_r_symbol,
           py::arg("a"),
           py::arg("b"),
           py::arg("c"),
           R"pydoc(
           Internal implementation of :meth:`r_symbol`. Can assume that inputs are valid.
           )pydoc")
      .def("as_Symmetry", &BaseSymmetry::as_Symmetry)
      .def("is_valid_sector",
           &BaseSymmetry::is_valid_sector,
           py::arg("a"),
           R"pydoc(
           Whether `a` is a valid sector of this symmetry
           )pydoc")
      .def("fusion_outcomes",
           &BaseSymmetry::fusion_outcomes,
           py::arg("a"),
           py::arg("b"),
           R"pydoc(
           Returns all outcomes for the fusion of sectors

           Each sector appears only once, regardless of its multiplicity (given by n_symbol) in the fusion
           )pydoc")
      .def("_fusion_tensor",
           &BaseSymmetry::_fusion_tensor,
           py::arg("a"),
           py::arg("b"),
           py::arg("c"),
           py::arg("Z_a"),
           py::arg("Z_b"),
           R"pydoc(
           Internal implementation of :meth:`fusion_tensor`. Can assume that inputs are valid.
           )pydoc")
      .def("swap_gate",
           &BaseSymmetry::swap_gate,
           py::arg("a"),
           py::arg("b"),
           R"pydoc(
           The swap gate (numpy representation of the braid) of single sectors.

               |   a   b
               |   │   │
               |   v   v
               |    ╲ ╱
               |     ╲          <-  overbraid == underbraid is assumed
               |    ╱ ╲
               |   v   v
               |   │   │
               |   b   a

           Returns
           -------
           A numpy representation of the above tensor with axes ``[b, a, b*, a*]``.
           )pydoc")
      .def("Z_iso",
           &BaseSymmetry::Z_iso,
           py::arg("a"),
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
           )pydoc")
      .def("all_sectors",
           &BaseSymmetry::all_sectors,
           R"pydoc(
           Assume there are finitely many sectors, return all of them.

           .. warning ::
               Do not perform inplace operations on the output. That may invalidate caches.
           )pydoc")
      .def("n_symbol",
           &BaseSymmetry::n_symbol,
           py::arg("a"),
           py::arg("b"),
           py::arg("c"),
           R"pydoc(
           The N-symbol N^{ab}_c, i.e. how often c appears in the fusion of a and b.
           )pydoc")
      .def("f_symbol",
           &BaseSymmetry::f_symbol,
           py::arg("a"),
           py::arg("b"),
           py::arg("c"),
           py::arg("d"),
           py::arg("e"),
           py::arg("f"),
           R"pydoc(
           Coefficients :math:`[F^{abc}_d]^e_f` related to recoupling of fusion.

           The F symbol relates the following two maps::

               m1 := [a ⊗ b ⊗ c] --(1 ⊗ X_μ)--> [a ⊗ e] --(X_ν)--> d
               m2 := [a ⊗ b ⊗ c] --(X_κ ⊗ 1)--> [f ⊗ c] --(X_λ)--> d

           Such that :math:`m_1 = \sum_{f\kappa\lambda} [F^{abc}_d]^{e\mu\nu}_{f\kappa\lambda} m_2`.

           The F symbol is unitary as a matrix from indices :math:`(f\kappa\lambda)`
           to :math:`(e\mu\nu)`.

           .. warning ::
               Do not perform inplace operations on the output. That may invalidate caches.

           Parameters
           ----------
           a, b, c, d, e, f
               Sectors. Must be compatible with the fusion described above.

           Returns
           -------
           F : 4D array
               The F symbol as an array of the multiplicity indices [μ,ν,κ,λ]
           )pydoc")
      .def("b_symbol",
           &BaseSymmetry::b_symbol,
           py::arg("a"),
           py::arg("b"),
           py::arg("c"),
           R"pydoc(
           Coefficients :math:`B^{ab}_c` related to bending the right leg on a fusion tensor.

           The B symbol relates the following two maps::

               m1 := a --(1 ⊗ η_b)--> [a ⊗ b ⊗ b^*] --(X_μ ⊗ 1)--> [c ⊗ b^*]
               m2 := a --(Y_ν)--> [c ⊗ \bar{b}] --(1 ⊗ Z_b^†)--> [c ⊗ b^*]

           such that :math:`m_1 = \sum_{\nu} [B^{ab}_c]^\mu_\nu m_2`.

           The related A-symbol for bending left legs is not needed, since we always
           work with fusion trees in form

           .. warning ::
               Do not perform inplace operations on the output. That may invalidate caches.

           Parameters
           ----------
           a, b, c
               Sectors. Must be compatible with the fusion described above.

           Returns
           -------
           B : 2D array
               The B symbol as an array of the multiplicity indices [μ,ν]
           )pydoc")
      .def("r_symbol",
           &BaseSymmetry::r_symbol,
           py::arg("a"),
           py::arg("b"),
           py::arg("c"),
           R"pydoc(
           Coefficients :math:`R^{ab}_c` related to braiding on a single fusion tensor.

           The R symbol relates the following two maps::

               m1 := [b ⊗ a] --τ--> [a ⊗ b] --X_μ--> c
               m2 := [b ⊗ a] --X_ν--> c

           such that :math:`m_1 = \sum_{\nu} [R^{ab}_c]^\mu_\nu m_2`.

           We can use the unitary gauge freedom of the fusion tensors
           .. math ::

               X_μ \mapsto \sum_ν U_{μ,ν} X_ν

           to enforce that the R symbol is diagonal.

           .. warning ::
               Do not perform inplace operations on the output. That may invalidate caches.

           Parameters
           ----------
           a, b, c
               Sectors. Must be compatible with the fusion described above.

           Returns
           -------
           R : 1D array
               The diagonal entries of the R symbol as an array of the multiplicity index [μ].
           )pydoc")
      .def("c_symbol",
           &BaseSymmetry::c_symbol,
           py::arg("a"),
           py::arg("b"),
           py::arg("c"),
           py::arg("d"),
           py::arg("e"),
           py::arg("f"),
           R"pydoc(
           Coefficients :math:`[C^{abc}_d]^e_f` related to braiding on a pair of fusion tensors.

           The C symbol relates the following two maps::

               m1 := [a ⊗ c ⊗ b] --(1 ⊗ τ)--> [a ⊗ b ⊗ c] --(X_μ ⊗ 1)--> [e ⊗ c] --X_ν--> d
               m2 := [a ⊗ c ⊗ b] --(X_κ ⊗ 1)--> [f ⊗ b] --X_λ--> d

           such that :math:`m_1 = \sum_{f\kappa\lambda} C^{e\mu\nu}_{f\kappa\lambda} m_2`.

           .. warning ::
               Do not perform inplace operations on the output. That may invalidate caches.

           Parameters
           ----------
           a, b, c, d, e, f
               Sectors. Must be compatible with the fusion described above.

           Returns
           -------
           C : 4D array
               The C symbol as an array of the multiplicity indices [μ,ν,κ,λ]
           )pydoc")
      .def("fusion_tensor",
           &BaseSymmetry::fusion_tensor,
           py::arg("a"),
           py::arg("b"),
           py::arg("c"),
           py::arg("Z_a") = false,
           py::arg("Z_b") = false,
           R"pydoc(
           Matrix elements of the fusion tensor :math:`X^{ab}_{c,\mu}` for all :math:`\mu`.

           May not be well defined for anyons.

           .. warning ::
               Do not perform inplace operations on the output. That may invalidate caches.

           Parameters
           ----------
           a, b, c
               Sectors. Must be compatible with the fusion described above.
           Z_a : bool
               If we should include a Z isomorphism :math:`Z_{\bar{a}} : \bar{a}^* -> a` below the
               sector a. If so, the composite is a map from :math:`\bar{a}^* \otimes b \to c`.
           Z_b : bool
               Analogously to `Z_a`.

           Returns
           -------
           X : 4D ndarray
               Axis [μ, m_a, m_b, m_c] where μ is the multiplicity index of the fusion tensor and
               m_a goes over a basis for sector a, etc.
           )pydoc")
      .def("are_valid_sectors", &BaseSymmetry::are_valid_sectors, py::arg("sectors"))
      .def("fusion_outcomes_broadcast",
           &BaseSymmetry::fusion_outcomes_broadcast,
           py::arg("a"),
           py::arg("b"),
           R"pydoc(
           Allows optimized fusion in the case of FusionStyle.single.

           For two SectorArrays, return the element-wise fusion outcome of each pair of Sectors,
           which is a single unique Sector, as a new SectorArray.
           Subclasses may override this with more efficient implementations.
           )pydoc")
      .def("multiple_fusion",
           [](BaseSymmetry const& self, py::args args) {
               std::vector<Sector> sectors;
               sectors.reserve(args.size());
               for (auto h : args) {
                   sectors.push_back(h.cast<Sector>());
               }
               return self.multiple_fusion(sectors);
           })
      .def(
        "multiple_fusion_broadcast",
        [](BaseSymmetry const& self, py::args args) {
            std::vector<SectorArray> sectors;
            sectors.reserve(args.size());
            for (auto h : args) {
                sectors.push_back(h.cast<SectorArray>());
            }
            return self.multiple_fusion_broadcast(sectors);
        },
        R"pydoc(
        Allows optimized fusion in the case of FusionStyle.single.

        It generalizes :meth:`fusion_outcomes_broadcast` to more than two fusion inputs.
        )pydoc")
      .def(
        "_multiple_fusion_broadcast",
        [](BaseSymmetry const& self, py::args args) {
            std::vector<SectorArray> sectors;
            sectors.reserve(args.size());
            for (auto h : args) {
                sectors.push_back(h.cast<SectorArray>());
            }
            return self._multiple_fusion_broadcast(sectors);
        },
        R"pydoc(
        Internal version of :meth:`multiple_fusion_broadcast`. May assume ``len(sectors) >= 2``.
        )pydoc")
      .def("can_fuse_to",
           &BaseSymmetry::can_fuse_to,
           py::arg("a"),
           py::arg("b"),
           py::arg("c"),
           R"pydoc(
           Whether c is a valid fusion outcome, i.e. if it appears in ``self.fusion_outcomes(a, b)``
           )pydoc")
      .def("sector_dim",
           &BaseSymmetry::sector_dim,
           py::arg("a"),
           R"pydoc(
           The dimension of a sector, as an unstructured space (i.e. if we drop the symmetry).

           For bosonic braiding style, e.g. for group symmetries, this coincides with the quantum
           dimension computed by :meth:`qdim`.
           For other braiding styles,
           )pydoc")
      .def("batch_sector_dim",
           &BaseSymmetry::batch_sector_dim,
           py::arg("a"),
           R"pydoc(
           sector_dim of every sector (row) in a
           )pydoc")
      .def("batch_qdim",
           &BaseSymmetry::batch_qdim,
           py::arg("a"),
           R"pydoc(
           Quantum dimension of every sector (row) in `a`
           )pydoc")
      .def("sector_str",
           &BaseSymmetry::sector_str,
           py::arg("a"),
           R"pydoc(
           Short and readable string for the sector. Is used in __str__ of symmetry-related objects.
           )pydoc")
      .def("dual_sectors",
           &BaseSymmetry::dual_sectors,
           py::arg("sectors"),
           R"pydoc(
           dual_sector for multiple sectors

           subclasses my override this.
           )pydoc")
      .def("frobenius_schur",
           &BaseSymmetry::frobenius_schur,
           py::arg("a"),
           R"pydoc(
           The Frobenius Schur indicator of a sector.
           )pydoc")
      .def("qdim",
           &BaseSymmetry::qdim,
           py::arg("a"),
           R"pydoc(
           The quantum dimension ``Tr(id_a)`` of a sector
           )pydoc")
      .def("sqrt_qdim",
           &BaseSymmetry::sqrt_qdim,
           py::arg("a"),
           R"pydoc(
           The square root of the quantum dimension.
           )pydoc")
      .def("inv_sqrt_qdim",
           &BaseSymmetry::inv_sqrt_qdim,
           py::arg("a"),
           R"pydoc(
           The inverse square root of the quantum dimension.
           )pydoc")
      .def("total_qdim",
           &BaseSymmetry::total_qdim,
           R"pydoc(
           Total quantum dimension, :math:`D = \sqrt{\sum_a d_a^2}`.
           )pydoc")
      .def("_b_symbol",
           &BaseSymmetry::_b_symbol,
           py::arg("a"),
           py::arg("b"),
           py::arg("c"),
           R"pydoc(
           Internal implementation of :meth:`b_symbol`. Can assume that inputs are valid.
           )pydoc")
      .def("_c_symbol",
           &BaseSymmetry::_c_symbol,
           py::arg("a"),
           py::arg("b"),
           py::arg("c"),
           py::arg("d"),
           py::arg("e"),
           py::arg("f"),
           R"pydoc(
           Internal implementation of :meth:`c_symbol`. Can assume that inputs are valid.
           )pydoc")
      .def("topological_twist",
           &BaseSymmetry::topological_twist,
           py::arg("a"),
           R"pydoc(
           The prefactor that relates the twist on a single sector to the identity.

           Graphically::

               |   │   ╭─╮                |
               |    ╲ ╱  │                |
               |     ╱   │   =   theta_a  |
               |    ╱ ╲  │                |
               |   │   ╰─╯                |
               |   a                      a

           Notes
           -----
           For a twist with opposite chirality, the prefactor is conjugated.

               |   │   ╭─╮                      |
               |    ╲ ╱  │                      |
               |     ╲   │   =   conj(theta_a)  |
               |    ╱ ╲  │                      |
               |   │   ╰─╯                      |
               |   a                            a
           )pydoc")
      .def("s_matrix_element",
           &BaseSymmetry::s_matrix_element,
           py::arg("a"),
           py::arg("b"),
           R"pydoc(
           Single matrix-element of the S-matrix.

           See Also
           --------
           s_matrix
           )pydoc")
      .def("s_matrix",
           &BaseSymmetry::s_matrix,
           R"pydoc(
           The modular S-matrix. Only defined for modular tensor categories.

           See Also
           --------
           s_matrix_element
           )pydoc");
}

} // namespace cyten
