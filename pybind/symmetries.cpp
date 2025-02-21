#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <exception>

#include "cyten/symmetries.h"


using namespace std;
using namespace cyten;
namespace py = pybind11;
using namespace pybind11::literals; // provides "arg"_a literals
                                

/// @brief pybind11 trampoline class to allow subclassing Symmetry in Python
template<class SymmetryBase=Symmetry>
class PySymmetry : public Symmetry {
    public:
        using SymmetryBase::SymmetryBase;  // inherit constructors
        bool can_be_dropped() const override { PYBIND11_OVERRIDE(bool, SymmetryBase, can_be_dropped, ); }
        std::string group_name() const override { PYBIND11_OVERRIDE_PURE(std::string, SymmetryBase, group_name, ); }
        Sector trivial_sector() const override { PYBIND11_OVERRIDE_PURE(Sector, SymmetryBase, trivial_sector, ); }
        bool is_valid_sector(Sector sector) const override { PYBIND11_OVERRIDE_PURE(bool, SymmetryBase, is_valid_sector, sector); }
        SectorArray fusion_outcomes(Sector a, Sector b) const override { PYBIND11_OVERRIDE_PURE(SectorArray, SymmetryBase, fusion_outcomes, a, b); }
        std::string __repr__() const override { PYBIND11_OVERRIDE_PURE(std::string, SymmetryBase, __repr__, ); }
        bool is_same_symmetry(Symmetry const & other) const override { PYBIND11_OVERRIDE_PURE(bool, SymmetryBase, is_same_symmetry, other); }
        Sector dual_sector(Sector a) const override { PYBIND11_OVERRIDE_PURE(Sector, SymmetryBase, dual_sector, a); }
        cyten_int _n_symbol(Sector a, Sector b, Sector c) const override { PYBIND11_OVERRIDE_PURE(cyten_int, SymmetryBase, _n_symbol, a, b, c); }
        py::array _f_symbol(Sector a, Sector b, Sector c, Sector d, Sector e, Sector f) const override { PYBIND11_OVERRIDE_PURE(py::array, SymmetryBase, _f_symbol, a, b, c, d, e, f); }
        py::array _r_symbol(Sector a, Sector b, Sector c) const override { PYBIND11_OVERRIDE_PURE(py::array, SymmetryBase, _r_symbol, a, b, c); }
        py::array _fusion_tensor(Sector a, Sector b, Sector c, bool Z_a, bool Z_b) const override { PYBIND11_OVERRIDE(py::array, SymmetryBase, _fusion_tensor, a, b, c, Z_a, Z_b); }
        py::array Z_iso(Sector a) override { PYBIND11_OVERRIDE_PURE(py::array, SymmetryBase, Z_iso, a); }
        SectorArray all_sectors() const override { PYBIND11_OVERRIDE(SectorArray, SymmetryBase, all_sectors, ); }
        Sector compress_sector(std::vector<Sector> const & decompressed) const override { PYBIND11_OVERRIDE(Sector, SymmetryBase, compress_sector, decompressed); }
        std::vector<Sector> decompress_sector(Sector compressed) const override { PYBIND11_OVERRIDE(std::vector<Sector>, SymmetryBase, decompress_sector, compressed); }
        bool are_valid_sectors(SectorArray const & sectors) const override { PYBIND11_OVERRIDE(bool, SymmetryBase, are_valid_sectors, sectors); }
        SectorArray fusion_outcomes_broadcast(SectorArray const & a, SectorArray const & b) const override { PYBIND11_OVERRIDE(SectorArray, SymmetryBase, fusion_outcomes_broadcast, a, b); }
        SectorArray _multiple_fusion_broadcast(std::vector<SectorArray> const & sectors) const override { PYBIND11_OVERRIDE(SectorArray, SymmetryBase, _multiple_fusion_broadcast, sectors); }
        bool can_fuse_to(Sector a, Sector b, Sector c) const override { PYBIND11_OVERRIDE(bool, SymmetryBase, can_fuse_to, a, b, c); }
        cyten_int sector_dim(Sector a) const override{ PYBIND11_OVERRIDE(cyten_int, SymmetryBase, sector_dim, a); }
        std::vector<cyten_int> sector_dim(SectorArray const& a) const override { PYBIND11_OVERRIDE(std::vector<cyten_int>, SymmetryBase, sector_dim, a); }
        cyten_float qdim(Sector a) const override { PYBIND11_OVERRIDE(cyten_float, SymmetryBase, qdim, a); }
        std::vector<cyten_float> qdim(SectorArray const& a) const override { PYBIND11_OVERRIDE(std::vector<cyten_float>, SymmetryBase, qdim, a); }
        std::string sector_str(Sector a) const override { PYBIND11_OVERRIDE(std::string, SymmetryBase, sector_str, a); }
        cyten_int frobenius_schur(Sector a) const override { PYBIND11_OVERRIDE(cyten_int, SymmetryBase, frobenius_schur, a); }
        py::array _b_symbol(Sector a, Sector b, Sector c) const override { PYBIND11_OVERRIDE(py::array, SymmetryBase, _b_symbol, a, b, c); }
        py::array _c_symbol(Sector a, Sector b, Sector c, Sector d, Sector e, Sector f) const override { PYBIND11_OVERRIDE(py::array, SymmetryBase, _c_symbol, a, b, c, d, e, f); }
};



void bind_symmetries(py::module_ &m){
    py::options options;
    //options.disable_enum_members_docstring();

    py::register_exception<SymmetryError>(m, "SymmetryError");

    py::enum_<FusionStyle>(m, "FusionStyle", py::arithmetic(), R"pydoc(
        Describes properties of fusion, i.e. of the tensor product.
        )pydoc")
        .value("single", FusionStyle::single,
            "Fusing sectors results in a single sector ``a ⊗ b = c``, e.g. abelian groups.")
        .value("multiple_unique", FusionStyle::multiple_unique,
            "Every sector appears at most once in pairwise fusion, ``N_symbol in [0, 1]``.")
        .value("general", FusionStyle::general,
            "No assumptions, ``N_symbol in [0, 1, 2, 3, ...]``.")
        .export_values();

    py::enum_<BraidingStyle>(m, "BraidingStyle", py::arithmetic(), R"pydoc(
        Describes properties of braiding, i.e. behaviour under twisting.
        )pydoc")
        .value("bosonic", BraidingStyle::bosonic, "Symmetric braiding with trivial twist")
        .value("fermionic", BraidingStyle::fermionic, "Symmetric braiding with non-trivial twist")
        .value("anyonic", BraidingStyle::anyonic, "General, non-symmetric braiding") 
        .value("no_braiding", BraidingStyle::no_braiding, "Braiding is not defined")
        .export_values();



    py::class_<Symmetry, PySymmetry<>> symmetry(m, "Symmetry");
    symmetry.doc() = R"pydoc(
        Base class for symmetries that impose a block-structure on tensors
    )pydoc";

    symmetry.def(py::init<FusionStyle, BraidingStyle>())
        .def_property_readonly("can_be_dropped", &Symmetry::can_be_dropped)
        // TODO: fusion tensor dtype?
        .def_readonly("fusion_style", &Symmetry::fusion_style)
        .def_readonly("brading_style", &Symmetry::braiding_style)
        .def_property_readonly("trivial_sector", &Symmetry::trivial_sector)
        .def_property_readonly("group_name", &Symmetry::group_name)
        .def_property_readonly("num_sectors", &Symmetry::num_sectors)
        .def_readwrite("descriptive_name", &Symmetry::descriptive_name)
        .def_property_readonly("sector_ind_len", &Symmetry::sector_ind_len)
        .def_property_readonly("empty_sector_array", [](){return SectorArray(); })
        .def_property_readonly("is_abelian", &Symmetry::is_abelian)
        .def("is_valid_sector", &Symmetry::is_valid_sector, 
            "Whether `a` is a valid sector of this symmetry")
        .def("fusion_outcomes", &Symmetry::fusion_outcomes, R"pydoc(
            Returns all outcomes for the fusion of sectors

            Each sector appears only once, regardless of its multiplicity (given by n_symbol) in the fusion
            )pydoc")
        .def("__repr__", &Symmetry::__repr__)
        .def("is_same_symmetry", &Symmetry::is_same_symmetry, R"pydoc(
            Whether self and other describe the same mathematical structure.

            In contrast to `==`, the :attr:`descriptive_name` is ignored.
            )pydoc")
        .def("dual_sector", &Symmetry::dual_sector, R"pydoc(
            The sector dual to a, such that N^{a,dual(a)}_u = 1.

            Note that the dual space :math:`a^\star` to a sector :math:`a` may not itself be one of
            the sectors, but it must be isomorphic to one of the sectors. This method returns that
            representative :math:`\bar{a}` of the equivalence class.
            )pydoc")
        .def("_n_symbol", &Symmetry::_n_symbol, R"pydoc(
            Optimized version of self.n_symbol that assumes that c is a valid fusion outcome.

            If it is not, the results may be nonsensical. We do this for optimization purposes
            )pydoc")
        ;

}
