#pragma once

#include <cyten/symmetries/base_symmetry.h>

#include <pybind11/pybind11.h>

#include <string>
#include <vector>

namespace cyten {

/// pybind11 trampoline so Python subclasses (SymmetryFactor, Symmetry, …) can override.
class PyBaseSymmetry
  : public BaseSymmetry
  , public py::trampoline_self_life_support
{
  public:
    using BaseSymmetry::BaseSymmetry;

    bool can_be_dropped() const override
    {
        PYBIND11_OVERRIDE(bool, BaseSymmetry, can_be_dropped);
    }
    bool has_symmetric_braid() const override
    {
        PYBIND11_OVERRIDE(bool, BaseSymmetry, has_symmetric_braid);
    }
    bool has_trivial_braid() const override
    {
        PYBIND11_OVERRIDE(bool, BaseSymmetry, has_trivial_braid);
    }
    bool is_abelian() const override { PYBIND11_OVERRIDE(bool, BaseSymmetry, is_abelian); }
    bool has_unique_fusion() const override
    {
        PYBIND11_OVERRIDE(bool, BaseSymmetry, has_unique_fusion);
    }

    Sector dual_sector(Sector a) const override
    {
        PYBIND11_OVERRIDE_PURE(Sector, BaseSymmetry, dual_sector, a);
    }
    int64 _n_symbol(Sector a, Sector b, Sector c) const override
    {
        PYBIND11_OVERRIDE_PURE(int64, BaseSymmetry, _n_symbol, a, b, c);
    }
    py::array _f_symbol(Sector a, Sector b, Sector c, Sector d, Sector e, Sector f) const override
    {
        PYBIND11_OVERRIDE_PURE(py::array, BaseSymmetry, _f_symbol, a, b, c, d, e, f);
    }
    py::array _r_symbol(Sector a, Sector b, Sector c) const override
    {
        PYBIND11_OVERRIDE_PURE(py::array, BaseSymmetry, _r_symbol, a, b, c);
    }
    py::object as_Symmetry() override
    {
        PYBIND11_OVERRIDE_PURE(py::object, BaseSymmetry, as_Symmetry);
    }
    bool is_valid_sector(Sector a) const override
    {
        PYBIND11_OVERRIDE_PURE(bool, BaseSymmetry, is_valid_sector, a);
    }
    SectorArray fusion_outcomes(Sector a, Sector b) const override
    {
        PYBIND11_OVERRIDE_PURE(SectorArray, BaseSymmetry, fusion_outcomes, a, b);
    }

    py::array _fusion_tensor(Sector a, Sector b, Sector c, bool Z_a, bool Z_b) const override
    {
        PYBIND11_OVERRIDE(py::array, BaseSymmetry, _fusion_tensor, a, b, c, Z_a, Z_b);
    }
    py::array swap_gate(Sector a, Sector b) const override
    {
        PYBIND11_OVERRIDE(py::array, BaseSymmetry, swap_gate, a, b);
    }
    py::array Z_iso(Sector a) const override
    {
        PYBIND11_OVERRIDE(py::array, BaseSymmetry, Z_iso, a);
    }
    SectorArray all_sectors() const override
    {
        PYBIND11_OVERRIDE(SectorArray, BaseSymmetry, all_sectors);
    }
    bool are_valid_sectors(SectorArray const& sectors) const override
    {
        PYBIND11_OVERRIDE(bool, BaseSymmetry, are_valid_sectors, sectors);
    }
    SectorArray fusion_outcomes_broadcast(SectorArray const& a,
                                          SectorArray const& b) const override
    {
        PYBIND11_OVERRIDE(SectorArray, BaseSymmetry, fusion_outcomes_broadcast, a, b);
    }
    SectorArray _multiple_fusion_broadcast(std::vector<SectorArray> const& sectors) const override
    {
        PYBIND11_OVERRIDE(SectorArray, BaseSymmetry, _multiple_fusion_broadcast, sectors);
    }
    bool can_fuse_to(Sector a, Sector b, Sector c) const override
    {
        PYBIND11_OVERRIDE(bool, BaseSymmetry, can_fuse_to, a, b, c);
    }
    int64 sector_dim(Sector a) const override
    {
        PYBIND11_OVERRIDE(int64, BaseSymmetry, sector_dim, a);
    }
    py::array batch_sector_dim(SectorArray const& a) const override
    {
        PYBIND11_OVERRIDE(py::array, BaseSymmetry, batch_sector_dim, a);
    }
    py::array batch_qdim(SectorArray const& a) const override
    {
        PYBIND11_OVERRIDE(py::array, BaseSymmetry, batch_qdim, a);
    }
    std::string sector_str(Sector a) const override
    {
        PYBIND11_OVERRIDE(std::string, BaseSymmetry, sector_str, a);
    }
    SectorArray dual_sectors(SectorArray const& sectors) const override
    {
        PYBIND11_OVERRIDE(SectorArray, BaseSymmetry, dual_sectors, sectors);
    }
    int64 frobenius_schur(Sector a) const override
    {
        PYBIND11_OVERRIDE(int64, BaseSymmetry, frobenius_schur, a);
    }
    float64 qdim(Sector a) const override
    {
        PYBIND11_OVERRIDE(float64, BaseSymmetry, qdim, a);
    }
    float64 sqrt_qdim(Sector a) const override
    {
        PYBIND11_OVERRIDE(float64, BaseSymmetry, sqrt_qdim, a);
    }
    float64 inv_sqrt_qdim(Sector a) const override
    {
        PYBIND11_OVERRIDE(float64, BaseSymmetry, inv_sqrt_qdim, a);
    }
    py::array _b_symbol(Sector a, Sector b, Sector c) const override
    {
        PYBIND11_OVERRIDE(py::array, BaseSymmetry, _b_symbol, a, b, c);
    }
    py::array _c_symbol(Sector a, Sector b, Sector c, Sector d, Sector e, Sector f) const override
    {
        PYBIND11_OVERRIDE(py::array, BaseSymmetry, _c_symbol, a, b, c, d, e, f);
    }
    complex128 topological_twist(Sector a) const override
    {
        PYBIND11_OVERRIDE(complex128, BaseSymmetry, topological_twist, a);
    }
};

} // namespace cyten
