#pragma once

#include <cyten/symmetries/abelian_group.h>
#include <cyten/symmetries/base_symmetry.h>
#include <cyten/symmetries/group.h>
#include <cyten/symmetries/sector_numpy.h>
#include <cyten/symmetries/spaces.h>
#include <cyten/symmetries/symmetry_factor.h>

#include <pybind11/pybind11.h>

#include <optional>
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

    // Note: do NOT trampoline can_be_dropped / has_*_braid / is_abelian / has_unique_fusion.
    // Those are bound as def_property_readonly; PYBIND11_OVERRIDE then finds the property
    // and raises TypeError: bool is not an instance of function.

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
        // Python overrides take ``*sectors``; unpack instead of passing one list.
        py::gil_scoped_acquire gil;
        py::function override =
          py::get_override(static_cast<BaseSymmetry const*>(this), "_multiple_fusion_broadcast");
        if (override) {
            py::tuple args(sectors.size());
            for (std::size_t i = 0; i < sectors.size(); ++i) {
                args[i] = sector_array_to_numpy(sectors[i]);
            }
            return sector_array_from_numpy(override(*args));
        }
        return BaseSymmetry::_multiple_fusion_broadcast(sectors);
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
    float64 qdim(Sector a) const override { PYBIND11_OVERRIDE(float64, BaseSymmetry, qdim, a); }
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

/// Trampoline for Python subclasses of SymmetryFactor (Group, anyons, …).
class PySymmetryFactor
  : public SymmetryFactor
  , public py::trampoline_self_life_support
{
  public:
    using SymmetryFactor::SymmetryFactor;

    Sector dual_sector(Sector a) const override
    {
        PYBIND11_OVERRIDE_PURE(Sector, SymmetryFactor, dual_sector, a);
    }
    int64 _n_symbol(Sector a, Sector b, Sector c) const override
    {
        PYBIND11_OVERRIDE_PURE(int64, SymmetryFactor, _n_symbol, a, b, c);
    }
    py::array _f_symbol(Sector a, Sector b, Sector c, Sector d, Sector e, Sector f) const override
    {
        PYBIND11_OVERRIDE_PURE(py::array, SymmetryFactor, _f_symbol, a, b, c, d, e, f);
    }
    py::array _r_symbol(Sector a, Sector b, Sector c) const override
    {
        PYBIND11_OVERRIDE_PURE(py::array, SymmetryFactor, _r_symbol, a, b, c);
    }
    py::object as_Symmetry() override
    {
        PYBIND11_OVERRIDE(py::object, SymmetryFactor, as_Symmetry);
    }
    bool is_valid_sector(Sector a) const override
    {
        PYBIND11_OVERRIDE_PURE(bool, SymmetryFactor, is_valid_sector, a);
    }
    SectorArray fusion_outcomes(Sector a, Sector b) const override
    {
        PYBIND11_OVERRIDE_PURE(SectorArray, SymmetryFactor, fusion_outcomes, a, b);
    }
    std::string repr() const override
    {
        PYBIND11_OVERRIDE_PURE_NAME(std::string, SymmetryFactor, "__repr__", repr);
    }
    bool _is_equivalent_factor(SymmetryFactor const& other) const override
    {
        PYBIND11_OVERRIDE_PURE(bool, SymmetryFactor, _is_equivalent_factor, other);
    }

    // Optional overrides commonly customized by factors (mirror PyBaseSymmetry).
    // Do not trampoline property-bound methods (is_abelian, can_be_dropped, …).
    py::array _fusion_tensor(Sector a, Sector b, Sector c, bool Z_a, bool Z_b) const override
    {
        PYBIND11_OVERRIDE(py::array, SymmetryFactor, _fusion_tensor, a, b, c, Z_a, Z_b);
    }
    py::array swap_gate(Sector a, Sector b) const override
    {
        PYBIND11_OVERRIDE(py::array, SymmetryFactor, swap_gate, a, b);
    }
    py::array Z_iso(Sector a) const override
    {
        PYBIND11_OVERRIDE(py::array, SymmetryFactor, Z_iso, a);
    }
    SectorArray all_sectors() const override
    {
        PYBIND11_OVERRIDE(SectorArray, SymmetryFactor, all_sectors);
    }
    bool are_valid_sectors(SectorArray const& sectors) const override
    {
        PYBIND11_OVERRIDE(bool, SymmetryFactor, are_valid_sectors, sectors);
    }
    SectorArray fusion_outcomes_broadcast(SectorArray const& a,
                                          SectorArray const& b) const override
    {
        PYBIND11_OVERRIDE(SectorArray, SymmetryFactor, fusion_outcomes_broadcast, a, b);
    }
    SectorArray _multiple_fusion_broadcast(std::vector<SectorArray> const& sectors) const override
    {
        // Python overrides take ``*sectors``; unpack instead of passing one list.
        py::gil_scoped_acquire gil;
        py::function override =
          py::get_override(static_cast<SymmetryFactor const*>(this), "_multiple_fusion_broadcast");
        if (override) {
            py::tuple args(sectors.size());
            for (std::size_t i = 0; i < sectors.size(); ++i) {
                args[i] = sector_array_to_numpy(sectors[i]);
            }
            return sector_array_from_numpy(override(*args));
        }
        return SymmetryFactor::_multiple_fusion_broadcast(sectors);
    }
    bool can_fuse_to(Sector a, Sector b, Sector c) const override
    {
        PYBIND11_OVERRIDE(bool, SymmetryFactor, can_fuse_to, a, b, c);
    }
    int64 sector_dim(Sector a) const override
    {
        PYBIND11_OVERRIDE(int64, SymmetryFactor, sector_dim, a);
    }
    py::array batch_sector_dim(SectorArray const& a) const override
    {
        PYBIND11_OVERRIDE(py::array, SymmetryFactor, batch_sector_dim, a);
    }
    py::array batch_qdim(SectorArray const& a) const override
    {
        PYBIND11_OVERRIDE(py::array, SymmetryFactor, batch_qdim, a);
    }
    std::string sector_str(Sector a) const override
    {
        PYBIND11_OVERRIDE(std::string, SymmetryFactor, sector_str, a);
    }
    SectorArray dual_sectors(SectorArray const& sectors) const override
    {
        PYBIND11_OVERRIDE(SectorArray, SymmetryFactor, dual_sectors, sectors);
    }
    int64 frobenius_schur(Sector a) const override
    {
        PYBIND11_OVERRIDE(int64, SymmetryFactor, frobenius_schur, a);
    }
    float64 qdim(Sector a) const override { PYBIND11_OVERRIDE(float64, SymmetryFactor, qdim, a); }
    float64 sqrt_qdim(Sector a) const override
    {
        PYBIND11_OVERRIDE(float64, SymmetryFactor, sqrt_qdim, a);
    }
    float64 inv_sqrt_qdim(Sector a) const override
    {
        PYBIND11_OVERRIDE(float64, SymmetryFactor, inv_sqrt_qdim, a);
    }
    py::array _b_symbol(Sector a, Sector b, Sector c) const override
    {
        PYBIND11_OVERRIDE(py::array, SymmetryFactor, _b_symbol, a, b, c);
    }
    py::array _c_symbol(Sector a, Sector b, Sector c, Sector d, Sector e, Sector f) const override
    {
        PYBIND11_OVERRIDE(py::array, SymmetryFactor, _c_symbol, a, b, c, d, e, f);
    }
    complex128 topological_twist(Sector a) const override
    {
        PYBIND11_OVERRIDE(complex128, SymmetryFactor, topological_twist, a);
    }
};

/// Trampoline for Python subclasses of Group (SU2, SUN, …).
class PyGroup
  : public Group
  , public py::trampoline_self_life_support
{
  public:
    using Group::Group;

    Sector dual_sector(Sector a) const override
    {
        PYBIND11_OVERRIDE_PURE(Sector, Group, dual_sector, a);
    }
    int64 _n_symbol(Sector a, Sector b, Sector c) const override
    {
        PYBIND11_OVERRIDE_PURE(int64, Group, _n_symbol, a, b, c);
    }
    py::array _f_symbol(Sector a, Sector b, Sector c, Sector d, Sector e, Sector f) const override
    {
        PYBIND11_OVERRIDE_PURE(py::array, Group, _f_symbol, a, b, c, d, e, f);
    }
    py::array _r_symbol(Sector a, Sector b, Sector c) const override
    {
        PYBIND11_OVERRIDE_PURE(py::array, Group, _r_symbol, a, b, c);
    }
    py::object as_Symmetry() override { PYBIND11_OVERRIDE(py::object, Group, as_Symmetry); }
    bool is_valid_sector(Sector a) const override
    {
        PYBIND11_OVERRIDE_PURE(bool, Group, is_valid_sector, a);
    }
    SectorArray fusion_outcomes(Sector a, Sector b) const override
    {
        PYBIND11_OVERRIDE_PURE(SectorArray, Group, fusion_outcomes, a, b);
    }
    std::string repr() const override
    {
        PYBIND11_OVERRIDE_PURE_NAME(std::string, Group, "__repr__", repr);
    }
    bool _is_equivalent_factor(SymmetryFactor const& other) const override
    {
        PYBIND11_OVERRIDE_PURE(bool, Group, _is_equivalent_factor, other);
    }

    py::array _fusion_tensor(Sector a, Sector b, Sector c, bool Z_a, bool Z_b) const override
    {
        PYBIND11_OVERRIDE_PURE(py::array, Group, _fusion_tensor, a, b, c, Z_a, Z_b);
    }
    py::array swap_gate(Sector a, Sector b) const override
    {
        PYBIND11_OVERRIDE(py::array, Group, swap_gate, a, b);
    }
    py::array Z_iso(Sector a) const override { PYBIND11_OVERRIDE(py::array, Group, Z_iso, a); }
    SectorArray all_sectors() const override
    {
        PYBIND11_OVERRIDE(SectorArray, Group, all_sectors);
    }
    bool are_valid_sectors(SectorArray const& sectors) const override
    {
        PYBIND11_OVERRIDE(bool, Group, are_valid_sectors, sectors);
    }
    SectorArray fusion_outcomes_broadcast(SectorArray const& a,
                                          SectorArray const& b) const override
    {
        PYBIND11_OVERRIDE(SectorArray, Group, fusion_outcomes_broadcast, a, b);
    }
    SectorArray _multiple_fusion_broadcast(std::vector<SectorArray> const& sectors) const override
    {
        py::gil_scoped_acquire gil;
        py::function override =
          py::get_override(static_cast<Group const*>(this), "_multiple_fusion_broadcast");
        if (override) {
            py::tuple args(sectors.size());
            for (std::size_t i = 0; i < sectors.size(); ++i) {
                args[i] = sector_array_to_numpy(sectors[i]);
            }
            return sector_array_from_numpy(override(*args));
        }
        return Group::_multiple_fusion_broadcast(sectors);
    }
    bool can_fuse_to(Sector a, Sector b, Sector c) const override
    {
        PYBIND11_OVERRIDE(bool, Group, can_fuse_to, a, b, c);
    }
    int64 sector_dim(Sector a) const override { PYBIND11_OVERRIDE(int64, Group, sector_dim, a); }
    py::array batch_sector_dim(SectorArray const& a) const override
    {
        PYBIND11_OVERRIDE(py::array, Group, batch_sector_dim, a);
    }
    py::array batch_qdim(SectorArray const& a) const override
    {
        PYBIND11_OVERRIDE(py::array, Group, batch_qdim, a);
    }
    std::string sector_str(Sector a) const override
    {
        PYBIND11_OVERRIDE(std::string, Group, sector_str, a);
    }
    SectorArray dual_sectors(SectorArray const& sectors) const override
    {
        PYBIND11_OVERRIDE(SectorArray, Group, dual_sectors, sectors);
    }
    int64 frobenius_schur(Sector a) const override
    {
        PYBIND11_OVERRIDE(int64, Group, frobenius_schur, a);
    }
    float64 qdim(Sector a) const override { PYBIND11_OVERRIDE(float64, Group, qdim, a); }
    float64 sqrt_qdim(Sector a) const override { PYBIND11_OVERRIDE(float64, Group, sqrt_qdim, a); }
    float64 inv_sqrt_qdim(Sector a) const override
    {
        PYBIND11_OVERRIDE(float64, Group, inv_sqrt_qdim, a);
    }
    py::array _b_symbol(Sector a, Sector b, Sector c) const override
    {
        PYBIND11_OVERRIDE(py::array, Group, _b_symbol, a, b, c);
    }
    py::array _c_symbol(Sector a, Sector b, Sector c, Sector d, Sector e, Sector f) const override
    {
        PYBIND11_OVERRIDE(py::array, Group, _c_symbol, a, b, c, d, e, f);
    }
    complex128 topological_twist(Sector a) const override
    {
        PYBIND11_OVERRIDE(complex128, Group, topological_twist, a);
    }
};

/// Trampoline for Python subclasses of AbelianGroup (if any remain after concretes).
class PyAbelianGroup
  : public AbelianGroup
  , public py::trampoline_self_life_support
{
  public:
    using AbelianGroup::AbelianGroup;

    Sector dual_sector(Sector a) const override
    {
        PYBIND11_OVERRIDE_PURE(Sector, AbelianGroup, dual_sector, a);
    }
    py::object as_Symmetry() override { PYBIND11_OVERRIDE(py::object, AbelianGroup, as_Symmetry); }
    bool is_valid_sector(Sector a) const override
    {
        PYBIND11_OVERRIDE_PURE(bool, AbelianGroup, is_valid_sector, a);
    }
    SectorArray fusion_outcomes(Sector a, Sector b) const override
    {
        PYBIND11_OVERRIDE_PURE(SectorArray, AbelianGroup, fusion_outcomes, a, b);
    }
    std::string repr() const override
    {
        PYBIND11_OVERRIDE_PURE_NAME(std::string, AbelianGroup, "__repr__", repr);
    }
    bool _is_equivalent_factor(SymmetryFactor const& other) const override
    {
        PYBIND11_OVERRIDE_PURE(bool, AbelianGroup, _is_equivalent_factor, other);
    }

    int64 _n_symbol(Sector a, Sector b, Sector c) const override
    {
        PYBIND11_OVERRIDE(int64, AbelianGroup, _n_symbol, a, b, c);
    }
    py::array _f_symbol(Sector a, Sector b, Sector c, Sector d, Sector e, Sector f) const override
    {
        PYBIND11_OVERRIDE(py::array, AbelianGroup, _f_symbol, a, b, c, d, e, f);
    }
    py::array _r_symbol(Sector a, Sector b, Sector c) const override
    {
        PYBIND11_OVERRIDE(py::array, AbelianGroup, _r_symbol, a, b, c);
    }
    py::array _fusion_tensor(Sector a, Sector b, Sector c, bool Z_a, bool Z_b) const override
    {
        PYBIND11_OVERRIDE(py::array, AbelianGroup, _fusion_tensor, a, b, c, Z_a, Z_b);
    }
    py::array swap_gate(Sector a, Sector b) const override
    {
        PYBIND11_OVERRIDE(py::array, AbelianGroup, swap_gate, a, b);
    }
    py::array Z_iso(Sector a) const override
    {
        PYBIND11_OVERRIDE(py::array, AbelianGroup, Z_iso, a);
    }
    SectorArray all_sectors() const override
    {
        PYBIND11_OVERRIDE(SectorArray, AbelianGroup, all_sectors);
    }
    bool are_valid_sectors(SectorArray const& sectors) const override
    {
        PYBIND11_OVERRIDE(bool, AbelianGroup, are_valid_sectors, sectors);
    }
    SectorArray fusion_outcomes_broadcast(SectorArray const& a,
                                          SectorArray const& b) const override
    {
        PYBIND11_OVERRIDE(SectorArray, AbelianGroup, fusion_outcomes_broadcast, a, b);
    }
    SectorArray _multiple_fusion_broadcast(std::vector<SectorArray> const& sectors) const override
    {
        py::gil_scoped_acquire gil;
        py::function override =
          py::get_override(static_cast<AbelianGroup const*>(this), "_multiple_fusion_broadcast");
        if (override) {
            py::tuple args(sectors.size());
            for (std::size_t i = 0; i < sectors.size(); ++i) {
                args[i] = sector_array_to_numpy(sectors[i]);
            }
            return sector_array_from_numpy(override(*args));
        }
        return AbelianGroup::_multiple_fusion_broadcast(sectors);
    }
    bool can_fuse_to(Sector a, Sector b, Sector c) const override
    {
        PYBIND11_OVERRIDE(bool, AbelianGroup, can_fuse_to, a, b, c);
    }
    int64 sector_dim(Sector a) const override
    {
        PYBIND11_OVERRIDE(int64, AbelianGroup, sector_dim, a);
    }
    py::array batch_sector_dim(SectorArray const& a) const override
    {
        PYBIND11_OVERRIDE(py::array, AbelianGroup, batch_sector_dim, a);
    }
    py::array batch_qdim(SectorArray const& a) const override
    {
        PYBIND11_OVERRIDE(py::array, AbelianGroup, batch_qdim, a);
    }
    std::string sector_str(Sector a) const override
    {
        PYBIND11_OVERRIDE(std::string, AbelianGroup, sector_str, a);
    }
    SectorArray dual_sectors(SectorArray const& sectors) const override
    {
        PYBIND11_OVERRIDE(SectorArray, AbelianGroup, dual_sectors, sectors);
    }
    int64 frobenius_schur(Sector a) const override
    {
        PYBIND11_OVERRIDE(int64, AbelianGroup, frobenius_schur, a);
    }
    float64 qdim(Sector a) const override { PYBIND11_OVERRIDE(float64, AbelianGroup, qdim, a); }
    float64 sqrt_qdim(Sector a) const override
    {
        PYBIND11_OVERRIDE(float64, AbelianGroup, sqrt_qdim, a);
    }
    float64 inv_sqrt_qdim(Sector a) const override
    {
        PYBIND11_OVERRIDE(float64, AbelianGroup, inv_sqrt_qdim, a);
    }
    py::array _b_symbol(Sector a, Sector b, Sector c) const override
    {
        PYBIND11_OVERRIDE(py::array, AbelianGroup, _b_symbol, a, b, c);
    }
    py::array _c_symbol(Sector a, Sector b, Sector c, Sector d, Sector e, Sector f) const override
    {
        PYBIND11_OVERRIDE(py::array, AbelianGroup, _c_symbol, a, b, c, d, e, f);
    }
    complex128 topological_twist(Sector a) const override
    {
        PYBIND11_OVERRIDE(complex128, AbelianGroup, topological_twist, a);
    }
};

/// Trampoline for Python subclasses of Leg (LegPipe, ElementarySpace, …).
class PyLeg
  : public Leg
  , public py::trampoline_self_life_support
{
  public:
    using Leg::Leg;

    void test_sanity() const override { PYBIND11_OVERRIDE(void, Leg, test_sanity); }

    py::object as_Space() override { PYBIND11_OVERRIDE_PURE(py::object, Leg, as_Space); }

    py::object as_ElementarySpace(bool is_dual) override
    {
        PYBIND11_OVERRIDE(py::object, Leg, as_ElementarySpace, is_dual);
    }

    Ptr dual() const override { PYBIND11_OVERRIDE_PURE(Ptr, Leg, dual); }

    bool is_trivial() const override { PYBIND11_OVERRIDE_PURE(bool, Leg, is_trivial); }

    void set_basis_perm(std::optional<std::vector<int64>> basis_perm) override
    {
        PYBIND11_OVERRIDE(void, Leg, set_basis_perm, basis_perm);
    }

    void set_inverse_basis_perm(std::optional<std::vector<int64>> inverse_basis_perm) override
    {
        PYBIND11_OVERRIDE(void, Leg, set_inverse_basis_perm, inverse_basis_perm);
    }

    std::vector<Ptr> flat_legs() override { PYBIND11_OVERRIDE(std::vector<Ptr>, Leg, flat_legs); }

    std::vector<Ptr> flat_spaces() override
    {
        PYBIND11_OVERRIDE(std::vector<Ptr>, Leg, flat_spaces);
    }

    int64 num_flat_legs() const override { PYBIND11_OVERRIDE(int64, Leg, num_flat_legs); }

    std::vector<int64> _flat_leg_permutation(int64 offset) const override
    {
        PYBIND11_OVERRIDE(std::vector<int64>, Leg, _flat_leg_permutation, offset);
    }

    std::string ascii_arrow() const override { PYBIND11_OVERRIDE(std::string, Leg, ascii_arrow); }

    bool operator==(Leg const& other) const override
    {
        PYBIND11_OVERRIDE_PURE_NAME(bool, Leg, "__eq__", operator==, other);
    }
};

/// Trampoline for Python subclasses of Space (ElementarySpace, TensorProduct, …).
class PySpace
  : public Space
  , public py::trampoline_self_life_support
{
  public:
    using Space::Space;

    void test_sanity() const override { PYBIND11_OVERRIDE(void, Space, test_sanity); }

    Ptr dual() const override { PYBIND11_OVERRIDE_PURE(Ptr, Space, dual); }

    bool is_trivial() const override { PYBIND11_OVERRIDE(bool, Space, is_trivial); }

    bool operator==(Space const& other) const override
    {
        PYBIND11_OVERRIDE_NAME(bool, Space, "__eq__", operator==, other);
    }

    py::object as_ElementarySpace(bool is_dual) override
    {
        PYBIND11_OVERRIDE(py::object, Space, as_ElementarySpace, is_dual);
    }

    py::object change_symmetry(Symmetry::Ptr symmetry,
                               SectorMapFn sector_map,
                               bool injective) override
    {
        PYBIND11_OVERRIDE_PURE(
          py::object, Space, change_symmetry, symmetry, sector_map, injective);
    }

    py::object drop_symmetry(std::optional<std::vector<int64>> which) override
    {
        PYBIND11_OVERRIDE_PURE(py::object, Space, drop_symmetry, which);
    }

    Ptr as_Space() override { PYBIND11_OVERRIDE(Ptr, Space, as_Space); }
};

} // namespace cyten
