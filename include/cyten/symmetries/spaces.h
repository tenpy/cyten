#pragma once

#include "exceptions.h"
#include "sector.h"
#include "symmetry.h"

#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace cyten {

/// Common base class for a single leg of a tensor.
///
/// A single leg on a tensor can either be an :class:`ElementarySpace` or, e.g. as the result
/// of combining legs, a :class:`LegPipe`.
class Leg : public std::enable_shared_from_this<Leg>
{
  public:
    using Ptr = std::shared_ptr<Leg>;
    using CPtr = std::shared_ptr<const Leg>;

    Symmetry::Ptr symmetry;
    /// Quantum dimension; integer if ``symmetry->can_be_dropped()``, otherwise may be non-integer.
    float64 dim = 0.;
    bool is_dual = false;

    Leg(Symmetry::Ptr symmetry,
        float64 dim,
        bool is_dual,
        std::optional<std::vector<int64>> basis_perm = std::nullopt);
    virtual ~Leg() = default;

    /// Perform sanity checks.
    virtual void test_sanity() const;

    /// Convert to (an appropriate subclass of) :class:`Space`.
    /// Returns a Python object until :class:`Space` subclasses are fully migrated.
    virtual py::object as_Space() = 0;

    /// Convert to an isomorphic :class:`ElementarySpace`.
    /// Returns a Python object until :class:`ElementarySpace` is converted to C++.
    virtual py::object as_ElementarySpace(bool is_dual = false);

    /// The dual leg, that is obtained when bending this leg.
    virtual Ptr dual() const = 0;

    virtual bool is_trivial() const = 0;

    /// Permutation that translates between public and internal basis order.
    ///
    /// Raises :class:`SymmetryError` if the symmetry cannot be dropped.
    /// Returns the identity permutation when ``_basis_perm`` is empty.
    [[nodiscard]] std::vector<int64> basis_perm() const;

    /// Set :meth:`basis_perm` (and derive :meth:`inverse_basis_perm`).
    /// ``nullopt`` means the identity permutation.
    /// Overridden by pipes that forbid changing the permutation.
    virtual void set_basis_perm(std::optional<std::vector<int64>> basis_perm);

    /// Inverse permutation of :meth:`basis_perm`.
    [[nodiscard]] std::vector<int64> inverse_basis_perm() const;

    /// Set :meth:`inverse_basis_perm` (and derive :meth:`basis_perm`).
    /// ``nullopt`` means the identity permutation.
    /// Overridden by pipes that forbid changing the permutation.
    virtual void set_inverse_basis_perm(std::optional<std::vector<int64>> inverse_basis_perm);

    /// Flatten until there are no more pipes.
    virtual std::vector<Ptr> flat_legs();

    /// Flatten until we get spaces (keeps :class:`AbelianLegPipe` nested).
    virtual std::vector<Ptr> flat_spaces();

    /// The number of :meth:`flat_legs`.
    virtual int64 num_flat_legs() const;

    /// Leg permutation such that combining legs would be in C style.
    virtual std::vector<int64> _flat_leg_permutation(int64 offset = 0) const;

    /// A single character arrow, for use in tensor diagrams.
    ///
    /// Indicates (a) if the leg is a pipe and (b) for ElementarySpaces, the duality.
    /// Default implementation throws; subclasses override.
    virtual std::string ascii_arrow() const;

    virtual bool operator==(Leg const& other) const = 0;

    /// Apply the basis_perm, i.e. form ``arr[self.basis_perm]``.
    ///
    /// This is the preferred method of accessing the permutation, since we may skip applying
    /// trivial permutations.
    py::array apply_basis_perm(py::array arr,
                               int64 axis = 0,
                               bool inverse = false,
                               bool pre_compose = false) const;

    /// Whether ``_basis_perm`` is set (non-identity internal permutation).
    [[nodiscard]] bool has_custom_basis_perm() const noexcept { return _basis_perm.has_value(); }

  protected:
    /// Internal basis permutation; ``nullopt`` means identity. Meaningless if symmetry cannot be
    /// dropped.
    std::optional<std::vector<int64>> _basis_perm;
    std::optional<std::vector<int64>> _inverse_basis_perm;
};

/// Map of sector arrays for :meth:`Space::change_symmetry`.
using SectorMapFn = std::function<SectorArray(SectorArray const&)>;

/// Base class for symmetry spaces, see :class:`ElementarySpace` for the standard case.
///
/// A symmetry space is e.g. a vector space with a representation of a symmetry group.
///
/// Each symmetry space is equivalent to a direct sum of sectors.
class Space : public std::enable_shared_from_this<Space>
{
  public:
    using Ptr = std::shared_ptr<Space>;
    using CPtr = std::shared_ptr<const Space>;

    Symmetry::Ptr symmetry;
    SectorArray sector_decomposition;
    /// ``"sorted"``, ``"dual_sorted"``, or nullopt (no guaranteed order).
    std::optional<std::string> sector_order;
    std::vector<int64> multiplicities;
    int64 num_sectors = 0;
    /// Integer sector dims if ``symmetry->can_be_dropped()``; otherwise nullopt.
    std::optional<std::vector<int64>> sector_dims;
    std::vector<float64> sector_qdims;
    /// For each sector, ``[start, stop)`` indices in the internal basis order.
    std::optional<std::vector<std::array<int64, 2>>> slices;
    float64 dim = 0.;

    Space(Symmetry::Ptr symmetry,
          SectorArray sector_decomposition,
          std::optional<std::vector<int64>> multiplicities = std::nullopt,
          std::optional<std::string> sector_order = std::nullopt);
    virtual ~Space() = default;

    /// Perform sanity checks.
    virtual void test_sanity() const;

    /// The dual space of the same type.
    virtual Ptr dual() const = 0;

    /// If the space is trivial, i.e. isomorphic to the one-dimensional trivial sector.
    virtual bool is_trivial() const;

    /// Spaces do not support ``==``; use :meth:`is_isomorphic_to`. Subclasses may override.
    virtual bool operator==(Space const& other) const;

    /// If the two spaces are isomorphic, i.e. have the same :attr:`sector_decomposition`.
    [[nodiscard]] bool is_isomorphic_to(Space const& other) const;

    /// Whether self is (isomorphic to) a subspace of other.
    [[nodiscard]] bool is_subspace_of(Space const& other) const;

    /// Convert to an isomorphic :class:`ElementarySpace`.
    /// Returns a Python object until :class:`ElementarySpace` is converted to C++.
    virtual py::object as_ElementarySpace(bool is_dual = false);

    /// Change the symmetry by specifying how the sectors change.
    /// Returns a Python object until :class:`ElementarySpace` is converted to C++.
    virtual py::object change_symmetry(Symmetry::Ptr symmetry,
                                       SectorMapFn sector_map,
                                       bool injective = false) = 0;

    /// Drop some or all symmetries.
    /// ``nullopt`` means drop all (Python ``'all'``). Otherwise factor indices to drop.
    /// Returns a Python object until :class:`ElementarySpace` is converted to C++.
    virtual py::object drop_symmetry(std::optional<std::vector<int64>> which = std::nullopt) = 0;

    /// Identity for spaces.
    virtual Ptr as_Space();

    /// Find the index of a given sector in the :attr:`sector_decomposition`.
    [[nodiscard]] std::optional<int64> sector_decomposition_where(Sector sector) const;

    /// The multiplicity of a given sector in the :attr:`sector_decomposition`.
    [[nodiscard]] int64 sector_multiplicity(Sector sector) const;
};

/// A group of legs, i.e. resulting from :func:`~cyten.tensors.combine_legs`.
///
/// Note that the abelian backend defines a custom subclass.
///
/// The :attr:`dual` of a pipe is given by another :class:`LegPipe`, which consists of the
/// dual of each of the :attr:`legs`, *in reverse order*. We also flip the :attr:`is_dual`
/// attribute to keep track of that (but the attribute has no further meaning).
class LegPipe : public Leg
{
  public:
    using Ptr = std::shared_ptr<LegPipe>;
    using CPtr = std::shared_ptr<const LegPipe>;

    std::vector<Leg::Ptr> legs;
    int64 num_legs = 0;
    /// C-style combine varies the last leg fastest; F-style varies the first fastest.
    bool combine_cstyle = true;

    explicit LegPipe(std::vector<Leg::Ptr> legs, bool is_dual = false, bool combine_cstyle = true);
    ~LegPipe() override = default;

    void test_sanity() const override;

    py::object as_Space() override;

    Leg::Ptr dual() const override;

    bool is_trivial() const override;

    std::vector<Leg::Ptr> flat_legs() override;

    std::vector<Leg::Ptr> flat_spaces() override;

    int64 num_flat_legs() const override;

    std::vector<int64> _flat_leg_permutation(int64 offset = 0) const override;

    void set_basis_perm(std::optional<std::vector<int64>> basis_perm) override;

    void set_inverse_basis_perm(std::optional<std::vector<int64>> inverse_basis_perm) override;

    std::string ascii_arrow() const override;

    bool operator==(Leg const& other) const override;

    /// Distinguishes :class:`AbelianLegPipe` for equality checks. Override there.
    [[nodiscard]] virtual bool is_abelian_leg_pipe() const { return false; }

    Leg::Ptr operator[](int64 idx) const;

    [[nodiscard]] std::string repr(bool show_symmetry = true, bool one_line = false) const;
};

} // namespace cyten
// CHECKME: the following was generated by .cursor/skills/pybind11-codegen/pybind11_codegen.py gen_cpp_declaration --py-name ElementarySpace --header-file include/cyten/symmetries/spaces.h

/// A :class:`Space` that is defined as (the dual of) a direct sum of sectors.
class ElementarySpace : public Space, public Leg {
public:
public:
    ElementarySpace(
        Symmetry::Ptr symmetry,
        Sector defining_sectors,
        ndarray multiplicities=py::none(),
        bool is_dual=false,
        std::optional<std::vector<int64>> basis_perm=py::none()
    );
    virtual ~ElementarySpace() = default;
    /// The sector (from the :attr:`sector_decomposition`) of each basis vector.
    /// property getter
    TYPEOF_return get_sectors_of_basis();
    /// property getter
    virtual ElementarySpace get_dual() override;
    /// Perform sanity checks.
    virtual TYPEOF_return test_sanity() override;
    /// Create an ElementarySpace by specifying the sector of every basis element.
    virtual ElementarySpace from_basis(Symmetry::Ptr symmetry, Sequence_Sequence_int__ sectors_of_basis);
    /// Create an ElementarySpace with multiple independent symmetries.
    virtual ElementarySpace from_independent_symmetries(std::vector<Leg::Ptr> independent_descriptions);
    /// The largest common subspace of a list of spaces.
    ElementarySpace from_largest_common_subspace(bool is_dual=false);
    /// The zero-dimensional space, i.e. the span of the empty set.
    virtual ElementarySpace from_null_space(Symmetry::Ptr symmetry, bool is_dual=false);
    /// Similar to the constructor, but with fewer requirements.
    virtual ElementarySpace_tuple_ElementarySpace_ndarray_ from_defining_sectors(
        Symmetry::Ptr symmetry,
        Sector defining_sectors,
        Sequence_int_ multiplicities=py::none(),
        bool is_dual=false,
        ndarray basis_perm=py::none(),
        bool unique_sectors=false,
        bool return_sorting_perm=false
    );
    /// Create a :class:`ElementarySpace` that has a given :attr:`sector_decomposition`.
    ElementarySpace from_sector_decomposition(
        Symmetry::Ptr symmetry,
        Sector sector_decomposition,
        Sequence_int_ multiplicities=py::none(),
        bool is_dual=false,
        ndarray basis_perm=py::none(),
        bool unique_sectors=false
    );
    /// Create an ElementarySpace that lives in the trivial sector (i.e. it is symmetric).
    virtual ElementarySpace from_trivial_sector(
        int64 dim=1,
        Symmetry::Ptr symmetry=no_symmetry,
        bool is_dual=false,
        ndarray basis_perm=py::none()
    );
    /// The sector (from the :attr:`sector_decomposition`) of each basis vector.
    /// property getter
    TYPEOF_return get_sectors_of_basis();
    virtual str __repr__(bool show_symmetry=true, TYPEOF_one_line one_line=false);
    virtual bool operator==(TYPEOF_other other) override;
    virtual ElementarySpace as_ElementarySpace(bool is_dual=false) override;
    /// The ket space (``is_dual=False``) isomorphic or equal to self.
    TYPEOF_return as_ket_space();
    /// The bra space (``is_dual=False``) isomorphic or equal to self.
    TYPEOF_return as_bra_space();
    virtual ElementarySpace change_symmetry(Symmetry::Ptr symmetry, SectorMapFn sector_map, bool injective=false) override;
    /// Form the direct sum (i.e. stacking).
    ElementarySpace direct_sum();
    virtual TYPEOF_return drop_symmetry(int64 which="all") override;
    /// property getter
    virtual ElementarySpace get_dual() override;
    /// Utility function to translate an index.
    tuple_int_int_ parse_index(int64 idx);
    Sector idx_to_sector(int64 idx);
    /// Take a "slice" of the leg, keeping only some of the basis states.
    virtual ElementarySpace take_slice(BlockBackend::BlockPtr blockmask);
    /// A space isomorphic to self with opposite ``is_dual`` attribute.
    virtual TYPEOF_return with_opposite_duality();
    /// A space isomorphic to self with given ``is_dual`` attribute.
    ElementarySpace with_is_dual(bool is_dual);
    TYPEOF_return save_hdf5(TYPEOF_hdf5_saver hdf5_saver, TYPEOF_h5gr h5gr, TYPEOF_subpath subpath);
    TYPEOF_return from_hdf5(TYPEOF_hdf5_loader hdf5_loader, TYPEOF_h5gr h5gr, TYPEOF_subpath subpath);
};
