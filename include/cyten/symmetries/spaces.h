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
#include <tuple>
#include <utility>
#include <vector>

namespace cyten {

/// Common base of :class:`Leg` and :class:`Space`, providing ``shared_from_this``.
///
/// :class:`ElementarySpace` inherits from both :class:`Space` and :class:`Leg`. They must
/// therefore share a single ``std::enable_shared_from_this`` base; two independent ones would
/// be ambiguous, such that ``shared_from_this()`` throws ``std::bad_weak_ptr``.
/// Not exposed to Python.
class LegOrSpace : public std::enable_shared_from_this<LegOrSpace>
{
  public:
    using Ptr = std::shared_ptr<LegOrSpace>;

    virtual ~LegOrSpace() = default;
};

/// Common base class for a single leg of a tensor.
///
/// A single leg on a tensor can either be an :class:`ElementarySpace` or, e.g. as the result
/// of combining legs, a :class:`LegPipe`.
class Leg : public virtual LegOrSpace
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
    virtual py::object as_Space() = 0;

    /// Convert to an isomorphic :class:`ElementarySpace`.
    virtual py::object as_ElementarySpace(bool is_dual = false);

    /// The dual leg (hook for MI; override :meth:`dual_leg`).
    Ptr dual() const { return dual_leg(); }

    /// Implement dual for :class:`Leg` subclasses.
    virtual Ptr dual_leg() const = 0;

    virtual bool is_trivial() const = 0;

    [[nodiscard]] std::vector<int64> basis_perm() const;

    virtual void set_basis_perm(std::optional<std::vector<int64>> basis_perm);

    [[nodiscard]] std::vector<int64> inverse_basis_perm() const;

    virtual void set_inverse_basis_perm(std::optional<std::vector<int64>> inverse_basis_perm);

    virtual std::vector<Ptr> flat_legs();

    virtual std::vector<Ptr> flat_spaces();

    virtual int64 num_flat_legs() const;

    virtual std::vector<int64> _flat_leg_permutation(int64 offset = 0) const;

    virtual std::string ascii_arrow() const;

    virtual bool operator==(Leg const& other) const = 0;

    py::array apply_basis_perm(py::array arr,
                               int64 axis = 0,
                               bool inverse = false,
                               bool pre_compose = false) const;

    [[nodiscard]] bool has_custom_basis_perm() const noexcept { return _basis_perm.has_value(); }

    /// ``shared_from_this()``, downcast to :class:`Leg`.
    [[nodiscard]] Ptr shared_leg();

  protected:
    /// For subclasses that inherit :class:`Leg` *virtually*.
    ///
    /// A virtual base is initialized by the constructor of the most derived class, which for
    /// pybind11 trampolines is generated from an inherited constructor and can not pass any
    /// arguments. Such subclasses therefore leave the :class:`Leg` base default-constructed and
    /// call :meth:`init_leg` in their constructor body instead.
    Leg() = default;

    /// Set the state of the :class:`Leg` base, see :meth:`Leg::Leg`.
    void init_leg(Symmetry::Ptr symmetry,
                  float64 dim,
                  bool is_dual,
                  std::optional<std::vector<int64>> basis_perm = std::nullopt);

    std::optional<std::vector<int64>> _basis_perm;
    std::optional<std::vector<int64>> _inverse_basis_perm;
};

using SectorMapFn = std::function<SectorArray(SectorArray const&)>;

/// Base class for symmetry spaces, see :class:`ElementarySpace` for the standard case.
class Space : public virtual LegOrSpace
{
  public:
    using Ptr = std::shared_ptr<Space>;
    using CPtr = std::shared_ptr<const Space>;

    Symmetry::Ptr symmetry;
    SectorArray sector_decomposition;
    std::optional<std::string> sector_order;
    std::vector<int64> multiplicities;
    int64 num_sectors = 0;
    std::optional<std::vector<int64>> sector_dims;
    std::vector<float64> sector_qdims;
    std::optional<std::vector<std::array<int64, 2>>> slices;
    float64 dim = 0.;

    Space(Symmetry::Ptr symmetry,
          SectorArray sector_decomposition,
          std::optional<std::vector<int64>> multiplicities = std::nullopt,
          std::optional<std::string> sector_order = std::nullopt);
    virtual ~Space() = default;

    virtual void test_sanity() const;

    /// The dual space (hook for MI; override :meth:`dual_space`).
    Ptr dual() const { return dual_space(); }

    virtual Ptr dual_space() const = 0;

    virtual bool is_trivial() const;

    virtual bool operator==(Space const& other) const;

    [[nodiscard]] bool is_isomorphic_to(Space const& other) const;

    [[nodiscard]] bool is_subspace_of(Space const& other) const;

    virtual py::object as_ElementarySpace(bool is_dual = false);

    virtual py::object change_symmetry(Symmetry::Ptr symmetry,
                                       SectorMapFn sector_map,
                                       bool injective = false) = 0;

    virtual py::object drop_symmetry(std::optional<std::vector<int64>> which = std::nullopt) = 0;

    /// Not virtual: :class:`ElementarySpace` also inherits the (differently typed)
    /// :meth:`Leg::as_Space` and can only override one of the two.
    Ptr as_Space();

    /// ``shared_from_this()``, downcast to :class:`Space`.
    [[nodiscard]] Ptr shared_space();

    [[nodiscard]] std::optional<int64> sector_decomposition_where(Sector sector) const;

    [[nodiscard]] int64 sector_multiplicity(Sector sector) const;
};

/// A group of legs, i.e. resulting from :func:`~cyten.tensors.combine_legs`.
class LegPipe : public virtual Leg
{
  public:
    using Ptr = std::shared_ptr<LegPipe>;
    using CPtr = std::shared_ptr<const LegPipe>;

    std::vector<Leg::Ptr> legs;
    int64 num_legs = 0;
    bool combine_cstyle = true;

    explicit LegPipe(std::vector<Leg::Ptr> legs, bool is_dual = false, bool combine_cstyle = true);
    ~LegPipe() override = default;

    void test_sanity() const override;

    py::object as_Space() override;

    Leg::Ptr dual_leg() const override;

    bool is_trivial() const override;

    std::vector<Leg::Ptr> flat_legs() override;

    std::vector<Leg::Ptr> flat_spaces() override;

    int64 num_flat_legs() const override;

    std::vector<int64> _flat_leg_permutation(int64 offset = 0) const override;

    void set_basis_perm(std::optional<std::vector<int64>> basis_perm) override;

    void set_inverse_basis_perm(std::optional<std::vector<int64>> inverse_basis_perm) override;

    std::string ascii_arrow() const override;

    bool operator==(Leg const& other) const override;

    [[nodiscard]] virtual bool is_abelian_leg_pipe() const { return false; }

    Leg::Ptr operator[](int64 idx) const;

    [[nodiscard]] std::string repr(bool show_symmetry = true, bool one_line = false) const;
};

/// A :class:`Space` that is defined as (the dual of) a direct sum of sectors.
///
/// Note that :attr:`Space::symmetry` / :attr:`Leg::symmetry` and :attr:`Space::dim` /
/// :attr:`Leg::dim` are separate members of the two bases. They are kept in sync; access them
/// through the :class:`Space` base within this class.
class ElementarySpace
  : public Space
  , public virtual Leg
{
  public:
    using Ptr = std::shared_ptr<ElementarySpace>;
    using CPtr = std::shared_ptr<const ElementarySpace>;

    SectorArray defining_sectors;

    ElementarySpace(Symmetry::Ptr symmetry,
                    SectorArray defining_sectors,
                    std::optional<std::vector<int64>> multiplicities = std::nullopt,
                    bool is_dual = false,
                    std::optional<std::vector<int64>> basis_perm = std::nullopt);
    ~ElementarySpace() override = default;

    void test_sanity() const override;

    static Ptr from_basis(Symmetry::Ptr symmetry, SectorArray sectors_of_basis);

    static Ptr from_independent_symmetries(std::vector<Ptr> const& independent_descriptions);

    static Ptr from_largest_common_subspace(std::vector<Space::Ptr> const& spaces,
                                            bool is_dual = false);

    static Ptr from_null_space(Symmetry::Ptr symmetry, bool is_dual = false);

    static Ptr from_defining_sectors(Symmetry::Ptr symmetry,
                                     SectorArray defining_sectors,
                                     std::optional<std::vector<int64>> multiplicities = std::nullopt,
                                     bool is_dual = false,
                                     std::optional<std::vector<int64>> basis_perm = std::nullopt,
                                     bool unique_sectors = false,
                                     std::vector<std::size_t>* return_sorting_perm = nullptr);

    static Ptr from_sector_decomposition(
      Symmetry::Ptr symmetry,
      SectorArray sector_decomposition,
      std::optional<std::vector<int64>> multiplicities = std::nullopt,
      bool is_dual = false,
      std::optional<std::vector<int64>> basis_perm = std::nullopt,
      bool unique_sectors = false);

    static Ptr from_trivial_sector(int64 dim = 1,
                                   Symmetry::Ptr symmetry = nullptr,
                                   bool is_dual = false,
                                   std::optional<std::vector<int64>> basis_perm = std::nullopt);

    /// ``shared_from_this()``, downcast to :class:`ElementarySpace`.
    ///
    /// Const, since the Python methods that "return self" are conceptually const.
    [[nodiscard]] Ptr shared_es() const;

    [[nodiscard]] SectorArray sectors_of_basis() const;

    [[nodiscard]] std::string repr(bool show_symmetry = true, bool one_line = false) const;

    bool operator==(Leg const& other) const override;
    bool operator==(Space const& other) const override;

    /// Common implementation of both ``operator==`` overloads.
    [[nodiscard]] bool equals_es(ElementarySpace const& other) const;

    py::object as_ElementarySpace(bool is_dual = false) override;

    [[nodiscard]] Ptr as_ket_space();
    [[nodiscard]] Ptr as_bra_space();

    py::object change_symmetry(Symmetry::Ptr symmetry,
                               SectorMapFn sector_map,
                               bool injective = false) override;

    [[nodiscard]] Ptr direct_sum(std::vector<Ptr> const& others) const;

    py::object drop_symmetry(std::optional<std::vector<int64>> which = std::nullopt) override;

    Space::Ptr dual_space() const override;
    Leg::Ptr dual_leg() const override;

    [[nodiscard]] Ptr dual_es() const;

    std::pair<int64, int64> parse_index(int64 idx) const;

    Sector idx_to_sector(int64 idx) const;

    [[nodiscard]] Ptr take_slice(py::array blockmask) const;

    [[nodiscard]] Ptr with_opposite_duality() const;

    [[nodiscard]] Ptr with_is_dual(bool is_dual) const;

    py::object as_Space() override;

    bool is_trivial() const override;

    std::string ascii_arrow() const override;

    void save_hdf5(py::object hdf5_saver, py::object h5gr, std::string const& subpath) const;

    static Ptr from_hdf5(py::object hdf5_loader, py::object h5gr, std::string const& subpath);
};

} // namespace cyten
