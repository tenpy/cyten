#pragma once

#include "exceptions.h"
#include "symmetry.h"

#include <cstdint>
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
    /// Returns a Python object until :class:`Space` is converted to C++.
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

  protected:
    /// Internal basis permutation; ``nullopt`` means identity. Meaningless if symmetry cannot be
    /// dropped.
    std::optional<std::vector<int64>> _basis_perm;
    std::optional<std::vector<int64>> _inverse_basis_perm;
};

} // namespace cyten
