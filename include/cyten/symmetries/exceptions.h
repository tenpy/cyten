#pragma once

#include <stdexcept>
#include <string>

namespace cyten {

/// Raised when something is not possible or not allowed due to symmetry.
class SymmetryError : public std::runtime_error
{
  public:
    using std::runtime_error::runtime_error;
};

/// Raised when a braid chirality should be specified but was not.
class BraidChiralityUnspecifiedError : public SymmetryError
{
  public:
    using SymmetryError::SymmetryError;
};

} // namespace cyten
