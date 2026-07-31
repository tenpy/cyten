#pragma once

#include <cstddef>
#include <cyten/cyten.h>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace cyten {

class BigOPolynomial;

/// A symbolic representation of an algorithmic cost as a monomial.
///
/// A monomial is of the form ``x^a y^b z^c``, i.e. a product of integer powers.
class BigOMonomial
{
  public:
    /// Factor map: entry ``{"x", n}`` represents the symbol factor ``x^n``.
    std::map<std::string, int64> factors;

    explicit BigOMonomial(std::map<std::string, int64> factors = {});

    /// Initialize from a string representation like ``'x^2 y^3'``.
    static BigOMonomial from_str(std::string const& mono);

    BigOPolynomial operator+(BigOMonomial const& other) const;
    BigOMonomial operator*(BigOMonomial const& other) const;
    bool operator==(BigOMonomial const& other) const;
    bool operator!=(BigOMonomial const& other) const;
    bool operator<(BigOMonomial const& other) const;

    std::size_t hash() const;
    std::string str() const;
    std::string repr() const;

    /// If the given monomial is negligible compared to `others`, s.t. ``O(self + x) = O(x)``.
    bool is_negligible(std::vector<BigOMonomial> const& others,
                       std::optional<std::vector<std::pair<BigOMonomial, BigOMonomial>>>
                         relations = std::nullopt) const;
};

/// A symbolic representation of an algorithmic cost as a polynomial.
///
/// A polynomial is a sum of :class:`BigOMonomial`\ s, e.g. ``x^a y^b + y^c z^d``.
class BigOPolynomial
{
  public:
    /// The terms such that the polynomial is their sum (unique monomials).
    std::set<BigOMonomial> terms;

    explicit BigOPolynomial(std::set<BigOMonomial> terms = {});

    /// Simplify terms by dropping duplicates and negligible monomials.
    static std::set<BigOMonomial> simplify_terms(
      std::set<BigOMonomial> const& terms,
      std::optional<std::vector<std::pair<BigOMonomial, BigOMonomial>>> relations = std::nullopt);

    /// Initialize from a string representation like ``'x^2 y^3 + x^4'``.
    static BigOPolynomial from_str(std::string const& poly);

    std::string str() const;
    std::string repr() const;

    BigOPolynomial operator+(BigOPolynomial const& other) const;
    BigOPolynomial operator+(BigOMonomial const& other) const;
    BigOPolynomial operator+(std::string const& other) const;

    BigOPolynomial operator*(BigOPolynomial const& other) const;
    BigOPolynomial operator*(BigOMonomial const& other) const;
    BigOPolynomial operator*(std::string const& other) const;

    bool operator==(BigOPolynomial const& other) const;
    bool operator==(BigOMonomial const& other) const;
    bool operator!=(BigOPolynomial const& other) const;
    bool operator!=(BigOMonomial const& other) const;

    std::size_t hash() const;

    /// Product of this polynomial with zero or more others.
    BigOPolynomial prod(std::vector<BigOPolynomial> const& others = {}) const;
};

inline BigOPolynomial
operator+(std::string const& left, BigOPolynomial const& right)
{
    return right + left;
}

inline BigOPolynomial
operator*(std::string const& left, BigOPolynomial const& right)
{
    return right * left;
}

inline BigOPolynomial
operator+(BigOMonomial const& left, BigOPolynomial const& right)
{
    return right + left;
}

inline BigOPolynomial
operator*(BigOMonomial const& left, BigOPolynomial const& right)
{
    return right * left;
}

} // namespace cyten
