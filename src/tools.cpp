#include <cyten/tools.h>

#include <format>
#include <functional>
#include <ranges>
#include <sstream>

#include <pybind11/numpy.h>

namespace cyten {

NotImplemented::NotImplemented(std::string name)
  : std::logic_error(std::format("Not implemented: {}", name)) {};

/// Format elements of an iterable as if it were a plain list.
std::string
format_like_list(py::iterable it)
{
    std::string result = "[";
    bool first = true;
    for (auto&& item : it) {
        if (!first)
            result += ", ";
        result += py::str(item).cast<std::string>();
        first = false;
    }
    result += "]";
    return result;
}

bool
is_iterable(py::object a)
{
    try {
        py::iter(a);
        return true;
    } catch (py::error_already_set& m) {
        // expected error: TypeError if not iterable
        if (!m.matches(PyExc_TypeError))
            throw;
    }
    return false;
}

py::object
to_iterable(py::object a)
{
    if (!py::isinstance<py::str>(a) && is_iterable(a))
        return a;
    py::list result(1);
    result[0] = a;
    return result;
}

int64
to_valid_idx(int64 idx, int64 length)
{
    if (idx < -length || idx >= length)
        throw std::out_of_range("Index " + std::to_string(idx) + " out of bounds for length " +
                                std::to_string(length));
    if (idx < 0)
        idx += length;
    return idx;
}

namespace {

template<typename T>
std::size_t
hash_combine(std::size_t seed, T const& value)
{
    // boost::hash_combine style
    return seed ^ (std::hash<T>{}(value) + 0x9e3779b9 + (seed << 6) + (seed >> 2));
}

} // namespace

BigOMonomial::BigOMonomial(std::map<std::string, int64> factors)
  : factors(std::move(factors))
{
}

BigOMonomial
BigOMonomial::from_str(std::string const& mono)
{
    auto trimmed = mono;
    // strip whitespace from both ends
    auto start = trimmed.find_first_not_of(" \t\n\r");
    if (start == std::string::npos)
        return BigOMonomial{};
    auto end = trimmed.find_last_not_of(" \t\n\r");
    trimmed = trimmed.substr(start, end - start + 1);

    std::map<std::string, int64> factors;
    std::istringstream iss(trimmed);
    std::string token;
    while (iss >> token) {
        auto caret = token.find('^');
        std::string dim;
        int64 exp = 1;
        if (caret == std::string::npos) {
            dim = token;
        } else if (token.find('^', caret + 1) == std::string::npos) {
            dim = token.substr(0, caret);
            exp = std::stoll(token.substr(caret + 1));
            if (exp <= 0)
                throw std::invalid_argument(std::format("Invalid monomial: \"{}\"", mono));
        } else {
            throw std::invalid_argument(std::format("Invalid monomial: \"{}\"", mono));
        }
        factors[dim] += exp;
    }
    return BigOMonomial{ std::move(factors) };
}

BigOPolynomial
BigOMonomial::operator+(BigOMonomial const& other) const
{
    return BigOPolynomial{ { *this, other } };
}

BigOMonomial
BigOMonomial::operator*(BigOMonomial const& other) const
{
    auto result_factors = factors;
    for (auto const& [s, e] : other.factors)
        result_factors[s] += e;
    return BigOMonomial{ std::move(result_factors) };
}

bool
BigOMonomial::operator==(BigOMonomial const& other) const
{
    for (auto const& [s, e] : factors) {
        auto it = other.factors.find(s);
        if ((it == other.factors.end() ? 0 : it->second) != e)
            return false;
    }
    for (auto const& [s, e] : other.factors) {
        auto it = factors.find(s);
        if ((it == factors.end() ? 0 : it->second) != e)
            return false;
    }
    return true;
}

bool
BigOMonomial::operator!=(BigOMonomial const& other) const
{
    return !(*this == other);
}

std::size_t
BigOMonomial::hash() const
{
    std::size_t h = 0;
    for (auto const& [dim, exp] : factors) {
        h = hash_combine(h, dim);
        h = hash_combine(h, exp);
    }
    return h;
}

std::string
BigOMonomial::str() const
{
    std::string result;
    bool first = true;
    for (auto const& [dim, exp] : factors) {
        if (!first)
            result += ' ';
        result += std::format("{}^{}", dim, exp);
        first = false;
    }
    return result;
}

std::string
BigOMonomial::repr() const
{
    return std::format("<BigOMonomial {} >", str());
}

bool
BigOMonomial::is_negligible(
  std::vector<BigOMonomial> const& others,
  std::optional<std::vector<std::pair<BigOMonomial, BigOMonomial>>> relations) const
{
    if (relations.has_value())
        throw NotImplemented("BigOMonomial::is_negligible with relations");
    for (auto const& o : others) {
        bool all_le = true;
        for (auto const& [x, n] : factors) {
            auto it = o.factors.find(x);
            int64 other_exp = (it == o.factors.end() ? 0 : it->second);
            if (n > other_exp) {
                all_le = false;
                break;
            }
        }
        if (all_le)
            return true;
    }
    return false;
}

BigOPolynomial::BigOPolynomial(std::vector<BigOMonomial> terms)
  : terms(simplify_terms(terms))
{
}

std::vector<BigOMonomial>
BigOPolynomial::simplify_terms(
  std::vector<BigOMonomial> const& terms,
  std::optional<std::vector<std::pair<BigOMonomial, BigOMonomial>>> relations)
{
    std::vector<BigOMonomial> non_negligible;
    for (auto const& t : terms) {
        if (!t.is_negligible(non_negligible, relations))
            non_negligible.push_back(t);
    }
    return non_negligible;
}

BigOPolynomial
BigOPolynomial::from_str(std::string const& poly)
{
    std::vector<BigOMonomial> terms;
    std::size_t start = 0;
    while (start <= poly.size()) {
        auto plus = poly.find('+', start);
        auto part =
          (plus == std::string::npos) ? poly.substr(start) : poly.substr(start, plus - start);
        // strip
        auto s = part.find_first_not_of(" \t\n\r");
        if (s != std::string::npos) {
            auto e = part.find_last_not_of(" \t\n\r");
            terms.push_back(BigOMonomial::from_str(part.substr(s, e - s + 1)));
        }
        if (plus == std::string::npos)
            break;
        start = plus + 1;
    }
    return BigOPolynomial{ std::move(terms) };
}

std::string
BigOPolynomial::str() const
{
    if (terms.empty())
        return "";
    std::string result = terms.front().str();
    for (auto const& t : terms | std::views::drop(1))
        result += " + " + t.str();
    return result;
}

std::string
BigOPolynomial::repr() const
{
    return std::format("<BigOPolynomial {} >", str());
}

BigOPolynomial
BigOPolynomial::operator+(BigOPolynomial const& other) const
{
    std::vector<BigOMonomial> combined = terms;
    combined.insert(combined.end(), other.terms.begin(), other.terms.end());
    return BigOPolynomial{ std::move(combined) };
}

BigOPolynomial
BigOPolynomial::operator+(BigOMonomial const& other) const
{
    return *this + BigOPolynomial{ { other } };
}

BigOPolynomial
BigOPolynomial::operator+(std::string const& other) const
{
    return *this + BigOPolynomial::from_str(other);
}

BigOPolynomial
BigOPolynomial::operator*(BigOPolynomial const& other) const
{
    std::vector<BigOMonomial> product_terms;
    product_terms.reserve(terms.size() * other.terms.size());
    for (auto const& m1 : terms)
        for (auto const& m2 : other.terms)
            product_terms.push_back(m1 * m2);
    return BigOPolynomial{ std::move(product_terms) };
}

BigOPolynomial
BigOPolynomial::operator*(BigOMonomial const& other) const
{
    return *this * BigOPolynomial{ { other } };
}

BigOPolynomial
BigOPolynomial::operator*(std::string const& other) const
{
    return *this * BigOPolynomial::from_str(other);
}

bool
BigOPolynomial::operator==(BigOPolynomial const& other) const
{
    for (auto const& t : terms) {
        bool found = false;
        for (auto const& t2 : other.terms) {
            if (t == t2) {
                found = true;
                break;
            }
        }
        if (!found)
            return false;
    }
    // Match Python's second loop (compares other.terms against other.terms — always true).
    // Kept intentionally for behavioral parity with the original Python.
    for (auto const& t2 : other.terms) {
        bool found = false;
        for (auto const& t : other.terms) {
            if (t == t2) {
                found = true;
                break;
            }
        }
        if (!found)
            return false;
    }
    return true;
}

bool
BigOPolynomial::operator==(BigOMonomial const& other) const
{
    if (terms.size() == 1)
        return terms[0] == other;
    return false;
}

bool
BigOPolynomial::operator!=(BigOPolynomial const& other) const
{
    return !(*this == other);
}

bool
BigOPolynomial::operator!=(BigOMonomial const& other) const
{
    return !(*this == other);
}

std::size_t
BigOPolynomial::hash() const
{
    std::size_t h = 0;
    for (auto const& t : terms)
        h = hash_combine(h, t.hash());
    return h;
}

BigOPolynomial
BigOPolynomial::prod(std::vector<BigOPolynomial> const& others) const
{
    if (others.empty())
        return *this;
    BigOPolynomial result = *this * others.front();
    if (others.size() == 1)
        return result;
    return result.prod(std::vector<BigOPolynomial>(others.begin() + 1, others.end()));
}

} // namespace cyten
