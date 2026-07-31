#include <cyten/tools/cost_polynomials.h>

#include <cyten/tools.h>

#include <format>
#include <functional>
#include <ranges>
#include <sstream>

namespace cyten {

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

bool
BigOMonomial::operator<(BigOMonomial const& other) const
{
    return factors < other.factors;
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

BigOPolynomial::BigOPolynomial(std::set<BigOMonomial> terms)
  : terms(simplify_terms(std::move(terms)))
{
}

std::set<BigOMonomial>
BigOPolynomial::simplify_terms(
  std::set<BigOMonomial> const& terms,
  std::optional<std::vector<std::pair<BigOMonomial, BigOMonomial>>> relations)
{
    // Order-independent: keep t iff it is not negligible compared to any other term.
    std::set<BigOMonomial> non_negligible;
    for (auto const& t : terms) {
        std::vector<BigOMonomial> others;
        others.reserve(terms.size() > 0 ? terms.size() - 1 : 0);
        for (auto const& u : terms) {
            if (u != t)
                others.push_back(u);
        }
        if (!t.is_negligible(others, relations))
            non_negligible.insert(t);
    }
    return non_negligible;
}

BigOPolynomial
BigOPolynomial::from_str(std::string const& poly)
{
    std::set<BigOMonomial> terms;
    std::size_t start = 0;
    while (start <= poly.size()) {
        auto plus = poly.find('+', start);
        auto part =
          (plus == std::string::npos) ? poly.substr(start) : poly.substr(start, plus - start);
        // strip
        auto s = part.find_first_not_of(" \t\n\r");
        if (s != std::string::npos) {
            auto e = part.find_last_not_of(" \t\n\r");
            terms.insert(BigOMonomial::from_str(part.substr(s, e - s + 1)));
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
    std::string result = terms.begin()->str();
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
    std::set<BigOMonomial> combined = terms;
    combined.insert(other.terms.begin(), other.terms.end());
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
    std::set<BigOMonomial> product_terms;
    for (auto const& m1 : terms)
        for (auto const& m2 : other.terms)
            product_terms.insert(m1 * m2);
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
    return terms == other.terms;
}

bool
BigOPolynomial::operator==(BigOMonomial const& other) const
{
    return terms.size() == 1 && *terms.begin() == other;
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
