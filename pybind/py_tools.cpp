#include <cyten/tools.h>

#include "py_cyten_pybind11.h"

namespace py = pybind11;
namespace cyten {

namespace {

py::object
not_implemented()
{
    return py::reinterpret_borrow<py::object>(Py_NotImplemented);
}

template<typename Self>
py::object
poly_add(Self const& self, py::handle other)
{
    if (py::isinstance<py::str>(other))
        return py::cast(self + other.cast<std::string>());
    if (py::isinstance<BigOMonomial>(other))
        return py::cast(self + other.cast<BigOMonomial>());
    if (py::isinstance<BigOPolynomial>(other))
        return py::cast(self + other.cast<BigOPolynomial>());
    return not_implemented();
}

template<typename Self>
py::object
poly_mul(Self const& self, py::handle other)
{
    if (py::isinstance<py::str>(other))
        return py::cast(self * other.cast<std::string>());
    if (py::isinstance<BigOMonomial>(other))
        return py::cast(self * other.cast<BigOMonomial>());
    if (py::isinstance<BigOPolynomial>(other))
        return py::cast(self * other.cast<BigOPolynomial>());
    return not_implemented();
}

BigOPolynomial
as_polynomial(py::handle obj)
{
    if (py::isinstance<BigOPolynomial>(obj))
        return obj.cast<BigOPolynomial>();
    if (py::isinstance<BigOMonomial>(obj))
        return BigOPolynomial{{obj.cast<BigOMonomial>()}};
    if (py::isinstance<py::str>(obj))
        return BigOPolynomial::from_str(obj.cast<std::string>());
    throw py::type_error("expected BigOPolynomial, BigOMonomial, or str");
}

std::optional<std::vector<std::pair<BigOMonomial, BigOMonomial>>>
optional_relations(py::object relations)
{
    if (relations.is_none())
        return std::nullopt;
    return relations.cast<std::vector<std::pair<BigOMonomial, BigOMonomial>>>();
}

} // namespace

void
bind_tools(py::module_& m)
{

    m.def("format_like_list",
          &cyten::format_like_list,
          R"pydoc(
          Format elements of an iterable as if it were a plain list.

          This means surrounding them with brackets and separating them by `', '`.
          )pydoc",
          py::arg("it"));

    m.def("is_iterable", &cyten::is_iterable, py::arg("a"), "If the given object is iterable.");

    m.def("to_iterable",
          &cyten::to_iterable,
          py::arg("a"),
          "If `a` is a not iterable or a string, return [a], else return a.");

    m.def("to_valid_idx",
          &cyten::to_valid_idx,
          py::arg("idx"),
          py::arg("length"),
          "Convert to a valid non-negative index into the given length.");

    py::class_<BigOMonomial> big_omonomial(m, "BigOMonomial");
    py::class_<BigOPolynomial> big_opolynomial(m, "BigOPolynomial");

    big_omonomial.doc() = R"pydoc(
        A symbolic representation of an algorithmic cost as a monomial.

        A monomial is of the form ``x^a y^b z^c``, i.e. a product of integer powers.

        Attributes
        ----------
        factors : dict {str: int}
            The factor, where an entry ``{'x': n}`` represents the symbol factor ``x^n``.
        )pydoc";

    big_omonomial
      .def(py::init<std::map<std::string, int64>>(), py::arg("factors"))
      .def_readwrite("factors", &BigOMonomial::factors)
      .def_static(
        "from_str",
        [](py::handle mono) {
            if (py::isinstance<BigOMonomial>(mono))
                return mono.cast<BigOMonomial>();
            return BigOMonomial::from_str(py::str(mono).cast<std::string>());
        },
        py::arg("mono"),
        "Initialize from a string representation like ``'x^2 y^3'``.")
      .def("__add__",
           [](BigOMonomial const& self, py::handle other) -> py::object {
               if (!py::isinstance<BigOMonomial>(other))
                   return not_implemented();
               return py::cast(self + other.cast<BigOMonomial>());
           },
           py::arg("other"))
      .def("__mul__",
           [](BigOMonomial const& self, py::handle other) -> py::object {
               if (!py::isinstance<BigOMonomial>(other))
                   return not_implemented();
               return py::cast(self * other.cast<BigOMonomial>());
           },
           py::arg("other"))
      .def("__eq__",
           [](BigOMonomial const& self, py::handle other) -> py::object {
               if (!py::isinstance<BigOMonomial>(other))
                   return not_implemented();
               return py::cast(self == other.cast<BigOMonomial>());
           },
           py::arg("other"))
      .def("__hash__", &BigOMonomial::hash)
      .def("__str__", &BigOMonomial::str)
      .def("__repr__", &BigOMonomial::repr)
      .def(
        "is_negligible",
        [](BigOMonomial const& self, py::args others, py::kwargs kwargs) {
            py::object relations = py::none();
            if (kwargs.contains("relations")) {
                relations = kwargs["relations"];
                kwargs.attr("__delitem__")("relations");
            }
            if (kwargs.size() != 0)
                throw py::type_error("is_negligible() got unexpected keyword argument(s)");
            std::vector<BigOMonomial> ovec;
            ovec.reserve(others.size());
            for (auto item : others)
                ovec.push_back(item.cast<BigOMonomial>());
            return self.is_negligible(ovec, optional_relations(relations));
        },
        R"pydoc(
        If the given monomial is negligible compared to `others`, s.t. ``O(self + x) = O(x)``.
        )pydoc");

    big_opolynomial.doc() = R"pydoc(
        A symbolic representation of an algorithmic cost as a monomial.

        A polynomial is a sum of :class:`BigOMonomials`\ s, i.e. it is of the form::

            x^a y^b + y^c z^d

        i.e. a sum of terms, which consist of integer powers of symbols.

        Polynomials can be added and multiplied and compared via :meth:`is_negligible`.

        Attributes
        ----------
        terms : list of BigOMonomial
            The terms such that the polynomial is their sum.
        )pydoc";

    big_opolynomial
      .def(py::init([](py::object terms) {
               if (terms.is_none())
                   return BigOPolynomial{};
               return BigOPolynomial{terms.cast<std::vector<BigOMonomial>>()};
           }),
           py::arg("terms") = py::none())
      .def_readwrite("terms", &BigOPolynomial::terms)
      .def_static(
        "simplify_terms",
        [](std::vector<BigOMonomial> const& terms, py::object relations) {
            return BigOPolynomial::simplify_terms(terms, optional_relations(relations));
        },
        py::arg("terms"),
        py::arg("relations") = py::none(),
        "Simplify a list of terms by dropping negligible terms.")
      .def_static(
        "from_str",
        [](py::handle poly) {
            if (py::isinstance<BigOPolynomial>(poly))
                return poly.cast<BigOPolynomial>();
            if (py::isinstance<BigOMonomial>(poly))
                return BigOPolynomial{{poly.cast<BigOMonomial>()}};
            return BigOPolynomial::from_str(py::str(poly).cast<std::string>());
        },
        py::arg("poly"),
        "Initialize from a string representation like ``'x^2 y^3 + x^4'``.")
      .def("__str__", &BigOPolynomial::str)
      .def("__repr__", &BigOPolynomial::repr)
      .def("__add__", &poly_add<BigOPolynomial>, py::arg("other"))
      .def("__radd__", &poly_add<BigOPolynomial>, py::arg("other"))
      .def("__mul__", &poly_mul<BigOPolynomial>, py::arg("other"))
      .def("__rmul__", &poly_mul<BigOPolynomial>, py::arg("other"))
      .def(
        "__eq__",
        [](BigOPolynomial const& self, py::handle other) -> py::object {
            if (py::isinstance<BigOMonomial>(other))
                return py::cast(self == other.cast<BigOMonomial>());
            if (py::isinstance<BigOPolynomial>(other))
                return py::cast(self == other.cast<BigOPolynomial>());
            return not_implemented();
        },
        py::arg("other"))
      .def("__hash__", &BigOPolynomial::hash)
      .def(
        "prod",
        [](BigOPolynomial const& self, py::args others) {
            std::vector<BigOPolynomial> ovec;
            ovec.reserve(others.size());
            for (auto item : others)
                ovec.push_back(as_polynomial(item));
            return self.prod(ovec);
        },
        "Product of multiply symmetries");
}

} // namespace cyten
