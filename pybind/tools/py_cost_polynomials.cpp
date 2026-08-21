#include <cyten/tools/cost_polynomials.h>

#include "../doc_plus.h"
#include "../py_cyten_pybind11.h"
#include "docstrings/tools/cost_polynomials.h"

namespace py = pybind11;
namespace cyten {

namespace {

py::object
not_implemented()
{
    return py::reinterpret_borrow<py::object>(Py_NotImplemented);
}

py::object
poly_add(BigOPolynomial const& self, py::handle other)
{
    if (py::isinstance<py::str>(other))
        return py::cast(self + other.cast<std::string>());
    if (py::isinstance<BigOMonomial>(other))
        return py::cast(self + other.cast<BigOMonomial>());
    if (py::isinstance<BigOPolynomial>(other))
        return py::cast(self + other.cast<BigOPolynomial>());
    return not_implemented();
}

py::object
poly_mul(BigOPolynomial const& self, py::handle other)
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
        return BigOPolynomial{ { obj.cast<BigOMonomial>() } };
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
bind_cost_polynomials(py::module_& m)
{
    py::class_<BigOMonomial> big_omonomial(m, "BigOMonomial");
    py::class_<BigOPolynomial> big_opolynomial(m, "BigOPolynomial");

    big_omonomial.doc() = doc_cpp_ref(R"pydoc(BigOPolynomial)pydoc", "cyten::BigOPolynomial");

    big_omonomial.def(py::init<std::map<std::string, int64>>(), py::arg("factors"))
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
      .def(
        "__add__",
        [](BigOMonomial const& self, py::handle other) -> py::object {
            if (!py::isinstance<BigOMonomial>(other))
                return not_implemented();
            return py::cast(self + other.cast<BigOMonomial>());
        },
        py::arg("other"))
      .def(
        "__mul__",
        [](BigOMonomial const& self, py::handle other) -> py::object {
            if (!py::isinstance<BigOMonomial>(other))
                return not_implemented();
            return py::cast(self * other.cast<BigOMonomial>());
        },
        py::arg("other"))
      .def(
        "__eq__",
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
        doc_cpp_ref(R"pydoc(is_negligible)pydoc", "cyten::BigOPolynomial::is_negligible()"));

    big_opolynomial.doc() = doc_cpp_ref(R"pydoc(BigOPolynomial)pydoc", "cyten::BigOPolynomial");

    big_opolynomial
      .def(py::init([](py::object terms) {
               if (terms.is_none())
                   return BigOPolynomial{};
               if (py::isinstance<py::set>(terms))
                   return BigOPolynomial{ terms.cast<std::set<BigOMonomial>>() };
               std::set<BigOMonomial> as_set;
               for (auto item : terms)
                   as_set.insert(item.cast<BigOMonomial>());
               return BigOPolynomial{ std::move(as_set) };
           }),
           py::arg("terms") = py::none())
      .def_readwrite("terms", &BigOPolynomial::terms)
      .def_static(
        "simplify_terms",
        [](py::object terms, py::object relations) {
            std::set<BigOMonomial> as_set;
            if (py::isinstance<py::set>(terms))
                as_set = terms.cast<std::set<BigOMonomial>>();
            else
                for (auto item : terms)
                    as_set.insert(item.cast<BigOMonomial>());
            return BigOPolynomial::simplify_terms(as_set, optional_relations(relations));
        },
        py::arg("terms"),
        py::arg("relations") = py::none(),
        "Simplify terms by dropping duplicates and negligible monomials.")
      .def_static(
        "from_str",
        [](py::handle poly) {
            if (py::isinstance<BigOPolynomial>(poly))
                return poly.cast<BigOPolynomial>();
            if (py::isinstance<BigOMonomial>(poly))
                return BigOPolynomial{ { poly.cast<BigOMonomial>() } };
            return BigOPolynomial::from_str(py::str(poly).cast<std::string>());
        },
        py::arg("poly"),
        "Initialize from a string representation like ``'x^2 y^3 + x^4'``.")
      .def("__str__", &BigOPolynomial::str)
      .def("__repr__", &BigOPolynomial::repr)
      .def("__add__", &poly_add, py::arg("other"))
      .def("__radd__", &poly_add, py::arg("other"))
      .def("__mul__", &poly_mul, py::arg("other"))
      .def("__rmul__", &poly_mul, py::arg("other"))
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
