#include <cyten/tensors/direct_sum.h>
#include <cyten/tensors/ops_algebra.h>
#include <cyten/tensors/tensor.h>

#include "../py_cyten_pybind11.h"

#include <pybind11/stl.h>

#include <sstream>
#include <string>
#include <vector>

namespace cyten {

void
bind_tensors_direct_sum(py::module_& m)
{
    py::class_<DirectSum, VectorLike, py::smart_holder> cls(m, "DirectSum");
    cls.doc() = R"pydoc(
Direct-sum vector: a list of tensors treated as one :class:`~cyten.tensors.VectorLike`

Addition and scalar multiplication are defined componentwise (zip over :attr:`components`).
Components need not share legs (they may live on different spaces / sites).
Nested :class:`DirectSum` is not supported.
Inner product and norm are the standard direct-sum Hilbert-space inner product
:math:`\langle X \mid Y \rangle = \sum_i \langle X_i \mid Y_i \rangle` and
:math:`\|X\| = \sqrt{\sum_i \|X_i\|^2}`, where the sum runs over components of the DirectSum
and each :math:`\langle X_i \mid Y_i \rangle` is the usual (Frobenius) tensor inner product
on the corresponding component.

Intended for Krylov / :class:`~cyten.sparse.LinearOperator` algorithms.
An example usage is a a VUMPS calculation or tangent-space excitations
with a multi-site unit cell: after gauge-fixing, we are left with an orthonormal
parametrization of the ground state/excitations with one tensor per site in the unit cell,
corresponding to one `component` entry in this DirectSum class.

Note that this class **assumes* that cross terms of the inner product between different
components are zero!
This is only sensible if you really have such an orthonormal parametrization
of vectors in the full (many-body) Hilbert space.
)pydoc";

    cls.def(py::init<std::vector<TensorPtr>>(),
            py::arg("components"),
            R"pydoc(Construct from a non-empty sequence of tensors.)pydoc");

    cls.def_property_readonly(
      "components",
      [](DirectSum const& self) { return self.components(); },
      R"pydoc(The component tensors.)pydoc");
    cls.def_property_readonly("dtype", &DirectSum::dtype);
    cls.def_property_readonly("device", &DirectSum::device);
    cls.def_property_readonly("backend", &DirectSum::backend);

    cls.def("copy",
            &DirectSum::copy,
            py::arg("deep") = true,
            R"pydoc(Copy all component tensors.)pydoc");
    cls.def("__len__", &DirectSum::size);
    cls.def(
      "__getitem__", [](DirectSum const& self, int64 i) { return self.at(i); }, py::arg("i"));
    cls.def("__repr__", [](DirectSum const& self) {
        std::ostringstream os;
        os << "<DirectSum of " << self.size() << " tensors, dtype=" << dtype::repr(self.dtype())
           << '>';
        return os.str();
    });
    // DirectSum is a sequence (__len__/__getitem__), so numpy would otherwise treat
    // ``np.float64 * ds`` as iterating the components into an object array.
    cls.attr("__array_ufunc__") = py::none();
}

} // namespace cyten
