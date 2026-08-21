#include <cyten/tensors/direct_sum.h>
#include <cyten/tensors/ops_algebra.h>
#include <cyten/tensors/tensor.h>

#include "../doc_plus.h"
#include "../py_cyten_pybind11.h"

#include <pybind11/stl.h>

#include "docstrings/tensors/direct_sum.h"

#include <sstream>
#include <string>
#include <vector>

namespace cyten {

void
bind_tensors_direct_sum(py::module_& m)
{
    py::class_<DirectSum, VectorLike, py::smart_holder> cls(m, "DirectSum");
    cls.doc() = DOC(cyten, DirectSum);

    cls.def(py::init<std::vector<TensorPtr>>(),
            py::arg("components"),
            DOC(cyten, DirectSum, DirectSum));

    cls.def_property_readonly(
      "components",
      [](DirectSum const& self) { return self.components(); },
      DOC(cyten, DirectSum, components));
    cls.def_property_readonly("dtype", &DirectSum::dtype);
    cls.def_property_readonly("device", &DirectSum::device);
    cls.def_property_readonly("backend", &DirectSum::backend);

    cls.def("copy",
            &DirectSum::copy,
            py::arg("deep") = true,
            DOC(cyten, DirectSum, copy));
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
