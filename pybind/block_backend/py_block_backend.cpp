#include "../doc_plus.h"
#include "../py_cyten_pybind11.h"
#include "docstrings/block_backend/block_backend.h"
#include "py_array_api.cpp"
#include "py_dtypes.cpp"
#include "py_numpy.cpp"
#include "py_torch.cpp"
#include "py_trampolines.hpp"

#include <cyten/block_backend/array_api.h>
#include <cyten/block_backend/block_backend.h>
#include <cyten/block_backend/numpy.h>
#include <cyten/block_backend/torch.h>
#include <cyten/symmetries/spaces.h>
#include <pybind11/detail/common.h>
#include <span>
#include <sstream>

namespace cyten {

namespace {

bool
is_py_integer_scalar(const py::handle& obj)
{
    if (obj.is_none())
        return false;
    // Numpy arrays are sequences and may expose __index__, but are not scalar indices.
    if (py::isinstance<py::array>(obj))
        return false;
    if (py::isinstance<py::slice>(obj))
        return false;
    try {
        (void)py::cast<int64>(obj);
        return true;
    } catch (const py::cast_error&) {
        return false;
    }
}

/// Parse ``key`` as an integer or tuple/list of integer scalars (not slices, strings, or arrays).
bool
try_parse_int_index_sequence(const py::object& key, std::vector<int64>& indices)
{
    if (is_py_integer_scalar(key)) {
        indices = { key.cast<int64>() };
        return true;
    }
    if (!py::isinstance<py::tuple>(key) && !py::isinstance<py::list>(key))
        return false;
    const py::sequence seq = key;
    indices.clear();
    indices.reserve(static_cast<std::size_t>(seq.size()));
    for (py::ssize_t i = 0; i < seq.size(); ++i) {
        const py::handle item = seq[i];
        if (!is_py_integer_scalar(item))
            return false;
        indices.push_back(item.cast<int64>());
    }
    return true;
}

bool
is_scalar_element_index(const BlockBackend::Block& block,
                        const py::object& key,
                        const std::vector<int64>& indices)
{
    if (indices.size() == 1 && block.ndim() == 1 && is_py_integer_scalar(key))
        return true;
    return indices.size() == static_cast<std::size_t>(block.ndim());
}

} // namespace

void
bind_block_backend(py::module_& m)
{
    bind_block_backend_dtypes(m);

    py::class_<BlockBackend, PyBlockBackend, py::smart_holder> block_backend(m, "BlockBackend");
    block_backend.doc() = "Abstract base class that defines the operation on dense blocks.";

    py::class_<BlockBackend::Block, PyBlock, py::smart_holder>(
      block_backend, "BlockCls", "Abstract base for dense blocks.")
      .def_property_readonly(
        "shape",
        [&](const BlockBackend::Block& self) {
            return py::cast<py::tuple>(py::cast(self.shape()));
        },
        "The shape of the block.")
      .def_property_readonly("dtype", &BlockBackend::Block::dtype)
      .def_property_readonly("device", &BlockBackend::Block::device)
      .def("get_backend",
           &BlockBackend::Block::get_backend,
           py::return_value_policy::reference,
           "Return the backend for this block's device.")
      .def(
        "__add__",
        [](const BlockBackend::Block& self, const BlockCPtr& other) { return self + (*other); },
        py::arg("other"),
        "Elementwise addition with another block.")
      .def(
        "__sub__",
        [](const BlockBackend::Block& self, const BlockCPtr& other) { return self - (*other); },
        py::arg("other"),
        "Elementwise subtraction with another block.")
      .def(
        "__mul__",
        [](const BlockBackend::Block& self, const BlockBackend::Scalar& s) { return self * s; },
        py::arg("other"),
        "Multiplication by a scalar.")
      .def(
        "__mul__",
        [](const BlockBackend::Block& self, float64 s) {
            return self * self.get_backend()->as_scalar(s);
        },
        py::arg("other"))
      .def(
        "__mul__",
        [](const BlockBackend::Block& self, complex128 s) {
            return self * self.get_backend()->as_scalar(s);
        },
        py::arg("other"))
      .def(
        "__rmul__",
        [](const BlockBackend::Block& self, BlockBackend::Scalar s) { return self * s; },
        py::arg("other"))
      .def(
        "__rmul__",
        [](const BlockBackend::Block& self, float64 s) {
            return self.get_backend()->as_scalar(s) * self;
        },
        py::arg("other"))
      .def(
        "__rmul__",
        [](const BlockBackend::Block& self, complex128 s) {
            return self.get_backend()->as_scalar(s) * self;
        },
        py::arg("other"))
      .def(
        "__mul__",
        [](const BlockBackend::Block& self, const BlockBackend::Block& s) { return self * s; },
        py::arg("other"),
        "Elementwise multiplication with another block.")
      .def(
        "__truediv__",
        [](const BlockBackend::Block& self, const BlockCPtr& other) {
            return self.operator/(*other);
        },
        py::arg("other"),
        "Elementwise division with another block.")
      .def(
        "__truediv__",
        [](const BlockBackend::Block& self, const BlockBackend::Scalar& s) { return self / s; },
        py::arg("other"),
        "Division by a scalar.")
      .def(
        "__truediv__",
        [](const BlockBackend::Block& self, float64 s) {
            return self / self.get_backend()->as_scalar(s);
        },
        py::arg("other"))
      .def(
        "__truediv__",
        [](const BlockBackend::Block& self, complex128 s) {
            return self / self.get_backend()->as_scalar(s);
        },
        py::arg("other"))
      .def("__abs__", &BlockBackend::Block::abs, "Elementwise absolute value.")
      .def(
        "__pow__",
        [](const BlockBackend::Block& self, const BlockBackend::Scalar& exponent) {
            return self.pow(exponent);
        },
        py::arg("exponent"),
        "Elementwise power with scalar exponent.")
      .def(
        "__pow__",
        [](const BlockBackend::Block& self, float64 exponent) {
            return self.pow(self.get_backend()->as_scalar(exponent));
        },
        py::arg("exponent"))
      .def(
        "__pow__",
        [](const BlockBackend::Block& self, complex128 exponent) {
            return self.pow(self.get_backend()->as_scalar(exponent));
        },
        py::arg("exponent"))
      .def(
        "__pow__",
        [](const BlockBackend::Block& self, const BlockBackend::Block& exponent) {
            return self.pow(exponent);
        },
        py::arg("exponent"),
        "Elementwise power with block exponent.")
      .def(
        "__lt__",
        [](const BlockBackend::Block& self, const BlockBackend::Block& other) {
            return self < other;
        },
        py::arg("other"),
        "Less than with another block.")
      .def(
        "__lt__",
        [](const BlockBackend::Block& self, const BlockBackend::Scalar& other) {
            return self < other;
        },
        py::arg("other"))
      .def(
        "__lt__",
        [](const BlockBackend::Block& self, float64 other) { return self < other; },
        py::arg("other"))
      .def(
        "__le__",
        [](const BlockBackend::Block& self, const BlockBackend::Block& other) {
            return self <= other;
        },
        py::arg("other"),
        "Less than or equal to with another block.")
      .def(
        "__le__",
        [](const BlockBackend::Block& self, const BlockBackend::Scalar& other) {
            return self <= other;
        },
        py::arg("other"))
      .def(
        "__le__",
        [](const BlockBackend::Block& self, float64 other) { return self <= other; },
        py::arg("other"))
      .def(
        "__gt__",
        [](const BlockBackend::Block& self, const BlockBackend::Block& other) {
            return self > other;
        },
        py::arg("other"),
        "Greater than with another block.")
      .def(
        "__gt__",
        [](const BlockBackend::Block& self, const BlockBackend::Scalar& other) {
            return self > other;
        },
        py::arg("other"))
      .def(
        "__gt__",
        [](const BlockBackend::Block& self, float64 other) { return self > other; },
        py::arg("other"))
      .def(
        "__ge__",
        [](const BlockBackend::Block& self, const BlockBackend::Block& other) {
            return self >= other;
        },
        py::arg("other"),
        "Greater than or equal to with another block.")
      .def(
        "__ge__",
        [](const BlockBackend::Block& self, const BlockBackend::Scalar& other) {
            return self >= other;
        },
        py::arg("other"))
      .def(
        "__ge__",
        [](const BlockBackend::Block& self, float64 other) { return self >= other; },
        py::arg("other"))
      .def(
        "__eq__",
        [](const BlockBackend::Block& self, const BlockBackend::Block& other) {
            return self == other;
        },
        py::arg("other"),
        "Equality comparison with another block.")
      .def(
        "__ne__",
        [](const BlockBackend::Block& self, const BlockBackend::Block& other) {
            return self != other;
        },
        py::arg("other"),
        "Inequality comparison with another block.")
      .def(
        "__getitem__",
        [](BlockBackend::Block& self, py::object key) -> py::object {
            std::vector<int64> indices;
            if (try_parse_int_index_sequence(key, indices) &&
                is_scalar_element_index(self, key, indices)) {
                if (indices.size() == 1 && self.ndim() == 1 && is_py_integer_scalar(key))
                    return py::cast(self.get_item(indices[0]));
                return py::cast(self.get_item(indices));
            }
            if (auto idcs = BlockBackend::try_py_key_to_block_indices(key))
                return py::cast(self.get_item(std::span<const BlockBackend::BlockIndex>(*idcs)));
            return py::cast(self.get_item(key));
        },
        py::arg("key"))
      .def(
        "__setitem__",
        [](BlockBackend::Block& self, py::object key, py::object value) {
            if (py::isinstance<BlockBackend::Scalar>(value)) {
                std::vector<int64> indices;
                if (try_parse_int_index_sequence(key, indices) &&
                    is_scalar_element_index(self, key, indices)) {
                    const auto& scalar = value.cast<BlockBackend::Scalar>();
                    if (indices.size() == 1 && self.ndim() == 1 && is_py_integer_scalar(key))
                        self.set_item(indices[0], scalar);
                    else
                        self.set_item(indices, scalar);
                    return;
                }
            }
            if (auto idcs = BlockBackend::try_py_key_to_block_indices(key)) {
                if (py::isinstance<BlockBackend::Block>(value)) {
                    self.set_item(std::span<const BlockBackend::BlockIndex>(*idcs),
                                  value.cast<BlockBackend::Block&>());
                    return;
                }
                if (py::isinstance<BlockBackend::Scalar>(value)) {
                    self.set_item(std::span<const BlockBackend::BlockIndex>(*idcs),
                                  value.cast<BlockBackend::Scalar&>());
                    return;
                }
            }
            self.set_item(key, value);
        },
        py::arg("key"),
        py::arg("value"))
      .def("to_numpy",
           py::overload_cast<Dtype>(&BlockBackend::Block::to_numpy, py::const_),
           py::arg("dtype"),
           "Convert to numpy array with the given Dtype.")
      .def("_item_as_complex128",
           &BlockBackend::Block::_item_as_complex128,
           "Return the element of a zero-dimensional block as a complex128.")
      .def("_item_as_int64",
           &BlockBackend::Block::_item_as_int64,
           "Return the element of a zero-dimensional block as a int64.")
      .def("save_hdf5",
           &BlockBackend::Block::save_hdf5,
           py::arg("hdf5_saver"),
           py::arg("h5gr"),
           py::arg("subpath"),
           "Save block state to HDF5.")
      .def_static("from_hdf5",
                  &BlockBackend::Block::from_hdf5,
                  py::arg("hdf5_loader"),
                  py::arg("h5gr"),
                  py::arg("subpath"),
                  "Load block from HDF5 (subclass must implement).")
      .def(
        "__str__",
        [](const BlockBackend::Block& self) {
            std::ostringstream oss;
            oss << self;
            return oss.str();
        },
        "String representation of the block.")
      .def(
        "__repr__",
        [](const BlockBackend::Block& self) {
            std::ostringstream oss;
            oss << self;
            return oss.str();
        },
        "String representation of the block.");

    py::class_<BlockBackend::Scalar, py::smart_holder>(
      block_backend,
      "Scalar",
      "Scalar value with Dtype; use accessors to cast to float, complex, or bool.")
      .def(py::init<std::shared_ptr<BlockBackend::Block>>(),
           py::arg("block"),
           "Construct from a 0-d block (ndim == 0). Raises if block is null or ndim != 0.")
      .def_property_readonly("dtype", &BlockBackend::Scalar::dtype)
      .def(
        "__str__",
        [](const BlockBackend::Scalar& self) {
            std::ostringstream oss;
            oss << self;
            return oss.str();
        },
        "String representation of the scalar.")
      .def(
        "__repr__",
        [](const BlockBackend::Scalar& self) {
            std::ostringstream oss;
            oss << self;
            return oss.str();
        },
        "String representation of the scalar.")
      .def("as_float64",
           &BlockBackend::Scalar::as_float64,
           "As float; raises if dtype is not Float32/Float64.")
      .def("as_complex128",
           &BlockBackend::Scalar::as_complex128,
           "As complex (real/bool have zero imaginary part).")
      .def("as_int64", &BlockBackend::Scalar::as_int64, "As int64; raises if dtype is not Int64.")
      .def("as_bool", &BlockBackend::Scalar::as_bool, "As bool; raises if dtype is not Bool.")
      .def("to_numpy",
           &BlockBackend::Scalar::to_numpy,
           "Return as numpy scalar (np.bool_, np.float64, etc.).")
      .def(
        "__bool__",
        [](const BlockBackend::Scalar& self) {
            return self.as_bool(); // throws if dtype is not Bool!
        },
        "Return value of boolean scalar. Raises if dtype != bool.")
      .def(
        "__neg__", [](const BlockBackend::Scalar& self) { return -self; }, "Unary negation.")
      .def("real", &BlockBackend::Scalar::real, "Real part as a Scalar (valid for any dtype).")
      .def(
        "imag", &BlockBackend::Scalar::imag, "Imaginary part as a Scalar (valid for any dtype).")
      .def("__abs__", &BlockBackend::Scalar::abs, "Absolute value.")
      .def("sqrt", &BlockBackend::Scalar::sqrt, "Square root.")
      .def("exp", &BlockBackend::Scalar::exp, "Elementwise / scalar exponential.")
      .def("log", &BlockBackend::Scalar::log, "Natural logarithm.")
      .def("pow", &BlockBackend::Scalar::pow, py::arg("exponent"), "Raise to a scalar power.")
      .def(
        "__pow__",
        [](const BlockBackend::Scalar& self, const BlockBackend::Scalar& exponent) {
            return self.pow(exponent);
        },
        py::arg("exponent"),
        "Raise to a scalar power.")
      .def(
        "__pow__",
        [](const BlockBackend::Scalar& self, float64 exponent) {
            return self.pow(self._block()->get_backend()->as_scalar(exponent));
        },
        py::arg("exponent"))
      .def(
        "__pow__",
        [](const BlockBackend::Scalar& self, complex128 exponent) {
            return self.pow(self._block()->get_backend()->as_scalar(exponent));
        },
        py::arg("exponent"))
      .def(
        "__add__",
        [](const BlockBackend::Scalar& self, const BlockBackend::Scalar& other) {
            return self + other;
        },
        py::arg("other"),
        "Addition with another scalar.")
      .def(
        "__add__",
        [](const BlockBackend::Scalar& self, float64 other) { return self + other; },
        py::arg("other"))
      .def(
        "__add__",
        [](const BlockBackend::Scalar& self, complex128 other) { return self + other; },
        py::arg("other"))
      .def(
        "__radd__",
        [](const BlockBackend::Scalar& self, float64 other) { return other + self; },
        py::arg("other"))
      .def(
        "__radd__",
        [](const BlockBackend::Scalar& self, complex128 other) { return other + self; },
        py::arg("other"))
      .def(
        "__sub__",
        [](const BlockBackend::Scalar& self, const BlockBackend::Scalar& other) {
            return self - other;
        },
        py::arg("other"),
        "Subtraction with another scalar.")
      .def(
        "__sub__",
        [](const BlockBackend::Scalar& self, float64 other) { return self - other; },
        py::arg("other"))
      .def(
        "__sub__",
        [](const BlockBackend::Scalar& self, complex128 other) { return self - other; },
        py::arg("other"))
      .def(
        "__rsub__",
        [](const BlockBackend::Scalar& self, float64 other) { return other - self; }, // reversed!
        py::arg("other"))
      .def(
        "__rsub__",
        [](const BlockBackend::Scalar& self, complex128 other) {
            return other - self;
        }, // reversed!
        py::arg("other"))
      .def(
        "__mul__",
        [](const BlockBackend::Scalar& self, const BlockBackend::Scalar& other) {
            return self * other;
        },
        py::arg("other"),
        "Multiplication with another scalar.")
      .def(
        "__mul__",
        [](const BlockBackend::Scalar& self, const BlockBackend::Block& other) {
            return other * self;
        },
        py::arg("other"),
        "Multiplication with another scalar.")
      .def(
        "__mul__",
        [](const BlockBackend::Scalar& self, float64 other) { return self * other; },
        py::arg("other"))
      .def(
        "__mul__",
        [](const BlockBackend::Scalar& self, complex128 other) { return self * other; },
        py::arg("other"))
      .def(
        "__mul__",
        [](const BlockBackend::Scalar&, py::object) -> py::object {
            return py::reinterpret_borrow<py::object>(Py_NotImplemented);
        },
        py::arg("other"))
      .def(
        "__rmul__",
        [](const BlockBackend::Scalar& self, float64 other) { return other * self; },
        py::arg("other"))
      .def(
        "__rmul__",
        [](const BlockBackend::Scalar& self, complex128 other) { return other * self; },
        py::arg("other"))
      .def(
        "__truediv__",
        [](const BlockBackend::Scalar& self, const BlockBackend::Scalar& other) {
            return self / other;
        },
        py::arg("other"))
      .def(
        "__truediv__",
        [](const BlockBackend::Scalar& self, float64 other) { return self / other; },
        py::arg("other"))
      .def(
        "__truediv__",
        [](const BlockBackend::Scalar& self, complex128 other) { return self / other; },
        py::arg("other"))
      .def(
        "__rtruediv__",
        [](const BlockBackend::Scalar& self, float64 other) {
            return other / self; // reversed!
        },
        py::arg("other"))
      .def(
        "__rtruediv__",
        [](const BlockBackend::Scalar& self, complex128 other) {
            return other / self; // reversed!
        },
        py::arg("other"))
      .def(
        "__lt__",
        [](const BlockBackend::Scalar& self, const BlockBackend::Scalar& other) {
            return self < other;
        },
        py::arg("other"))
      .def(
        "__lt__",
        [](const BlockBackend::Scalar& self, float64 other) { return self < other; },
        py::arg("other"))
      .def(
        "__gt__",
        [](const BlockBackend::Scalar& self, const BlockBackend::Scalar& other) {
            return self > other;
        },
        py::arg("other"))
      .def(
        "__gt__",
        [](const BlockBackend::Scalar& self, float64 other) { return self > other; },
        py::arg("other"))
      .def(
        "__le__",
        [](const BlockBackend::Scalar& self, const BlockBackend::Scalar& other) {
            return self <= other;
        },
        py::arg("other"))
      .def(
        "__le__",
        [](const BlockBackend::Scalar& self, float64 other) { return self <= other; },
        py::arg("other"))
      .def(
        "__ge__",
        [](const BlockBackend::Scalar& self, const BlockBackend::Scalar& other) {
            return self >= other;
        },
        py::arg("other"))
      .def(
        "__ge__",
        [](const BlockBackend::Scalar& self, float64 other) { return self >= other; },
        py::arg("other"))
      .def("inverse", &BlockBackend::Scalar::inverse, "The inverse of the scalar, 1./self")
      .def_property_readonly(
        "_block", &BlockBackend::Scalar::_block, "Return the underlying block.")
      .def("save_hdf5",
           &BlockBackend::Scalar::save_hdf5,
           py::arg("hdf5_saver"),
           py::arg("h5gr"),
           py::arg("subpath"),
           "Save scalar to HDF5.")
      .def_static("from_hdf5",
                  &BlockBackend::Scalar::from_hdf5,
                  py::arg("hdf5_loader"),
                  py::arg("h5gr"),
                  py::arg("subpath"),
                  "Load scalar from HDF5.");

    block_backend // init and attributes
      .def(py::init<std::string>(), py::arg("device") = "cpu")
      .def_readonly("default_device", &BlockBackend::default_device);

    block_backend //  methods
      .def("__repr__",
           [](const BlockBackend& self) { return self.get_backend_name() + std::string("()"); })
      .def("__str__",
           [](const BlockBackend& self) { return self.get_backend_name() + std::string("()"); })
      .def("__eq__",
           [](BlockBackend const& self, py::object other) {
               if (!py::isinstance<BlockBackend>(other)) {
                   return false;
               }
               return self == other.cast<BlockBackend const&>();
           })
      .def(
        "as_scalar",
        [](BlockBackend& self, py::bool_ value) { return self.as_scalar(value.cast<bool>()); },
        py::arg("value"),
        "Convert a Python bool to a scalar block.")
      .def("as_scalar",
           py::overload_cast<int64>(&BlockBackend::as_scalar),
           py::arg("value"),
           "Convert an int64 to a scalar block.")
      .def("as_scalar",
           py::overload_cast<float64>(&BlockBackend::as_scalar),
           py::arg("value"),
           "Convert a float64 to a scalar block.")
      .def("as_scalar",
           py::overload_cast<complex128>(&BlockBackend::as_scalar),
           py::arg("value"),
           "Convert a complex128 to a scalar block.")
      .def("as_scalar",
           py::overload_cast<py::object, Dtype>(&BlockBackend::as_scalar),
           py::arg("value"),
           py::arg("dtype"),
           "Convert a Python object to a scalar block with the given Dtype.")
      .def("as_scalar",
           py::overload_cast<const BlockBackend::Scalar&>(&BlockBackend::as_scalar),
           py::arg("value"),
           "Return a Scalar unchanged.")
      .def("abs",
           &BlockBackend::abs,
           py::arg("a"),
           "The absolute value of a complex number, elementwise.")
      .def(
        "apply_basis_perm",
        [](BlockBackend& self, BlockBackend::BlockCPtr block, py::iterable legs_obj, bool inv) {
            // Accept sequences or generators (e.g. conventional_leg_order(...)).
            std::vector<BlockBackend::LegCPtr> legs;
            for (py::handle item : legs_obj) {
                // Legs may be ElementarySpace / LegPipe / AbelianLegPipe; all register as Leg.
                legs.push_back(item.cast<Leg::Ptr>());
            }
            return self.apply_basis_perm(block, legs, inv);
        },
        py::arg("block"),
        py::arg("legs"),
        py::arg("inv") = false,
        DOC(cyten, BlockBackend, apply_basis_perm))
      .def("apply_leg_permutations",
           &BlockBackend::apply_leg_permutations,
           py::arg("block"),
           py::arg("perms"),
           "Apply permutations to every axis of a dense block")
      .def("as_block",
           &BlockBackend::as_block,
           py::arg("a"),
           py::arg("dtype") = py::none(),
           py::arg("device") = py::none(),
           DOC(cyten, BlockBackend, as_block))
      .def("as_device",
           &BlockBackend::as_device,
           py::arg("device"),
           DOC(cyten, BlockBackend, as_device))
      .def("abs_argmax",
           &BlockBackend::abs_argmax,
           py::arg("block"),
           "Return the indices (one per axis) of the largest entry (by magnitude) of the block")
      .def("argmin",
           &BlockBackend::argmin,
           py::arg("block"),
           "Return the indices (one per axis) of the smallest entry of the block")
      .def("add_axis", &BlockBackend::add_axis, py::arg("a"), py::arg("pos"))
      .def("all",
           &BlockBackend::all,
           py::arg("a"),
           "Require a boolean block. If all of its entries are True")
      .def("allclose",
           &BlockBackend::allclose,
           py::arg("a"),
           py::arg("b"),
           py::arg("rtol") = 1e-05,
           py::arg("atol") = 1e-08)
      .def("angle",
           &BlockBackend::angle,
           py::arg("a"),
           "The angle of a complex number such that ``a == exp(1.j * angle)``. Elementwise.")
      .def("any",
           &BlockBackend::any,
           py::arg("a"),
           "Require a boolean block. If any of its entries are True")
      .def("apply_mask",
           &BlockBackend::apply_mask,
           py::arg("block"),
           py::arg("mask"),
           py::arg("ax"),
           "Apply a mask (1D boolean block) to a block, slicing/projecting that axis")
      .def("argsort",
           &BlockBackend::argsort,
           py::arg("block"),
           py::arg("sort") = py::none(),
           py::arg("axis") = 0,
           DOC(cyten, BlockBackend, argsort))
      .def("_argsort",
           &BlockBackend::_argsort,
           py::arg("block"),
           py::arg("axis"),
           "Like :meth:`block_argsort` but can assume real valued block, and sort ascending")
      .def("combine_legs",
           py::overload_cast<const BlockCPtr&,
                             const std::vector<std::vector<int64>>&,
                             const std::vector<bool>&>(&BlockBackend::combine_legs),
           py::arg("a"),
           py::arg("leg_idcs_combine"),
           py::arg("cstyles"),
           DOC(cyten, BlockBackend, combine_legs))
      .def("combine_legs",
           py::overload_cast<const BlockCPtr&, const std::vector<std::vector<int64>>&, bool>(
             &BlockBackend::combine_legs),
           py::arg("a"),
           py::arg("leg_idcs_combine"),
           py::arg("cstyles") = true)
      .def("conj", &BlockBackend::conj, py::arg("a"), "Complex conjugate of a block")
      .def("copy_block",
           &BlockBackend::copy_block,
           py::arg("a"),
           py::arg("device") = py::none(),
           DOC(cyten, BlockBackend, copy_block))
      .def(
        "cutoff_inverse",
        &BlockBackend::cutoff_inverse,
        py::arg("a"),
        py::arg("cutoff"),
        "The elementwise cutoff-inverse: ``1 / a`` where ``abs(a) >= cutoff``, otherwise ``0``.")
      .def("dagger",
           &BlockBackend::dagger,
           py::arg("a"),
           "Permute axes to reverse order and elementwise conj.")
      .def("get_dtype", &BlockBackend::get_dtype, py::arg("a"))
      .def("eigh",
           &BlockBackend::eigh,
           py::arg("block"),
           py::arg("sort") = py::none(),
           DOC(cyten, BlockBackend, eigh))
      .def("eigvalsh",
           &BlockBackend::eigvalsh,
           py::arg("block"),
           py::arg("sort") = py::none(),
           DOC(cyten, BlockBackend, eigvalsh))
      .def("enlarge_leg",
           &BlockBackend::enlarge_leg,
           py::arg("block"),
           py::arg("mask"),
           py::arg("axis"))
      .def("exp", &BlockBackend::exp, py::arg("a"), DOC(cyten, BlockBackend, exp))
      .def("block_from_diagonal",
           &BlockBackend::block_from_diagonal,
           py::arg("diag"),
           "Return a 2D square block that has the 1D ``diag`` on the diagonal")
      .def("block_from_mask",
           &BlockBackend::block_from_mask,
           py::arg("mask"),
           py::arg("dtype"),
           DOC(cyten, BlockBackend, block_from_mask))
      .def("block_from_numpy",
           &BlockBackend::block_from_numpy,
           py::arg("a"),
           py::arg("dtype") = py::none(),
           py::arg("device") = py::none())
      .def("get_device", &BlockBackend::get_device, py::arg("a"))
      .def("get_diagonal",
           &BlockBackend::get_diagonal,
           py::arg("a"),
           py::arg("tol"),
           "Get the diagonal of a 2D block as a 1D block")
      .def("imag",
           &BlockBackend::imag,
           py::arg("a"),
           "The imaginary part of a complex number, elementwise.")
      .def("inner",
           &BlockBackend::inner,
           py::arg("a"),
           py::arg("b"),
           py::arg("do_dagger"),
           DOC(cyten, BlockBackend, inner))
      .def("is_real", &BlockBackend::is_real, py::arg("a"), DOC(cyten, BlockBackend, is_real))
      .def("item",
           &BlockBackend::item,
           py::arg("a"),
           "Assumes that data is a scalar (i.e. has only one entry). Returns that scalar as "
           "python float or complex")
      .def("kron", &BlockBackend::kron, py::arg("a"), py::arg("b"), DOC(cyten, BlockBackend, kron))
      .def("linear_combination",
           &BlockBackend::linear_combination,
           py::arg("a"),
           py::arg("v"),
           py::arg("b"),
           py::arg("w"))
      .def("log", &BlockBackend::log, py::arg("a"), DOC(cyten, BlockBackend, log))
      .def("max", &BlockBackend::max, py::arg("a"))
      .def("max_abs", &BlockBackend::max_abs, py::arg("a"))
      .def("min", &BlockBackend::min, py::arg("a"))
      .def("mul",
           py::overload_cast<const BlockBackend::Scalar&, const BlockCPtr&>(&BlockBackend::mul),
           py::arg("a"),
           py::arg("b"))
      .def("mul",
           py::overload_cast<float64, const BlockCPtr&>(&BlockBackend::mul),
           py::arg("a"),
           py::arg("b"))
      .def("mul",
           py::overload_cast<complex128, const BlockCPtr&>(&BlockBackend::mul),
           py::arg("a"),
           py::arg("b"))
      .def("norm",
           &BlockBackend::norm,
           py::arg("a"),
           py::arg("order") = 2,
           py::arg("axis") = py::none(),
           DOC(cyten, BlockBackend, norm))
      .def(
        "outer", &BlockBackend::outer, py::arg("a"), py::arg("b"), DOC(cyten, BlockBackend, outer))
      .def("permute_axes", &BlockBackend::permute_axes, py::arg("a"), py::arg("permutation"))
      .def("permute_combined_matrix",
           &BlockBackend::permute_combined_matrix,
           py::arg("block"),
           py::arg("dims1"),
           py::arg("idcs1"),
           py::arg("dims2"),
           py::arg("idcs2"),
           DOC(cyten, BlockBackend, permute_combined_matrix))
      .def("permute_combined_idx",
           &BlockBackend::permute_combined_idx,
           py::arg("block"),
           py::arg("axis"),
           py::arg("dims"),
           py::arg("idcs"),
           DOC(cyten, BlockBackend, permute_combined_idx))
      .def("random_normal",
           &BlockBackend::random_normal,
           py::arg("dims"),
           py::arg("dtype"),
           py::arg("sigma"),
           py::arg("device") = py::none())
      .def("random_uniform",
           &BlockBackend::random_uniform,
           py::arg("dims"),
           py::arg("dtype"),
           py::arg("device") = py::none())
      .def("real",
           &BlockBackend::real,
           py::arg("a"),
           "The real part of a complex number, elementwise.")
      .def("real_if_close",
           &BlockBackend::real_if_close,
           py::arg("a"),
           py::arg("tol"),
           DOC(cyten, BlockBackend, real_if_close))
      .def("tile",
           &BlockBackend::tile,
           py::arg("a"),
           py::arg("repeats"),
           "Repeat a (1d) block multiple times. Similar to numpy.tile and torch.Tensor.repeat.")
      .def("_block_repr_lines",
           &BlockBackend::_block_repr_lines,
           py::arg("a"),
           py::arg("indent"),
           py::arg("max_width"),
           py::arg("max_lines"))
      .def("reshape", &BlockBackend::reshape, py::arg("a"), py::arg("shape"))
      .def("scale_axis",
           &BlockBackend::scale_axis,
           py::arg("block"),
           py::arg("factors"),
           py::arg("axis"),
           DOC(cyten, BlockBackend, scale_axis))
      .def(
        "get_shape",
        [](BlockBackend& self, const BlockCPtr& a) {
            return py::cast<py::tuple>(py::cast(self.get_shape(a)));
        },
        py::arg("a"))
      .def("split_legs",
           py::overload_cast<const BlockCPtr&,
                             const std::vector<int64>&,
                             const std::vector<std::vector<int64>>&,
                             const std::vector<bool>&>(&BlockBackend::split_legs),
           py::arg("a"),
           py::arg("idcs"),
           py::arg("dims"),
           py::arg("cstyles"),
           DOC(cyten, BlockBackend, split_legs))
      .def("split_legs",
           py::overload_cast<const BlockCPtr&,
                             const std::vector<int64>&,
                             const std::vector<std::vector<int64>>&,
                             bool>(&BlockBackend::split_legs),
           py::arg("a"),
           py::arg("idcs"),
           py::arg("dims"),
           py::arg("cstyles") = true)
      .def("sqrt", &BlockBackend::sqrt, py::arg("a"), "The elementwise square root")
      .def("squeeze_axes", &BlockBackend::squeeze_axes, py::arg("a"), py::arg("idcs"))
      .def("stable_log",
           &BlockBackend::stable_log,
           py::arg("block"),
           py::arg("cutoff"),
           "Elementwise stable log. For entries > cutoff, yield their natural log. Otherwise 0.")
      .def("sum", &BlockBackend::sum, py::arg("a"), py::arg("ax"), "The sum over a single axis.")
      .def("sum_all", &BlockBackend::sum_all, py::arg("a"), DOC(cyten, BlockBackend, sum_all))
      .def("tdot",
           &BlockBackend::tdot,
           py::arg("a"),
           py::arg("b"),
           py::arg("idcs_a"),
           py::arg("idcs_b"))
      .def("tensor_outer",
           &BlockBackend::tensor_outer,
           py::arg("a"),
           py::arg("b"),
           py::arg("K"),
           DOC(cyten, BlockBackend, tensor_outer))
      .def("to_dtype", &BlockBackend::to_dtype, py::arg("a"), py::arg("dtype"))
      .def("to_numpy", &BlockBackend::to_numpy, py::arg("a"), py::arg("numpy_dtype") = py::none())
      .def("trace_full", &BlockBackend::trace_full, py::arg("a"))
      .def("trace_partial",
           &BlockBackend::trace_partial,
           py::arg("a"),
           py::arg("idcs1"),
           py::arg("idcs2"),
           py::arg("remaining_idcs"))
      .def("eye_block",
           &BlockBackend::eye_block,
           py::arg("legs"),
           py::arg("dtype"),
           py::arg("device") = py::none(),
           DOC(cyten, BlockBackend, eye_block))
      .def("eye_matrix",
           &BlockBackend::eye_matrix,
           py::arg("dim"),
           py::arg("dtype"),
           py::arg("device") = py::none(),
           "The ``dim x dim`` identity matrix")
      .def("get_block_element", &BlockBackend::get_block_element, py::arg("a"), py::arg("idcs"))
      .def("get_block_mask_element",
           &BlockBackend::get_block_mask_element,
           py::arg("a"),
           py::arg("large_leg_idx"),
           py::arg("small_leg_idx"),
           py::arg("sum_block") = 0,
           DOC(cyten, BlockBackend, get_block_mask_element))
      .def("matrix_dot",
           &BlockBackend::matrix_dot,
           py::arg("a"),
           py::arg("b"),
           "As in numpy.dot, both a and b might be matrix or vector.")
      .def("matrix_exp", &BlockBackend::matrix_exp, py::arg("matrix"))
      .def("matrix_lq", &BlockBackend::matrix_lq, py::arg("a"), py::arg("full"))
      .def("matrix_qr",
           &BlockBackend::matrix_qr,
           py::arg("a"),
           py::arg("full"),
           "QR decomposition of a 2D block")
      .def("matrix_svd",
           &BlockBackend::matrix_svd,
           py::arg("a"),
           py::arg("algorithm"),
           "Perform a SVD decomposition of a matrix.")
      .def("possible_svd_algorithms",
           &BlockBackend::possible_svd_algorithms,
           "Possible algorithms for :meth:`matrix_svd`.")
      .def("ones_block",
           &BlockBackend::ones_block,
           py::arg("shape"),
           py::arg("dtype"),
           py::arg("device") = py::none())
      .def("synchronize",
           &BlockBackend::synchronize,
           "Wait for asynchronous processes (if any) to finish")
      .def("test_block_sanity",
           &BlockBackend::test_block_sanity,
           py::arg("block"),
           py::arg("expect_shape") = py::none(),
           py::arg("expect_dtype") = py::none(),
           py::arg("expect_device") = py::none())
      .def("zeros",
           &BlockBackend::zeros,
           py::arg("shape"),
           py::arg("dtype"),
           py::arg("device") = py::none())
      .def("save_hdf5",
           &BlockBackend::save_hdf5,
           py::arg("hdf5_saver"),
           py::arg("h5gr"),
           py::arg("subpath"))
      .def_static("from_hdf5",
                  &BlockBackend::from_hdf5,
                  py::arg("hdf5_loader"),
                  py::arg("h5gr"),
                  py::arg("subpath")); // completed block_backend methods

    bind_block_backend_numpy(m);
    bind_block_backend_torch(m);
    bind_block_backend_array_api(m);
}

} // namespace cyten
