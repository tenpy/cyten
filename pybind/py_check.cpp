#include "py_cyten_pybind11.h"
#include <check.h>

namespace cyten {

class PyCheckBase
  : public CheckBaseOp
  , public py::trampoline_self_life_support
{
  public:
    using CheckBaseOp::CheckBaseOp; // inherit constructors

    int64 check(int64 i, int64 j) const override
    {
        PYBIND11_OVERRIDE_PURE(int64, CheckBaseOp, check, i, j);
    }
};

class PyCheckAdd
  : public CheckAdd
  , public py::trampoline_self_life_support
{
  public:
    using CheckAdd::CheckAdd; // inherit constructors

    int64 check(int64 i, int64 j) const override
    {
        PYBIND11_OVERRIDE(int64, CheckAdd, check, i, j);
    }
};

void
bind_check(py::module_& m)
{

    /// check

    py::class_<CheckBaseOp, PyCheckBase, py::smart_holder>(m, "CheckBaseOp", "CheckBaseOp class")
      .def(py::init<>(), "Initialize the check operation");
    py::class_<CheckAdd, PyCheckAdd, CheckBaseOp, py::smart_holder>(
      m, "CheckAdd", "CheckAdd class")
      .def(py::init<>(), "Initialize the check operation")
      .def("check", &CheckAdd::check, "Check the addition of two numbers");
    m.def("add", &cyten::add, "A function that adds two numbers");
    m.def("apply_check_op",
          &cyten::apply_check_op,
          py::arg("op"),
          py::arg("i"),
          py::arg("j"),
          "Apply a check operation to two numbers");
}

} // namespace cyten
