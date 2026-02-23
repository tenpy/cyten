#include <cyten/config.h>

#include "cyten_pyb11.h"

namespace py = pybind11;
namespace cyten {

void
bind_config(pybind11::module_& m)
{
    py::class_<CytenConfig::PrintOptions>(m, "PrintOptions")
      .def(py::init<>())
      .def_readonly("linewidth", &CytenConfig::PrintOptions::linewidth)
      .def_readonly("indent", &CytenConfig::PrintOptions::indent)
      .def_readonly("precision", &CytenConfig::PrintOptions::precision)
      .def_readonly("maxlines_spaces", &CytenConfig::PrintOptions::maxlines_spaces)
      .def_readonly("maxlines_tensors", &CytenConfig::PrintOptions::maxlines_tensors)
      .def_readonly("skip_data", &CytenConfig::PrintOptions::skip_data)
      .def_readonly("summarize_blocks", &CytenConfig::PrintOptions::summarize_blocks);

    py::class_<CytenConfig>(m, "CytenConfig")
      .def(py::init<>())
      .def_readonly("print_options", &CytenConfig::print_options)
      .def_readonly("do_fusion_input_checks", &CytenConfig::do_fusion_input_checks)
      .def_readonly("default_symmetry_backend", &CytenConfig::default_symmetry_backend)
      .def_readonly("default_block_backend", &CytenConfig::default_block_backend);

    m.def("get_config", &get_config, "Get the global configuration object");
}

} // namespace cyten
