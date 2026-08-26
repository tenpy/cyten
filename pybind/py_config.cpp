#include <cyten/config.h>

#include "py_cyten_pybind11.h"

namespace py = pybind11;
namespace cyten {

namespace {

/// Context manager: saves a copy of _global_config on enter, restores on exit.
class TemporaryOptions
{
  public:
    explicit TemporaryOptions(py::dict options)
      : options_(std::move(options))
    {
    }

    TemporaryOptions& enter()
    {
        if (!entered_) {
            saved_ = get_config();
        }
        _global_config.update(options_);
        entered_ = true;
        return *this;
    }

    void exit(py::object /*exc_type*/, py::object /*exc_value*/, py::object /*traceback*/)
    {
        if (entered_) {
            _global_config = saved_;
            entered_ = false;
        }
    }

  private:
    CytenConfig saved_;
    py::dict options_;
    bool entered_ = false;
};

} // namespace

void
bind_config(py::module_& m)
{
    py::class_<CytenConfig>(m, "CytenConfig")
      .def(py::init<>())
      .def_readonly("print_linewidth", &CytenConfig::print_linewidth)
      .def_readonly("print_indent", &CytenConfig::print_indent)
      .def_readonly("maxlines_spaces", &CytenConfig::maxlines_spaces)
      .def_readonly("maxlines_tensors", &CytenConfig::maxlines_tensors)
      .def_readonly("check_fusion", &CytenConfig::check_fusion)
      .def_readonly("default_tensor_backend", &CytenConfig::default_tensor_backend)
      .def_readonly("default_block_backend", &CytenConfig::default_block_backend)
      .def_readonly("fusion_tree_eps", &CytenConfig::fusion_tree_eps)
      .def_readonly("coupling_cutoff", &CytenConfig::coupling_cutoff)
      .def_static(
        "all_option_keys", &CytenConfig::all_option_keys, "Names of all recognized config options")
      .def_static("env_var_name",
                  &CytenConfig::env_var_name,
                  py::arg("key"),
                  "Environment variable name for a config option")
      .def("set_option",
           py::overload_cast<const std::string&, bool>(&CytenConfig::set_option),
           py::arg("key"),
           py::arg("value"))
      .def("set_option",
           py::overload_cast<const std::string&, int64>(&CytenConfig::set_option),
           py::arg("key"),
           py::arg("value"))
      .def("set_option",
           py::overload_cast<const std::string&, float64>(&CytenConfig::set_option),
           py::arg("key"),
           py::arg("value"))
      .def("set_option",
           py::overload_cast<const std::string&, const std::string&>(&CytenConfig::set_option),
           py::arg("key"),
           py::arg("value"))
      .def("update", py::overload_cast<py::dict>(&CytenConfig::update), py::arg("options"))
      .def("update", py::overload_cast<const CytenConfig&>(&CytenConfig::update), py::arg("other"))
      .def("update_from_env", &CytenConfig::update_from_env)
      .def("update_from_yaml", &CytenConfig::update_from_yaml, py::arg("yaml_text"))
      .def("update_from_file", &CytenConfig::update_from_file, py::arg("filename"))
      .def("get_option", &CytenConfig::get_option, py::arg("key"))
      .def("str", &CytenConfig::str)
      .def("__str__", &CytenConfig::str)
      .def("__repr__", &CytenConfig::str)
      .def("save_hdf5",
           &CytenConfig::save_hdf5,
           py::arg("hdf5_saver"),
           py::arg("h5gr"),
           py::arg("subpath"),
           "Export config to hdf5 such that it can be re-imported with from_hdf5")
      .def_static("from_hdf5",
                  &CytenConfig::from_hdf5,
                  py::arg("hdf5_loader"),
                  py::arg("h5gr"),
                  py::arg("subpath"),
                  "Load config from hdf5");

    py::class_<TemporaryOptions>(m, "TemporaryOptions")
      .def(py::init<py::dict>(), py::arg("options"))
      .def("__enter__", &TemporaryOptions::enter)
      .def("__exit__", &TemporaryOptions::exit);

    m.def("get_config",
          &get_config,
          py::return_value_policy::reference,
          "Get the global configuration object");
    m.def("get_option", &get_option, py::arg("key"), "Get a config option by name");
    m.def(
      "set_option",
      [](const std::string& key, py::handle value) { set_option(key, value); },
      py::arg("key"),
      py::arg("value"),
      "Set a single config option on the global config");
    m.def(
      "set_options",
      [](py::kwargs kwargs) { set_options(py::dict(kwargs)); },
      "Set config options on the global config");
    m.def(
      "temporary_options",
      [](py::kwargs kwargs) { return TemporaryOptions(py::dict(kwargs)); },
      "Context manager for temporary config overrides");
    m.def("restore_defaults",
          &restore_defaults,
          py::arg("use_user_file") = true,
          py::arg("use_local_file") = true,
          py::arg("use_env_vars") = true,
          "Reset config to defaults and optionally reload files/env");
}

} // namespace cyten
