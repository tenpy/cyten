config
======

Declared in ``include/lasyten/config.h``.

lasyten::LasytenConfig
------------------

.. doxygenclass:: lasyten::LasytenConfig
   :project: lasyten
   :members:
   :undoc-members:

Free functions
--------------

.. doxygenfunction:: lasyten::get_config
   :project: lasyten

.. doxygenfunction:: lasyten::set_option(const std::string &, const std::string &)
   :project: lasyten

.. doxygenfunction:: lasyten::set_option(const std::string &, int64)
   :project: lasyten

.. doxygenfunction:: lasyten::set_option(const std::string &, bool)
   :project: lasyten

.. doxygenfunction:: lasyten::set_option(const std::string &, py::handle)
   :project: lasyten

.. doxygenfunction:: lasyten::set_options
   :project: lasyten

.. doxygenfunction:: lasyten::get_option
   :project: lasyten

.. doxygenfunction:: lasyten::restore_defaults
   :project: lasyten
