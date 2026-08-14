config.h
========

Declared in ``include/cyten/config.h``.

cyten::CytenConfig
------------------

.. doxygenclass:: cyten::CytenConfig
   :project: cyten
   :members:
   :undoc-members:

Free functions
--------------

.. doxygenfunction:: cyten::get_config
   :project: cyten

.. doxygenfunction:: cyten::set_option(const std::string &, const std::string &)
   :project: cyten

.. doxygenfunction:: cyten::set_option(const std::string &, int64)
   :project: cyten

.. doxygenfunction:: cyten::set_option(const std::string &, bool)
   :project: cyten

.. doxygenfunction:: cyten::set_option(const std::string &, py::handle)
   :project: cyten

.. doxygenfunction:: cyten::set_options
   :project: cyten

.. doxygenfunction:: cyten::get_option
   :project: cyten

.. doxygenfunction:: cyten::restore_defaults
   :project: cyten
