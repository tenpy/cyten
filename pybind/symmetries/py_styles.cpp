#include "py_cyten_pybind11.h"
#include "../doc_plus.h"
#include "docstrings/symmetries/styles.h"

#include <cyten/symmetries/styles.h>

#include <pybind11/native_enum.h>

namespace cyten {

void
bind_symmetries_styles(py::module_& m)
{
    py::native_enum<FusionStyle> fusion_enum(m,
                                             "FusionStyle",
                                             "enum.IntEnum",
                                             DOC(cyten, FusionStyle));
    fusion_enum.value("single", FusionStyle::single)
      .value("multiple_unique", FusionStyle::multiple_unique)
      .value("general", FusionStyle::general)
      .export_values()
      .finalize();

    py::native_enum<BraidingStyle> braid_enum(m,
                                              "BraidingStyle",
                                              "enum.IntEnum",
                                              DOC(cyten, BraidingStyle));
    braid_enum.value("bosonic", BraidingStyle::bosonic)
      .value("fermionic", BraidingStyle::fermionic)
      .value("anyonic", BraidingStyle::anyonic)
      .value("no_braiding", BraidingStyle::no_braiding)
      .export_values()
      .finalize();
}

} // namespace cyten
