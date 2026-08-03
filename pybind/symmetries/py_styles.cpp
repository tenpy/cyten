#include "py_cyten_pybind11.h"

#include <cyten/symmetries/styles.h>

#include <pybind11/native_enum.h>

namespace cyten {

void
bind_symmetries_styles(py::module_& m)
{
    py::native_enum<FusionStyle> fusion_enum(m,
                                             "FusionStyle",
                                             "enum.IntEnum",
                                             R"pydoc(
                                             Describes properties of fusion, i.e. of the tensor product.

                                             =================  =============================================================================
                                             Value              Meaning
                                             =================  =============================================================================
                                             single             Fusing sectors results in a single sector ``a ⊗ b = c``, e.g. abelian groups.
                                             -----------------  -----------------------------------------------------------------------------
                                             multiple_unique    Every sector appears at most once in pairwise fusion, ``N_symbol in [0, 1]``.
                                             -----------------  -----------------------------------------------------------------------------
                                             general            No assumptions, ``N_symbol in [0, 1, 2, 3, ...]``.
                                             =================  =============================================================================
                                             )pydoc");
    fusion_enum.value("single", FusionStyle::single)
      .value("multiple_unique", FusionStyle::multiple_unique)
      .value("general", FusionStyle::general)
      .export_values()
      .finalize();

    py::native_enum<BraidingStyle> braid_enum(m,
                                              "BraidingStyle",
                                              "enum.IntEnum",
                                              R"pydoc(
                                              Describes properties of braiding.

                                              =============  ===========================================
                                              Value
                                              =============  ===========================================
                                              bosonic        Symmetric braiding with trivial twist
                                              -------------  -------------------------------------------
                                              fermionic      Symmetric braiding with non-trivial twist
                                              -------------  -------------------------------------------
                                              anyonic        General, non-symmetric braiding
                                              -------------  -------------------------------------------
                                              no_braiding    Braiding is not defined
                                              =============  ===========================================
                                              )pydoc");
    braid_enum.value("bosonic", BraidingStyle::bosonic)
      .value("fermionic", BraidingStyle::fermionic)
      .value("anyonic", BraidingStyle::anyonic)
      .value("no_braiding", BraidingStyle::no_braiding)
      .export_values()
      .finalize();
}

} // namespace cyten
