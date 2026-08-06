#include <cyten/symmetries/abelian_group.h>

#include <utility>

namespace cyten {

namespace {

py::module_
numpy()
{
    return py::module_::import("numpy");
}

/// Cached ones arrays matching Python ``one_1D`` / ``one_2D`` / … in ``_symmetries.py``.
py::array
ones_array(py::tuple shape, py::object dtype)
{
    return numpy().attr("ones")(shape, py::arg("dtype") = dtype).cast<py::array>();
}

py::array
one_1D()
{
    return ones_array(py::make_tuple(1), numpy().attr("intp"));
}

py::array
one_2D()
{
    return ones_array(py::make_tuple(1, 1), numpy().attr("intp"));
}

py::array
one_2D_float()
{
    return ones_array(py::make_tuple(1, 1), numpy().attr("float64"));
}

py::array
one_4D()
{
    return ones_array(py::make_tuple(1, 1, 1, 1), numpy().attr("intp"));
}

py::array
one_4D_float()
{
    return ones_array(py::make_tuple(1, 1, 1, 1), numpy().attr("float64"));
}

} // namespace

AbelianGroup::AbelianGroup(Sector trivial_sector,
                           std::string group_name,
                           float64 num_sectors,
                           std::optional<std::string> descriptive_name,
                           bool trivial_shift)
  : Group(FusionStyle::single,
          trivial_sector,
          std::move(group_name),
          num_sectors,
          /*has_complex_topological_data=*/false,
          std::move(descriptive_name),
          trivial_shift)
{
    fusion_tensor_dtype = Dtype::Float64;
}

std::string
AbelianGroup::sector_str(Sector a) const
{
    // Sectors labelled by a single number.
    if (a.len() == 0) {
        return "";
    }
    return std::to_string(a.q[0]);
}

int64
AbelianGroup::sector_dim(Sector /*a*/) const
{
    return 1;
}

py::array
AbelianGroup::batch_sector_dim(SectorArray const& a) const
{
    return numpy()
      .attr("ones")(py::make_tuple(static_cast<py::ssize_t>(a.size())),
                    py::arg("dtype") = numpy().attr("intp"))
      .cast<py::array>();
}

int64
AbelianGroup::_n_symbol(Sector /*a*/, Sector /*b*/, Sector /*c*/) const
{
    return 1;
}

py::array
AbelianGroup::_f_symbol(Sector /*a*/,
                        Sector /*b*/,
                        Sector /*c*/,
                        Sector /*d*/,
                        Sector /*e*/,
                        Sector /*f*/) const
{
    return one_4D();
}

int64
AbelianGroup::frobenius_schur(Sector /*a*/) const
{
    return 1;
}

float64
AbelianGroup::qdim(Sector /*a*/) const
{
    return 1.0;
}

float64
AbelianGroup::sqrt_qdim(Sector /*a*/) const
{
    return 1.0;
}

float64
AbelianGroup::inv_sqrt_qdim(Sector /*a*/) const
{
    return 1.0;
}

py::array
AbelianGroup::_b_symbol(Sector /*a*/, Sector /*b*/, Sector /*c*/) const
{
    return one_2D();
}

py::array
AbelianGroup::_r_symbol(Sector /*a*/, Sector /*b*/, Sector /*c*/) const
{
    // For abelian groups, the R symbol is always 1.
    return one_1D();
}

py::array
AbelianGroup::_c_symbol(Sector /*a*/,
                        Sector /*b*/,
                        Sector /*c*/,
                        Sector /*d*/,
                        Sector /*e*/,
                        Sector /*f*/) const
{
    return one_4D();
}

py::array
AbelianGroup::_fusion_tensor(Sector /*a*/, Sector /*b*/, Sector /*c*/, bool /*Z_a*/, bool /*Z_b*/)
  const
{
    return one_4D_float();
}

py::array
AbelianGroup::Z_iso(Sector /*a*/) const
{
    return one_2D_float();
}

} // namespace cyten
