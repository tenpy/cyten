#include <cyten/symmetries/group.h>

#include <utility>

namespace cyten {

Group::Group(FusionStyle fusion_style,
             Sector trivial_sector,
             std::string group_name,
             float64 num_sectors,
             bool has_complex_topological_data,
             std::optional<std::string> descriptive_name,
             bool trivial_shift)
  : SymmetryFactor(fusion_style,
                   BraidingStyle::bosonic,
                   trivial_sector,
                   std::move(group_name),
                   num_sectors,
                   has_complex_topological_data,
                   std::move(descriptive_name),
                   trivial_shift)
{
}

py::array
Group::swap_gate(Sector a, Sector b) const
{
    // [b, a, b*, a*] = eye(dim_a)[None, :, None, :] * eye(dim_b)[:, None, :, None]
    auto np = py::module_::import("numpy");
    py::object colon = py::slice(py::none(), py::none(), py::none());
    py::array eye_a = np.attr("eye")(static_cast<py::ssize_t>(sector_dim(a)));
    py::array eye_b = np.attr("eye")(static_cast<py::ssize_t>(sector_dim(b)));
    py::object ea =
      eye_a.attr("__getitem__")(py::make_tuple(py::none(), colon, py::none(), colon));
    py::object eb =
      eye_b.attr("__getitem__")(py::make_tuple(colon, py::none(), colon, py::none()));
    return (ea.attr("__mul__")(eb)).cast<py::array>();
}

float64
Group::qdim(Sector a) const
{
    return static_cast<float64>(sector_dim(a));
}

py::array
Group::batch_qdim(SectorArray const& a) const
{
    return batch_sector_dim(a);
}

complex128
Group::topological_twist(Sector /*a*/) const
{
    return complex128{ 1.0, 0.0 };
}

} // namespace cyten
