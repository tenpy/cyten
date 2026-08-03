#include <cyten/symmetries/symmetry_factor.h>

#include <stdexcept>

namespace cyten {

SymmetryFactor::SymmetryFactor(FusionStyle fusion_style,
                               BraidingStyle braiding_style,
                               Sector trivial_sector,
                               std::string group_name_,
                               float64 num_sectors,
                               bool has_complex_topological_data,
                               std::optional<std::string> descriptive_name_,
                               bool trivial_shift)
    : BaseSymmetry(fusion_style,
                   braiding_style,
                   trivial_sector,
                   num_sectors,
                   has_complex_topological_data,
                   trivial_shift)
    , group_name(std::move(group_name_))
    , descriptive_name(std::move(descriptive_name_))
{}

bool
SymmetryFactor::is_equivalent_to(BaseSymmetry const& other) const
{
    if (auto const* factor = dynamic_cast<SymmetryFactor const*>(&other)) {
        return _is_equivalent_factor(*factor);
    }
    // Product Symmetry (or unknown): ask the other side via Python when available.
    // Until Symmetry is C++, callers should use the py::object overload from bindings.
    return false;
}

py::object
SymmetryFactor::as_Symmetry()
{
    auto self = std::static_pointer_cast<SymmetryFactor>(shared_from_this());
    py::object self_py = py::cast(self);
    auto Symmetry = py::module_::import("cyten.symmetries").attr("Symmetry");
    return Symmetry(py::make_tuple(self_py));
}

std::string
SymmetryFactor::str() const
{
    if (descriptive_name.has_value()) {
        return group_name + " (\"" + *descriptive_name + "\")";
    }
    return group_name;
}

py::object
SymmetryFactor::mul(py::object other) const
{
    auto self = std::static_pointer_cast<SymmetryFactor const>(shared_from_this());
    py::object self_py = py::cast(self);
    auto Symmetry = py::module_::import("cyten.symmetries").attr("Symmetry");
    // isinstance checks against Python SymmetryFactor / Symmetry
    py::object SymmetryFactor_py = py::module_::import("cyten.symmetries").attr("SymmetryFactor");
    if (py::isinstance(other, SymmetryFactor_py)) {
        return Symmetry(py::make_tuple(self_py, other));
    }
    if (py::isinstance(other, Symmetry)) {
        py::list factors;
        factors.append(self_py);
        for (auto f : other.attr("factors")) {
            factors.append(f);
        }
        return Symmetry(factors);
    }
    return py::none(); // NotImplemented → binding maps to NotImplemented
}

bool
SymmetryFactor::equals(SymmetryFactor const& other) const
{
    if (descriptive_name != other.descriptive_name) {
        return false;
    }
    return _is_equivalent_factor(other);
}

void
SymmetryFactor::save_hdf5(py::object hdf5_saver, py::object h5gr, std::string const& subpath) const
{
    hdf5_saver.attr("save")(group_name, subpath + "group_name");
    hdf5_saver.attr("save")(static_cast<int>(fusion_style), subpath + "fusion_style");
    hdf5_saver.attr("save")(static_cast<int>(braiding_style), subpath + "braiding_style");
    hdf5_saver.attr("save")(trivial_sector, subpath + "trivial_sector");
    hdf5_saver.attr("save")(num_sectors, subpath + "num_sectors");
    hdf5_saver.attr("save")(static_cast<int>(sector_ind_len), subpath + "sector_ind_len");
    std::string descr = descriptive_name.has_value() ? *descriptive_name : "None";
    h5gr.attr("attrs")["descriptive_name"] = descr;
    h5gr.attr("attrs")["has_complex_topological_data"] = has_complex_topological_data;
}

void
SymmetryFactor::load_hdf5_common(py::object hdf5_loader,
                                 py::object h5gr,
                                 std::string const& subpath)
{
    group_name = hdf5_loader.attr("load")(subpath + "group_name").cast<std::string>();
    fusion_style =
      static_cast<FusionStyle>(hdf5_loader.attr("load")(subpath + "fusion_style").cast<int>());
    braiding_style =
      static_cast<BraidingStyle>(hdf5_loader.attr("load")(subpath + "braiding_style").cast<int>());
    trivial_sector = hdf5_loader.attr("load")(subpath + "trivial_sector").cast<Sector>();
    num_sectors = hdf5_loader.attr("load")(subpath + "num_sectors").cast<float64>();
    sector_ind_len =
      static_cast<std::uint8_t>(hdf5_loader.attr("load")(subpath + "sector_ind_len").cast<int>());
    empty_sector_array = SectorArray::empty(sector_ind_len);
    auto descr = h5gr.attr("attrs")["descriptive_name"].cast<std::string>();
    if (descr == "None") {
        descriptive_name = std::nullopt;
    } else {
        descriptive_name = descr;
    }
    has_complex_topological_data =
      h5gr.attr("attrs")["has_complex_topological_data"].cast<bool>();
}

} // namespace cyten
