#include <cyten/symmetries/symmetry_factor.h>

#include <cyten/symmetries/symmetry.h>

#include <cyten/tools/hdf5.h>
#include <cyten/tools/hdf5_py_bridge.h>
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
{
}

bool
SymmetryFactor::is_equivalent_to(BaseSymmetry const& other) const
{
    if (auto const* factor = dynamic_cast<SymmetryFactor const*>(&other)) {
        return _is_equivalent_factor(*factor);
    }
    // Product Symmetry: ask the other side (Symmetry::is_equivalent_to handles factors).
    return false;
}

BaseSymmetry::SymmetryPtr
SymmetryFactor::as_Symmetry()
{
    // Prefer the Python binding (takes py::object) for trampoline instances: smart_holder
    // does not always initialize enable_shared_from_this. C++-only shared_ptr owners work here.
    try {
        auto self = std::static_pointer_cast<SymmetryFactor>(shared_from_this());
        return std::make_shared<Symmetry>(std::vector<SymmetryFactor::Ptr>{ self });
    } catch (std::bad_weak_ptr const&) {
        throw std::runtime_error(
          "SymmetryFactor::as_Symmetry: call via Python bindings (no shared_from_this)");
    }
}

std::string
SymmetryFactor::str() const
{
    if (descriptive_name.has_value()) {
        return group_name + " (\"" + *descriptive_name + "\")";
    }
    return group_name;
}

BaseSymmetry::SymmetryPtr
SymmetryFactor::mul(Ptr other)
{
    try {
        auto self = std::static_pointer_cast<SymmetryFactor>(shared_from_this());
        return std::make_shared<Symmetry>(
          std::vector<SymmetryFactor::Ptr>{ self, std::move(other) });
    } catch (std::bad_weak_ptr const&) {
        throw std::runtime_error(
          "SymmetryFactor::mul: call via Python bindings (no shared_from_this)");
    }
}

BaseSymmetry::SymmetryPtr
SymmetryFactor::mul(Symmetry const& other)
{
    try {
        auto self = std::static_pointer_cast<SymmetryFactor>(shared_from_this());
        std::vector<SymmetryFactor::Ptr> factors;
        factors.reserve(1 + other.factors.size());
        factors.push_back(self);
        factors.insert(factors.end(), other.factors.begin(), other.factors.end());
        return std::make_shared<Symmetry>(std::move(factors));
    } catch (std::bad_weak_ptr const&) {
        throw std::runtime_error(
          "SymmetryFactor::mul: call via Python bindings (no shared_from_this)");
    }
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
SymmetryFactor::save_hdf5(cyten::hdf5::Saver& saver,
                          HighFive::Group& h5gr,
                          std::string const& subpath) const
{
    cyten::hdf5::py_save(subpath + "group_name", group_name);
    cyten::hdf5::py_save(subpath + "fusion_style", static_cast<int>(fusion_style));
    cyten::hdf5::py_save(subpath + "braiding_style", static_cast<int>(braiding_style));
    // Bound Sector so Hdf5Saver finds Sector.save_hdf5.
    cyten::hdf5::py_save(subpath + "trivial_sector", py::cast(trivial_sector));
    cyten::hdf5::py_save(subpath + "num_sectors", num_sectors);
    cyten::hdf5::py_save(subpath + "sector_ind_len", static_cast<int>(sector_ind_len));
    cyten::hdf5::py_save(subpath + "trivial_shift", trivial_shift);
    std::string descr = descriptive_name.has_value() ? *descriptive_name : "None";
    cyten::hdf5::py_set_group_attr("descriptive_name", py::cast(descr));
    cyten::hdf5::py_set_group_attr("has_complex_topological_data",
                                   py::cast(has_complex_topological_data));
}

void
SymmetryFactor::load_hdf5_common(cyten::hdf5::Loader& loader,
                                 HighFive::Group& h5gr,
                                 std::string const& subpath)
{
    group_name = cyten::hdf5::py_load(subpath + "group_name").cast<std::string>();
    fusion_style =
      static_cast<FusionStyle>(cyten::hdf5::py_load(subpath + "fusion_style").cast<int>());
    braiding_style =
      static_cast<BraidingStyle>(cyten::hdf5::py_load(subpath + "braiding_style").cast<int>());
    trivial_sector = cyten::hdf5::py_load(subpath + "trivial_sector").cast<Sector>();
    num_sectors = cyten::hdf5::py_load(subpath + "num_sectors").cast<float64>();
    sector_ind_len =
      static_cast<std::uint8_t>(cyten::hdf5::py_load(subpath + "sector_ind_len").cast<int>());
    empty_sector_array = SectorArray::empty(sector_ind_len);
    // trivial_shift was added later; default true if missing.
    try {
        trivial_shift = cyten::hdf5::py_load(subpath + "trivial_shift").cast<bool>();
    } catch (py::error_already_set&) {
        PyErr_Clear();
        trivial_shift = true;
    }
    auto descr = cyten::hdf5::py_get_group_attr("descriptive_name").cast<std::string>();
    if (descr == "None") {
        descriptive_name = std::nullopt;
    } else {
        descriptive_name = descr;
    }
    has_complex_topological_data =
      cyten::hdf5::py_get_group_attr("has_complex_topological_data").cast<bool>();
}

std::optional<std::string>
descriptive_name_from_hdf5_attrs(HighFive::Group& /*h5gr*/)
{
    auto descr = cyten::hdf5::py_get_group_attr("descriptive_name").cast<std::string>();
    if (descr == "None") {
        return std::nullopt;
    }
    return descr;
}

bool
trivial_shift_from_hdf5(cyten::hdf5::Loader& loader, std::string const& subpath)
{
    try {
        return cyten::hdf5::py_load(subpath + "trivial_shift").cast<bool>();
    } catch (py::error_already_set&) {
        PyErr_Clear();
        return true;
    }
}

} // namespace cyten
