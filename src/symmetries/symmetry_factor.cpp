#include <cyten/symmetries/symmetry_factor.h>

#include <cyten/symmetries/sector_numpy.h>
#include <cyten/symmetries/symmetry.h>

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
    // Product Symmetry (or unknown): ask the other side via Python when available.
    // Until Symmetry is C++, callers should use the py::object overload from bindings.
    return false;
}

py::object
SymmetryFactor::as_Symmetry()
{
    // Prefer the Python binding (takes py::object) for trampoline instances: smart_holder
    // does not always initialize enable_shared_from_this. C++-only shared_ptr owners work here.
    try {
        auto self = std::static_pointer_cast<SymmetryFactor>(shared_from_this());
        return py::cast(std::make_shared<Symmetry>(std::vector<SymmetryFactor::Ptr>{ self }));
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

py::object
SymmetryFactor::mul(py::object other)
{
    // Prefer the Python ``__mul__`` binding for trampoline instances (see as_Symmetry).
    try {
        auto self = std::static_pointer_cast<SymmetryFactor>(shared_from_this());
        if (py::isinstance<SymmetryFactor>(other)) {
            return py::cast(std::make_shared<Symmetry>(
              std::vector<SymmetryFactor::Ptr>{ self, other.cast<SymmetryFactor::Ptr>() }));
        }
        if (py::isinstance<Symmetry>(other)) {
            auto const& sym = other.cast<Symmetry const&>();
            std::vector<SymmetryFactor::Ptr> factors;
            factors.reserve(1 + sym.factors.size());
            factors.push_back(self);
            factors.insert(factors.end(), sym.factors.begin(), sym.factors.end());
            return py::cast(std::make_shared<Symmetry>(std::move(factors)));
        }
        return py::none(); // binding maps None → NotImplemented
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
SymmetryFactor::save_hdf5(py::object hdf5_saver, py::object h5gr, std::string const& subpath) const
{
    hdf5_saver.attr("save")(group_name, subpath + "group_name");
    hdf5_saver.attr("save")(static_cast<int>(fusion_style), subpath + "fusion_style");
    hdf5_saver.attr("save")(static_cast<int>(braiding_style), subpath + "braiding_style");
    // Bound Sector so Hdf5Saver finds Sector.save_hdf5 (libcyten has no type_casters).
    hdf5_saver.attr("save")(sector_as_hdf5_exportable(trivial_sector), subpath + "trivial_sector");
    hdf5_saver.attr("save")(num_sectors, subpath + "num_sectors");
    hdf5_saver.attr("save")(static_cast<int>(sector_ind_len), subpath + "sector_ind_len");
    hdf5_saver.attr("save")(trivial_shift, subpath + "trivial_shift");
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
    trivial_sector = sector_from_hdf5_object(hdf5_loader.attr("load")(subpath + "trivial_sector"));
    num_sectors = hdf5_loader.attr("load")(subpath + "num_sectors").cast<float64>();
    sector_ind_len =
      static_cast<std::uint8_t>(hdf5_loader.attr("load")(subpath + "sector_ind_len").cast<int>());
    empty_sector_array = SectorArray::empty(sector_ind_len);
    // trivial_shift was added later; default true if missing.
    try {
        trivial_shift = hdf5_loader.attr("load")(subpath + "trivial_shift").cast<bool>();
    } catch (py::error_already_set&) {
        PyErr_Clear();
        trivial_shift = true;
    }
    auto descr = h5gr.attr("attrs")["descriptive_name"].cast<std::string>();
    if (descr == "None") {
        descriptive_name = std::nullopt;
    } else {
        descriptive_name = descr;
    }
    has_complex_topological_data = h5gr.attr("attrs")["has_complex_topological_data"].cast<bool>();
}

std::optional<std::string>
descriptive_name_from_hdf5_attrs(py::object h5gr)
{
    auto descr = h5gr.attr("attrs")["descriptive_name"].cast<std::string>();
    if (descr == "None") {
        return std::nullopt;
    }
    return descr;
}

bool
trivial_shift_from_hdf5(py::object hdf5_loader, std::string const& subpath)
{
    try {
        return hdf5_loader.attr("load")(subpath + "trivial_shift").cast<bool>();
    } catch (py::error_already_set&) {
        PyErr_Clear();
        return true;
    }
}

} // namespace cyten
