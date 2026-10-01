#pragma once

/// Native TeNPy-format HDF5 helpers used by cyten ``save_hdf5`` / ``from_hdf5``.

#include <cyten/backends/block_inds.h>
#include <cyten/backends/tensor_backend.h>
#include <cyten/block_backend/block_backend.h>
#include <cyten/block_backend/dtypes.h>
#include <cyten/symmetries/sector.h>
#include <cyten/symmetries/spaces.h>
#include <cyten/symmetries/symmetry.h>
#include <cyten/symmetries/symmetry_factor.h>
#include <cyten/tensors/labels.h>
#include <cyten/tools/hdf5.h>

#include <hdf5_io/constants.h>
#include <hdf5_io/h5_ops.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace cyten::hdf5_export {

inline constexpr char const* kModule = "cyten._core";

[[nodiscard]] hdf5_io::Hdf5Buffer i64_vector_to_buffer(std::vector<std::int64_t> const& v);
[[nodiscard]] std::vector<std::int64_t> i64_vector_from_buffer(hdf5_io::Hdf5Buffer const& buf);

[[nodiscard]] hdf5_io::Hdf5Buffer i64_matrix_to_buffer(std::vector<std::int64_t> const& flat,
                                                       std::size_t rows,
                                                       std::size_t cols);
[[nodiscard]] std::pair<std::vector<std::int64_t>, std::pair<std::size_t, std::size_t>>
i64_matrix_from_buffer(hdf5_io::Hdf5Buffer const& buf);

[[nodiscard]] hdf5_io::Hdf5Buffer f64_vector_to_buffer(std::vector<double> const& v);
[[nodiscard]] std::vector<double> f64_vector_from_buffer(hdf5_io::Hdf5Buffer const& buf);

void save_i64_vector(cyten::hdf5::Saver& saver,
                     std::string const& path,
                     std::vector<std::int64_t> const& v);
[[nodiscard]] std::vector<std::int64_t> load_i64_vector(cyten::hdf5::Loader& loader,
                                                        std::string const& path);

void save_f64_vector(cyten::hdf5::Saver& saver,
                     std::string const& path,
                     std::vector<double> const& v);
[[nodiscard]] std::vector<double> load_f64_vector(cyten::hdf5::Loader& loader,
                                                  std::string const& path);

void save_optional_i64_vector(cyten::hdf5::Saver& saver,
                              std::string const& path,
                              std::optional<std::vector<std::int64_t>> const& v);
[[nodiscard]] std::optional<std::vector<std::int64_t>> load_optional_i64_vector(
  cyten::hdf5::Loader& loader,
  std::string const& path);

void save_optional_labels(cyten::hdf5::Saver& saver,
                          std::string const& path,
                          OptionalLabels const& labels);
[[nodiscard]] OptionalLabels load_optional_labels(cyten::hdf5::Loader& loader,
                                                  std::string const& path);

void save_dtype_string(cyten::hdf5::Saver& saver, std::string const& path, Dtype dt);
[[nodiscard]] Dtype load_dtype_string(cyten::hdf5::Loader& loader, std::string const& path);

void save_sector(cyten::hdf5::Saver& saver, std::string const& path, Sector const& sector);
[[nodiscard]] Sector load_sector(cyten::hdf5::Loader& loader, std::string const& path);

void save_sector_array(cyten::hdf5::Saver& saver,
                       std::string const& path,
                       SectorArray const& sectors);
[[nodiscard]] SectorArray load_sector_array(cyten::hdf5::Loader& loader, std::string const& path);

void save_block_inds(cyten::hdf5::Saver& saver, std::string const& path, BlockInds const& bi);
[[nodiscard]] BlockInds load_block_inds(cyten::hdf5::Loader& loader, std::string const& path);

void save_symmetry_factor(cyten::hdf5::Saver& saver,
                          std::string const& path,
                          SymmetryFactor::CPtr const& factor);
[[nodiscard]] SymmetryFactor::Ptr load_symmetry_factor(cyten::hdf5::Loader& loader,
                                                       std::string const& path);

void save_symmetry(cyten::hdf5::Saver& saver,
                   std::string const& path,
                   Symmetry::CPtr const& symmetry);
[[nodiscard]] Symmetry::Ptr load_symmetry(cyten::hdf5::Loader& loader, std::string const& path);

void save_elementary_space(cyten::hdf5::Saver& saver,
                           std::string const& path,
                           ElementarySpace::CPtr const& space);
[[nodiscard]] ElementarySpace::Ptr load_elementary_space(cyten::hdf5::Loader& loader,
                                                         std::string const& path);

void save_leg(cyten::hdf5::Saver& saver, std::string const& path, Leg::CPtr const& leg);
[[nodiscard]] Leg::Ptr load_leg(cyten::hdf5::Loader& loader, std::string const& path);

void save_tensor_product(cyten::hdf5::Saver& saver,
                         std::string const& path,
                         TensorProduct::CPtr const& tp);
[[nodiscard]] TensorProduct::Ptr load_tensor_product(cyten::hdf5::Loader& loader,
                                                     std::string const& path);

void save_block(cyten::hdf5::Saver& saver, std::string const& path, BlockCPtr const& block);
[[nodiscard]] BlockPtr load_block(cyten::hdf5::Loader& loader, std::string const& path);

void save_block_backend(cyten::hdf5::Saver& saver,
                        std::string const& path,
                        std::shared_ptr<BlockBackend const> const& backend);
[[nodiscard]] std::shared_ptr<BlockBackend> load_block_backend(cyten::hdf5::Loader& loader,
                                                               std::string const& path);

void save_tensor_backend(cyten::hdf5::Saver& saver,
                         std::string const& path,
                         TensorBackend::CPtr const& backend);
[[nodiscard]] TensorBackend::Ptr load_tensor_backend(cyten::hdf5::Loader& loader,
                                                     std::string const& path);

void save_tensor_backend_data(cyten::hdf5::Saver& saver,
                              std::string const& path,
                              TensorBackend::DataCPtr const& data);
[[nodiscard]] TensorBackend::DataPtr load_tensor_backend_data(cyten::hdf5::Loader& loader,
                                                              std::string const& path);

/// Open child path; if already memoized as ``T``, return it. Otherwise call ``make(group, sub)``.
template<typename T, typename Make>
std::shared_ptr<T>
load_instance(cyten::hdf5::Loader& loader, std::string const& path, Make&& make)
{
    hid_t id = loader.open(path);
    if (auto memo = loader.lookup_memo(id)) {
        H5Idec_ref(id);
        return std::static_pointer_cast<T>(memo);
    }
    HighFive::Group group = hdf5_io::group_from_hid(id);
    return make(group, cyten::hdf5::ensure_slash(path));
}

[[nodiscard]] std::string class_attr(HighFive::Group& h5gr);

/// NumPy ↔ ``Hdf5Buffer`` (used by numpy/torch/array_api block backends).
[[nodiscard]] hdf5_io::Hdf5Buffer buffer_from_numpy(py::array arr);
[[nodiscard]] py::array numpy_from_buffer(hdf5_io::Hdf5Buffer const& buf);

} // namespace cyten::hdf5_export
