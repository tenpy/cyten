#pragma once

#include <string>
#include <vector>

#include <cyten/cyten.h>

namespace cyten {

// NOLINTBEGIN(readability-magic-numbers)

class CytenConfig
{
  public:
    int64 print_linewidth = 100;
    int64 print_indent = 2;
    int64 maxlines_spaces = 15;
    int64 maxlines_tensors = 30;
    /// If the symmetry methods should check their inputs are valid
    bool check_fusion = true;
    std::string default_tensor_backend = "abelian";
    std::string default_block_backend = "numpy";
    /// Threshold for discarding near-zero fusion-tree blocks after topological moves.
    /// Default is based on tests for 4-leg tensors: smaller values produced extra blocks from
    /// numerical noise when bending legs and restoring the original configuration.
    float64 fusion_tree_eps = 5.0e-14;

    CytenConfig() = default;

    /// Names of all recognized config options.
    static const std::vector<std::string>& all_option_keys();

    /// Environment variable name for a config option (``CYTEN_`` + uppercased key).
    static std::string env_var_name(const std::string& key);

    void set_option(const std::string& key, const std::string& value);
    void set_option(const std::string& key, int64 value);
    void set_option(const std::string& key, bool value);
    void set_option(const std::string& key, float64 value);

    void update(py::dict options);
    void update(const CytenConfig& other);
    void update_from_env();
    void update_from_yaml(const std::string& yaml_text);
    void update_from_file(const std::string& filename);

    py::object get_option(const std::string& key) const;

    std::string str() const;

    void save_hdf5(py::object hdf5_saver, py::object h5gr, const std::string& subpath) const;
    static CytenConfig from_hdf5(py::object hdf5_loader,
                                 py::object h5gr,
                                 const std::string& subpath);
};

// NOLINTEND(readability-magic-numbers)

/// Global mutable config (initialized on first get_config / restore_defaults).
/// Prefer get_config() for reads; use set_option / set_options / temporary_options to mutate.
extern CytenConfig _global_config;

const CytenConfig& get_config();

void set_option(const std::string& key, const std::string& value);
void set_option(const std::string& key, int64 value);
void set_option(const std::string& key, bool value);
void set_option(const std::string& key, float64 value);
void set_option(const std::string& key, py::handle value);
void set_options(py::dict options);

py::object get_option(const std::string& key);

void restore_defaults(bool use_user_file = true,
                      bool use_local_file = true,
                      bool use_env_vars = true);

} // namespace cyten
