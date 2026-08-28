#include <cyten/config.h>
#include <cyten/tools/warn.h>

#include <cctype>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <fstream>
#include <ranges>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace cyten {

namespace {

namespace fs = std::filesystem;

bool g_config_initialized = false;

std::string
to_upper(std::string s)
{
    for (char& c : s)
        c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
    return s;
}

bool
coerce_bool(const std::string& value)
{
    std::string lower = value;
    for (char& c : lower)
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return lower == "true" || lower == "1" || lower == "y" || lower == "yes";
}

int64
parse_int64(const std::string& value)
{
    try {
        size_t idx = 0;
        long long v = std::stoll(value, &idx);
        if (idx != value.size())
            throw std::invalid_argument("trailing characters");
        return static_cast<int64>(v);
    } catch (const std::exception& e) {
        throw py::value_error(std::string("Invalid integer config value: ") + value + " (" +
                              e.what() + ")");
    }
}

float64
parse_float64(const std::string& value)
{
    try {
        size_t idx = 0;
        double v = std::stod(value, &idx);
        if (idx != value.size())
            throw std::invalid_argument("trailing characters");
        return static_cast<float64>(v);
    } catch (const std::exception& e) {
        throw py::value_error(std::string("Invalid float config value: ") + value + " (" +
                              e.what() + ")");
    }
}

void
check_min(int64 value, int64 min_value, const std::string& key)
{
    if (value < min_value) {
        throw py::value_error("Config option '" + key + "' must be >= " +
                              std::to_string(min_value) + ", got " + std::to_string(value));
    }
}

void
check_min(float64 value, float64 min_value, const std::string& key)
{
    if (value < min_value) {
        throw py::value_error("Config option '" + key + "' must be >= " +
                              std::to_string(min_value) + ", got " + std::to_string(value));
    }
}

bool
is_allowed(const std::string& value, const std::vector<std::string>& allowed)
{
    for (const auto& a : allowed) {
        if (value == a)
            return true;
    }
    return false;
}

std::string
home_directory()
{
    if (const char* home = std::getenv("HOME"); home != nullptr && *home != '\0')
        return home;
#ifdef _WIN32
    if (const char* profile = std::getenv("USERPROFILE"); profile != nullptr && *profile != '\0')
        return profile;
    const char* drive = std::getenv("HOMEDRIVE");
    const char* path = std::getenv("HOMEPATH");
    if (drive != nullptr && path != nullptr)
        return std::string(drive) + path;
#endif
    return {};
}

/// Login name, mirroring the lookup order of Python's ``getpass.getuser()``. Used only to build
/// the default SU(N) data path, which is defined (by the external data-generating repo) in terms
/// of ``getpass.getuser()`` -- matching the order keeps the two in agreement whenever several of
/// these are set to different values (e.g. under ``sudo -E``, or in some CI/container setups).
std::string
login_name()
{
    for (const char* var : { "LOGNAME", "USER", "LNAME", "USERNAME" }) {
        if (const char* v = std::getenv(var); v != nullptr && *v != '\0')
            return v;
    }
    return "unknown";
}

/// Empty string if no usable user config path.
std::string
resolve_user_config_path()
{
    if (const char* override = std::getenv("CYTEN_CONFIG_FILE")) {
        const fs::path p(expand_user(override));
        if (!fs::exists(p)) {
            throw py::value_error(
              std::string("User config file read from CYTEN_CONFIG_FILE does not exist: ") +
              p.string());
        }
        return p.string();
    }
    const std::string home = home_directory();
    if (home.empty())
        return {};
    const fs::path p = fs::path(home) / ".cytenconfig.yaml";
    if (!fs::exists(p))
        return {};
    return p.string();
}

/// Empty string if no local config file exists.
std::string
resolve_local_config_path()
{
    const fs::path p = fs::current_path() / ".cytenconfig.yaml";
    if (!fs::exists(p))
        return {};
    return p.string();
}

void
try_update_from_file(CytenConfig& config, const std::string& path)
{
    if (path.empty())
        return;
    try {
        config.update_from_file(path);
    } catch (py::error_already_set& e) {
        std::string msg = e.what();
        e.discard_as_unraisable(__func__);
        warn(std::format("Invalid config in {}. Ignoring the file. Reason: {}", path, msg));
    } catch (const std::exception& e) {
        warn(std::format("Invalid config in {}. Ignoring the file. Reason: {}", path, e.what()));
    }
}

} // namespace

std::string
expand_user(std::string path)
{
    if (path.empty() || path[0] != '~')
        return path;
    const std::string home = home_directory();
    if (home.empty())
        return path;
    if (path.size() == 1)
        return home;
    if (path[1] == '/' || path[1] == '\\')
        return home + path.substr(1);
    return path; // ~user forms are not expanded
}

std::string
default_su_n_data_path()
{
    // Deliberately the literal POSIX form on all platforms -- see the doc comment in config.h.
    // Touches only std::getenv (via login_name()), never py:: calls: this runs during dynamic
    // initialization of _global_config, before the Python interpreter state can be relied on.
    return "/home/" + login_name() + "/.tenpy/su_n_symmetry_data";
}

CytenConfig _global_config;

const std::vector<std::string>&
CytenConfig::all_option_keys()
{
    static const std::vector<std::string> keys = {
        "print_linewidth",         "print_indent",    "maxlines_spaces",
        "maxlines_tensors",        "check_fusion",    "default_tensor_backend",
        "default_block_backend",   "fusion_tree_eps", "su_n_data_path",
        "su_n_data_filename_base", "coupling_cutoff",
    };
    return keys;
}

std::string
CytenConfig::env_var_name(const std::string& key)
{
    return "CYTEN_" + to_upper(key);
}

void
CytenConfig::set_option(const std::string& key, int64 value)
{
    if (key == "print_linewidth") {
        check_min(value, 10, key);
        print_linewidth = value;
    } else if (key == "print_indent") {
        check_min(value, 0, key);
        print_indent = value;
    } else if (key == "maxlines_spaces") {
        check_min(value, 0, key);
        maxlines_spaces = value;
    } else if (key == "maxlines_tensors") {
        check_min(value, 0, key);
        maxlines_tensors = value;
    } else if (key == "fusion_tree_eps" || key == "coupling_cutoff") {
        set_option(key, static_cast<float64>(value));
    } else if (std::ranges::contains(all_option_keys(), key)) {
        throw py::type_error("Config option '" + key + "' is not an int");
    } else {
        throw py::key_error("Invalid config option: " + key);
    }
}

void
CytenConfig::set_option(const std::string& key, float64 value)
{
    if (key == "fusion_tree_eps") {
        check_min(value, 0.0, key);
        fusion_tree_eps = value;
    } else if (key == "coupling_cutoff") {
        check_min(value, 0.0, key);
        coupling_cutoff = value;
    } else if (std::ranges::contains(all_option_keys(), key)) {
        throw py::type_error("Config option '" + key + "' is not a float");
    } else {
        throw py::key_error("Invalid config option: " + key);
    }
}

void
CytenConfig::set_option(const std::string& key, bool value)
{
    if (key == "check_fusion") {
        check_fusion = value;
    } else if (std::ranges::contains(all_option_keys(), key)) {
        throw py::type_error("Config option '" + key + "' is not a bool");
    } else {
        throw py::key_error("Invalid config option: " + key);
    }
}

void
CytenConfig::set_option(const std::string& key, const std::string& value)
{
    if (key == "print_linewidth" || key == "print_indent" || key == "maxlines_spaces" ||
        key == "maxlines_tensors") {
        set_option(key, parse_int64(value));
    } else if (key == "check_fusion") {
        set_option(key, coerce_bool(value));
    } else if (key == "fusion_tree_eps" || key == "coupling_cutoff") {
        set_option(key, parse_float64(value));
    } else if (key == "default_tensor_backend") {
        static const std::vector<std::string> allowed = { "no_symmetry",
                                                          "abelian",
                                                          "fusion_tree" };
        if (!is_allowed(value, allowed))
            throw py::value_error("Invalid default_tensor_backend: " + value);
        default_tensor_backend = value;
    } else if (key == "default_block_backend") {
        static const std::vector<std::string> allowed = {
            "numpy", "torch", "cpu", "gpu", "apple_silicon"
        };
        if (!is_allowed(value, allowed))
            throw py::value_error("Invalid default_block_backend: " + value);
        default_block_backend = value;
    } else if (key == "su_n_data_path") {
        su_n_data_path = value;
    } else if (key == "su_n_data_filename_base") {
        if (value.empty())
            throw py::value_error("Config option 'su_n_data_filename_base' must not be empty");
        su_n_data_filename_base = value;
    } else {
        throw py::key_error("Invalid config option: " + key);
    }
}

void
CytenConfig::update(py::dict options)
{
    for (auto item : options) {
        std::string key = py::cast<std::string>(item.first);
        py::handle val = item.second;
        // bool is a subclass of int in Python; check bool first.
        if (py::isinstance<py::bool_>(val)) {
            set_option(key, py::cast<bool>(val));
        } else if (py::isinstance<py::int_>(val)) {
            set_option(key, py::cast<int64>(val));
        } else if (py::isinstance<py::float_>(val)) {
            set_option(key, py::cast<float64>(val));
        } else if (py::isinstance<py::str>(val)) {
            set_option(key, py::cast<std::string>(val));
        } else {
            set_option(key, std::string(py::str(val)));
        }
    }
}

void
CytenConfig::update(const CytenConfig& other)
{
    *this = other;
}

void
CytenConfig::update_from_env()
{
    for (const auto& key : all_option_keys()) {
        const char* val = std::getenv(env_var_name(key).c_str());
        if (val == nullptr)
            continue;
        try {
            set_option(key, std::string(val));
        } catch (py::error_already_set& e) {
            std::string msg = e.what();
            e.discard_as_unraisable(__func__);
            warn(std::format(
              "Invalid config option in envvar {}. Reason {}", env_var_name(key), msg));
        } catch (const std::exception& e) {
            warn(std::format(
              "Invalid config option in envvar {}. Reason {}", env_var_name(key), e.what()));
        }
    }
}

void
CytenConfig::update_from_yaml(const std::string& yaml_text)
{
    py::module_ yaml = py::module_::import("yaml");
    py::object data = yaml.attr("safe_load")(yaml_text);
    if (data.is_none())
        return;
    if (!py::isinstance<py::dict>(data))
        throw py::type_error("Config must contain a mapping");
    update(py::reinterpret_borrow<py::dict>(data));
}

void
CytenConfig::update_from_file(const std::string& filename)
{
    const std::filesystem::path path(filename);
    if (!std::filesystem::exists(path))
        return;
    std::ifstream in(path);
    if (!in) {
        throw py::value_error("Could not open config file: " + path.string());
    }
    std::ostringstream ss;
    ss << in.rdbuf();
    update_from_yaml(ss.str());
}

py::object
CytenConfig::get_option(const std::string& key) const
{
    if (key == "print_linewidth")
        return py::cast(print_linewidth);
    if (key == "print_indent")
        return py::cast(print_indent);
    if (key == "maxlines_spaces")
        return py::cast(maxlines_spaces);
    if (key == "maxlines_tensors")
        return py::cast(maxlines_tensors);
    if (key == "check_fusion")
        return py::cast(check_fusion);
    if (key == "default_tensor_backend")
        return py::cast(default_tensor_backend);
    if (key == "default_block_backend")
        return py::cast(default_block_backend);
    if (key == "fusion_tree_eps")
        return py::cast(fusion_tree_eps);
    if (key == "su_n_data_path")
        return py::cast(su_n_data_path);
    if (key == "su_n_data_filename_base")
        return py::cast(su_n_data_filename_base);
    if (key == "coupling_cutoff")
        return py::cast(coupling_cutoff);
    throw py::key_error("Invalid option name: " + key);
}

std::string
CytenConfig::str() const
{
    std::ostringstream ss;
    ss << "CytenConfig(";
    bool first = true;
    for (const auto& key : all_option_keys()) {
        if (!first)
            ss << ", ";
        first = false;
        ss << key << "=";
        py::object val = get_option(key);
        if (py::isinstance<py::str>(val))
            ss << "'" << std::string(py::cast<std::string>(val)) << "'";
        else
            ss << std::string(py::str(val));
    }
    ss << ")";
    return ss.str();
}

void
CytenConfig::save_hdf5(py::object hdf5_saver,
                       py::object /*h5gr*/,
                       const std::string& subpath) const
{
    for (const auto& key : all_option_keys())
        hdf5_saver.attr("save")(get_option(key), subpath + key);
}

CytenConfig
CytenConfig::from_hdf5(py::object hdf5_loader, py::object h5gr, const std::string& subpath)
{
    CytenConfig obj;
    py::dict options;
    for (const auto& key : all_option_keys())
        options[py::str(key)] = hdf5_loader.attr("load")(subpath + key);
    obj.update(options);
    hdf5_loader.attr("memorize_load")(h5gr, py::cast(obj));
    return obj;
}

const CytenConfig&
get_config()
{
    if (!g_config_initialized)
        restore_defaults();
    return _global_config;
}

void
set_option(const std::string& key, const std::string& value)
{
    get_config(); // ensure initialized
    _global_config.set_option(key, value);
}

void
set_option(const std::string& key, int64 value)
{
    get_config();
    _global_config.set_option(key, value);
}

void
set_option(const std::string& key, bool value)
{
    get_config();
    _global_config.set_option(key, value);
}

void
set_option(const std::string& key, float64 value)
{
    get_config();
    _global_config.set_option(key, value);
}

void
set_option(const std::string& key, py::handle value)
{
    get_config();
    py::dict d;
    d[py::str(key)] = value;
    _global_config.update(d);
}

void
set_options(py::dict options)
{
    get_config();
    _global_config.update(options);
}

py::object
get_option(const std::string& key)
{
    return get_config().get_option(key);
}

void
restore_defaults(bool use_user_file, bool use_local_file, bool use_env_vars)
{
    _global_config = CytenConfig{}; // default values

    // Precedence (later wins): defaults -> user file -> local file -> env
    if (use_user_file)
        try_update_from_file(_global_config, resolve_user_config_path());
    if (use_local_file)
        try_update_from_file(_global_config, resolve_local_config_path());
    if (use_env_vars)
        _global_config.update_from_env();

    g_config_initialized = true;
}

} // namespace cyten
