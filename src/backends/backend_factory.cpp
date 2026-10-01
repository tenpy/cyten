#include <cyten/backends/abelian.h>
#include <cyten/backends/backend_factory.h>
#include <cyten/backends/fusion_tree_backend.h>
#include <cyten/backends/no_symmetry.h>
#include <cyten/block_backend/numpy.h>
#include <cyten/block_backend/torch.h>
#include <cyten/config.h>
#include <cyten/symmetries/factors/no_symmetry.h>
#include <cyten/tools.h>

#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace cyten {

namespace {

std::shared_ptr<BlockBackend>
make_block_backend(std::string const& block_backend)
{
    if (block_backend == "numpy" || block_backend == "cpu") {
        return NumpyBlockBackend::from_factory_shared("cpu");
    }
    if (block_backend == "torch") {
        return TorchBlockBackend::from_factory_shared("cpu:0");
    }
    if (block_backend == "gpu") {
        return TorchBlockBackend::from_factory_shared("cuda");
    }
    if (block_backend == "apple_silicon") {
        return TorchBlockBackend::from_factory_shared("mps");
    }
    if (block_backend == "tensorflow" || block_backend == "jax" || block_backend == "tpu") {
        throw NotImplemented(std::string("block backend ") + block_backend);
    }
    throw std::invalid_argument("Unknown block_backend: " + block_backend);
}

bool
is_no_symmetry(Symmetry const& symmetry)
{
    Symmetry no_sym{ std::vector<SymmetryFactor::Ptr>{ std::make_shared<NoSymmetry>() } };
    return symmetry.is_equivalent_to(no_sym);
}

struct BackendCacheKey
{
    std::string tensor_backend;
    std::string block_backend;

    bool operator==(BackendCacheKey const& other) const noexcept
    {
        return tensor_backend == other.tensor_backend && block_backend == other.block_backend;
    }
};

struct BackendCacheKeyHash
{
    std::size_t operator()(BackendCacheKey const& key) const noexcept
    {
        return std::hash<std::string>{}(key.tensor_backend) ^
               (std::hash<std::string>{}(key.block_backend) << 1);
    }
};

std::unordered_map<BackendCacheKey, TensorBackend::Ptr, BackendCacheKeyHash>&
backend_cache()
{
    static std::unordered_map<BackendCacheKey, TensorBackend::Ptr, BackendCacheKeyHash> cache;
    return cache;
}

std::mutex&
backend_cache_mutex()
{
    static std::mutex mu;
    return mu;
}

TensorBackend::Ptr
make_tensor_backend(std::string const& tensor_backend,
                    std::shared_ptr<BlockBackend> block_backend_instance)
{
    if (tensor_backend == "no_symmetry") {
        return std::make_shared<NoSymmetryBackend>(std::move(block_backend_instance));
    }
    if (tensor_backend == "abelian") {
        return std::make_shared<AbelianBackend>(std::move(block_backend_instance));
    }
    if (tensor_backend == "fusion_tree") {
        return std::make_shared<FusionTreeBackend>(std::move(block_backend_instance));
    }
    throw std::invalid_argument("Unknown tensor_backend: " + tensor_backend);
}

} // namespace

py::object
get_backend(py::object symmetry, py::object block_backend)
{
    // --- hints from Python get_backend ---
    // figure out minimal symmetry_backend that supports that symmetry
    // ---
    if (symmetry.is_none()) {
        symmetry = py::cast(get_config().default_tensor_backend);
    }
    if (block_backend.is_none()) {
        block_backend = py::cast(get_config().default_block_backend);
    }

    std::string tensor_backend;
    Symmetry::Ptr sym_ptr;
    if (py::isinstance<Symmetry>(symmetry)) {
        sym_ptr = symmetry.cast<Symmetry::Ptr>();
        if (is_no_symmetry(*sym_ptr)) {
            tensor_backend = "no_symmetry";
        } else if (sym_ptr->is_abelian() && sym_ptr->has_trivial_braid()) {
            tensor_backend = "abelian";
        } else {
            tensor_backend = "fusion_tree";
        }
    } else if (py::isinstance<py::str>(symmetry)) {
        tensor_backend = symmetry.cast<std::string>();
    } else {
        throw py::type_error("Invalid type for symmetry. Expected Symmetry or str");
    }

    std::string block_backend_str = block_backend.cast<std::string>();
    BackendCacheKey key{ tensor_backend, block_backend_str };

    {
        std::lock_guard<std::mutex> lock(backend_cache_mutex());
        auto& cache = backend_cache();
        auto it = cache.find(key);
        if (it != cache.end()) {
            return py::cast(it->second);
        }
    }

    auto block_backend_instance = make_block_backend(block_backend_str);
    auto backend = make_tensor_backend(tensor_backend, std::move(block_backend_instance));

    if (sym_ptr) {
        if (!backend->supports_symmetry(sym_ptr)) {
            throw std::runtime_error("backend does not support the given symmetry");
        }
    }

    {
        std::lock_guard<std::mutex> lock(backend_cache_mutex());
        auto& cache = backend_cache();
        auto [it, inserted] = cache.emplace(key, backend);
        if (!inserted) {
            backend = it->second;
        }
    }
    return py::cast(backend);
}

TensorBackend::Ptr
get_backend(Symmetry::Ptr symmetry, std::optional<std::string> block_backend)
{
    py::object bb = block_backend.has_value() ? py::cast(*block_backend) : py::none();
    return get_backend(py::cast(std::move(symmetry)), bb).cast<TensorBackend::Ptr>();
}

} // namespace cyten
