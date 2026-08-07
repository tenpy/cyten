#include <cyten/backends/backend_factory.h>
#include <cyten/backends/abelian.h>
#include <cyten/backends/fusion_tree_backend.h>
#include <cyten/backends/no_symmetry.h>
#include <cyten/block_backend/numpy.h>
#include <cyten/block_backend/torch.h>
#include <cyten/config.h>
#include <cyten/symmetries/factors/no_symmetry.h>
#include <cyten/tools.h>

#include <stdexcept>
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

py::object
make_python_tensor_backend(std::string const& tensor_backend,
                           std::shared_ptr<BlockBackend> block_backend_instance)
{
    if (tensor_backend != "fusion_tree") {
        throw std::invalid_argument("Unknown tensor_backend: " + tensor_backend);
    }
    py::object cls =
      py::module_::import("cyten.backends.fusion_tree_backend").attr("FusionTreeBackend");
    return cls(py::arg("block_backend") = block_backend_instance);
}

bool
is_no_symmetry(Symmetry const& symmetry)
{
    Symmetry no_sym{ std::vector<SymmetryFactor::Ptr>{ std::make_shared<NoSymmetry>() } };
    return symmetry.is_equivalent_to(no_sym);
}

/// Module-lifetime cache (avoids static ``py::object`` destruction at interpreter shutdown).
py::dict
backend_cache()
{
    py::module_ core = py::module_::import("cyten._core");
    if (!py::hasattr(core, "_tensor_backend_cache")) {
        core.attr("_tensor_backend_cache") = py::dict();
    }
    return core.attr("_tensor_backend_cache").cast<py::dict>();
}

} // namespace

py::object
get_backend(py::object symmetry, py::object block_backend)
{
    if (symmetry.is_none()) {
        symmetry = py::cast(get_config().default_tensor_backend);
    }
    if (block_backend.is_none()) {
        block_backend = py::cast(get_config().default_block_backend);
    }

    std::string tensor_backend;
    if (py::isinstance<Symmetry>(symmetry)) {
        Symmetry::Ptr sym_ptr = symmetry.cast<Symmetry::Ptr>();
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
    py::tuple key = py::make_tuple(tensor_backend, block_backend_str);
    py::dict cache = backend_cache();
    if (cache.contains(key))
        return cache[key];

    auto block_backend_instance = make_block_backend(block_backend_str);
    py::object backend;
    if (tensor_backend == "no_symmetry") {
        backend = py::cast(std::make_shared<NoSymmetryBackend>(block_backend_instance));
    } else if (tensor_backend == "abelian") {
        auto ab = std::make_shared<AbelianBackend>(block_backend_instance);
        // DataCls is filled when the type object exists (bindings); factory may run before
        // AbelianBackendData is fully usable from Python — set via py::type if available.
        try {
            ab->DataCls = py::type::of<AbelianBackendData>();
        } catch (py::error_already_set const&) {
            ab->DataCls = py::none();
        }
        backend = py::cast(std::move(ab));
    } else if (tensor_backend == "fusion_tree") {
        auto ft = std::make_shared<FusionTreeBackend>(block_backend_instance);
        try {
            ft->DataCls = py::type::of<FusionTreeData>();
        } catch (py::error_already_set const&) {
            ft->DataCls = py::none();
        }
        backend = py::cast(std::move(ft));
    } else {
        throw std::invalid_argument("Unknown tensor_backend: " + tensor_backend);
    }

    if (py::isinstance<Symmetry>(symmetry)) {
        if (!backend.attr("supports_symmetry")(symmetry).cast<bool>()) {
            throw std::runtime_error("backend does not support the given symmetry");
        }
    }

    cache[key] = backend;
    return backend;
}

} // namespace cyten
