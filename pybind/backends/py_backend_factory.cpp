#include "../py_cyten_pybind11.h"

#include <cyten/backends/backend_factory.h>

namespace cyten {

void
bind_backend_factory(py::module_& m)
{
    m.def("get_backend",
          &get_backend,
          py::arg("symmetry") = py::none(),
          py::arg("block_backend") = py::none(),
          R"pydoc(
          Get an instance of an appropriate backend.

          Backends are instantiated only once and then cached. If a suitable backend instance is in
          the cache, that same instance is returned.

          Parameters
          ----------
          symmetry : {'no_symmetry', 'abelian', 'fusion_tree'} | Symmetry
              Specifies which subclass of :class:`TensorBackend` to use, either directly via string,
              or as the minimal version which supports the given symmetry.
          block_backend : {None, 'numpy', 'torch', 'tensorflow', 'jax', 'cpu', 'gpu', 'tpu'}
              Specify which block backend to use.
          )pydoc");
}

} // namespace cyten
