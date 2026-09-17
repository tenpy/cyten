#include <cyten/tools/hdf5_py_bridge.h>

#include <stdexcept>

namespace cyten::hdf5 {

namespace {

thread_local py::object* tls_saver = nullptr;
thread_local py::object* tls_loader = nullptr;
thread_local py::object* tls_h5gr = nullptr;
thread_local py::object tls_saver_obj;
thread_local py::object tls_loader_obj;
thread_local py::object tls_h5gr_obj;

} // namespace

PyBridge::PyBridge(py::object saver, py::object loader, py::object h5gr)
{
    prev_saver_obj_ = tls_saver_obj;
    prev_loader_obj_ = tls_loader_obj;
    prev_h5gr_obj_ = tls_h5gr_obj;
    had_saver_ = tls_saver != nullptr;
    had_loader_ = tls_loader != nullptr;
    had_h5gr_ = tls_h5gr != nullptr;

    tls_saver_obj = std::move(saver);
    tls_loader_obj = std::move(loader);
    tls_h5gr_obj = std::move(h5gr);
    tls_saver = tls_saver_obj.is_none() ? nullptr : &tls_saver_obj;
    tls_loader = tls_loader_obj.is_none() ? nullptr : &tls_loader_obj;
    tls_h5gr = tls_h5gr_obj.is_none() ? nullptr : &tls_h5gr_obj;
}

PyBridge::~PyBridge()
{
    tls_saver_obj = std::move(prev_saver_obj_);
    tls_loader_obj = std::move(prev_loader_obj_);
    tls_h5gr_obj = std::move(prev_h5gr_obj_);
    tls_saver = had_saver_ ? &tls_saver_obj : nullptr;
    tls_loader = had_loader_ ? &tls_loader_obj : nullptr;
    tls_h5gr = had_h5gr_ ? &tls_h5gr_obj : nullptr;
}

void
py_save_handle(std::string const& path, py::handle obj)
{
    if (!tls_saver)
        throw std::runtime_error("cyten::hdf5::py_save: no Python Hdf5Saver bound (PyBridge)");
    tls_saver->attr("save")(obj, path);
}

py::object
py_load(std::string const& path)
{
    if (!tls_loader)
        throw std::runtime_error("cyten::hdf5::py_load: no Python Hdf5Loader bound (PyBridge)");
    return tls_loader->attr("load")(path);
}

void
py_memorize_load(HighFive::Group& /*h5gr*/, py::handle obj)
{
    if (!tls_loader || !tls_h5gr)
        throw std::runtime_error("cyten::hdf5::py_memorize_load: no Python loader/group bound");
    tls_loader->attr("memorize_load")(*tls_h5gr, obj);
}

py::object
py_get_attr(HighFive::Group& /*h5gr*/, std::string const& name)
{
    if (!tls_loader || !tls_h5gr)
        throw std::runtime_error("cyten::hdf5::py_get_attr: no Python loader/group bound");
    return tls_loader->attr("get_attr")(*tls_h5gr, name);
}

void
py_set_group_attr(std::string const& name, py::handle value)
{
    if (!tls_h5gr)
        throw std::runtime_error("cyten::hdf5::py_set_group_attr: no Python group bound");
    (*tls_h5gr).attr("attrs")[py::str(name)] = value;
}

py::object
py_get_group_attr(std::string const& name)
{
    if (!tls_h5gr)
        throw std::runtime_error("cyten::hdf5::py_get_group_attr: no Python group bound");
    return (*tls_h5gr).attr("attrs")[py::str(name)];
}

py::object*
tls_py_saver()
{
    return tls_saver;
}

py::object*
tls_py_loader()
{
    return tls_loader;
}

py::object*
tls_py_h5gr()
{
    return tls_h5gr;
}

} // namespace cyten::hdf5
