#pragma once

#include <string>

namespace cyten {

/// Concatenate a header-extracted docstring with a Python-only appendix.
///
/// Returns a process-lifetime C string (leaked) so it is safe to pass to
/// pybind11 as a docstring without dangling pointers.
[[nodiscard]] inline char const*
doc_plus(char const* shared, char const* python_extra)
{
    if (python_extra == nullptr || python_extra[0] == '\0') {
        return shared;
    }
    if (shared == nullptr || shared[0] == '\0') {
        return python_extra;
    }
    std::string combined;
    combined.reserve(std::char_traits<char>::length(shared) + 2 +
                     std::char_traits<char>::length(python_extra));
    combined.append(shared);
    if (combined.back() != '\n') {
        combined.push_back('\n');
    }
    combined.push_back('\n');
    combined.append(python_extra);
    return (new std::string(std::move(combined)))->c_str();
}

} // namespace cyten
