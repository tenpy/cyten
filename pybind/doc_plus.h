#pragma once

#include <string>
#include <string_view>

namespace cyten {

namespace detail {

/// Marker produced by ``doxygen_xml_to_docstrings.py`` / ``doc_cpp_ref``.
inline constexpr std::string_view kCppRefMarker = ".. cyten-cpp-ref::";

[[nodiscard]] inline char const*
leak_string(std::string s)
{
    return (new std::string(std::move(s)))->c_str();
}

[[nodiscard]] inline std::string
format_cpp_ref_marker(char const* cpp_symbol, char const* role)
{
    // Inventory keys are bare names (cyten::compose), not cyten::compose().
    std::string_view sym(cpp_symbol);
    auto paren = sym.find('(');
    if (paren != std::string_view::npos) {
        sym = sym.substr(0, paren);
    }
    while (!sym.empty() && (sym.back() == ' ' || sym.back() == '\t')) {
        sym.remove_suffix(1);
    }

    std::string marker = ".. cyten-cpp-ref:: ";
    marker.append(sym);
    marker += '\n';
    char const* r = (role != nullptr && role[0] != '\0') ? role : "func";
    if (std::string_view(r) != "func") {
        marker += "   :role: ";
        marker += r;
        marker += '\n';
    }
    return marker;
}

} // namespace detail

/// Concatenate a header-extracted docstring with a Python-only appendix.
///
/// If ``shared`` already ends with a ``.. cyten-cpp-ref::`` marker (as produced
/// by ``doxygen_xml_to_docstrings.py``), ``python_extra`` is inserted *before*
/// that marker so the Sphinx ``[C++]`` badge metadata stays last.
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

    std::string_view shared_sv(shared);
    std::string_view extra_sv(python_extra);
    auto marker_pos = shared_sv.rfind(detail::kCppRefMarker);

    std::string combined;
    if (marker_pos == std::string_view::npos) {
        combined.reserve(shared_sv.size() + 2 + extra_sv.size());
        combined.append(shared_sv);
        if (combined.back() != '\n') {
            combined.push_back('\n');
        }
        combined.push_back('\n');
        combined.append(extra_sv);
    } else {
        std::string_view before = shared_sv.substr(0, marker_pos);
        while (!before.empty() && (before.back() == '\n' || before.back() == ' ')) {
            before.remove_suffix(1);
        }
        std::string_view after = shared_sv.substr(marker_pos);
        combined.reserve(before.size() + 2 + extra_sv.size() + 1 + after.size());
        combined.append(before);
        combined.push_back('\n');
        combined.push_back('\n');
        combined.append(extra_sv);
        if (!extra_sv.empty() && extra_sv.back() != '\n') {
            combined.push_back('\n');
        }
        combined.push_back('\n');
        combined.append(after);
    }
    return detail::leak_string(std::move(combined));
}

/// Append (or replace) a ``.. cyten-cpp-ref::`` marker for Sphinx ``[C++]`` badges.
///
/// Use for wrappers / lambdas that are not 1:1 ``DOC(...)`` bindings, or to
/// override the auto-generated marker from ``doxygen_xml_to_docstrings.py``.
///
/// @param doc Base docstring (often from ``DOC(...)`` or a short ``R"pydoc"``).
/// @param cpp_symbol Qualified C++ name (``"cyten::compose"``). A trailing
///     ``()`` or parameter list is stripped for the Sphinx inventory key.
/// @param role Sphinx C++ domain role without colons (default ``"func"``;
///     also ``"class"``, ``"enum"``, …).
///
/// Returns a process-lifetime C string (leaked).
[[nodiscard]] inline char const*
doc_cpp_ref(char const* doc, char const* cpp_symbol, char const* role = "func")
{
    if (cpp_symbol == nullptr || cpp_symbol[0] == '\0') {
        return doc;
    }

    std::string marker = detail::format_cpp_ref_marker(cpp_symbol, role);

    if (doc == nullptr || doc[0] == '\0') {
        return detail::leak_string(std::move(marker));
    }

    std::string_view doc_sv(doc);
    auto marker_pos = doc_sv.rfind(detail::kCppRefMarker);
    std::string combined;
    if (marker_pos == std::string_view::npos) {
        combined.reserve(doc_sv.size() + 2 + marker.size());
        combined.append(doc_sv);
        if (combined.back() != '\n') {
            combined.push_back('\n');
        }
        combined.push_back('\n');
        combined.append(marker);
    } else {
        std::string_view before = doc_sv.substr(0, marker_pos);
        while (!before.empty() && (before.back() == '\n' || before.back() == ' ')) {
            before.remove_suffix(1);
        }
        combined.reserve(before.size() + 2 + marker.size());
        combined.append(before);
        combined.push_back('\n');
        combined.push_back('\n');
        combined.append(marker);
    }
    return detail::leak_string(std::move(combined));
}

} // namespace cyten
