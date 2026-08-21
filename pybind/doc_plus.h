#pragma once

#include <cstddef>
#include <string>
#include <string_view>

namespace cyten {

namespace detail {

[[nodiscard]] inline std::size_t
find_trailing_see_also(std::string_view doc)
{
    // Prefer the last NumPy "See Also" underline block.
    constexpr std::string_view markers[] = {
        "See Also\n--------",
        "See also\n--------",
    };
    std::size_t best = std::string_view::npos;
    for (auto marker : markers) {
        auto pos = doc.rfind(marker);
        if (pos != std::string_view::npos && (best == std::string_view::npos || pos > best)) {
            best = pos;
        }
    }
    return best;
}

[[nodiscard]] inline char const*
leak_string(std::string s)
{
    return (new std::string(std::move(s)))->c_str();
}

} // namespace detail

/// Concatenate a header-extracted docstring with a Python-only appendix.
///
/// If ``shared`` already ends with a NumPy ``See Also`` section (as produced by
/// ``doxygen_xml_to_docstrings.py``), ``python_extra`` is inserted *before* that
/// section so cross-links stay last.
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
    auto see_pos = detail::find_trailing_see_also(shared_sv);

    std::string combined;
    if (see_pos == std::string_view::npos) {
        combined.reserve(shared_sv.size() + 2 + extra_sv.size());
        combined.append(shared_sv);
        if (combined.back() != '\n') {
            combined.push_back('\n');
        }
        combined.push_back('\n');
        combined.append(extra_sv);
    } else {
        std::string_view before = shared_sv.substr(0, see_pos);
        while (!before.empty() && (before.back() == '\n' || before.back() == ' ')) {
            before.remove_suffix(1);
        }
        std::string_view after = shared_sv.substr(see_pos);
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

/// Append (or replace) a NumPy ``See Also`` link to a C++ symbol for Sphinx.
///
/// Use for wrappers / lambdas that are not 1:1 ``DOC(...)`` bindings, or to
/// override the auto-generated link from ``doxygen_xml_to_docstrings.py``.
///
/// @param doc Base docstring (often from ``DOC(...)`` or a short ``R"pydoc"``).
/// @param cpp_symbol Qualified C++ target as accepted by Sphinx, including
///     parentheses for functions. Examples:
///     ``"cyten::compose()"``,
///     ``"cyten::inner(TensorCPtr, TensorCPtr, bool)"``.
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

    std::string see_also = "See Also\n--------\n:cpp:";
    see_also += (role != nullptr && role[0] != '\0') ? role : "func";
    see_also += ":`";
    see_also += cpp_symbol;
    see_also += "`\n";

    if (doc == nullptr || doc[0] == '\0') {
        return detail::leak_string(std::move(see_also));
    }

    std::string_view doc_sv(doc);
    auto see_pos = detail::find_trailing_see_also(doc_sv);
    std::string combined;
    if (see_pos == std::string_view::npos) {
        combined.reserve(doc_sv.size() + 2 + see_also.size());
        combined.append(doc_sv);
        if (combined.back() != '\n') {
            combined.push_back('\n');
        }
        combined.push_back('\n');
        combined.append(see_also);
    } else {
        std::string_view before = doc_sv.substr(0, see_pos);
        while (!before.empty() && (before.back() == '\n' || before.back() == ' ')) {
            before.remove_suffix(1);
        }
        combined.reserve(before.size() + 2 + see_also.size());
        combined.append(before);
        combined.push_back('\n');
        combined.push_back('\n');
        combined.append(see_also);
    }
    return detail::leak_string(std::move(combined));
}

} // namespace cyten
