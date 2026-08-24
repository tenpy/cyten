#pragma once

#include <stdexcept>
#include <string>
#include <string_view>

namespace cyten {

/// Always-on check for public API preconditions.
///
/// Throws ``std::invalid_argument`` (Python ``ValueError``). Unlike ``assert``, this is not
/// compiled out under ``NDEBUG``.
///
/// Use ``check(cond, "msg")`` for a fixed message; for messages that include values, prefer
/// ``if (!cond) throw std::invalid_argument(std::format(...))`` so formatting runs only on
/// failure. Keep C ``assert`` for hot-path internal invariants.
inline void
check(bool condition, std::string_view message)
{
    if (!condition) {
        throw std::invalid_argument(std::string(message));
    }
}

} // namespace cyten
