#pragma once

#include <string>

#include <cyten/cyten.h>

namespace cyten {

// NOLINTBEGIN(readability-magic-numbers)

class CytenConfig
{
  public:
    class PrintOptions
    {
      public:
        int64 linewidth = 100;
        int64 indent = 2;
        /// #digits
        int64 precision = 8;
        int64 maxlines_spaces = 15;
        int64 maxlines_tensors = 30;
        /// skip Data section in Tensor prints
        bool skip_data = false;
        /// True -> always summarize (show only shape, not entries)
        bool summarize_blocks = false;
        PrintOptions() = default;
    };

    PrintOptions print_options = PrintOptions();
    /// If the symmetry methods should check their inputs are valid
    bool do_fusion_input_checks = true;
    std::string default_symmetry_backend = "abelian";
    std::string default_block_backend = "numpy";
    CytenConfig() = default;
};

// NOLINTEND(readability-magic-numbers)

const CytenConfig& get_config();

} // namespace cyten
