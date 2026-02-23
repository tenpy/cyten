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
        int linewidth = 100;
        int indent = 2;
        int precision = 8; // #digits
        int maxlines_spaces = 15;
        int maxlines_tensors = 30;
        bool skip_data = false;        // skip Data section in Tensor prints
        bool summarize_blocks = false; // True -> always summarize (show only shape, not entries)
        PrintOptions() = default;
    };

    PrintOptions print_options = PrintOptions();
    bool do_fusion_input_checks =
      true; // If the symmetry methods should check their inputs are valid
    std::string default_symmetry_backend = "abelian";
    std::string default_block_backend = "numpy";
    CytenConfig() = default;
};

// NOLINTEND(readability-magic-numbers)

const CytenConfig& get_config();

} // namespace cyten
