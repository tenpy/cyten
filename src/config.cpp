#include <cyten/config.h>

namespace cyten {

const CytenConfig&
get_config()
{
    static CytenConfig config;
    static bool config_loaded = false;
    if (config_loaded)
        return config;
    if (!config_loaded) {
        config_loaded = true;
        // TODO (issue #209): load config from environment variables or config (yaml) file (or
        // both).
    }
    return config;
}

} // namespace cyten
