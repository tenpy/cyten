#include <string>

#include "version.h"
#include <cyten/version.h>

namespace cyten {

const std::string&
get_build_version()
{

    static std::string version = (GIT_TAG != "" ? GIT_TAG : GIT_REV "-" GIT_BRANCH);
    return version;
}

} // namespace cyten
