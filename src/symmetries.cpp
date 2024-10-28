
#include "symmetries.h"

#include <cassert>

namespace cyten {

SectorArray::SectorArray(size_t sector_ind_len)
    : data(), sector_len(sector_ind_len)
{
}

size_t SectorArray::size() const
{
    size_t s = data.size();
    assert(s % sector_len == 0);
    return s / sector_len ; 
}

Sector SectorArray::operator[](size_t i)
{
    assert(data.size() >= (i+1)*sector_len);
    charge * first = &data[i*sector_len];
    return Sector(first, first + sector_len);
}

void SectorArray::push_back(Sector const & s)
{
    assert(s.size() == sector_len);
    data.reserve(data.size() + sector_len);
    for (charge i : s)
        data.push_back(i);
}

charge *SectorArray::raw_pointer()
{
    return &data[0];
}

} // namespace cyten
