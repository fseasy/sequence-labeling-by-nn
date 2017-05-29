#include "symnn/device.h"
#include "misc/strutils.h"
using namespace std;

namespace symnn {

DeviceMemPoolSizes::DeviceMemPoolSizes(size_t total_size)
{
    for (unsigned i = 0U; i < NrMemPoolType; ++i) 
    {
        used[i] = total_size / NrMemPoolType;
    }
}

DeviceMemPoolSizes::DeviceMemPoolSizes(size_t fx_sz,
                                       size_t dEdfs_sz,
                                       size_t ps_sz)
{
    used[MemPoolType::FXS] = fx_sz;
    used[MemPoolType::DEDFS] = dEdfs_sz;
    used[MemPoolType::PS] = ps_sz;
}

DeviceMemPoolSizes::DeviceMemPoolSizes(const std::string& descriptor)
{
    vector<string> values;
    misc::split(values, descriptor, ",");
    if (values.size() == 1U)
    {
        size_t total_size = std::stoul(values[0]);

    }
}



} // end of namespace symnn