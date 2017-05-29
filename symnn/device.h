#ifndef SYMNN_DEVICE_H_INCLUDED_
#define SYMNN_DEVICE_H_INCLUDED_

#include <string>
#include "symnn/aligned_mem_pool.h"

namespace symnn {

enum class DeviceType: unsigned char
{ 
    CPU = 0U
};
enum class MemPoolType: unsigned char
{
    FXS = 0U,
    DEDFS = 1U,
    PS = 2U,
    NONE = 3U
};
constexpr unsigned NrMemPoolType = 3U;

class ComputationGraph;
class Tensor;

struct DeviceMemPoolSizes
{
    DeviceMemPoolSizes() = default;
    DeviceMemPoolSizes(std::size_t total_sz);
    DeviceMemPoolSizes(std::size_t fsx_sz,
                        std::size_t dEdfs_sz,
                        std::size_t ps_sz);
    DeviceMemPoolSizes(const std::string& descriptor);
    
    std::size_t used[NrMemPoolType];
};

class Device
{
public:
    Device(DeviceType t, MemAllocator* allocator);
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
    virtual ~Device() {};

    virtual DeviceMemPoolSizes mark(ComputationGraph* pcg);
    virtual void revert(const DeviceMemPoolSizes& ps);
    void allocate_tensor(MemPoolType mem_pool, Tensor& t);
protected:
    DeviceType type;
    MemAllocator* allocator;
    std::string name;
    std::vector<AlignedMemoryPool*> pools;
};


/**
 * inline implementation
 */

inline
Device::Device(DeviceType t, MemAllocator* allocator_)
    :type(t), allocator(allocator_),
    pools(NrMemPoolType, nullptr){}




} // end of namespace symnn


#endif