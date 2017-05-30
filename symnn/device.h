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

class DeviceMemPoolSizes
{
public:
    DeviceMemPoolSizes() = default;
    DeviceMemPoolSizes(std::size_t total_sz);
    DeviceMemPoolSizes(std::size_t fxs_sz,
                       std::size_t dEdfs_sz,
                       std::size_t ps_sz);
    DeviceMemPoolSizes(const std::string& descriptor);
    std::size_t get_fxs_sz() const { return capacity[static_cast<int>(MemPoolType::FXS)]; }
    std::size_t get_dedfs_sz() const { return capacity[static_cast<int>(MemPoolType::DEDFS)]; }
    std::size_t get_ps_sz() const { return capacity[static_cast<int>(MemPoolType::PS)]; }
private:
    void set_value(std::size_t fxs_sz,
                   std::size_t dEdfs_sz,
                   std::size_t ps_sz);
    std::size_t capacity[NrMemPoolType];
};

class Device
{
public:
    Device(DeviceType t, MemAllocator* allocator);
    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;
    virtual ~Device() {};

    virtual DeviceMemPoolSizes mark(ComputationGraph* pcg);
    //virtual void revert(const DeviceMemPoolSizes& ps);
    void allocate_tensor(MemPoolType mpt, Tensor& t);
    real_t* allocate_space4tensor(MemPoolType mpt, const Dim& d);
protected:
    DeviceType type;
    MemAllocator* pallocator;
    std::string name;
    std::vector<AlignedMemoryPool*> pools;
};

class DeviceCPU : public Device
{
public:
    explicit DeviceCPU(const DeviceMemPoolSizes& mpsz);
    ~DeviceCPU() {};

    CPUAllocator allocator;
};

/**
 * inline implementation
 */

inline
Device::Device(DeviceType t, MemAllocator* allocator_)
    :type(t), pallocator(allocator_),
    pools(NrMemPoolType, nullptr){}




} // end of namespace symnn


#endif