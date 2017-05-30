#include "symnn/device.h"
#include "misc/strutils.h"
#include "symnn/symnn.h"
#include "symnn/expr.h"
#include "symnn/dim.h"
using namespace std;

namespace symnn {

void DeviceMemPoolSizes::set_value(size_t fx_sz,
                              size_t dEdfs_sz,
                              size_t ps_sz)
{
    capacity[static_cast<int>(MemPoolType::FXS)] = fx_sz;
    capacity[static_cast<int>(MemPoolType::DEDFS)] = dEdfs_sz;
    capacity[static_cast<int>(MemPoolType::PS)] = ps_sz;
}

DeviceMemPoolSizes::DeviceMemPoolSizes(size_t total_size)
{
    size_t mean_val = total_size / NrMemPoolType;
    set_value(mean_val, mean_val, mean_val);
}

DeviceMemPoolSizes::DeviceMemPoolSizes(size_t fx_sz,
                                       size_t dEdfs_sz,
                                       size_t ps_sz)
{
    set_value(fx_sz, dEdfs_sz, ps_sz);
}

DeviceMemPoolSizes::DeviceMemPoolSizes(const std::string& descriptor)
{
    vector<string> values;
    misc::split(values, descriptor, ",");
    if (values.size() == 1U)
    {
        size_t total_size = std::stoul(values[0]);
        size_t mean_val = total_size / NrMemPoolType;
        set_value(mean_val, mean_val, mean_val);
    }
    else if (values.size() == NrMemPoolType)
    {
        size_t fx_sz = stoul(values[static_cast<int>(MemPoolType::FXS)]),
            dEdfs_sz = stoul(values[static_cast<int>(MemPoolType::DEDFS)]),
            ps_sz = stoul(values[static_cast<int>(MemPoolType::PS)]);
        set_value(fx_sz, dEdfs_sz, ps_sz);
    }
    else
    {
        SLNN_INVALID_ARG(MODULE_SYMNN_NAME,
                         "invalid memory pool size, "
                         "set a toatl size, "
                         "or use 'fx_sz,dEdfs_sz,ps_sz' set individual size");
    }
}

DeviceMemPoolSizes Device::mark(ComputationGraph* pcg)
{
    size_t node_num = pcg->get_node_num();
    if (node_num == 0U) 
    {
        return DeviceMemPoolSizes(0U, 0U, 0U);
    }
    expr::Expression last_expr(pcg, node_num - 1U);
    pcg->incremental_forward(last_expr);
    return DeviceMemPoolSizes(pools[static_cast<int>(MemPoolType::FXS)]->get_used_capacity(),
                              pools[static_cast<int>(MemPoolType::DEDFS)]->get_used_capacity(),
                              pools[static_cast<int>(MemPoolType::PS)]->get_used_capacity());
}

void Device::allocate_tensor(MemPoolType mpt, Tensor& t)
{
    Tensor::REAL_TYPE* mem_ptr = allocate_space4tensor(mpt, t.get_dim());
    t.set_raw_ptr(mem_ptr);
    t.set_mpt(mpt);

}

Tensor::REAL_TYPE* Device::allocate_space4tensor(MemPoolType mpt, const Dim& dim)
{
    AlignedMemoryPool * p = pools[static_cast<int>(mpt)];
    SLNN_ASSERT(p, MODULE_SYMNN_NAME,
                "mempool is nullptr");
    Tensor::REAL_TYPE* mem_ptr = static_cast<Tensor::REAL_TYPE*>(
        p->allocate(dim.size() * sizeof(Tensor::REAL_TYPE))
        );
    SLNN_ASSERT(mem_ptr != nullptr, MODULE_SYMNN_NAME,
                "Device::allocate_tensor failed.");
    return mem_ptr;
}


DeviceCPU::DeviceCPU(const DeviceMemPoolSizes& mpsz)
    :Device(DeviceType::CPU, nullptr)
{
    Device::pallocator = &this->allocator;

    size_t fxs_sz_kb = mpsz.get_fxs_sz() << 20;
    size_t dedfs_sz_kb = mpsz.get_dedfs_sz() << 20;
    size_t ps_sz_kb = mpsz.get_ps_sz() << 20;
    pools[static_cast<int>(MemPoolType::FXS)] = new AlignedMemoryPool("Forward MP",
                                                                      fxs_sz_kb,
                                                                      &allocator);
    if (dedfs_sz_kb != 0U)
    {
        pools[static_cast<int>(MemPoolType::DEDFS)] = new AlignedMemoryPool("Backward MP",
                                                                            dedfs_sz_kb,
                                                                            &allocator);
    }
    if (ps_sz_kb != 0U)
    {
        pools[static_cast<int>(MemPoolType::PS)] = new AlignedMemoryPool("Parameter MP",
                                                                         ps_sz_kb,
                                                                         &allocator);
    }
    
}

} // end of namespace symnn