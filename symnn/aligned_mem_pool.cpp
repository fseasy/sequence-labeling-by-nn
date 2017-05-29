#include <sstream>
#include "symnn/aligned_mem_pool.h"

using namespace std;
namespace symnn {

InternalMemoryPool::InternalMemoryPool(const std::string& name,
                                       std::size_t capacity_,
                                       MemAllocator* alloc)
    :name(name), capacity(capacity_),
    allocator(alloc)
{
    // set the capacity to fit the alignment.
    capacity = allocator->round_up_align(capacity_);
    sys_alloc();
    zero_all();
    used = 0U;
}

void* InternalMemoryPool::allocate(size_t n)
{
    size_t rounded_n = allocator->round_up_align(n);
    if (rounded_n + used > capacity) 
    {
        return nullptr;
    }
    // void* can't do offset oprating.
    void* res = static_cast<char*>(mem_space) + used;
    used += rounded_n;
    return res;
}

void InternalMemoryPool::sys_alloc()
{
    mem_space = allocator->malloc(capacity);
    SLNN_ASSERT(mem_space, MODULE_SYMNN_NAME,
                "InternalMemoryPool::sys_alloc failed "
                "on the memory " + to_string(capacity) +
                " Bytes allocation.");
}

AlignedMemoryPool::AlignedMemoryPool(const std::string& name,
                                     size_t capacity,
                                     MemAllocator* allocator)
    :name(name),
    single_capacity(capacity),
    allocator(allocator)
{
    single_capacity = allocator->round_up_align(capacity);
    SLNN_ASSERT(capacity > 0U, MODULE_SYMNN_NAME,
                "AlignedMemoryPool constructing failed: "
                "attempting to allocate zero memory");
    pools.push_back(
        new InternalMemoryPool(name, single_capacity, allocator)
    );
}

void* AlignedMemoryPool::allocate(size_t n)
{
    void* res = pools.back()->allocate(n);
    if (res == nullptr)
    {
        SLNN_ASSERT(n <= single_capacity, MODULE_SYMNN_NAME,
                    "AlignedMemoryPool::allocate failed: "
                    "attempt to allocate memory over single capacity.");
        pools.push_back(new InternalMemoryPool(
            name, single_capacity, allocator));
        res = pools.back()->allocate(n);
    }
    return res;
}

void AlignedMemoryPool::free()
{
    if (pools.size() > 1U)
    {
        /**
         * in fact, we can still use free
         * for every InternalMemoryPool
         * if we do it, we should change the allocate
         * logic and add a variable to indicate the 
         * current position of pools.
         * blow just delete it, more easy and clear.
         **/
        // firly delete all.
        // truly delete, instead of clear.
        for (InternalMemoryPool* p : pools) { delete p };
        // expand the capacity
        single_capacity *= pools.size();
        single_capacity = allocator->round_up_align(single_capacity);
        // clear vector
        pools.clear();
        pools.push_back(
            new InternalMemoryPool(name, single_capacity, allocator)
        );
    }
    else { pools[0]->free(); }
}

size_t AlignedMemoryPool::get_used_capacity() const
{
    if (pools.size() == 1U) { return pools[0]->used(); }
    else
    {
        size_t used = 0U;
        for (auto p : pools) { used += p->get_used_capacity(); }
        returrn used;
    }
}



} // end of namespace symnn