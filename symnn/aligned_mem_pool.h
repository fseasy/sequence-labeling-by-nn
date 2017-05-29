#ifndef SYMNN_ALIGNED_MEM_POOL_H_INCLUDED_
#define SYMNN_ALIGNED_MEM_POOL_H_INCLUDED_

#include <string>
#include "symnn/mem.h"
#include "misc/except_macros.h"

namespace symnn {

class InternalMemoryPool 
{
public:
    InternalMemoryPool(const std::string& name,
                       std::size_t capacity,
                       MemAllocator* alloc);
    ~InternalMemoryPool();

    void* allocate(std::size_t n);
    void free();
    void zero_allocated_memory();
    std::size_t get_truly_capacity() const;
    std::size_t get_used_capacity() const;
private:
    void sys_alloc();
    void zero_all();
private:
    std::string name;
    std::size_t capacity;
    MemAllocator* allocator;
    void* mem_space;
    std::size_t used;
};

class AlignedMemoryPool 
{
    /**
     * This is built for dynamically 
     * INCREASING MEMORY! 
     * by push new fresh InternalMemoryPool!
     * what a amazing and easy WAY!!
     */

public:
    AlignedMemoryPool(const std::string& name,
                      std::size_t capacity,
                      MemAllocator* allocator);
    ~AlignedMemoryPool();
    void* allocate(std::size_t n);
    void free();
    void zero_allocated_memory();

    std::size_t get_truly_capacity() const;
    std::size_t get_used_capacity() const;
private:
    std::string name;
    std::vector<InternalMemoryPool*> pools;
    std::size_t single_capacity;
    MemAllocator* allocator;
};


/**
 * inline implememtation
 **/

inline
InternalMemoryPool::~InternalMemoryPool()
{
    allocator->free(mem_space);
}

inline
void InternalMemoryPool::free()
{
    used = 0U;
}


inline
std::size_t InternalMemoryPool::get_truly_capacity() const
{
    return capacity;
}

inline
std::size_t InternalMemoryPool::get_used_capacity() const
{
    return used;
}

inline
void InternalMemoryPool::zero_allocated_memory()
{
    if(used > 0U)
    { 
        allocator->zero(mem_space, used); 
    }
}

inline
void InternalMemoryPool::zero_all()
{
    allocator->zero(mem_space, capacity);
}


inline
AlignedMemoryPool::~AlignedMemoryPool()
{
    for (InternalMemoryPool* p : pools) { delete p; }
}

inline
void AlignedMemoryPool::zero_allocated_memory()
{
    for (auto p : pools) { p->zero_allocated_memory(); };
}

inline
std::size_t AlignedMemoryPool::get_truly_capacity() const
{
    return single_capacity * pools.size();
}

} // end of namespace symnn

#endif