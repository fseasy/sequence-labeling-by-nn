#ifndef SYMNN_MEM_H_INCLUDED_
#define SYMNN_MEM_H_INCLUDED_

#ifndef WIN32
#include <sys/shm.h>
#include <sys/mman.h>
#endif

#include <fcntl.h>

#ifndef WIN32
#include <mm_malloc.h>
#endif

#include <vector>

namespace symnn {

class MemAllocator
{
public:
    explicit MemAllocator(unsigned align);
    MemAllocator(const MemAllocator&) = delete;
    MemAllocator& operator=(const MemAllocator&) = delete;
    virtual ~MemAllocator() {};
    virtual void* malloc(std::size_t n) = 0;
    virtual void free(void* p) = 0;
    virtual void zero(void* p, std::size_t n) = 0;
    
    std::size_t round_up_align(std::size_t n) const;
protected:
    const unsigned align;
};

class CPUAllocator : public MemAllocator
{
public:
    CPUAllocator();
    void* malloc(std::size_t n) override;
    void free(void *p) override;
    void zero(void *p, std::size_t n) override;
};

/**
* inline implementation
***/
MemAllocator::MemAllocator(unsigned align) : align(align) {}

inline
std::size_t MemAllocator::round_up_align(std::size_t n) const
{
    if (align < 2U) { return n; }
    return (n + align - 1U) / align * align; // may overflow in extremely 
}

CPUAllocator::CPUAllocator()
    :MemAllocator(32U) {}

inline
void CPUAllocator::free(void* p)
{
    _mm_free(p);
}

inline
void CPUAllocator::zero(void* p, std::size_t n) {
    memset(p, 0, n);
}

} // end of namespace symnn



#endif