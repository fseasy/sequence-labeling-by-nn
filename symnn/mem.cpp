#include <cstdlib>
#include <cstring>
#include <iostream>

#include "symnn/mem.h"
#include "misc/except_macros.h"

using namespace std;

namespace symnn {

void* CPUAllocator::malloc(size_t n) {
    /**
     * std::aligned_alloc -> C++2017
     * g++4.8.5 supports
     * vs2017 only have aligned_malloc? EXM??
     */
    void* ptr = _mm_malloc(n, align);
    SLNN_ASSERT(ptr, MODULE_SYMNN_NAME,
                "_mm_malloc failed. "
                "Out-Of-Memory for size " +
                to_string(n) + " with aligned " +
                to_string(align));
    return ptr;
}






} // end of namespace symnn