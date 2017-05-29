#ifndef MISC_EXCEPT_MACROS_H_INLCUDED_
#define MISC_EXCEPT_MACROS_H_INCLUDED_

#include <sstream>
#include <stdexcept>

namespace misc {

#define ENABLE_ASSERT

#define MODULE_SYMNN_NAME "SYMNN"

#ifdef ENABLE_ASSERT
#define SLNN_ASSERT(expr, module_name, msg) do {\
        if(!(expr)) {                               \
            std::ostringstream oss;                 \
            oss << "[" << module_name << "] "       \
            << msg;                                 \
            throw std::runtime_error(oss.str());    \
        }                                           \
     } while (0);                                   
#else
#define SLNN_ASSERT(expr, module_name, msg)
#endif

#define SLNN_RUNTIME_ERROR(module_name, msg) do {   \
    std::ostringstream oss;                         \
    oss << "[" << module_name << "]"                \
    << msg;                                         \
    throw std::runtime_error(oss.str())             \
} while(0);



}




} // end of namespace misc

#endif
