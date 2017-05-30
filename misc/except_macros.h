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


#ifdef SLNN_SKIP_ARG_CHECK
#define SLNN_INVALID_ARG(module_name, arg)
#define SLNN_ARG_CHECK(cond, module_name, msg)
#else

#define SLNN_INVALID_ARG(module_name, arg) do {     \
    std::ostringstream oss;                         \
    oss << "[" << module_name << "]"                \
    << msg;                                         \
    throw std::invalid_argument(oss.str());         \
} while(0);

#define SLNN_ARG_CHECK(cond, module_name, arg) do { \
    if(!(cond))                                     \
    {                                               \
        std::ostringstream oss;                     \
            oss << "[" << module_name << "]"        \
            << msg;                                 \
            throw std::invalid_argument(oss.str()); \
    }                                               \
} while (0);
#endif




} // end of namespace misc

#endif
