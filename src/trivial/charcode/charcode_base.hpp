#ifndef SLNN_TRIVIAL_CHARCODE_CHARCODE_BASE_HPP_
#define SLNN_TRIVIAL_CHARCODE_CHARCODE_BASE_HPP_
#include <string>

namespace slnn{
namespace charcode{
namespace base{

using uint8_t = unsigned char;

// octec stands for 8 bits (a byte)

template <typename octet_type>
inline  
uint8_t mask8(octet_type c)
{
    return static_cast<uint8_t>(c); 
}

} // end of namespace base

} // end of namespace charcode
} // end of namespace slnn



#endif