#ifndef SLNN_TRIVIAL_CHARCODE_CHARCODE_BASE_HPP_
#define SLNN_TRIVIAL_CHARCODE_CHARCODE_BASE_HPP_
#include <string>

namespace slnn{
namespace charcode{
namespace base{

enum class EncodingType
{
    UTF8 = 1,
    GB18030 = 2,
    UNSUPPORT = 9999
};

using uint8_t = unsigned char;
constexpr int UTF8MaxByteSize = 4;
// octec stands for 8 bits (a byte)

template <typename octet_type>
inline  
uint8_t mask8(octet_type c)
{
    return static_cast<uint8_t>(c); 
}

/*****************************
 * utilities 
 *****************************/

inline
std::string encoding_type2str(EncodingType encoding_type)
{
    switch( encoding_type )
    {
    case EncodingType::UTF8 :
        return "UTF8";
    case EncodingType::GB18030:
        return "GB18030";
    default:
        return "unsupport(value=" + std::to_string(static_cast<int>(encoding_type)) + ")";
    }
}

inline
EncodingType str2encoding_type(const std::string &str)
{
    std::string upper_name(str);
    for( char &c : upper_name ){ c = ::toupper(c); }
    if( upper_name == "UTF8" ){ return EncodingType::UTF8; }
    else if( upper_name == "GB18030" || upper_name == "GB2312" ){ return EncodingType::GB18030; }
    else{ return EncodingType::UNSUPPORT; }
}


} // end of namespace base
using base::EncodingType;
} // end of namespace charcode
} // end of namespace slnn



#endif