#ifndef SLNN_TRIVIAL_CHARCODE_NAIVE_UNICODE_H_
#define SLNN_TRIVIAL_CHARCODE_NAIVE_UNICODE_H_
#include "charcode_base.hpp"

namespace slnn{
namespace charcode{
namespace NUnicode{

char32_t UnicodeErrorValue = -1;

char32_t unicode_from_u8(const std::string &u8_bytes, int offset, int bytes_length) noexcept;
std::string unicode2u8(char32_t uchar) noexcept;

char32_t next_unicode_from_u8_bytes(const std::string &u8_bytes, int &offset, int bytes_length) noexcept;

void decode_from_u8_bytes(const std::string &u8_bytes);

using slnn::charcode::base::mask8;
inline
char32_t unicode_from_u8(const std::string &u8_bytes, int offset, int bytes_length) noexcept
{
    return next_unicode_from_u8_bytes(u8_bytes, offset, bytes_length);
}

} // end of namespace NUnicode
}// end of namespace charcode
} // end of namespace slnn



#endif