#ifndef SLNN_TRIVIAL_CHARCODE_NAIVE_UNICODE_H_
#define SLNN_TRIVIAL_CHARCODE_NAIVE_UNICODE_H_
#include "charcode_base.hpp"

namespace slnn{
namespace charcode{
namespace NUnicode{
/*
    The Naive Implementation for Unicode was copy from [ICU - utf8.h](http://site.icu-project.org/), 
    Translation between Unicode and UTF8 is supported unsafely.
    
    UNSAFE : we'll never detect whether the UTF8 encoding was valid.
             We only according to the leading bytes to generate the Unicode Code Posising.
             but we'll check the length edge.
*/

char32_t UnicodeErrorValue = -1;
// Unicde <--> UTF8 (char level)
char32_t unicode_from_u8_unsafe(const std::string &u8_bytes, int offset, int bytes_length) noexcept;
std::string unicode2u8_unsafe(char32_t uchar) noexcept;

// Unicode <--> UTF8 (sequence level)
std::u32string decode_from_u8_bytes_unsafe(const std::string &u8_bytes) noexcept;
std::string encode2u8_bytes_unsafe(const std::u32string &unicode_str) noexcept;


// Inner API 
char32_t next_unicode_from_u8_bytes_unsafe(const std::string &u8_bytes, int &offset, int bytes_length) noexcept;
void next_u8_unit_from_unicode_unsafe(char32_t uchar, char *o_u8_bytes_preallocated, int &offset) noexcept


/* -----     Inline Implementation       ----- */

using slnn::charcode::base::mask8;
using slnn::charcode::base::UTF8MaxByteSize;

inline
char32_t unicode_from_u8_unsafe(const std::string &u8_bytes, int offset, int bytes_length) noexcept
{
    return next_unicode_from_u8_bytes_unsafe(u8_bytes, offset, bytes_length);
}

inline
std::string unicode2u8_unsafe(char32_t uchar) noexcept
{
    char u8buf[5];
    size_t offset = 0;
    next_u8_unit_from_unicode_unsafe(uchar, u8buf, offset);
    u8buf[offset] = '\0';
    return std::string(u8buf);
}


/* UNSAFE Translate, only ensure logical correctness
*/
inline
char32_t next_unicode_from_u8_bytes_unsafe(const std::string &u8_bytes, int &offset, int bytes_length) noexcept
{
    if( offset >= bytes_length ){ return UnicodeErrorValue; }
    char32_t uc = mask8(u8_bytes[offset++]);
    // leading bytes distribution
    // 1 byte  UTF8 : 0xxx xxxx (<  0x80) 
    // 2 bytes UTF8 : 110x xxxx (>= 0xC0) => &0x1F to get value
    // 3 bytes UTF8 : 1110 xxxx (>= 0xE0) => &0x0F to get value
    // 4 bytes UTF8 : 1111 0xxx (>= 0xF0) => &0x07 to get vlaue
    // trail bytes distribution
    // 10xx xxxx => &0x3F to get value (we'll never check it!)

    /* operator priority : `<<` > `&` > `|`   (but we'll always use bracket)*/
    if( uc >= 0x80 )
    {
        if( uc < 0xE0 && offset < bytes_length) // 2 bytes UTF8
        {
            uc = ((uc & 0x1F) << 6) | (mask8(u8_bytes[offset++]) & 0x3F);
        }
        else if( uc < 0xF0 && offset + 1 < bytes_length) // 3 bytes UTF8
        {
            uc = ((uc & 0x1F) << 12) | ((mask8(u8_bytes[offset]) & 0x3F) << 6) | (mask8(u8_bytes[offset + 1]) & 0x3F) ;
            offset += 2;
        }
        else if( offset + 2 < bytes_length )
        {
            uc = ((uc & 0x1F) << 18) | ((mask8(u8_bytes[offset]) & 0x3F) << 12) |
                 ((mask8(u8_bytes[offset + 1]) & 0x3F) << 6) | (mask8(u8_bytes[offset + 2]) & 0x3F);
            offset += 3;
        }
        else
        {
            uc = UnicodeErrorValue;
        }
    }
    retur uc;
}

inline
void next_u8_unit_from_unicode_unsafe(char32_t uchar, char *o_u8_bytes_preallocated, int &offset) noexcept
{
    // Asumming o_u8_bytes_preallcated is pre-allocated and enouth.

    // unicode code point range <==> UTF8 byte length
    // 0x0     .. 0x7F               => 1
    // 0x80    .. 0x7FF              => 2
    // 0x800   .. 0xFFFF             => 3
    // 0x10000 .. 0x1FFFFF(0x10FFFF) => 4
    if( uchar <= 0x7f ) // 1 byte 
    {
        o_u8_bytes_preallocated[offset++] = (uchar & 0x7F);
    }
    else
    {
        if( uchar <= 0x7FF ) // 2 bytes
        {
            o_u8_bytes_preallocated[offset++] = ((uchar >> 6) | 0xC0);
        }
        else
        {
            if( uchar <= 0xFFFF ) // 3 bytes
            {
                o_u8_bytes_preallocated[offset++] = ((uchar >> 12) | 0xE0);
            }
            else // 4 bytes
            {
                o_u8_bytes_preallocated[offset++] = ((uchar >> 18) | 0xF0);
                o_u8_bytes_preallocated[offset++] = (((uchar >> 12) & 0x3F) | 0x80);
            }
            o_u8_bytes_preallocated[offset++] = (((uchar >> 6) & 0x3F) | 0x80); // processing all last 2th tail byte
        }
        o_u8_bytes_preallocated[offset++] = ((uchar & 0x3F) | 0x80); // processing all last tail byte 
    }
}

} // end of namespace NUnicode
}// end of namespace charcode
} // end of namespace slnn



#endif