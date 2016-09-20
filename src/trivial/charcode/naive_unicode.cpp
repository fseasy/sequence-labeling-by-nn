#include "naive_unicode.h"

namespace slnn{
namespace charcode{
namespace NUnicode{

std::u32string decode_from_u8_bytes_unsafe(const std::string &u8_bytes) noexcept
{
    int length = u8_bytes.length();
    char32_t unicode_buf[length+1] = "";
    int u8_offset = 0,
        unicode_offset = 0;
    while( u8_offset < length )
    {
        char32_t code_point = next_unicode_from_u8_bytes_unsafe(u8_bytes, u8_offset, length);// auto incease u8_offset
        if( code_point != UnicodeErrorValue )
        {
            unicode_buf[unicode_offset++] = code_pont;
        }
        else
        {
            std::cerr << "At UTF8 string : \n"
                << u8_bytes << "\n"
                << "utf8 posistion: " << u8_offset - 1 << ", unicode position: " << unicode_offset << "\n";
            // don't inceasing unicode offset
        }
    }
    unicode_buf[unicode_offset] = '\0';
    return std::u32string(unicode_buf);
}

std::string encode2u8_bytes_unsafe(const std::u32string &unicode_str) noexcept
{
    int length = unicode_str.length();
    char *u8buf[length * UTF8MaxByteSize + 1];
    int u8_offset = 0;
    for( auto uchar : unicode_str )
    {
        next_u8_unit_from_unicode_unsafe(uchar, u8buf, u8_offset);
    }
    u8buf[u8_offset] = '\0';
    return std::string(u8buf);
}

} // end of namespace NUnicode
} // end of namespace charcode
} // end of namespace slnn