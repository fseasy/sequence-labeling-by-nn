#ifndef SLNN_TRIVIAL_CHARCODE_CHARCODE_CONVERTOR_H_
#define SLNN_TRIVIAL_CHARCODE_CHARCODE_CONVERTOR_H_
#include "naive_unicode.h"
namespace slnn{
namespace charcode
{

struct CharcodeConvertor
{
    virtual char32_t decode1(const std::string &encoded_bytes, int start_pos, int bytes_length) = 0;
    virtual std::string encode1(char32_t unicode_char) = 0;
    virtual std::u32string decode(const std::string &encoded_bytes) = 0;
    virtual std::string encode(const std::u32string &unicode_str) = 0;
};

struct UTF8Convertor : public CharcodeConvertor
{
    char32_t decode1(const std::string &u8_bytes, int start_pos, int bytes_length) noexcept override;
    std::string encode1(char32_t unicode_char) noexcept override;
    std::u32string decode(const std::string &encoded_bytes) noexcept override;
    std::string encode(const std::u32string &unicode_str) noexcept override;
};

/* TODO */
struct GB18030Convertor : public CharcodeConvertor
{};



/*************************************
*      Inline Implementation
**************************************/
inline
char32_t UTF8Convertor::decode1(const std::string &u8_bytes, int start_pos, int bytes_length) noexcept
{
    return slnn::charcode::NUnicode::unicode_from_u8_unsafe(u8_bytes, start_pos, bytes_length);
}

inline
std::string UTF8Convertor::encode1(char32_t unicode_char) noexcept
{
    return slnn::charcode::NUnicode::unicode2u8_unsafe(unicode_char);
}

inline
std::u32string UTF8Convertor::decode(const std::string &u8_bytes) noexcept
{
    return slnn::charcode::NUnicode::decode_from_u8_bytes_unsafe(u8_bytes);
}

inline
std::string UTF8Convertor::encode(const std::u32string &unicode_str) noexcept
{
    return slnn::charcode::NUnicode::encode2u8_bytes_unsafe(unicode_str);
}

} // end of namespace charcode
} // end of namespace slnn



#endif