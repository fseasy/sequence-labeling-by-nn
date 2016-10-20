#ifndef SLNN_TRIVIAL_CHARCODE_CHARCODE_CONVERTOR_H_
#define SLNN_TRIVIAL_CHARCODE_CHARCODE_CONVERTOR_H_
#include <memory>
#include <stdexcept>
#include "naive_unicode.h"
#include "charcode_detector.h"
namespace slnn{
namespace charcode{
namespace convertor{
struct CharcodeConvertor
{
    /**
     * factory for convertor.
     * factory pattern for C++, see: https://sourcemaking.com/design_patterns/factory_method/cpp/1
     */
    static std::shared_ptr<CharcodeConvertor> create_convertor(base::EncodingType encoding_type);
    virtual ~CharcodeConvertor(){};
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
{
    char32_t decode1(const std::string &u8_bytes, int start_pos, int bytes_length) noexcept override { return 0; }
    std::string encode1(char32_t unicode_char) noexcept override { return ""; }
    std::u32string decode(const std::string &encoded_bytes) noexcept override { return U""; }
    std::string encode(const std::u32string &unicode_str) noexcept override { return ""; }
};



/*************************************
*      Inline Implementation
**************************************/

inline
std::shared_ptr<CharcodeConvertor>
CharcodeConvertor::create_convertor(base::EncodingType encoding_type)
{
    if( encoding_type == base::EncodingType::UTF8 )
    {
        return std::make_shared<UTF8Convertor>();
    }
    else if( encoding_type == base::EncodingType::GB18030 )
    {
        return std::make_shared<GB18030Convertor>();
    }
    else
    {
        throw std::invalid_argument(std::string("un-supported encoding type: ") + 
                                    base::encoding_type2str(encoding_type));
    }
}


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

} // end of namespace convertor
using convertor::CharcodeConvertor;
} // end of namespace charcode
} // end of namespace slnn



#endif