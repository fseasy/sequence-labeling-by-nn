#ifndef UTF8PROCESSING_HPP_INCLUDED
#define UTF8PROCESSING_HPP_INCLUDED
#include <string>
#include <vector>
#include "utils/typedeclaration.h"
#include <boost/log/trivial.hpp>
/**
 * uint8_t , mask8 , get_length is copy from `utf8.h`
 */
/*********************UNICODE <---> utf8 translate table******************
unicode(U+)	              utf-8
U+00000000 - U+0000007F:	0xxxxxxx
U+00000080 - U+000007FF:	110xxxxx10xxxxxx
U+00000800 - U+0000FFFF:	1110xxxx10xxxxxx10xxxxxx
U+00010000 - U+001FFFFF:	11110xxx10xxxxxx10xxxxxx10xxxxxx
U+00200000 - U+03FFFFFF:	111110xx10xxxxxx10xxxxxx10xxxxxx10xxxxxx
U+04000000 - U+7FFFFFFF:	1111110x10xxxxxx10xxxxxx10xxxxxx10xxxxxx10xxxxxx
***************************************************************************/
namespace slnn
{

struct UTF8Processing
{
    using uint8_t = unsigned char ;
    template<typename octet_type>
    static uint8_t mask8(octet_type) ;
    static std::string::difference_type get_number_byte_width(const std::string::const_iterator &start_ite, 
            const std::string::const_iterator &end_ite);
    static size_t get_number_byte_width(const std::string &str, size_t start_pos);
    
    static std::string::difference_type get_utf8_char_length(const std::string::const_iterator &start_ite , 
            const std::string::const_iterator &end_ite);
    static size_t get_utf8_char_length(const std::string &str , size_t start_pos);
    
    static std::string::difference_type get_utf8_char_length_checked(const std::string::const_iterator &start_ite , 
            const std::string::const_iterator &end_ite) ;
    static size_t get_utf8_char_length_checked(const std::string &str , size_t start_pos) ;

    // utils
    static size_t utf8_char_len(const std::string &utf8_str);
    static void utf8_str2char_seq(const std::string &utf8_str, Seq &utf8_seq) ;
    static std::string replace_number(const std::string &str ,
                                      const std::string number_transform_str="##",
                                      const size_t length_transform_str=2);

};


template<typename octet_type>
inline
uint8_t UTF8Processing::mask8(octet_type oc)
{
    return static_cast<uint8_t>(oc & 0xff) ;
}

/******
 * get_number_byte_width
 * return string::difference_type
 * if character from this start_pos is not number , return 0
 * else , if UTF8 number , return 3
 * else , Ascii number , return 1
 */
inline 
std::string::difference_type UTF8Processing::get_number_byte_width(const std::string::const_iterator &pos, 
        const std::string::const_iterator &end_pos)
{
    if (pos == end_pos) return 0u ;
    if (mask8(*pos) == 0xef &&
        pos + 1 != end_pos && mask8(*(pos + 1)) == 0xbc &&
        pos + 2 != end_pos && mask8(*(pos + 2)) >= 0x90 && mask8(*(pos + 2)) <= 0x99)
    {
        // UTF8 number
        return 3u ;
    }
    else if (mask8(*pos) <= 0x39 && mask8(*pos) >= 0x30)
    {
        // Ascii number
        return 1u ;
    }
    else
    {
        // Not number
        return 0u;
    }
}

inline
size_t UTF8Processing::get_number_byte_width(const std::string &str , size_t start_pos)
{
    size_t len = str.length() ;
    if(start_pos >= len) return 0u ;
    if( mask8(str[start_pos]) == 0xef &&
        start_pos + 1 < len && mask8(str[start_pos+1]) == 0xbc &&
        start_pos + 2 < len && mask8(str[start_pos+2]) >= 0x90 && mask8(str[start_pos+2]) <= 0x99 ) 
    {
        return 3u ;
    }
    else if( mask8(str[start_pos]) <= 0x39 && mask8(str[start_pos]) >= 0x30 )
    {
        return 1u ;
    }
    else return 0u ;
}

inline
std::string::difference_type UTF8Processing::get_utf8_char_length(const std::string::const_iterator &start_ite , 
    const std::string::const_iterator &end_ite )
{
   if(start_ite >= end_ite) return 0 ;
   uint8_t lead = mask8(*start_ite) ;
    if(lead >> 7 == 0) return 1 ; // ASCII , 1 bit
    else if( (lead >> 5) == 0x6) return 2 ; // Infact , bracket is not necessary .
    else if( (lead >> 4) == 0xE) return 3 ;
    else if( (lead >> 3) == 0x1E) return 4 ;
    else if( (lead >> 2) == 0x3E) return 5 ;
    else if( (lead >> 1) == 0x7E) return 6 ;
    else return 0 ;
}

inline
size_t UTF8Processing::get_utf8_char_length(const std::string &str , size_t start_pos)
{
    return get_utf8_char_length(str.cbegin() + start_pos , str.cend()) ;
}

inline
std::string::difference_type UTF8Processing::get_utf8_char_length_checked(const std::string::const_iterator &start_iter , 
        const std::string::const_iterator &end_iter)
{
    size_t utf8_char_len = get_utf8_char_length(start_iter , end_iter) ;
    std::string::const_iterator utf8_end_iter = start_iter + utf8_char_len ;
    if( utf8_end_iter > end_iter) return 0 ;
    std::string::const_iterator checking_iter = start_iter;
    while(++checking_iter != utf8_end_iter)
    {
        uint8_t check_byte = mask8(*checking_iter) ;
        if( (check_byte >> 6) != 0x2 ) return 0 ;
    }
    return utf8_char_len ;
}

inline
size_t UTF8Processing::get_utf8_char_length_checked(const std::string &str , size_t start_pos)
{
    return get_utf8_char_length( str.cbegin() + start_pos , str.cend() ) ;
}

inline
size_t UTF8Processing::utf8_char_len(const std::string &utf8_str)
{
    size_t char_len = 0;
    std::string::const_iterator iter = utf8_str.cbegin();
    while( iter < utf8_str.cend() )
    {
        size_t sz = get_utf8_char_length_checked(iter, utf8_str.cend());
        if( sz == 0 ){ return 0; }
        ++char_len;
        iter += sz;
    }
    return char_len;
}

inline
void UTF8Processing::utf8_str2char_seq(const std::string &utf8_str, Seq &utf8_seq)
{
    Seq tmp_word_cont ;
    std::string::const_iterator start_iter = utf8_str.cbegin() ;
    while( start_iter < utf8_str.cend() )
    {
        size_t utf8_char_len = UTF8Processing::get_utf8_char_length_checked(start_iter, utf8_str.cend()) ;
        if( utf8_char_len > 0 )
        {
            tmp_word_cont.emplace_back(start_iter, start_iter + utf8_char_len) ;
            start_iter += utf8_char_len ;
        }
        else
        {
            BOOST_LOG_TRIVIAL(warning) << "illegal utf8 character at position " 
                << start_iter - utf8_str.cbegin() + 1
                << " of words : " << utf8_str ;
            start_iter += 1 ; // skip this position
        }
    }
    std::swap(tmp_word_cont, utf8_seq) ;
}

inline
std::string UTF8Processing::replace_number(const std::string &str,
                                           const std::string number_transform_str,
                                           const size_t length_transform_str )
{
    std::string tmp_str = str;
    size_t start_pos = 0;
    while (start_pos < tmp_str.length())
    {
        size_t end_pos = start_pos;

        while (true)
        {
            size_t byte_len = UTF8Processing::get_number_byte_width(tmp_str, end_pos);
            if (0 == byte_len) break;
            else end_pos += byte_len;
        }
        size_t number_byte_len = end_pos - start_pos;
        if (0 != number_byte_len)
        {
            // replace
            tmp_str.replace(start_pos, number_byte_len, number_transform_str);
            start_pos += length_transform_str;
        }
        else ++start_pos;
    }
    return tmp_str;
}

} // end of namespace slnn
#endif
