#define CATCH_CONFIG_MAIN
#include "trivial/charcode/naive_unicode.h"
#include "../3rdparty/catch/include/catch.hpp"
#include <iostream>
using namespace std;
using namespace slnn::charcode::NUnicode;

/*
    For compatibility, we never write hard Chinese Character. 
    Using Pinyin + English to represent it.
*/

TEST_CASE("charcode-1", "[UTF8-UNICODE]-SPECIFIC")
{
    char32_t unicode_normal_cn = 0x4E70, // <==> UTF8 char 'Mai' (buy in English) 
        unicode_extend_cn = 0x20005, //   U+20005, see http://www.utf8-chartable.de/unicode-utf8-table.pl
        unicode_ascii = 'b';
    string u8_normal_cn = u8"\u4E70",
        u8_extend_cn = u8"\u20005",
        u8_ascii = u8"b";

    cout << hex << unicode_normal_cn << " " << unicode_extend_cn << " " << unicode_ascii << endl;

    u32string unicode_str_cn = U"\u4E70\u20005", // the two continues Chinese Characters
        unicode_str_mix = U"\u4E70\u0062\u20005";     // the Chinese - English - Chinese Characters
    string u8_str_cn = u8"\u4E70\u20005",
        u8_str_mix = u8"\u4E70\u0062\u20005";
    

    // char level, UTF8 <==> UNICODE
    REQUIRE(unicode_from_u8_unsafe(u8_normal_cn, 0, u8_normal_cn.size()) == unicode_normal_cn);
    REQUIRE(unicode_from_u8_unsafe(u8_extend_cn, 0, u8_extend_cn.size()) == unicode_extend_cn);
    REQUIRE(unicode_from_u8_unsafe(u8_ascii, 0, u8_ascii.size()) == unicode_ascii);

    REQUIRE(unicode2u8_unsafe(unicode_normal_cn) == u8_normal_cn);
    REQUIRE(unicode2u8_unsafe(unicode_extend_cn) == u8_extend_cn);
    REQUIRE(unicode2u8_unsafe(unicode_ascii) == u8_ascii);

    // sentence level, UTF8 <==> UNICODE
    REQUIRE(decode_from_u8_bytes_unsafe(u8_str_cn) == unicode_str_cn);
    REQUIRE(decode_from_u8_bytes_unsafe(u8_str_mix) == unicode_str_mix);

    REQUIRE(encode2u8_bytes_unsafe(unicode_str_cn) == u8_str_cn);
    REQUIRE(encode2u8_bytes_unsafe(unicode_str_mix) == u8_str_mix);
}

TEST_CASE("charcode-2", "[UTF8-UNICODE]-RANGE_RANDOM")
{
    /* Randomized Unicode Generating Script (Python)
    import random
    if __name__ == "__main__" :
        max_ordinal = 0x10FFFF
        step_size_min = 0xFF
        step_size_max = 0x8FF
        factor = 1.02
        cnt = 0
        random.seed(1234)
        next_ord = random.randint(0, 0x7F)
        while next_ord < max_ordinal :
            step_size = random.randint(step_size_min, step_size_max)
            try :
                unichr(next_ord)
            except ValueError, e:
                next_ord += step_size
                continue
            next_ord_str = "%x" %(next_ord)
            print "0x%s," %(next_ord_str.upper())
            next_ord += step_size
            next_ord = int(next_ord * factor ** cnt) # exponential increasing
            cnt += 1    
    */
    char32_t unicode_range_list[] = 
    {
        0x7B,
        0x501,
        0x62E,
        0xF0C,
        0x1900,
        0x212D,
        0x2BA9,
        0x330A,
        0x42D1,
        0x51AC,
        0x6317,
        0x81B2,
        0xA5F1,
        0xC80B,
        0x121B8,
        0x18128,
        0x209AE,
        0x2CEC6,
        0x3F002,
        0x5A6AB,
        0x84839,
        0xC50C9
    };
    string u8_range_list[] = 
    {
        u8"\u007B",
        u8"\u0501",
        u8"\u062E",
        u8"\u0F0C",
        u8"\u1900",
        u8"\u212D",
        u8"\u2BA9",
        u8"\u330A",
        u8"\u42D1",
        u8"\u51AC",
        u8"\u6317",
        u8"\u81B2",
        u8"\uA5F1",
        u8"\uC80B",
        u8"\u121B8",
        u8"\u18128",
        u8"\u209AE",
        u8"\u2CEC6",
        u8"\u3F002",
        u8"\u5A6AB",
        u8"\u84839",
        u8"\uC50C9"
    };
    // char level
    bool check_result = true;
    int unicode_char_cnt = end(unicode_range_list) - begin(unicode_range_list);
    for( int i = 0; i < unicode_char_cnt; ++i )
    {
        bool u82unicode_check = (unicode_from_u8_unsafe(u8_range_list[i], 0, u8_range_list[i].size()) 
                                 == unicode_range_list[i]);
        bool unicode2u8_check = (unicode2u8_unsafe(unicode_range_list[i])
                                 == u8_range_list[i]);
        check_result &= u82unicode_check & unicode2u8_check;
    }
    REQUIRE(check_result == true);

    // sentence level
    u32string unicode_str(unicode_range_list, unicode_char_cnt);
    string u8_str;
    for( const string u8_char_bytes : u8_range_list ){ u8_str.append(u8_char_bytes); }
    check_result = true;
    bool u82unicode_check = encode2u8_bytes_unsafe(unicode_str) == u8_str;
    bool unicode2u8_check = decode_from_u8_bytes_unsafe(u8_str) == unicode_str;
    check_result &= u82unicode_check & unicode2u8_check;
    REQUIRE(check_result);
}
