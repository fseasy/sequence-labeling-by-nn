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

TEST_CASE("charcode", "[UTF8UNICODE]")
{
    char32_t unicode_normal_cn = 0x4E70, // <==> UTF8 char 'Mai' (buy in Englist) 
        unicode_extend_cn = 0x20005, //   U+20005, see http://www.utf8-chartable.de/unicode-utf8-table.pl
        unicode_ascii = 'b';
    string u8_normal_cn = u8"\x4E70",
        u8_extend_cn = u8"\x20005",
        u8_ascii = u8"b";

    cout << hex << unicode_normal_cn << " " << unicode_extend_cn << " " << unicode_ascii << endl;

    u32string unicode_str_cn = U"\x4E70\x20005", // the two continues Chinese Characters
        unicode_str_mix = U"\x4E70\x62\x20005";     // the Chinese - English - Chinese Characters
    string u8_str_cn = u8"\x4E70\x20005",
        u8_str_mix = u8"\x4E70\x62\x20005";
    
    auto printHex = [](const string &u8str)
    {
        for( unsigned char c : u8str ){ cout << hex << (c & 0xff) <<" "; }
        cout << endl;
    };

    cerr << u8_normal_cn.size() << endl; printHex(u8_normal_cn);
    cout << u8_extend_cn.size() << endl; printHex(u8_extend_cn) ;

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