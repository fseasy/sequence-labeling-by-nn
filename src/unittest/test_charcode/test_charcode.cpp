#define CACHE_CONFIG_MAIN
#include "trivial/charcode/naive_unicode.h"
#include "../3rdparty/catch/include/catch.hpp"

using namespace std;
using namespace slnn::charcode::NUnicode;

/*
    For compatibility, we never write hard Chinese Character. 
    Using Pinyin + English to represent it.
*/

TEST_CASE("charcode", "[UTF8<->UNICODE]")
{
    char32_t unicode_normal_cn = U'\xE4B9B0', // <==> UTF8 char 'Mai' (buy in Englist) 
        unicode_extend_cn = U'\xF0A08085', //   U+20005, see http://www.utf8-chartable.de/unicode-utf8-table.pl
        unicode_ascii = U'b';
    string u8_normal_cn = u8"\xE4\xB9\xB0",
        u8_extend_cn = u8"\xF0\xA0\x80\x85",
        u8_ascii = u8"b";
}