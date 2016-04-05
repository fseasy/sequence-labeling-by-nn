#pragma once
#ifndef UTF8_PROCESSING_INCLUDED
#define UTF8_PROCESSING_INCLUDED
#include <string>
#include <vector>
using namespace std;

struct UTF8Processing
{
    using Range = pair<string::iterator, string::iterator>;
    inline static string::difference_type get_number_byte_width(string::iterator &start_pos, string::iterator end_pos);
    inline static size_t get_number_byte_width(const string &str , size_t start_pos) ;



};


/******
 * get_number_byte_width
 * return string::difference_type
 * if character from this start_pos is not number , return 0
 * else , if UTF8 number , return 3
 * else , Ascii number , return 1
 */
string::difference_type UTF8Processing::get_number_byte_width(string::iterator &pos, string::iterator end_pos)
{
    if (pos == end_pos) return 0u ;
    if ((*pos & 0xff) == 0xef &&
        pos + 1 != end_pos && (*(pos + 1) & 0xff) == 0xbc &&
        pos + 2 != end_pos && (*(pos + 2) & 0xff) >= 0x90 && (*(pos + 2)) <= 0x99)
    {
        // UTF8 number
        return 3u ;
    }
    else if ((*pos & 0xff) <= 0x39 && (*pos & 0xff) >= 0x30)
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

size_t UTF8Processing::get_number_byte_width(const string &str , size_t start_pos)
{
    size_t len = str.length() ;
    if(start_pos >= len) return 0u ;
    if( (str[start_pos] & 0xff) == 0xef &&
         start_pos + 1 < len && (str[start_pos+1] & 0xff) == 0xbc &&
         start_pos + 2 < len && (str[start_pos+2] & 0xff) >= 0x90 && (str[start_pos+2] & 0xff) <= 0x99 ) 
    {
        return 3u ;
    }
    else if( (str[start_pos] & 0xff ) <= 0x39 && (str[start_pos] & 0xff ) >= 0x30 )
    {
        return 1u ;
    }
    else return 0u ;
}



#endif
