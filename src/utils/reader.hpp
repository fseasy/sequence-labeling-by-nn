#ifndef UTILS_READER_HPP_
#define UTILS_READER_HPP_

#include <iostream>
#include <algorithm>
namespace slnn{

struct Reader
{
    Reader(std::istream &is);
    bool good();
    size_t count_line();

    std::istream &is;
};

inline
Reader::Reader(std::istream &is)
    :is(is)
{}

inline
bool Reader::good()
{
    return is.good();
}

inline
size_t Reader::count_line()
{
    // skip when bad
    if( is.bad() ) return 0; 
    // save state
    std::istream::iostate state_backup = is.rdstate();
    // clear state
    is.clear();
    std::istream::streampos pos_backup = is.tellg();

    is.seekg(0);
    size_t line_cnt;
    size_t lf_cnt = std::count(std::istreambuf_iterator<char>(is), std::istreambuf_iterator<char>(), '\n');
    line_cnt = lf_cnt;
    // if the file is not end with '\n' , then line_cnt should plus 1  
    is.unget();
    if( is.get() != '\n' ) { ++line_cnt ; }

    // recover state
    is.clear() ; // previous reading may set eofbit
    is.seekg(pos_backup);
    is.setstate(state_backup);

    return line_cnt;
}

} // end of namespace slnn
#endif