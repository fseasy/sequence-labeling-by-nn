#ifndef SLNN_UTILS_WRITER_H_
#define SLNN_UTILS_WRITER_H_
#include <iostream>
#include <string>
namespace slnn{
namespace utils{
/**
 * writer.
 * not thread safe.
 */
struct Writer
{
    // constructor
    Writer(std::ostream &os) :os(os){}
    // interface
    void writeline(const std::string &line){ os << line << std::endl; }
    // data
    std::ostream os;
};

} // end of namespace utils
} // end of namespace slnn

#endif