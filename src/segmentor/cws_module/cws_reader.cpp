#ifndef SLNN_SEGMENTOR_CWS_MODULE_CWS_READER_H_
#define SLNN_SEGMENTOR_CWS_MODULE_CWS_READER_H_
#include "cws_reader.h"
#include "utils/utf8processing.hpp"
namespace slnn{
const char* CWSReader::CWSWordSeperator = " \t";

CWSReader::CWSReader(std::istream &is)
    :Reader(is)
{}

bool CWSReader::read_segmented_line(Seq &word_seq)
{
    using std::swap;
    Seq tmp_sent;
    std::string line;
    if( !getline(is, line) ){ return false;  } ; // static_cast<ifstream> == !fail() && !bad() , not equal to good() , especially on EOF bit 
    std::vector<std::string> tmp_word_seq;
    boost::algorithm::split(tmp_word_seq, line, boost::is_any_of(CWSWordSeperator));
    swap(word_seq, tmp_word_seq);
    return true ;
}

bool CWSReader::readline(Seq &char_seq)
{
    using std::swap;
    Seq tmp_sent;
    std::string line;
    if( !getline(is, line) ){ return false;  } ;
    UTF8Processing::utf8_str2char_seq(line, char_seq);
    return true;
}

} // end of namespace slnn

#endif
