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

namespace segmentor{
namespace reader{

SegmentorUnicodeReader::SegmentorUnicodeReader(std::istream &is, charcode::EncodingType f_encoding, predicateT pred_func)
    :Reader(is),
    conv(charcode::CharcodeConvertor::create_convertor(f_encoding)),
    pred_func(pred_func)
{}

bool SegmentorUnicodeReader::read_segmented_line(std::vector<std::u32string> &out_wordseq)
{
    using std::swap;
    std::string line;
    if( !getline(is, line) ){ return false; }
    std::u32string uline = conv->decode(line);
    std::vector<std::u32string> wordseq;
    boost::split(wordseq, uline, pred_func);
    swap(out_wordseq, wordseq);
    return true;
}

bool SegmentorUnicodeReader::readline(std::u32string &out_charseq)
{
    using std::swap;
    std::string line;
    if( !getline(is, line) ){ return false; }
    std::u32string uline = conv->decode(line);
    swap(out_charseq, uline);
    return true;
}

} // end of namespace reader
} // end of namespace segmentor
} // end of namespace slnn

#endif