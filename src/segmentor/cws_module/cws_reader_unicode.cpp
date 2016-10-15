#include "cws_reader_unicode.h"
namespace slnn{
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
