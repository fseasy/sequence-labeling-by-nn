#include "ner_reader.h"

namespace slnn{
namespace ner{
namespace reader{

NerUnicodeReader::NerUnicodeReader(std::istream &is, 
    charcode::EncodingType f_encoding)
    :Reader(is),
    conv(charcode::CharcodeConvertor::create_convertor(f_encoding))
{}

bool NerUnicodeReader::readline(std::u32string &out_line)
{
    using std::swap;
    std::string line;
    if( !std::getline(is, line) ){ return false; }
    std::u32string u_line = conv->decode(line);
    swap(u_line, out_line);
    return true;
}

} // end of namespace reader
} // end of namespace ner
} // end of namespace slnn