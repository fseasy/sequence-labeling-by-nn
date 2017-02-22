#ifndef SLNN_NER_MODULE_READER_H_
#define SLNN_NER_MODULE_READER_H_
#include <functional>
#include <memory>
#include <vector>
#include "utils/reader.hpp"
#include "trivial/charcode/charcode_base.hpp"
#include "trivial/charcode/charcode_convertor.h"
namespace slnn{
namespace ner{
namespace reader{


class NerUnicodeReader : public Reader
{

public:
    NerUnicodeReader(std::istream &is, 
        charcode::EncodingType file_encoding=charcode::EncodingType::UTF8);
    bool readline(std::u32string &out_charseq);
private:
    std::shared_ptr<charcode::CharcodeConvertor> conv;
};

} // end of namespace reader
} // end of namespace segmenter
} // end of namespace slnn
#endif
