#ifndef SLNN_SEGMENTER_CWS_MODULE_CWS_WRITER_H_
#define SLNN_SEGMENTER_CWS_MODULE_CWS_WRITER_H_
#include <string>
#include <vector>
#include <ostream>
#include <sstream>
#include <memory>
#include "utils/writer.h"
#include "token_module/cws_tag_definition.h"
#include "trivial/charcode/charcode_base.hpp"
#include "trivial/charcode/charcode_convertor.h"
namespace slnn{
namespace segmenter{
namespace writer{

class SegmentorWriter : private utils::Writer
{
public:
    // constructor
    SegmentorWriter(std::ostream &os, charcode::EncodingType encoding_type, const std::u32string &uni_delimiter=U"\t");
    using Writer::writeline;
    void write(const std::u32string &uni_str, const std::vector<Index> &tagseq);
private:
    std::shared_ptr<charcode::CharcodeConvertor> conv;
    std::string out_delimiter;
};

/*****************************************
 * Inline Implementation
 *****************************************/

inline 
SegmentorWriter::SegmentorWriter(std::ostream &os, charcode::EncodingType encoding_type, const std::u32string &uni_delimiter)
    :utils::Writer(os),
    conv(charcode::CharcodeConvertor::create_convertor(encoding_type)),
    out_delimiter(conv->encode(uni_delimiter))
{}

inline
void SegmentorWriter::write(const std::u32string &charseq, const std::vector<Index> &tagseq)
{
    if( charseq.size() == 0 ){ writeline(""); return; }
    std::vector<std::u32string> wordseq = token_module::generate_wordseq_from_chartagseq(charseq, tagseq);
    std::ostringstream oss;
    oss << conv->encode(wordseq[0]);
    for( std::size_t i = 1; i < wordseq.size(); ++i )
    {
        oss << out_delimiter << conv->encode(wordseq[i]);
    }
    writeline(oss.str());
}


} // end of namespace 
} // enf of namespace segmenter
} // end of namespace slnn



#endif
