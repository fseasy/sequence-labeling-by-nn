#ifndef SEGMENTOR_CWS_MODULE_CWS_READER_H_
#define SEGMENTOR_CWS_MODULE_CWS_READER_H_

#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/classification.hpp>
#include "utils/reader.hpp"
#include "utils/typedeclaration.h"
namespace slnn{

class CWSReader : public Reader
{
public:
    static const char *CWSWordSeperator;

public:
    CWSReader(std::istream &is);
    bool read_segmented_line(Seq &word_seq);
    bool readline(Seq &char_seq);
};

} // end of namespace slnn
#endif