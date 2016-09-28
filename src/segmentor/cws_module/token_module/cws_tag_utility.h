#ifndef SLNN_SEGMENTOR_CWS_MODULE_TOKEN_MODULE_TAG_UTILITY_H_
#define SLNN_SEGMENTOR_CWS_MODULE_TOKEN_MODULE_TAG_UTILITY_H_
#include "cws_tag_definition.h"

namespace slnn{
namespace segmentor{
namespace token_module{

inline 
std::vector<std::u32string> generate_wordseq_from_chartagseq(const std::u32string &charseq,
    const std::vector<Index> &tagseq) noexcept
{
    assert(charseq.size() == tagseq.size());
    std::vector<std::u32string> wordseq;
    std::u32string word = U"";
    for(size_t i =  0U; i <)
}




}
}
} // end of namespace slnn

#endif