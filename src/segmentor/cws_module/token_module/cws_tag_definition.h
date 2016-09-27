#ifndef SLNN_SEGMENTOR_TOKEN_MODULE_TAG_DEFINITION_H_
#define SLNN_SEGMENTOR_TOKEN_MODULE_TAG_DEFINITION_H_
#include "utils/typedeclaration.h"
#include <string>
#include <vector>
namespace slnn{
namespace segmentor{
namespace token_module{

constexpr Index TagBId = 0;
constexpr Index TagMId = 1;
constexpr Index TagEId = 2;
constexpr Index TagSId = 3;
constexpr Index TagNoneId = -1;

constexpr ::std::size_t TagSize = 4U;

inline GenerateTagSeqFromWord(const std::u32string &word, std::vector<Index> &tag_seq4out)
{

}


} // end of segmentor token module
} // end of namespace segmentor
namespace CWSTaggingHelper{
    void word2char_tag(const std::u32string& word, std::vector<std::u32string> &char_seq, std::vector<Index> &tag_seq);

    void char_tag_seq2word_seq(const std::vector<std::u32string> &char_seq, const std::vector<Index> &tag_seq,
        std::vector<std::u32string> &word_seq);
    bool can_emit(int cur_pos, Index cur_tag_id);
    bool can_trans(Index pre_tag_id, Index cur_tag_id);
    Index select_tag_constrained(std::vector<cnn::real> &dist, int time, Index pre_tag_id=NONE_ID);

}; // end of namespace CWSTaggingHelper

} // end of namespace slnn

#endif