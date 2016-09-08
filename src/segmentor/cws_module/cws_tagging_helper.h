#ifndef SLNN_SEGMENTOR_CWS_TAGGING_HELPER_H_
#define SLNN_SEGMENTOR_CWS_TAGGING_HELPER_H_
#include "utils/typedeclaration.h"
namespace slnn{

namespace CWSTaggingHelper{

    constexpr Index B_ID = 0;
    constexpr Index M_ID = 1;
    constexpr Index E_ID = 2;
    constexpr Index S_ID = 3;
    constexpr Index NONE_ID = -1;

    constexpr std::size_t get_tag_num(){ return 4U;  }

    void word2char_tag(const std::u32string& word, std::vector<std::u32string> &char_seq, std::vector<Index> &tag_seq);

    void char_tag_seq2word_seq(const std::vector<std::u32string> &char_seq, const std::vector<Index> &tag_seq,
        std::vector<std::u32string> &word_seq);
    bool can_emit(int cur_pos, Index cur_tag_id);
    bool can_trans(Index pre_tag_id, Index cur_tag_id);
    Index select_tag_constrained(std::vector<cnn::real> &dist, int time, Index pre_tag_id=NONE_ID);

}; // end of namespace CWSTaggingHelper

} // end of namespace slnn

#endif