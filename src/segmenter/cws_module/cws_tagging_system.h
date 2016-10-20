#ifndef SLNN_SEGMENTER_CWS_UTILS_CWS_UTILS_HPP_
#define SLNN_SEGMENTER_CWS_UTILS_CWS_UTILS_HPP_

#include <string>

#include <boost/log/trivial.hpp>
#include <boost/log/core.hpp>

#include "dynet/dict.h"
#include "dynet/dynet.h"
#include "utils/utf8processing.hpp"
#include "utils/typedeclaration.h"

namespace slnn{

struct CWSTaggingSystem
{
    // static member
    static const std::string B_TAG ;
    static const std::string M_TAG ;
    static const std::string E_TAG ;
    static const std::string S_TAG ;
    
    static constexpr Index STATIC_B_ID = 0;
    static constexpr Index STATIC_M_ID = 1;
    static constexpr Index STATIC_E_ID = 2;
    static constexpr Index STATIC_S_ID = 3;
    static constexpr Index STATIC_NONE_ID = -1;

    static constexpr size_t get_tag_num(){ return 4;  }

    static void static_parse_word2chars_indextag(const std::string &word, Seq &word_cont, IndexSeq &tag_cont);
    static void static_parse_chars_indextag2word_seq(const Seq &char_seq, const IndexSeq &static_tag_indices, Seq &word_seq);
    static bool static_can_emit(size_t cur_pos, Index cur_static_tag_id);
    static bool static_can_trans(Index pre_static_tag_id, Index cur_static_tag_id);
    static Index static_select_tag_constrained(std::vector<dynet::real> &dist, size_t time, Index pre_tag_id=STATIC_NONE_ID);


    // below is previous implementation . it is not good ! ( no need to use dynamic tag id for CWS TASK )
    // For compatibility, we still preserve it . But we should never use it any more ;
    static void parse_words2word_tag(const std::string &words, Seq &word_cont, Seq &tag_cont) ;
    static void split_word(const std::string &utf8_str, Seq &utf8_seq) ;
    static void parse_word_tag2words(const Seq &raw_words, const Seq &tags, Seq &words) ;

    // class member
    Index B_ID,
        M_ID,
        E_ID,
        S_ID ;

    void build(dynet::Dict &tag_dict) ;
    bool can_emit(size_t cur_pos , Index cur_tag_id) ;
    bool can_trans(Index pre_tag_id, Index cur_tag_id) ;
    void parse_word_tag2words(const Seq &raw_words, const IndexSeq &tag_ids, Seq &o_words) ; // overide

};

}// end of namespace slnn

#endif 
