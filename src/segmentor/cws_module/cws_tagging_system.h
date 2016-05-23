#ifndef SLNN_SEGMENTOR_CWS_UTILS_CWS_UTILS_HPP_
#define SLNN_SEGMENTOR_CWS_UTILS_CWS_UTILS_HPP_

#include <string>

#include <boost/log/trivial.hpp>
#include <boost/log/core.hpp>

#include "cnn/dict.h"

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
    static void parse_words2word_tag(const std::string &words, Seq &word_cont, Seq &tag_cont) ;
    static void split_word(const std::string &utf8_str, Seq &utf8_seq) ;
    static void parse_word_tag2words(const Seq &raw_words, const Seq &tags, Seq &words) ;

    // class member
    Index B_ID,
        M_ID,
        E_ID,
        S_ID ;

    void build(cnn::Dict &tag_dict) ;
    bool can_emit(size_t cur_pos , Index cur_tag_id) ;
    bool can_trans(Index pre_tag_id, Index cur_tag_id) ;
    void parse_word_tag2words(const Seq &raw_words, const IndexSeq &tag_ids, Seq &o_words) ; // overide

};

}// end of namespace slnn

#endif 
