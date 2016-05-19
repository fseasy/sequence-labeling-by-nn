#ifndef SLNN_SEGMENTOR_CWS_UTILS_CWS_UTILS_HPP_
#define SLNN_SEGMENTOR_CWS_UTILS_CWS_UTILS_HPP_

#include <string>

#include <boost/log/trivial.hpp>
#include <boost/log/core.hpp>

#include "utils/utf8processing.hpp"
#include "utils/typedeclaration.h"

namespace slnn{

struct CWSUtils
{
    static const std::string B_TAG ;
    static const std::string M_TAG ;
    static const std::string E_TAG ;
    static const std::string S_TAG ;
    static void parse_words2word_tag(const std::string &words, Seq &word_cont, Seq &tag_cont) ;
    static void split_word(const std::string &utf8_str, Seq &utf8_seq) ;
    static void parse_word_tag2words(const Seq &raw_words, const Seq &tags, Seq &words) ;
};

}// end of namespace slnn

#endif 
