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

const std::string CWSUtils::B_TAG = "B" ;
const std::string CWSUtils::M_TAG = "M" ;
const std::string CWSUtils::E_TAG = "E" ;
const std::string CWSUtils::S_TAG = "S" ;

void CWSUtils::split_word(const std::string &utf8_str, Seq &utf8_seq)
{
    Seq tmp_word_cont ;
    std::string::const_iterator start_iter = utf8_str.cbegin() ;
    while( start_iter < utf8_str.cend() )
    {
        size_t utf8_char_len = UTF8Processing::get_utf8_char_length_checked(start_iter, utf8_str.cend()) ;
        if( utf8_char_len > 0 )
        {
            tmp_word_cont.emplace_back(start_iter, start_iter + utf8_char_len) ;
            start_iter += utf8_char_len ;
        }
        else
        {
            BOOST_LOG_TRIVIAL(warning) << "illegal utf8 character at position " 
                << start_iter - utf8_str.cbegin() + 1
                << " of words : " << utf8_str ;
            start_iter += 1 ; // skip this position
        }
    }
    std::swap(tmp_word_cont, utf8_seq) ;
}

void CWSUtils::parse_words2word_tag(const std::string &words, Seq &word_cont, Seq &tag_cont)
{
    Seq tmp_word_cont,
        tmp_tag_cont ;
    split_word(words, tmp_word_cont) ;
    if( tmp_word_cont.size() == 1 )
    {
        tmp_tag_cont.push_back(S_TAG) ;
    }
    else if( tmp_word_cont.size() > 1 )
    {
        tmp_tag_cont.push_back(B_TAG) ;
        for( size_t i = 1 ; i < tmp_word_cont.size() - 1 ; ++i )
        {
            tmp_tag_cont.push_back(M_TAG) ;
        }
        tmp_tag_cont.push_back(E_TAG) ;
    }
    std::swap(word_cont, tmp_word_cont) ;
    std::swap(tag_cont, tmp_tag_cont) ;
}

void CWSUtils::parse_word_tag2words(const Seq &raw_words, const Seq &tags, Seq &words)
{
    Seq tmp_words ;
    assert(raw_words.size() == tags.size()) ;
    std::string word ;
    for( size_t i = 0 ; i < raw_words.size() ; ++i )
    {
        word.append(raw_words[i]) ;
        const std::string &tag = tags[i] ;
        if( tag == S_TAG || tag == E_TAG )
        {
            tmp_words.push_back(word) ;
            word = "" ;
        }
    }
    std::swap(words, tmp_words) ;
}

}// end of namespace slnn

#endif 