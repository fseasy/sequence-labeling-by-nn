#include "cws_tagging_system.h"

namespace slnn{

const std::string CWSTaggingSystem::B_TAG = "B" ;
const std::string CWSTaggingSystem::M_TAG = "M" ;
const std::string CWSTaggingSystem::E_TAG = "E" ;
const std::string CWSTaggingSystem::S_TAG = "S" ;

void CWSTaggingSystem::split_word(const std::string &utf8_str, Seq &utf8_seq)
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

void CWSTaggingSystem::parse_words2word_tag(const std::string &words, Seq &word_cont, Seq &tag_cont)
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

void CWSTaggingSystem::parse_word_tag2words(const Seq &raw_words, const Seq &tags, Seq &words)
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

void CWSTaggingSystem::build(cnn::Dict &tag_dict)
{
    B_ID = tag_dict.Convert(B_TAG) ;
    M_ID = tag_dict.Convert(M_TAG) ;
    E_ID = tag_dict.Convert(E_TAG) ;
    S_ID = tag_dict.Convert(S_TAG) ;
}

bool CWSTaggingSystem::can_emit(size_t cur_pos, Index cur_tag_id )
{
    if( cur_pos == 0 ) return cur_tag_id == B_ID || cur_tag_id == S_ID ; // if first position , only `S` or `B` are valid 
    return true ; // the others is all valid
}

bool CWSTaggingSystem::can_trans(size_t pre_tag_id, size_t cur_tag_id)
{
    return (((pre_tag_id == B_ID || pre_tag_id == M_ID) && (cur_tag_id == M_ID || cur_tag_id == E_ID)) ||
        ((pre_tag_id == M_ID || pre_tag_id == S_ID) && (cur_tag_id == B_ID || cur_tag_id == S_ID))) ;
}

void CWSTaggingSystem::parse_word_tag2words(const Seq &raw_words, const IndexSeq &tag_ids, Seq &o_words)
{
    Seq tmp_words ;
    assert(raw_words.size() == tag_ids.size()) ;
    std::string word ;
    for( size_t i = 0 ; i < raw_words.size() ; ++i )
    {
        word.append(raw_words[i]) ;
        Index tag = tag_ids[i] ;
        if( tag == S_ID || tag == E_ID )
        {
            tmp_words.push_back(word) ;
            word = "" ;
        }
    }
    std::swap(o_words, tmp_words) ;
}

}// end of namespace slnn
