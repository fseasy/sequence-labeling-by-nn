#include "cws_tagging_system.h"

namespace slnn{

const std::string CWSTaggingSystem::B_TAG = "B" ;
const std::string CWSTaggingSystem::M_TAG = "M" ;
const std::string CWSTaggingSystem::E_TAG = "E" ;
const std::string CWSTaggingSystem::S_TAG = "S" ;

const Index CWSTaggingSystem::STATIC_B_ID;
const Index CWSTaggingSystem::STATIC_M_ID;
const Index CWSTaggingSystem::STATIC_E_ID;
const Index CWSTaggingSystem::STATIC_S_ID;

void CWSTaggingSystem::static_parse_word2chars_indextag(const std::string &word, Seq &word_cont, IndexSeq &tag_cont)
{
    Seq tmp_word_cont;
    IndexSeq tmp_tag_cont;
    UTF8Processing::utf8_str2char_seq(word, tmp_word_cont);
    if( tmp_word_cont.size() == 1 )
    {
        tmp_tag_cont.push_back(STATIC_S_ID) ;
    }
    else if( tmp_word_cont.size() > 1 )
    {
        tmp_tag_cont.push_back(STATIC_B_ID) ;
        for( size_t i = 1 ; i < tmp_word_cont.size() - 1 ; ++i )
        {
            tmp_tag_cont.push_back(STATIC_M_ID) ;
        }
        tmp_tag_cont.push_back(STATIC_E_ID) ;
    }
    std::swap(word_cont, tmp_word_cont) ;
    std::swap(tag_cont, tmp_tag_cont) ;
}
void CWSTaggingSystem::static_parse_chars_indextag2word_seq(const Seq &char_seq, const IndexSeq &static_tag_indices, Seq &word_seq)
{
    Seq tmp_word_seq ;
    assert(char_seq.size() == static_tag_indices.size()) ;
    std::string word ;
    for( size_t i = 0 ; i < char_seq.size() ; ++i )
    {
        word += char_seq[i] ;
        const Index &tag = static_tag_indices[i] ;
        if( tag == STATIC_S_ID || tag == STATIC_E_ID )
        {
            tmp_word_seq.push_back(word) ;
            word = "" ;
        }
    }
    std::swap(word_seq, tmp_word_seq) ;
}
bool CWSTaggingSystem::static_can_emit(size_t cur_pos, Index cur_static_tag_id)
{
    if( cur_pos == 0 ) {return (cur_static_tag_id == STATIC_B_ID || cur_static_tag_id == STATIC_S_ID) ; } // if first position , only `S` or `B` are valid 
    return true ; // the others is all valid
}
bool CWSTaggingSystem::static_can_trans(Index pre_static_tag_id, Index cur_static_tag_id)
{
    return (
        ((pre_static_tag_id == STATIC_B_ID || pre_static_tag_id == STATIC_M_ID) 
        && (cur_static_tag_id == STATIC_M_ID || cur_static_tag_id == STATIC_E_ID)) ||
        ((pre_static_tag_id == STATIC_E_ID || pre_static_tag_id == STATIC_S_ID) 
        && (cur_static_tag_id == STATIC_B_ID || cur_static_tag_id == STATIC_S_ID))
        ) ;
}


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

bool CWSTaggingSystem::can_trans(Index pre_tag_id, Index cur_tag_id)
{
    return (((pre_tag_id == B_ID || pre_tag_id == M_ID) && (cur_tag_id == M_ID || cur_tag_id == E_ID)) ||
        ((pre_tag_id == E_ID || pre_tag_id == S_ID) && (cur_tag_id == B_ID || cur_tag_id == S_ID))) ;
}

}// end of namespace slnn
