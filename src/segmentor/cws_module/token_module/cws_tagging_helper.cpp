#include "cws_tagging_helper.h"
#include "utils/utf8processing.hpp"
#include <codecvt>
using namespace std;
using namespace slnn;

namespace CWSTaggingHelper{
void word2char_tag(const u32string &word, vector<u32string> &char_seq, vector<Index> &tag_seq) noexcept
{
    int word_len = word.size();
    vector<u32string> &tmp_char_seq();
    vector<Index> tmp_tag_seq;

}
}// end of namespace CWSTaggingHelper

void CWSTaggingHelper::parse_word2chars_indextag(const std::string &word, Seq &word_cont, IndexSeq &tag_cont)
{
    Seq tmp_word_cont;
    IndexSeq tmp_tag_cont;
    UTF8Processing::utf8_str2char_seq(word, tmp_word_cont);
    if( tmp_word_cont.size() == 1 )
    {
        tmp_tag_cont.push_back(S_ID) ;
    }
    else if( tmp_word_cont.size() > 1 )
    {
        tmp_tag_cont.push_back(B_ID) ;
        for( size_t i = 1 ; i < tmp_word_cont.size() - 1 ; ++i )
        {
            tmp_tag_cont.push_back(M_ID) ;
        }
        tmp_tag_cont.push_back(E_ID) ;
    }
    std::swap(word_cont, tmp_word_cont) ;
    std::swap(tag_cont, tmp_tag_cont) ;
}
void CWSTaggingHelper::parse_chars_indextag2word_seq(const Seq &char_seq, const IndexSeq &tag_indices, Seq &word_seq)
{
    Seq tmp_word_seq ;
    assert(char_seq.size() == tag_indices.size()) ;
    std::string word ;
    for( size_t i = 0 ; i < char_seq.size() ; ++i )
    {
        word += char_seq[i] ;
        const Index &tag = tag_indices[i] ;
        if( tag == S_ID || tag == E_ID )
        {
            tmp_word_seq.push_back(word) ;
            word = "" ;
        }
    }
    std::swap(word_seq, tmp_word_seq) ;
}
bool CWSTaggingHelper::can_emit(size_t cur_time, Index cur_tag_id)
{
    if( cur_time == 0 ) {return (cur_tag_id == B_ID || cur_tag_id == S_ID) ; } // if first position , only `S` or `B` are valid 
    return true ; // the others is all valid
}
bool CWSTaggingHelper::can_trans(Index pre_tag_id, Index cur_tag_id)
{
    return (
        ((pre_tag_id == B_ID || pre_tag_id == M_ID) 
        && (cur_tag_id == M_ID || cur_tag_id == E_ID)) ||
        ((pre_tag_id == E_ID || pre_tag_id == S_ID) 
        && (cur_tag_id == B_ID || cur_tag_id == S_ID))
        ) ;
}

Index CWSTaggingHelper::select_tag_constrained(std::vector<cnn::real> &dist, size_t time, Index pre_time_tag_id)
{
    cnn::real max_prob = std::numeric_limits<cnn::real>::lowest();
    Index tag_with_max_prob = NONE_ID;
    constexpr Index max_tag_id = get_tag_num() - 1;
    for( Index tag_id = 0; tag_id <= max_tag_id; ++tag_id )
    {
        if( !can_emit(time, tag_id) ){ continue; }
        if( time > 0 && !can_trans(pre_time_tag_id, tag_id) ){ continue; }
        if( dist[tag_id] >= max_prob )
        {
            tag_with_max_prob = tag_id;
            max_prob = dist[tag_id];
        }
    }
    // assert(tag_with_max_prob != NONE_ID);
    return tag_with_max_prob;
}