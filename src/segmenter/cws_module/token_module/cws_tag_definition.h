/**
 * tag_definition.h, tag system definition, including tag id, tag generation rule, tag relation constrains.
 * tag id : B, M, E, S
 * tag generation rule: word => (char, tag); 
 *                      word_seq => (char_seq, tag_seq);
 *                      (char_seq, tag_seq) => word_seq.
 * tag relation constrains : can_emit?, can_trans?
 */

#ifndef SLNN_SEGMENTER_TOKEN_MODULE_TAG_DEFINITION_H_
#define SLNN_SEGMENTER_TOKEN_MODULE_TAG_DEFINITION_H_
#include "utils/typedeclaration.h"
#include <string>
#include <vector>
namespace slnn{
namespace segmenter{
namespace token_module{

/**
 * enum for Tag in Chinese Word Segmentation.
 */
enum Tag : Index
{
    TAG_B_ID = 0,    /**< tag b index value. */
    TAG_M_ID = 1,    /**< tag m index value. */
    TAG_E_ID = 2,    /**< tag e index value. */
    TAG_S_ID = 3,    /**< tag s index value. */
    TAG_NONE_ID = -1 /**< tag for placeholder or invalid representation. */
};

/**
 * total tag number.
 */
constexpr std::size_t TAG_SIZE = 4U;

/**
 * pre-built transition table for quikly get can-trans result.
 * WARNING : if B, M, E, S value has changed, the table should be update!
 */

constexpr bool TRANS_TABLE[TAG_SIZE][TAG_SIZE] = {
                                    // | row tag( previous tag)
      {false, true , true , false}, // B
      {false, true , true , false}, // M
      {true , false, false, true }, // E
      {true , false, false, true }  // S
      // B      M      E      S   -> col tag (current tag)
};

/**
 * from word to generate the corresponding tag sequence.
 * according to the CWS tag definition, tag is the representation of char-position.
 * B(begin), M(middle), E(end), S(single)
 * @param word unicode string
 * @return generated tag sequence
 */
std::vector<Index> generate_tagseq_from_word(const std::u32string &word) noexcept;

/**
 * from word sequence to generate the tag sequence, ant put it to the pre-allocated memeory.
 * for efficiency, using pre-allocated memory instead of dynamic-inceasing.
 * @param word_seq list of word
 * @param out_preallocated_tagseq pre-allocated container for output tag sequence
 */
void generate_tagseq_from_wordseq2preallocated_space(const std::vector<std::u32string> &word_seq,
    std::vector<Index> &out_preallocated_tagseq) noexcept;


/**
 * from tagseq to word range list.
 * @param tagseq tag sequence
 * @return word_range_list, word range is a pair <unsigned, unsigned> indicating <start, end> position of a word.
 */
std::vector<std::pair<unsigned, unsigned>> 
tagseq2word_range_list(const std::vector<Index>& tagseq) noexcept;


/**
 * from not valid tagseq to word range list.
 * @param not_valid_tagseq not valid tag sequence
 * @return word_range_list, see @tagseq2word_range_list
 */
std::vector<std::pair<unsigned, unsigned>> 
not_valid_tagseq2word_range_list(const std::vector<Index>& not_valid_tagseq) noexcept;


/**
* from char and tag sequence to genrate word sequence
* @param charseq character sequence, unicode string
* @param tagseq  tag sequence, Index sequence
* @return word sequence, unicode string sequence
*/
std::vector<std::u32string> generate_wordseq_from_chartagseq(const std::u32string &charseq,
    const std::vector<Index> &tagseq) noexcept;

/**
 * from char and not valid tag sequence to word sequence.
 * the different from `generate_wordseq_from_chartagseq` is that the input tag sequence may be not a valid 
 * sequence meet the constraint of the segmneter tag definition.
 * @param charseq character sequence, unicode string
 * @param not_valid_tagseq tag sequence with may be not valid
 * @return wordseq unicode string sequence.
 ***********/
std::vector<std::u32string> generate_wordseq_from_not_valid_chartagseq(const std::u32string &charseq,
    const std::vector<Index> &not_valid_tagseq) noexcept;

/**
 * whethere can emit for the current tag id at the current time.
 * Only time 0 can be limited(B/S). 
 * in fact, the last time should also be limited, but we can't know here where is the end. so we have to
 * make decision according to the specific sequence
 * @param cur_time current timestamp
 * @param cur_tag_id current tag id
 * @return whethere can emit
 */
bool can_emit(std::size_t cur_time, Index cur_tag_id) noexcept;

/**
 * whethere can transfer from previous tag to current tag(unsafe version - using lookup table).
 * according to TRANS_TABLE, don't varify the id, so it is unsafe(may overflow).
 * but it is ok under the tag system difinition.
 * @param pre_tag_id previous tag id. the first place can't have the calling.
 * @param cur_tag_id current tag id.
 * @return whthere can transfer
 */
bool can_trans_unsafe(Index pre_tag_id, Index cur_tag_id) noexcept;

/**
 * whethere can transfer from previous tag to current tag(safe version - using condition judgment).
 * @param pre_tag_id
 * @param cur_tag_id
 * @return whethere can transfer
 */
bool can_trans(Index pre_tag_id, Index cur_tag_id) noexcept;

/***********************************************************
 * Inline Implementation
 ***********************************************************/

inline 
std::vector<Index> generate_tagseq_from_word(const std::u32string &word) noexcept
{
    std::size_t word_len = word.length();
    std::vector<Index> tagseq_tmp(word_len);
    if( word_len == 1U ) { tagseq_tmp[0] = TAG_S_ID; }
    else if(word_len > 1U )
    {
        tagseq_tmp[0] = TAG_B_ID;
        for( std::size_t i = 1U; i < word_len - 1U; ++i ){ tagseq_tmp[i] = TAG_M_ID; }
        tagseq_tmp[word_len - 1] = TAG_E_ID;
    }
    return tagseq_tmp;
}

inline 
void generate_tagseq_from_wordseq2preallocated_space(const std::vector<std::u32string> &word_seq,
    std::vector<Index> &out_preallocated_tagseq) noexcept
{
    int idx = 0;
    for( const auto &word : word_seq )
    {
        std::size_t word_len = word.length();
        if( word_len == 1U ){ out_preallocated_tagseq[idx++] = TAG_S_ID; }
        else if( word_len > 1U )
        {
            out_preallocated_tagseq[idx++] = TAG_B_ID;
            for( std::size_t i = 1U; i < word_len - 1U; ++i ){ out_preallocated_tagseq[idx++] = TAG_M_ID; }
            out_preallocated_tagseq[idx++] = TAG_E_ID;
        }
    }
}

inline
std::vector<std::pair<unsigned, unsigned>> 
tagseq2word_range_list(const std::vector<Index>& tagseq) noexcept
{
    std::vector<std::pair<unsigned, unsigned>> tmp_word_ranges ;
    unsigned range_s = 0 ;
    for( unsigned i = 0 ; i < tagseq.size() ; ++i )
    {
        Index tag_id = tagseq.at(i) ;
        if( tag_id == Tag::TAG_E_ID || tag_id == Tag::TAG_S_ID)
        {
            tmp_word_ranges.push_back({ range_s , i }) ;
            range_s = i + 1 ;
        }
    }
    return tmp_word_ranges;
}

inline 
std::vector<std::pair<unsigned, unsigned>> 
not_valid_tagseq2word_range_list(const std::vector<Index>& not_valid_tagseq) noexcept
{
    std::vector<std::pair<unsigned, unsigned>> word_range_list;
    unsigned range_spos = 0U;
    for( unsigned range_epos = 0U; range_epos < not_valid_tagseq.size(); ++range_epos )
    {
        Index cur_tag = not_valid_tagseq[range_epos];
        if( cur_tag == Tag::TAG_B_ID )
        {
            // if has not processed segmentation(not valid)
            if( range_epos > range_spos )
            {
                word_range_list.push_back(std::make_pair(range_spos, range_epos - 1));
                range_spos = range_epos;
            }
        }
        else if( cur_tag == Tag::TAG_S_ID )
        {
            // whether has not processed segmentation(not valid)
            if( range_epos > range_spos )
            {
                word_range_list.push_back(std::make_pair(range_spos, range_epos - 1));
            }
            // add current pos as the word
            word_range_list.push_back(std::make_pair(range_epos, range_epos));
            range_spos = range_epos + 1; // set the next start pos.
        }
        else if( cur_tag == Tag::TAG_E_ID )
        {
            // including current char
            word_range_list.push_back(std::make_pair(range_spos, range_epos));
            range_spos = range_epos + 1;
        }
    }
    // if has the left(not valid)
    if( range_spos < not_valid_tagseq.size() )
    {
        word_range_list.push_back(std::make_pair(range_spos, static_cast<unsigned>(not_valid_tagseq.size()) - 1));
    }
    return word_range_list;
}

inline 
std::vector<std::u32string> generate_wordseq_from_chartagseq(const std::u32string &charseq,
    const std::vector<Index> &tagseq) noexcept
{
    assert(charseq.size() == tagseq.size());
    std::vector<std::u32string> wordseq;
    std::size_t slice_spos = 0;
    for( std::size_t i = 0U; i < charseq.size(); ++i )
    {
        Index tag = tagseq[i];
        if( tag == Tag::TAG_E_ID || tag == Tag::TAG_S_ID )
        {
            std::size_t word_len = i - slice_spos + 1;
            wordseq.push_back(charseq.substr(slice_spos, word_len));
            slice_spos = i + 1;
        }
    }
    return wordseq;
}

inline
std::vector<std::u32string> generate_wordseq_from_not_valid_chartagseq(const std::u32string& charseq,
    const std::vector<Index>& not_valid_tagseq) noexcept
{
    std::vector<std::u32string> wordseq;
    std::size_t slice_spos = 0U;
    for( std::size_t slice_epos = 0; slice_epos < charseq.size(); ++slice_epos )
    {
        Index cur_tag = not_valid_tagseq[slice_epos];
        if( cur_tag == Tag::TAG_B_ID)
        {
            // if previous has the not processed segmentation. (not valid part)
            if( slice_epos > slice_spos )
            {
                wordseq.push_back(charseq.substr(slice_spos, slice_epos - slice_spos));
                slice_spos = slice_epos;
            }
        }
        else if( cur_tag == Tag::TAG_S_ID )
        {
            // if previous has the not processed segmentation. (not valid part)
            if( slice_epos > slice_spos )
            {
                wordseq.push_back(charseq.substr(slice_spos, slice_epos - slice_spos));
            }
            // the current char is a new segmentation.
            wordseq.push_back(charseq.substr(slice_epos, 1));
            slice_spos = slice_epos + 1; // set the next start pos.
        }
        else if( cur_tag == Tag::TAG_E_ID )
        {
            // including the current char, no matter what the start tag is.
            wordseq.push_back(charseq.substr(slice_spos, slice_epos - slice_spos + 1));
            slice_spos = slice_epos + 1; // set the next start pos.
        }
    }
    // process the last. (not valid part)
    if( slice_spos < charseq.size() )
    {
        wordseq.push_back(charseq.substr(slice_spos, charseq.size() - slice_spos));
    }
    return wordseq;
}


inline
bool can_emit(std::size_t cur_time, Index cur_tag_id) noexcept
{
    return cur_time != 0U || cur_tag_id == TAG_S_ID || cur_tag_id == TAG_B_ID;
    // Equivalent logic: 
    // if( cur_time == 0 ) {return (cur_tag_id == TAG_S_ID || cur_tag_id == TAG_B_ID) ; } 
    // return true ;
}

inline
bool can_trans_unsafe(Index pre_tag_id, Index cur_tag_id) noexcept
{
    // WARNING: we haven't check the id, may be out of range
    return TRANS_TABLE[pre_tag_id][cur_tag_id];
}

inline
bool can_trans(Index pre_tag_id, Index cur_tag_id) noexcept
{
    return (
           (    (pre_tag_id == TAG_B_ID || pre_tag_id == TAG_M_ID) 
             && (cur_tag_id == TAG_M_ID || cur_tag_id == TAG_E_ID)
           ) 
           ||
           (    (pre_tag_id == TAG_E_ID || pre_tag_id == TAG_S_ID) 
             && (cur_tag_id == TAG_B_ID || cur_tag_id == TAG_S_ID)
           )
        ) ;
}

} // end of segmenter token module
using token_module::Tag;
} // end of namespace segmenter
} // end of namespace slnn

#endif