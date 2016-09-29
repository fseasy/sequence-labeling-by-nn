#ifndef SLNN_CWS_BASE_MODEL_H_
#define SLNN_CWS_BASE_MODEL_H_
#include "trivial/lookup_table/lookup_table.h"
#include "cws_tag_definition.h"
#include "trivial/charcode/charcode_convertor.h"
#include "utils/typedeclaration.h"

namespace slnn{
namespace segmentor{
namespace token_module{

namespace token_module_inner{

inline 

}

/**
 * basic segmentor token processing module.
 * for char-based chinese word segmentation, token module contains
 * 1. the data : lookup table for text token to index
 * 2. operations : a) add the token to lookup table when reading annotated raw data.
 *                 b) translate annotated raw data to integer char-index and integer tag-index (X translate).
 *                 c) translate unannoatated raw data to interger char-index.
 *                 d) other interface for lookup table.
 */
class CWSBasicTokenModule
{
    friend class boost::serialization::access;
public:
    static const std::string UNK_STR;
public:
    /**
     * constructor, with an seed to init the replace randomization.
     * @param seed unsigned, to init the inner LookupTableWithReplace.
     */
    CWSBasicTokenModule(unsigned seed) noexcept;

    // DATA TRANSLATING
    /**
     * token to index(const).
     * @param token unicode token
     * @return index of the token
     */
    Index token2index(char32_t token) const;

    /**
     * replace char index with satisfying [cnt <= cnt_threshold] with unk in probality [<= prob_threshold].(const)
     * @param idx char index
     * @return unk index(if replace) or the original idx(not replace)
     */
    Index unk_replace_in_probability(Index idx) const;

    /**
     * processng annotated data, including add new-token to lookup table, count token, translate text token to integer index.
     * @param raw_in raw annotated data, type may be vector<u32string>
     * @param out processed data. including char-index and tag-index sequence
     */
    template <typename RawAnnotatedData, typename ProcessedAnnotatedData>
    void process_annotated_data(const RawAnnotatedData &raw_in, ProcessedAnnotatedData &out) noexcept;

    /**
     * process unannotated data, that is translating text-token to char-index.
     * @param raw_in raw unannotated data
     * @param out processed data, including char-index sequence
     */
    template <typename RawUnannotatedData, typename ProcessedUnannotatedData>
    void process_unannotated_data(const RawUnannotatedData &raw_in, ProcessedUnannotatedData &out) const noexcept;

    // DICT INTERFACE

    /**
     * do someting when has read all training data, including freeze lookup table, set unk.
     */
    void finish_read_training_data() noexcept;
    /**
     * set unk replace [cnt_threshold] and [prob_threshold].
     */
    void set_unk_replace_threshold(unsigned cnt_threshold, float prob_threshold) noexcept;

    // MODULE INFO
    std::string get_module_info() const noexcept;
    std::size_t get_charset_size() const noexcept;
    std::size_t get_tag_set_dize() const noexcept;
private:
    slnn::trivial::LookupTableWithReplace<char32_t> char_dict;
};

/******************************************
 * Inline Implementation
 ******************************************/


inline
Index CWSBasicTokenModule::token2index(char32_t token) const
{
    return char_dict.convert(token);
}

inline
Index CWSBasicTokenModule::unk_replace_in_probability(Index idx) const
{
    return char_dict.unk_replace_in_probability(idx);
}

/*
RawAnnotatedData :
vector<u32string>

ProcessedAnnotatedData :
struct
{
    IndexSeq char_index_seq;
    IndexSeq tag_index_seq;
    (others(such as feature info) will be added in the dirived class.)
};
*/
template <typename RawAnnotatedData, typename ProcessedAnnotatedData>
void process_annotated_data(const RawAnnotatedData& raw_data, ProcessedAnnotatedData& processed_data) noexcept
{
    using std::swap;
    IndexSeq tmp_word_index_seq,
        tmp_tag_index_seq;
    Seq tmp_char_seq;
    tmp_word_index_seq.reserve(SentMaxLen);
    tmp_tag_index_seq.reserve(SentMaxLen);
    tmp_char_seq.reserve(SentMaxLen);
    for( const std::string &word : word_seq )
    {
        Seq word_char_seq ;
        IndexSeq word_tag_index_seq;
        CWSTaggingSystem::static_parse_word2chars_indextag(word, word_char_seq, word_tag_index_seq);
        for( size_t i = 0; i < word_char_seq.size(); ++i )
        {
            tmp_tag_index_seq.push_back(word_tag_index_seq[i]);
            Index word_id = word_dict_wrapper.Convert(word_char_seq[i]);
            tmp_word_index_seq.push_back(word_id);
            tmp_char_seq.push_back(std::move(word_char_seq[i]));
        }
    }
    cws_feature.extract(tmp_char_seq, tmp_word_index_seq, feature_data_seq);
    swap(word_index_seq, tmp_word_index_seq);
    swap(tag_index_seq, tmp_tag_index_seq);
}

} // end of namespace token_module
} // end of namespace segmentor
} // end of namespace slnn

#endif