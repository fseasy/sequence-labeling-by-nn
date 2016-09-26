#ifndef SLNN_CWS_BASE_MODEL_H_
#define SLNN_CWS_BASE_MODEL_H_
#include "trivial/lookup_table/lookup_table.h"
#include "segmentor/cws_module/cws_tagging_system.h"
#include "utils/typedeclaration.h"

namespace slnn{

class CWSBasicTokenModule
{
    friend class boost::serialization::access;
public:
    static const std::string UNK_STR;
public:
    CWSBasicTokenModule(unsigned seed) noexcept;
    // data translate handlers
    Index token2index(const std::string &character) const;
    Index unk_replace_in_probability(Index idx) const; 
    template <typename RawAnnotatedData, typename ProcessedAnnotatedData>
    void process_annotated_data(const RawAnnotatedData&, ProcessedAnnotatedData&) noexcept;
    template <typename RawUnannotatedData, typename ProcessedUnannotatedData>
    void process_unannotated_data(const RawUnannotatedData&, ProcessedUnannotatedData&) const noexcept;
    void char_tag_seq2word_seq(const Seq &char_seq, const IndexSeq &tag_seq, Seq &word_seq) const noexcept;
    
    // dict handler
    void finish_read_training_data() noexcept;
    void set_unk_replace_threshold(unsigned cnt_threshold, float prob_threshold) noexcept;

    // model info
    std::string get_model_info() const noexcept;
    std::size_t get_charset_size() const noexcept;
    std::size_t get_tag_set_dize() const noexcept;
private:
    LookupTableWithReplace char_dict;
};


inline 
Index CWSBaseModel::char2index(const std::string &character) const
{
    return char_dict.convert(character);
}

inline
Index CWSBaseModel::unk_replace_in_probability(Index idx) const
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
    for(const std::string &word : word_seq )
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


} // end of namespace slnn

#endif