#ifndef SLNN_CWS_BASE_MODEL_H_
#define SLNN_CWS_BASE_MODEL_H_
#include "trivial/lookup_table.h"
#include "utils/typedeclaration.h"
namespace slnn{

/*
RawAnnotatedData
{

}

*/

class CWSBaseModel
{
    friend class boost::serialization::access;
public:
    static const std::string UNK_STR;
    static const unsigned SentMaxLen;
public:
    CWSBaseModel(unsigned seed) noexcept;
    // data translate handlers
    Index char2index(const std::string &character) const;
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



} // end of namespace slnn

#endif