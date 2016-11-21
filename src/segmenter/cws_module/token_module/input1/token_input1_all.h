#ifndef SLNN_SEGMENTER_CWS_MODULE_TOKEN_MODULE_INPUT1_ALL_H_
#define SLNN_SEGMENTER_CWS_MODULE_TOKEN_MODULE_INPUT1_ALL_H_
#include "trivial/lookup_table/lookup_table.h"
#include "segmenter/cws_module/token_module/cws_tag_definition.h"
#include "trivial/charcode/charcode_convertor.h"
#include "utils/typedeclaration.h"

namespace slnn{
namespace segmenter{
namespace token_module{

namespace input1_all_token_module_inner{

struct TokenModuleState
{
    bool enable_unigram;
    bool enable_bigram;
    bool enbale_lexicon;
    bool enable_type;
    
    unsigned unigram_dict_sz;
    unsigned bigram_dict_sz;
    unsigned lexicon_dict_sz;
    unsigned type_dict_sz;

    unsigned tag_dict_sz;

// serialization
    friend class boost::serialization::access;
private:
    template<class Archive>
    void serialize(Archive &ar, const unsigned int);
};


template<class Archive>
void TokenModuleState::serialize(Archive &ar, const unsigned int)
{
    ar &enable_unigram &enable_bigram &enable_lexicon &enable_type
        &unigram_dict_sz &bigram_dict_sz &lexicon_dict_sz &type_dict_sz
        &tag_dict_sz;
}

} // end of namespace input1_all_token_module_inner


class TokenSegmenterInput1All
{
    friend class boost::serialization::access;
public:
    struct AnnotatedDataProcessedT
    {
        std::shared_ptr<std::vector<Index>> punigramseq;
        std::shared_ptr<std::vector<Index>> pbigramseq;
        std::shared_ptr<std::vector<std::vector<int>>> plexiconseq;
        std::shared_ptr<std::vector<Index>> ptypeseq;
        std::shared_ptr<std::vector<Index>> ptagseq;
        AnnotatedDataProcessedT() : punigramseq(nullptr), pbigramseq(nullptr), 
            plexiconseq(nullptr), ptypeseq(nullptr), ptagseq(nullptr){}
        std::size_t size() const { return punigramseq ? punigramseq->size() : 0UL; }
    };
    using AnnotatedDataRawT = std::vector<std::u32string>;
    struct UnannotatedDataProcessedT
    {
        std::shared_ptr<std::vector<Index>> punigramseq;
        std::shared_ptr<std::vector<Index>> pbigramseq;
        std::shared_ptr<std::vector<std::vector<int>>> plexiconseq;
        std::shared_ptr<std::vector<Index>> ptypeseq;
        std::shared_ptr<std::vector<Index>> ptagseq;
        UnannotatedDataProcessedT() : punigramseq(nullptr), pbigramseq(nullptr), 
            plexiconseq(nullptr), ptypeseq(nullptr), ptagseq(nullptr){}
        std::size_t size() const { return punigramseq ? punigramseq->size() : 0UL; }
    };
    using UnannotatedDataRawT = std::u32string;
public:
    explicit TokenSegmenterInput1All(unsigned seed) noexcept;

public:
    // DATA TRANSLATING
    Index token2index(char32_t token) const;
    Index unk_replace_in_probability(Index idx) const;
    std::shared_ptr<UnannotatedDataProcessedT>
        extract_unannotated_data_from_annotated_data(const AnnotatedDataProcessedT &ann_data) const;
    // WE DO NOT use current class-defined data structure. 
    // ideally, we'll process the derived class-defined data.
    template <typename ProcessedAnnotatedDataT> 
    ProcessedAnnotatedDataT replace_low_freq_token2unk(const ProcessedAnnotatedDataT & in_data) const;
    template <typename ProcessedAnnotatedDataT>
    void process_annotated_data(const std::vector<std::u32string> &raw_in, ProcessedAnnotatedDataT &out);
    template <typename ProcessedUnannotatedDataT>
    void process_unannotated_data(const std::u32string &raw_in, ProcessedUnannotatedDataT &out) const;

    // DICT INTERFACE
    void finish_read_training_data();
    template <typename StructureParamT>
    void set_unk_replace_threshold(const StructureParamT& param) noexcept;
    void set_unk_replace_threshold(unsigned cnt_threshold, float prob_threshold) noexcept;

    // MODULE INFO
    std::string get_module_info() const noexcept;
    const input1_all_token_module_inner::TokenModuleState& get_token_state() const noexcept { return state; }
private:
    template<class Archive>
    void serialize(Archive& ar, const unsigned int);

private:
    //slnn::trivial::LookupTableWithReplace<char32_t> token_dict;
    /**
    * token dict: unicode-char => index.
    * because boost(<=1.58.0) doesn't support deserialize char32_t, so we use actual-euqal data type -> unsigned
    */
    slnn::trivial::LookupTableWithReplace<char32_t>  unigram_dict;
    slnn::trivial::LookupTableWithReplace<std::u32string> bigram_dict;
    // lexicon info

    // type


};


} // end of namespace token_module
} // end of namespace segmenter
} // end of namespace slnn

#endif
