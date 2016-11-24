#ifndef SLNN_SEGMENTER_CWS_MODULE_TOKEN_MODULE_INPUT1_ALL_H_
#define SLNN_SEGMENTER_CWS_MODULE_TOKEN_MODULE_INPUT1_ALL_H_
#include <functional>
#include "trivial/lookup_table/lookup_table.h"
#include "segmenter/cws_module/token_module/cws_tag_definition.h"
#include "trivial/charcode/charcode_convertor.h"
#include "utils/typedeclaration.h"
#include "segmenter/cws_module/token_module/token_lexicon.h"
#include "segmenter/cws_module/token_module/token_chartype.h"
namespace slnn{
namespace segmenter{
namespace token_module{

namespace input1_all_token_module_inner{

struct TokenModuleState
{
    bool enable_unigram;
    bool enable_bigram;
    bool enable_lexicon;
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

/**
 * Token Segmenter for input1 with all features.
 * have the ability to extract unigram, bigram, lexicon, chartype feature and can disable any of them from commandline \
 * with the constraint that we should ensure at least one of unigram and bigram is enbaled. 
 * 
 * call flow:
 * for training:
 * 1. init with one parameter: seed
 * 2. call `set_param`.
 * 3. call `build_lexicon_if_necessary`
 * 4. read training data and build feature.
 * 5. call `finish_read_training_data`
 * 6. call `replace_low_freq_token2unk` if necessary
 * 7. serialization
 * 
 * for de-serialization
 * 1. init
 * 2. un-serialize
 **/
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
        std::size_t size() const 
        { 
            if( punigramseq ){ return punigramseq->size(); }
            else if( pbigramseq ){ return pbigramseq->size(); }
            else{ throw std::logic_error("at least one of {unigram, bigram} should be enable."); }
        }
    };
    using AnnotatedDataRawT = std::vector<std::u32string>;
    struct UnannotatedDataProcessedT
    {
        std::shared_ptr<std::vector<Index>> punigramseq;
        std::shared_ptr<std::vector<Index>> pbigramseq;
        std::shared_ptr<std::vector<std::vector<int>>> plexiconseq;
        std::shared_ptr<std::vector<Index>> ptypeseq;
        UnannotatedDataProcessedT() : punigramseq(nullptr), pbigramseq(nullptr), 
            plexiconseq(nullptr), ptypeseq(nullptr){}
        std::size_t size() const
        {
            if( punigramseq ){ return punigramseq->size(); }
            else if( pbigramseq ){ return pbigramseq->size(); }
            else{ throw std::logic_error("at least one of {unigram, bigram} should be enable."); }
        }
    };
    using UnannotatedDataRawT = std::u32string;
public:
    explicit TokenSegmenterInput1All(unsigned seed) noexcept;

public:
    // DATA TRANSLATING
    UnannotatedDataProcessedT
        extract_unannotated_data_from_annotated_data(const AnnotatedDataProcessedT &ann_data) const;
    // WE DO NOT use current class-defined data structure. 
    // ideally, we'll process the derived class-defined data.
    template <typename ProcessedAnnotatedDataT> 
    ProcessedAnnotatedDataT replace_low_freq_token2unk(const ProcessedAnnotatedDataT & in_data) const;
    template <typename ProcessedAnnotatedDataT>
    void process_annotated_data(const std::vector<std::u32string> &raw_in, ProcessedAnnotatedDataT &out);
    template <typename ProcessedUnannotatedDataT>
    void process_unannotated_data(const std::u32string &raw_in, ProcessedUnannotatedDataT &out) const;

    // PARAM INTERFACE (only for training)
    template <typename StructureParamT>
    void set_param(const StructureParamT& param);
    void build_lexicon_if_necessary(std::ifstream &training_is);
    void finish_read_training_data();

    // MODULE INFO
    std::string get_module_info() const noexcept;
    const input1_all_token_module_inner::TokenModuleState& get_token_state() const noexcept { return state; }

protected:
    void set_unk_replace_threshold(unsigned cnt_threshold, float prob_threshold) noexcept;

private:
    template<class Archive>
    void serialize(Archive& ar, const unsigned int);

private:
    slnn::trivial::LookupTableWithReplace<char32_t>  unigram_dict;
    static std::u32string EOS_REPR;
    slnn::trivial::LookupTableWithReplace<std::u32string> bigram_dict;
    TokenLexicon lexicon_feat;
    // TokenChartype is a static class
    input1_all_token_module_inner::TokenModuleState state;
};


/*******************
 * Inline/Template Implementation
 *******************/

inline
TokenSegmenterInput1All::UnannotatedDataProcessedT
TokenSegmenterInput1All::extract_unannotated_data_from_annotated_data(const AnnotatedDataProcessedT &ann_data) const
{
    UnannotatedDataProcessedT unann_data;
    unann_data.punigramseq = ann_data.punigramseq;
    unann_data.pbigramseq = ann_data.pbigramseq;
    unann_data.plexiconseq = ann_data.plexiconseq;
    unann_data.ptypeseq = ann_data.ptypeseq;
    return unann_data;
}

template <typename ProcessedAnnotatedDataT> 
ProcessedAnnotatedDataT 
TokenSegmenterInput1All::replace_low_freq_token2unk(const ProcessedAnnotatedDataT & in_data) const
{
    ProcessedAnnotatedDataT rep_data;
    rep_data.plexiconseq = in_data.plexiconseq;
    rep_data.ptypeseq = in_data.ptypeseq;
    rep_data.tagseq = in_data.tagseq;
    rep_data.punigramseq = std::shared_ptr<std::vector<Index>>(new std::vector<Index>(*in_data.punigramseq));
    rep_data.pbigramseq = std::shared_ptr<std::vector<Index>>(new std::vector<Index>(*in_data.pbigramseq));
    unsigned seqlen = in_data.size();
    for( unsigned i = 0; i < seqlen; ++i )
    {
        Index &uni_idx = (*rep_data.punigramseq)[i],
            &bi_idx = (*rep_data.pbigramseq)[i];
            
        uni_idx = unigram_dict.unk_replace_in_probability(uni_idx);
        bi_idx = bigram_dict.unk_replace_in_probability(bi_idx);
    }
    return rep_data;
}


template <typename ProcessedAnnotatedDataT>
void 
TokenSegmenterInput1All::process_annotated_data(const std::vector<std::u32string>& wordseq, ProcessedAnnotatedDataT &ann_data)
{
    unsigned charseq_len = std::accumulate(wordseq.begin(), wordseq.end(), 0,
    [](const unsigned &lhs_len, const std::u32string& rhs)
    {
        return lhs_len + rhs.length();
    });
    // generate charseq (next will use)
    std::u32string charseq(charseq_len);
    unsigned pos = 0;
    for( const std::u32string& word : wordseq )
    {
        for( char32_t uc : word ){ charseq[pos++] = uc; }
    }
    // unigram seq
    if( state.enable_unigram )
    {
        std::shared_ptr<std::vector<Index>> &puni_seq = ann_data.punigramseq;
        puni_seq.reset(new std::vector<Index>(charseq_len));
        for( pos = 0; pos < charseq_len; ++pos )
        {
            (*puni_seq)[i] = unigram_dict.convert(charseq[pos]);
        }
    }
    // bigram seq
    if( state.enable_bigram )
    {
        std::shared_ptr<std::vector<Index>> &pbi_seq = ann_data.pbigramseq;
        for( pos = 0; pos < charseq_len - 1; ++pos )
        {
            (*pbi_seq)[i] = bigram_dict.convert(charseq.substr(pos, 2));
        }
        pbi_seq->back() = bigram_dict.convert(charseq.back() + EOS_REPR);
    }
    // lexicon seq
    if( state.enable_lexicon )
    {
        ann_data.plexiconseq = lexicon_feat.extract(charseq);
    }
    // type seq
    if( state.enable_type )
    {
        ann_data.ptypeseq = TokenChartype::extract(charseq);
    }
    // tag seq
    ann_data.ptagseq.reset(new std::vector<Index>(charseq_len));
    generate_tagseq_from_wordseq2preallocated_space(wordseq, *ann_data.ptagseq);
}


template <typename ProcessedUnannotatedDataT>
void 
TokenSegmenterInput1All::process_unannotated_data(const std::u32string &charseq, ProcessedUnannotatedDataT &unann_data) const
{
    unsigned charseq_len = charseq.length();
    // unigram seq
    if( state.enable_unigram )
    {
        std::shared_ptr<std::vector<Index>> &puni_seq = unann_data.punigramseq;
        puni_seq.reset(new std::vector<Index>(charseq_len));
        for( pos = 0; pos < charseq_len; ++pos )
        {
            (*puni_seq)[i] = unigram_dict.convert(charseq[pos]);
        }
    }
    // bigram seq
    if( state.enable_bigram )
    {
        std::shared_ptr<std::vector<Index>> &pbi_seq = unann_data.pbigramseq;
        for( pos = 0; pos < charseq_len - 1; ++pos )
        {
            (*pbi_seq)[i] = bigram_dict.convert(charseq.substr(pos, 2));
        }
        pbi_seq->back() = bigram_dict.convert(charseq.back() + EOS_REPR);
    }
    // lexicon seq
    if( state.enable_lexicon )
    {
        unann_data.plexiconseq = lexicon_feat.extract(charseq);
    }
    // type seq
    if( state.enable_type )
    {
        unann_data.ptypeseq = TokenChartype::extract(charseq);
    }
}

inline
void TokenSegmenterInput1All::set_unk_replace_threshold(unsigned cnt_threshold, float prob_threshold) noexcept
{
    unigram_dict.set_unk_replace_threshold(cnt_threshold, prob_threshold);
    bigram_dict.set_unk_replace_threshold(cnt_threshold, prob_threshold);
}

template <typename StructureParamT>
void TokenSegmenterInput1All::set_param(const StructureParamT& param)
{
    state.enable_unigram = param.enable_unigram;
    state.enable_bigram = param.enable_bigram;
    state.enable_lexicon = param.enable_lexicon;
    state.enable_type = param.enable_type;
    if( !state.enable_unigram && !state.enable_bigram )
    { 
        throw std::logic_error("at least one of {unigram, bigram} should be enable."); 
    }
    set_unk_replace_threshold(param.replace_freq_threshold, param.replace_prob_threshold);
    lexicon_feat.set_maxlen4feature(param.lexicon_feature_max_len);
}

inline
void TokenSegmenterInput1All::build_lexicon_if_necessary(std::ifstream &is)
{
    if( state.enable_lexicon ){ lexicon_feat.build_inner_lexicon_from_training_data(is); }
}


inline
void TokenSegmenterInput1All::finish_read_training_data()
{
    unigram_dict.freeze();
    unigram_dict.set_unk();
    bigram_dict.freeze();
    bigram_dict.set_unk();
    state.unigram_dict_sz = unigram_dict.size();
    state.bigram_dict_sz = bigram_dict.size();
    state.lexicon_dict_sz = lexicon_feat.size();
    state.type_dict_sz = TokenChartype::size();
}

inline
std::string TokenSegmenterInput1All::get_module_info() const noexcept
{
    std::ostringstream oss;
    oss << "+ Token module info: \n"
        << std::boolalpha
        << "| enable unigram(" << state.enable_unigram << ") unigram dict size(" << state.unigram_dict_sz << "\n"
        << "| enable bigram(" << state.enable_bigram << ") bigram dict size(" << state.bigram_dict_sz << "\n"
        << "| enable lexicon(" << state.enable_lexicon << ") lexicon feature size(" << state.lexicon_dict_sz << "\n"
        << "| enable chartype(" << state.enable_type << ") chartype feature size(" << state.type_dict_sz << "\n"
        << lexicon_feat.get_lexicon_feature_info() << "\n"
        << "== - - - - -";
    return oss.str();
}

template<class Archive>
void TokenSegmenterInput1All::serialize(Archive& ar, const unsigned int)
{
    ar &unigram_dict &bigram_dict &lexicon_feat &state;
}

} // end of namespace token_module
} // end of namespace segmenter
} // end of namespace slnn

#endif
