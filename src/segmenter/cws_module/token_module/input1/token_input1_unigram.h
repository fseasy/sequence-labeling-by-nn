#ifndef SLNN_SEGMENTER_CWS_MODULE_TOKEN_MODULE_INPUT1_UNIGRAM_H_
#define SLNN_SEGMENTER_CWS_MODULE_TOKEN_MODULE_INPUT1_UNIGRAM_H_
#include "trivial/lookup_table/lookup_table.h"
#include "segmenter/cws_module/token_module/cws_tag_definition.h"
#include "trivial/charcode/charcode_convertor.h"
#include "utils/typedeclaration.h"

namespace slnn{
namespace segmenter{
namespace token_module{

namespace input1_unigram_token_module_inner{

inline 
std::string token2str(char32_t token)
{
    auto conv = charcode::CharcodeConvertor::create_convertor(charcode::EncodingDetector::get_console_encoding());
    return conv->encode1(token);
}

inline
size_t count_token_from_wordseq(const std::vector<std::u32string> &wordseq)
{
    size_t cnt = 0;
    for( auto word : wordseq ){ cnt += word.length(); }
    return cnt;
}

} // end of namespace token2str

/**
 * basic segmenter token processing module.
 * for char-based chinese word segmentation, token module contains
 * 1. the data : lookup table for text token to index
 * 2. operations : a) add the token to lookup table when reading annotated raw data.
 *                 b) translate annotated raw data to integer char-index and integer tag-index (X translate).
 *                 c) translate unannoatated raw data to interger char-index.
 *                 d) other interface for lookup table.
 */
class TokenSegmenterInput1Unigram
{
    friend class boost::serialization::access;
public:
    struct AnnotatedDataProcessedT
    {
        std::shared_ptr<std::vector<Index>> pcharseq;
        std::shared_ptr<std::vector<Index>> ptagseq;
        AnnotatedDataProcessedT() : pcharseq(nullptr), ptagseq(nullptr){}
        std::size_t size() const { return pcharseq ? pcharseq->size() : 0UL; }
    };
    using AnnotatedDataRawT = std::vector<std::u32string>;
    using UnannotatedDataProcessedT = std::shared_ptr<std::vector<Index>>;
    using UnannotatedDataRawT = std::u32string;
public:

    explicit TokenSegmenterInput1Unigram(unsigned seed) noexcept;

    // DATA TRANSLATING
    Index token2index(char32_t token) const;
    Index unk_replace_in_probability(Index idx) const;
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

    // DICT INTERFACE
    void finish_read_training_data();
    template <typename StructureParamT>
    void set_param_before_process_training_data(const StructureParamT& param);

    // MODULE INFO
    std::string get_module_info() const noexcept;
    std::size_t get_charset_size() const noexcept;
    std::size_t get_tagset_size() const noexcept;

protected:
    void set_unk_replace_threshold(unsigned cnt_threshold, float prob_threshold) noexcept;

private:
    template<class Archive>
    void serialize(Archive& ar, const unsigned int);

private:
    //slnn::trivial::LookupTableWithReplace<char32_t> token_dict;
    /**
     * token dict: unicode-char => index.
     * because boost(<=1.58.0) doesn't support deserialize char32_t, so we use actual-euqal data type -> unsigned
     */
    slnn::trivial::LookupTableWithReplace<unsigned> token_dict;
};

/******************************************
 * Inline Implementation
 ******************************************/

 /**
 * token to index(const).
 * @param token unicode token
 * @return index of the token
 */
inline
Index TokenSegmenterInput1Unigram::token2index(char32_t token) const
{
    return token_dict.convert(token);
}

/**
* replace char index with satisfying [cnt <= cnt_threshold] with unk in probality [<= prob_threshold].(const)
* @param idx char index
* @return unk index(if replace) or the original idx(not replace)
*/
inline
Index TokenSegmenterInput1Unigram::unk_replace_in_probability(Index idx) const
{
    return token_dict.unk_replace_in_probability(idx);
}


/**
 * extract unannotated data from annotated data.
 * @param ann_data annotated data
 * @return unannotated data
 */
inline
TokenSegmenterInput1Unigram::UnannotatedDataProcessedT
TokenSegmenterInput1Unigram::extract_unannotated_data_from_annotated_data(const AnnotatedDataProcessedT &ann_data) const
{
    return ann_data.pcharseq;
}


template <typename ProcessedAnnotatedDataT> 
ProcessedAnnotatedDataT 
TokenSegmenterInput1Unigram::replace_low_freq_token2unk(const ProcessedAnnotatedDataT & in_data) const
{
    ProcessedAnnotatedDataT rep_data;
    rep_data.pcharseq.reset(new std::vector<Index>( *(in_data.pcharseq) ));
    rep_data.ptagseq = in_data.ptagseq;
    for( Index &charidx : *(rep_data.pcharseq) ){ charidx = token_dict.unk_replace_in_probability(charidx); }
    return rep_data;
}


/**
* processng annotated data, including add new-token to lookup table, count token, translate text token to integer index.
* @exception bad_alloc
* @param raw_in raw annotated data
* @param out processed data. including char-index and tag-index sequence
*/
template <typename ProcessedAnnotatedDataT>
void 
TokenSegmenterInput1Unigram::process_annotated_data(const std::vector<std::u32string>& wordseq, 
    ProcessedAnnotatedDataT& processed_data)
{
    /*
    ProcessedAnnotatedData :
    struct
    {
    std::vector<Index> *pcharseq;
    std::vector<Index> *ptagseq;
    (others(such as feature info) will be added in the dirived class.)
    ProcessedAnnotatedData()
    : charindex_seq(nullptr),
    tagindex_seq(nullptr)
    ...
    {}
    ~ProcessedAnnotatedData(){ delete *; }
    };
    */
    size_t token_cnt = input1_unigram_token_module_inner::count_token_from_wordseq(wordseq);
    std::shared_ptr<std::vector<Index>> &charindex_seq = processed_data.pcharseq;
    std::shared_ptr<std::vector<Index>> &tagindex_seq = processed_data.ptagseq;
    charindex_seq.reset(new std::vector<Index>(token_cnt)); 
    tagindex_seq.reset(new std::vector<Index>(token_cnt)); // exception may be throw for memory limit
    size_t offset = 0;
    // char text seq -> char index seq
    for( const std::u32string &word : wordseq )
    {
        for(const char32_t& uc : word ){ (*charindex_seq)[offset++] = token_dict.convert(uc); }
    }
    // char text seq -> tag index seq
    generate_tagseq_from_wordseq2preallocated_space(wordseq, *tagindex_seq);
}


/**
* process unannotated data, that is translating text-token to char-index.
* @param raw_in raw unannotated data
* @param out processed data, including char-index sequence
*/
template <typename ProcessedUnannotatedDataT>
void 
TokenSegmenterInput1Unigram::process_unannotated_data(const std::u32string &tokenseq,
    ProcessedUnannotatedDataT &processed_out) const
{
    /**
    struct ProcessedUnannotatedData
    {
    std::vector<Index> *charindex_seq;
    ...
    ProcessedUnannotatedData()
    : charindex_seq(nullptr)
    ~ProcessedUnannotatedData(){ delete *; }
    }
    */
    size_t token_cnt = tokenseq.size();
    std::shared_ptr<std::vector<Index>> &charindex_seq = processed_out.charindex_seq;
    charindex_seq.reset(new std::vector<Index>(token_cnt));
    size_t offset = 0;
    for(const char32_t& token : tokenseq ){ (*charindex_seq)[offset++] = token_dict.convert(token); }
}


/**
* specification for ProcessedUnannotatedData = std::vector<Index> (not an structure)
*/
template <>
inline
void
TokenSegmenterInput1Unigram::process_unannotated_data(const std::u32string &tokenseq,
    std::shared_ptr<std::vector<Index>> &charseq) const
{
    using std::swap;
    size_t token_cnt = tokenseq.size();
    charseq.reset(new std::vector<Index>(token_cnt));
    size_t offset = 0;
    for( char32_t token : tokenseq ){ (*charseq)[offset++] = token_dict.convert(token); }
}


/**
* do someting when has read all training data, including freeze lookup table, set unk.
*/
inline
void TokenSegmenterInput1Unigram::finish_read_training_data()
{
    token_dict.freeze();
    token_dict.set_unk();
}


/**
* set unk replace [cnt_threshold] and [prob_threshold] from StructureParam.
*/
template <typename StructureParamT>
inline
void TokenSegmenterInput1Unigram::set_param_before_process_training_data(const StructureParamT& param)
{
    set_unk_replace_threshold(param.replace_freq_threshold, param.replace_prob_threshold);
}

/**
* set unk replace [cnt_threshold] and [prob_threshold].
*/
inline
void TokenSegmenterInput1Unigram::set_unk_replace_threshold(unsigned cnt_threshold, float prob_threshold) noexcept
{
    token_dict.set_unk_replace_threshold(cnt_threshold, prob_threshold);
}

inline
std::string TokenSegmenterInput1Unigram::get_module_info() const noexcept
{
    std::stringstream oss;
    oss << "token module info: \n"
        << "  - charset size = " << get_charset_size() << "\n"
        << "  - tag set size = " << get_tagset_size();
    return oss.str();
}

inline
std::size_t TokenSegmenterInput1Unigram::get_charset_size() const noexcept
{
    return token_dict.size();
}

inline
std::size_t TokenSegmenterInput1Unigram::get_tagset_size() const noexcept
{
    return TAG_SIZE;
}

template<class Archive>
void TokenSegmenterInput1Unigram::serialize(Archive& ar, const unsigned int)
{
    ar & token_dict;
}

} // end of namespace token_module
} // end of namespace segmenter
} // end of namespace slnn

#endif
