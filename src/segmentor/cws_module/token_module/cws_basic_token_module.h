#ifndef SLNN_SEGMENTOR_CWS_MODULE_BASIC_TOKEN_MODULE_H_
#define SLNN_SEGMENTOR_CWS_MODULE_BASIC_TOKEN_MODULE_H_
#include "trivial/lookup_table/lookup_table.h"
#include "cws_tag_definition.h"
#include "trivial/charcode/charcode_convertor.h"
#include "utils/typedeclaration.h"

namespace slnn{
namespace segmentor{
namespace token_module{

namespace token_module_inner{

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
 * basic segmentor token processing module.
 * for char-based chinese word segmentation, token module contains
 * 1. the data : lookup table for text token to index
 * 2. operations : a) add the token to lookup table when reading annotated raw data.
 *                 b) translate annotated raw data to integer char-index and integer tag-index (X translate).
 *                 c) translate unannoatated raw data to interger char-index.
 *                 d) other interface for lookup table.
 */
class SegmentorBasicTokenModule
{
    friend class boost::serialization::access;
public:
    static const std::string UNK_STR;
public:
    /**
     * constructor, with an seed to init the replace randomization.
     * @param seed unsigned, to init the inner LookupTableWithReplace.
     */
    SegmentorBasicTokenModule(unsigned seed) noexcept;

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
     * @exception bad_alloc
     * @param raw_in raw annotated data
     * @param out processed data. including char-index and tag-index sequence
     */
    template <typename ProcessedAnnotatedData>
    void process_annotated_data(const std::vector<std::u32string> &raw_in, ProcessedAnnotatedData &out);

    /**
     * process unannotated data, that is translating text-token to char-index.
     * @param raw_in raw unannotated data
     * @param out processed data, including char-index sequence
     */
    template <typename ProcessedUnannotatedData>
    void process_unannotated_data(const std::u32string &raw_in, ProcessedUnannotatedData &out) const;

    // DICT INTERFACE

    /**
     * do someting when has read all training data, including freeze lookup table, set unk.
     */
    void finish_read_training_data();
    
    /**
     * set unk replace [cnt_threshold] and [prob_threshold].
     */
    void set_unk_replace_threshold(unsigned cnt_threshold, float prob_threshold) noexcept;

    // MODULE INFO
    std::string get_module_info() const noexcept;
    std::size_t get_charset_size() const noexcept;
    std::size_t get_tagset_size() const noexcept;
private:
    slnn::trivial::LookupTableWithReplace<char32_t> token_dict;
};

/******************************************
 * Inline Implementation
 ******************************************/


inline
Index SegmentorBasicTokenModule::token2index(char32_t token) const
{
    return token_dict.convert(token);
}

inline
Index SegmentorBasicTokenModule::unk_replace_in_probability(Index idx) const
{
    return token_dict.unk_replace_in_probability(idx);
}

/*
ProcessedAnnotatedData :
struct
{
    std::vector<Index> *charindex_seq;
    std::vector<Index> *tagindex_seq;
    (others(such as feature info) will be added in the dirived class.)
    ProcessedAnnotatedData()
        : charindex_seq(nullptr),
          tagindex_seq(nullptr)
          ...
    {}
    ~ProcessedAnnotatedData(){ delete *; }
};
*/
template <typename ProcessedAnnotatedData>
void 
SegmentorBasicTokenModule::process_annotated_data(const std::vector<std::u32string>& wordseq, 
    ProcessedAnnotatedData& processed_data)
{
    size_t token_cnt = token_module_inner::count_token_from_wordseq(wordseq);
    std::vector<Index> * &charindex_seq = processed_data.charindex_seq;
    std::vector<Index> * &tagindex_seq = processed_data.tagindex_seq;
    charindex_seq = new std::vector<Index>(token_cnt); 
    tagindex_seq = new std::vector<Index>(token_cnt); // exception may be throw
    size_t offset = 0;
    // char text seq -> char index seq
    for( const std::u32string &word : wordseq )
    {
        for( char32_t uc : word ){ (*charindex_seq)[offset++] = token_dict.convert(uc); }
    }
    // char text seq -> tag index seq
    generate_tagseq_from_wordseq2preallocated_space(wordseq, *tagindex_seq);
}


/**
    struct ProcessedUnannotatedData
    {
        std::vector<Index> *charindex_seq;
        ProcessedUnannotatedData()
            : charindex_seq(nullptr)
        ~ProcessedUnannotatedData(){ delete *; }
    }
*/
template <typename ProcessedUnannotatedData>
void 
SegmentorBasicTokenModule::process_unannotated_data(const std::u32string &tokenseq,
    ProcessedUnannotatedData &processed_out) const
{
    size_t token_cnt = tokenseq.size();
    std::vector<Index>* &charindex_seq = processed_out.charindex_seq;
    charindex_seq = new std::vector<Index>(token_cnt);
    size_t offset = 0;
    for( char32_t token : tokenseq ){ (*charindex_seq)[offset++] = token_dict.convert(token); }
}

void SegmentorBasicTokenModule::finish_read_training_data()
{
    token_dict.freeze();
    token_dict.set_unk();
}

void SegmentorBasicTokenModule::set_unk_replace_threshold(unsigned cnt_threshold, float prob_threshold) noexcept
{
    token_dict.set_unk_replace_threshold(cnt_threshold, prob_threshold);
}

inline
std::string SegmentorBasicTokenModule::get_module_info() const noexcept
{
    std::stringstream oss;
    oss << "token module info: \n"
        << "  - charset size = " << get_charset_size() << "\n"
        << "  - tag set size = " << get_tagset_size();
    return oss.str();
}

inline
std::size_t SegmentorBasicTokenModule::get_charset_size() const noexcept
{
    return token_dict.size();
}
std::size_t SegmentorBasicTokenModule::get_tagset_size() const noexcept
{
    return TAG_SIZE;
}

} // end of namespace token_module
} // end of namespace segmentor
} // end of namespace slnn

#endif