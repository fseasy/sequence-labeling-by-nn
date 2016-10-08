#include "cws_basic_token_module.h"
namespace slnn{
namespace segmentor{
namespace token_module{

SegmentorBasicTokenModule::SegmentorBasicTokenModule(unsigned seed) noexcept
    :token_dict(seed, 1, 0.2F, [](const char32_t &token){ return token_module_inner::token2str(token); })
{}

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
    ...
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

} // end of namespace token_module
} // end of namespace segmentor
} // end of namespace slnn
