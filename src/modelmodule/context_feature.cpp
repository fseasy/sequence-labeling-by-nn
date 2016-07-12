#include "context_feature.h"

namespace slnn{

const Index ContextFeature::WordSOSId;
const Index ContextFeature::WordEOSId;

ContextFeature::ContextFeature(DictWrapper &dict_wrapper, unsigned context_left_size, unsigned context_right_size,  unsigned word_dim)
    :context_size(context_left_size + context_right_size),
    context_left_size(context_left_size),
    context_right_size(context_right_size),
    rwrapper(dict_wrapper),
    word_dim(word_dim)
{}

void ContextFeature::set_parameters(unsigned context_left_size, unsigned context_right_size, unsigned word_dim)
{
    this->context_size = context_left_size + context_right_size;
    this->context_left_size = context_left_size;
    this->context_right_size = context_right_size;
    this->word_dim = word_dim;
}

std::string ContextFeature::get_context_feature_info() const 
{
    std::ostringstream oss;
    oss << "context size : " << context_size << ", left size : " << context_left_size << ", right size: " << context_right_size
        << "totally context feature dim : " << get_context_feature_dim();
    return oss.str();
}

void ContextFeature::extract(const IndexSeq &seq, ContextFeatureDataSeq &context_feature_data_seq)
{
    using std::swap;
    int sent_len = seq.size();
    ContextFeatureDataSeq tmp_feature_data_seq(sent_len,
        ContextFeatureData(context_size));
    for( Index i = 0; i < sent_len; ++i )
    {
        ContextFeatureData &feature_data = tmp_feature_data_seq.at(i);
        unsigned feature_idx = 0 ;
        for( Index left_context_offset = 1 ; left_context_offset <= context_left_size ; ++left_context_offset )
        {
            int word_pos = i - left_context_offset;
            feature_data.at(feature_idx) = (word_pos < 0 ? WordSOSId : seq.at(word_pos)) ;
            ++feature_idx;
        }
        for( Index right_context_offset = 1 ; right_context_offset <= context_right_size; ++right_context_offset )
        {
            int word_pos = i + right_context_offset;
            feature_data.at(feature_idx) = ( word_pos >= sent_len ? WordEOSId : seq.at(word_pos) );
            ++feature_idx;
        }
    }
    swap(context_feature_data_seq, tmp_feature_data_seq);
}

} // end of namespace slnn