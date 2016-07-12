#include "cws_feature.h"

namespace slnn{

CWSFeature::CWSFeature(DictWrapper &word_dict_wrapper)
    :context_feature(word_dict_wrapper)
{}

void CWSFeature::set_feature_parameters(unsigned lexicon_start_here_dim, unsigned lexicon_pass_here_dim, unsigned lexicon_end_here_dim,
    unsigned context_left_size, unsigned context_right_size, unsigned word_embedding_dim)
{
    lexicon_feature.set_dim(lexicon_start_here_dim, lexicon_pass_here_dim, lexicon_end_here_dim);
    context_feature.set_parameters(context_left_size, context_right_size, word_embedding_dim);
}


void CWSFeature::extract(const Seq &char_seq, const IndexSeq &index_char_seq, CWSFeatureDataSeq &cws_feature_seq)
{
    lexicon_feature.extract(char_seq, cws_feature_seq.get_lexicon_feature_data_seq());
    context_feature.extract(index_char_seq, cws_feature_seq.get_context_feature_data_seq());
}

std::string CWSFeature::get_feature_info() const
{
    std::ostringstream oss;
    oss << "lexicon feature info : \n" << lexicon_feature.get_lexicon_info() << "\n"
        << "context feature info : \n" << context_feature.get_context_feature_info() ;
    return oss.str();
}

} // end of namespace slnn