#include "cws_feature.h"

namespace slnn{

CWSFeature::CWSFeature(){}

void CWSFeature::set_dim(unsigned start_here_dim, unsigned pass_here_dim, unsigned end_here_dim)
{
    lexicon_feature.set_dim(start_here_dim, pass_here_dim, end_here_dim);
}

void CWSFeature::extract(const Seq &char_seq, CWSFeatureDataSeq &cws_feature_seq)
{
    lexicon_feature.extract(char_seq, cws_feature_seq.get_lexicon_feature_data_seq());
}

std::string CWSFeature::get_feature_info()
{
    std::ostringstream oss;
    oss << "lexicon feature info : \n" << lexicon_feature.get_lexicon_info();
    return oss.str();
}

} // end of namespace slnn