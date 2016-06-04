#ifndef POSTAGGER_POSTAGGER_MODULE_POS_FEATURE_EXTRACTOR_HPP_
#define POSTAGGER_POSTAGGER_MODULE_POS_FEATURE_EXTRACTOR_HPP_
#include <string>
#include "cnn/dict.h"
#include "utils/dict_wrapper.hpp"
#include "utils/typedeclaration.h"

namespace slnn{

class POSFeatureExtractor
{
public:
    static const std::string FEATURE_UNK_STR ;
    
public :
    POSFeatureExtractor();
    void extract(const Seq &raw_inputs, FeatureIndexSeq &features_seq);

private:
    cnn::Dict prefix_fdict; // UNK
    cnn::Dict suffix_fdict; // UNK
    cnn::Dict len_fdict;

    DictWrapper prefix_fdict_wrapper;
    DictWrapper suffix_fdict_wrapper;
};

const std::string POSFeatureExtractor::FEATURE_UNK_STR = "feature_unk";

POSFeatureExtractor::POSFeatureExtractor()
    :prefix_fdict_wrapper(prefix_fdict),
    suffix_fdict_wrapper(suffix_fdict)
{};

void POSFeatureExtractor(const Seq &raw_inputs, FeatureIndexSeq &features_seq)
{
    using std::swap;
    size_t nr_tokens = raw_inputs.size();
    FeatureIndexSeq tmp_features_seq(nr_tokens);
    // prefix , suffix
    for( size_t i = 0; i < nr_tokens; ++i )
    {

    }
}


} // end of namespace slnn


#endif