#include "cws_feature_layer.h"

namespace slnn{
CWSFeatureLayer::CWSFeatureLayer(dynet::Model *dynet_m, unsigned start_here_dict_size, unsigned start_here_dim,
    unsigned pass_here_dict_size, unsigned pass_here_dim,
    unsigned end_here_dict_size, unsigned end_here_dim,
    dynet::LookupParameter word_lookup_param,
    unsigned chartype_category_num, unsigned chartype_dim)
    :lexicon_feature_layer(dynet_m, start_here_dict_size, start_here_dim, pass_here_dict_size, pass_here_dim,
        end_here_dict_size, end_here_dim),
    context_feature_layer(dynet_m, word_lookup_param),
    chartype_feature_layer(dynet_m, chartype_category_num, chartype_dim)
{}
CWSFeatureLayer::CWSFeatureLayer(dynet::Model *dynet_m, const CWSFeature &cws_feature, dynet::LookupParameter word_lookup_param)
    :lexicon_feature_layer(dynet_m, cws_feature.lexicon_feature),
    context_feature_layer(dynet_m, word_lookup_param),
    chartype_feature_layer(dynet_m, cws_feature.chartype_feature.FeatureDictSize(), 
                                  cws_feature.chartype_feature.get_feature_dim())
{}
}
