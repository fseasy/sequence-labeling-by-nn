#include "lexicon_feature_layer.h"

namespace slnn{

LexiconFeatureLayer::LexiconFeatureLayer(dynet::Model * dynet_m, unsigned start_here_dict_size, unsigned start_here_dim,
    unsigned pass_here_dict_size, unsigned pass_here_dim,
    unsigned end_here_dict_size, unsigned end_here_dim)
    : start_here_lookup_param(dynet_m->add_lookup_parameters(start_here_dict_size, { start_here_dim })),
    pass_here_lookup_param(dynet_m->add_lookup_parameters(pass_here_dict_size, { pass_here_dim })),
    end_here_lookup_param(dynet_m->add_lookup_parameters(end_here_dict_size, {end_here_dim}))
{}

LexiconFeatureLayer::LexiconFeatureLayer(dynet::Model *dynet_m, const LexiconFeature &lexicon_feature)
    :LexiconFeatureLayer(dynet_m, lexicon_feature.get_start_here_dict_size(), lexicon_feature.get_start_here_feature_dim(),
        lexicon_feature.get_pass_here_dict_size(), lexicon_feature.get_pass_here_feature_dim(),
        lexicon_feature.get_end_here_dict_size(), lexicon_feature.get_end_here_feature_dim())
{}

}