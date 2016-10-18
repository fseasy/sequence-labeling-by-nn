#include "context_feature_layer.h"

namespace slnn{

ContextFeatureLayer::ContextFeatureLayer(dynet::Model *m, dynet::LookupParameters *word_lookup_param)
    :word_lookup_param(word_lookup_param),
    pcg(nullptr),
    word_sos_param(m->add_parameters(word_lookup_param->dim)),
    word_eos_param(m->add_parameters(word_lookup_param->dim))
{}

} // end of namespace slnn