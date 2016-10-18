#include "pos_feature_layer.h"

namespace slnn{

POSFeatureLayer::POSFeatureLayer(dynet::Model *m, 
                                 size_t prefix_suffix_len1_dict_size, unsigned prefix_suffix_len1_embedding_dim,
                                 size_t prefix_suffix_len2_dict_size, unsigned prefix_suffix_len2_embedding_dim,
                                 size_t prefix_suffix_len3_dict_size, unsigned prefix_suffix_len3_embedding_dim,
                                 size_t char_length_dict_size, unsigned char_length_embedding_dim)
    :prefix_suffix_len1_lookup_param(m->add_lookup_parameters(prefix_suffix_len1_dict_size, {prefix_suffix_len1_embedding_dim})),
    prefix_suffix_len2_lookup_param(m->add_lookup_parameters(prefix_suffix_len2_dict_size, {prefix_suffix_len2_embedding_dim})),
    prefix_suffix_len3_lookup_param(m->add_lookup_parameters(prefix_suffix_len3_dict_size, {prefix_suffix_len3_embedding_dim})),
    char_length_lookup_param(m->add_lookup_parameters(char_length_dict_size, {char_length_embedding_dim}))
{}

POSFeatureLayer::POSFeatureLayer(dynet::Model *m, POSFeature &pos_feature)
    :POSFeatureLayer(m, 
                     pos_feature.prefix_suffix_len1_dict.size(), pos_feature.prefix_suffix_len1_embedding_dim,
                     pos_feature.prefix_suffix_len2_dict.size(), pos_feature.prefix_suffix_len2_embedding_dim,
                     pos_feature.prefix_suffix_len3_dict.size(), pos_feature.prefix_suffix_len3_embedding_dim,
                     pos_feature.get_char_length_dict_size(), pos_feature.char_length_embedding_dim)
{}

} // end of namespace slnn
