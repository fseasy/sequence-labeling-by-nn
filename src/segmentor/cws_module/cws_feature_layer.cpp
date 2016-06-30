#include "cws_feature_layer.h"

namespace slnn{
CWSFeatureLayer::CWSFeatureLayer(cnn::Model *cnn_m, unsigned start_here_dict_size, unsigned start_here_dim,
    unsigned pass_here_dict_size, unsigned pass_here_dim,
    unsigned end_here_dict_size, unsigned end_here_dim)
    :lexicon_feature_layer(cnn_m, start_here_dict_size, start_here_dim, pass_here_dict_size, pass_here_dim,
        end_here_dict_size, end_here_dim)
{}
CWSFeatureLayer::CWSFeatureLayer(cnn::Model *cnn_m, const CWSFeature &cws_feature)
    :lexicon_feature_layer(cnn_m, cws_feature.lexicon_feature)
{}
}
