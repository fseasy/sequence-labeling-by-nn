#ifndef SLNN_SEGMENTER_CWS_MODULE_EXP_CWS_SPECIFIC_OUTPUT_LAYER_H_
#define SLNN_SEGMENTER_CWS_MODULE_EXP_CWS_SPECIFIC_OUTPUT_LAYER_H_
#include <memory>
#include <string>
#include "modelmodule/hyper_output_layers.h"
namespace slnn{
namespace segmenter{
namespace nn_module{
namespace experiment{

/**
 * This layers has the first abstract parent: BareOutputBase
 */

class SegmenterClassificationBareOutput : public SimpleBareOutput
{
public:
    SegmenterClassificationBareOutput(dynet::Model *m, unsigned input_dim, unsigned output_dim);
    void build_output(const std::vector<dynet::expr::Expression>& input_expr_seq, std::vector<Index>& out_pred_seq) override;
};

std::shared_ptr<BareOutputBase>
create_segmenter_output_layer(const std::string& layer_type, dynet::Model* dynet_model, unsigned input_dim, unsigned output_dim);

} // end of namespace experiment
} // end of namespace nn-module
} // end of namespace segmeter
} // end of namespace slnn




#endif