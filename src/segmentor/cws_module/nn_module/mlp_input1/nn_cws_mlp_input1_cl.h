#ifndef SLNN_SEGMENTOR_CWS_MODULE_NN_MODULE_CWS_MLP_INPUT1_CL_H_
#define SLNN_SEGMENTOR_CWS_MODULE_NN_MODULE_CWS_MLP_INPUT1_CL_H_
#include "nn_cws_mlp_input1_abstract.h"
#include "utils/nn_utility.h"
namespace slnn{
namespace segmentor{
namespace nn_module{

class NnSegmentorInput1Cl : public NnSegmentorInput1Abstract
{
public:
    NnSegmentorInput1Cl(int argc, char **argv, unsigned seed) : NnSegmentorInput1Abstract(argc, argv, seed){}
    template <typename StructureParamT>
    void build_model_structure(const StructureParamT &param);
};

/************************************
 * Inline Implementation
 ************************************/

template <typename StructureParamT>
void NnSegmentorInput1Cl::build_model_structure(const StructureParamT &param)
{
    this->word_expr_layer.reset(new Index2ExprLayer(this->get_cnn_model(), param.corpus_token_dict_size,
        param.corpus_token_embedding_dim));
    this->window_expr_generate_layer.reset(new WindowExprGenerateLayer(this->get_cnn_model(), param.window_size,
        param.corpus_token_embedding_dim));
    std::vector<unsigned> mlp_hidden_dim_list;
    this->mlp_hidden_layer.reset(new MLPHiddenLayer(this->get_cnn_model(), param.mlp_input_dim,
        param.mlp_hidden_dim_list, param.mlp_dropout_rate, 
        utils::get_nonlinear_function_from_name(param.mlp_nonlinear_function_str)));
    this->output_layer.reset(new SimpleBareOutput(this->get_cnn_model(), param.mlp_hidden_dim_list.back(),
        param.output_dim));
}


} // end of namespace nn_module
} // end of namespace segmentor
} // end of namespace slnn




#endif